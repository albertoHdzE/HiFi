"""
Regime classifier for HiFi Phase 14+ (E5-T1 / E2-T4, DJ-089b).

Deterministic classification of market regime at a given date using SPY
price series and macro indicator series (Fed Funds Rate, optionally VIX).

Regimes
-------
bull_low_vol  : SPY 52w return > 10%  AND VIX 20d avg < 20
bear_high_vol : SPY 52w return < -10% AND VIX 20d avg > 30
rate_shock    : Fed Funds Rate delta  > 2.0pp over trailing 180 calendar days
recovery      : SPY 52w return > 20%  AND prior-year (y-2 → y-1) 52w return < -10%
neutral       : none of the above

Evaluation order: rate_shock → bear_high_vol → recovery → bull_low_vol → neutral.
rate_shock is checked first because aggressive hiking cycles can occur during
bull markets (e.g. 2022); bear_high_vol would also fire then, but rate_shock
is more specific and actionable for the walk-forward strategy.

Known calibration dates (from Phase 14 E5-T1 spec):
  2020-03-16 → bear_high_vol (COVID crash)
  2022-06-30 → rate_shock    (Fed hiking cycle peak)
  2021-06-30 → bull_low_vol  (post-COVID recovery plateau)

VIX fallback
------------
If ``macro_series`` does not contain a 'vix' column, SPY 20-day realised
annualised volatility is used as a proxy: < 0.15 ≈ VIX < 20 (low vol),
> 0.30 ≈ VIX > 30 (high vol).
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

RegimeLabel = Literal["bull_low_vol", "bear_high_vol", "rate_shock", "recovery", "neutral"]

# Thresholds (all from DJ-089b spec)
_BULL_RETURN = 0.10      # 52w return > 10%
_BEAR_RETURN = -0.10     # 52w return < -10%
_RECOVERY_RETURN = 0.20  # 52w return > 20% following a bear period
_VIX_LOW = 20.0          # 20d VIX avg < 20 → low vol
_VIX_HIGH = 30.0         # 20d VIX avg > 30 → high vol
_RATE_SHOCK_PP = 2.0     # Fed Funds Rate delta > 2.0pp over 180 calendar days
_RATE_WINDOW_DAYS = 180  # calendar days for rate delta window
_TRADING_YEAR = 252      # trading days in 52 weeks
_VOL_WINDOW = 20         # trading days for realised vol
_VOL_LOW = 0.15          # annualised realised vol proxy for VIX < 20
_VOL_HIGH = 0.30         # annualised realised vol proxy for VIX > 30


def _spy_52w_return(ohlcv: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    """Compute 252-day (52-week) return for SPY ending on or before as_of."""
    col = _close_col(ohlcv)
    data = ohlcv[col].dropna()
    data = data[data.index <= as_of]
    if len(data) < _TRADING_YEAR:
        return None
    price_now = data.iloc[-1]
    price_year_ago = data.iloc[-_TRADING_YEAR]
    if price_year_ago == 0:
        return None
    return float((price_now - price_year_ago) / price_year_ago)


def _close_col(df: pd.DataFrame) -> str:
    """Return the name of the close-price column (case-insensitive)."""
    for c in df.columns:
        if c.lower() == "close":
            return c
    # fallback: use first column
    return df.columns[0]


def _vix_value(
    ohlcv: pd.DataFrame,
    macro: pd.DataFrame,
    as_of: pd.Timestamp,
) -> float | None:
    """
    Return 20-day average VIX (from macro series) or SPY realised vol proxy.

    Returns None when insufficient data exists.
    """
    # Try explicit VIX column in macro series
    for col in macro.columns:
        if col.lower() == "vix":
            vix = macro[col].dropna()
            vix = vix[vix.index <= as_of]
            if len(vix) >= _VOL_WINDOW:
                return float(vix.iloc[-_VOL_WINDOW:].mean())
            break

    # Fallback: SPY 20d realised vol (annualised), scaled to VIX-like units × 100
    spy_col = _close_col(ohlcv)
    prices = ohlcv[spy_col].dropna()
    prices = prices[prices.index <= as_of]
    if len(prices) < _VOL_WINDOW + 1:
        return None
    returns = prices.pct_change().dropna()
    rv = float(returns.iloc[-_VOL_WINDOW:].std() * (252 ** 0.5))
    # Map annualised sigma to VIX-like scale: VIX ≈ σ × 100
    return rv * 100.0


def _rate_delta(macro: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    """
    Fed Funds Rate change (pp) over the trailing 180 calendar days.

    Returns None when the macro series does not span the window.
    """
    # Try common column names for Fed Funds Rate
    rate_col = None
    for col in macro.columns:
        if col.lower() in ("fed_funds_rate", "fedfunds", "ffr", "rate"):
            rate_col = col
            break
    if rate_col is None:
        return None

    rates = macro[rate_col].dropna()
    rates = rates[rates.index <= as_of]
    if rates.empty:
        return None

    cutoff = as_of - pd.Timedelta(days=_RATE_WINDOW_DAYS)
    old_rates = rates[rates.index >= cutoff]
    if old_rates.empty:
        return None

    return float(rates.iloc[-1]) - float(old_rates.iloc[0])


def classify_regime(
    date: str,
    ohlcv_series: pd.DataFrame,
    macro_series: pd.DataFrame,
) -> RegimeLabel:
    """
    Classify market regime at ``date`` using SPY OHLCV and macro indicators.

    Parameters
    ----------
    date : str
        ISO 8601 date string (e.g. "2022-06-30").
    ohlcv_series : pd.DataFrame
        SPY daily OHLCV with a DatetimeIndex and at least a 'Close' column.
        Must span at least 2 years before ``date`` for reliable classification.
    macro_series : pd.DataFrame
        Daily macro indicators with a DatetimeIndex.
        Required: 'fed_funds_rate' (or 'fedfunds' / 'ffr' / 'rate') column.
        Optional: 'vix' column for direct VIX-based vol classification.

    Returns
    -------
    RegimeLabel
        One of: "bull_low_vol", "bear_high_vol", "rate_shock", "recovery", "neutral".
    """
    as_of = pd.Timestamp(date)

    # 1. Rate shock (most distinctive; check before equity-based regimes)
    rd = _rate_delta(macro_series, as_of)
    if rd is not None and rd > _RATE_SHOCK_PP:
        return "rate_shock"

    # 2. Equity-based regimes
    ret_1y = _spy_52w_return(ohlcv_series, as_of)
    vix_val = _vix_value(ohlcv_series, macro_series, as_of)

    if ret_1y is not None and ret_1y < _BEAR_RETURN and (vix_val is None or vix_val > _VIX_HIGH):
        return "bear_high_vol"

    if ret_1y is not None and ret_1y > _RECOVERY_RETURN:
        # Recovery: verify prior year was bearish
        prior_as_of = as_of - pd.Timedelta(days=365)
        ret_prior = _spy_52w_return(ohlcv_series, prior_as_of)
        if ret_prior is not None and ret_prior < _BEAR_RETURN:
            return "recovery"

    if ret_1y is not None and ret_1y > _BULL_RETURN and (vix_val is None or vix_val < _VIX_LOW):
        return "bull_low_vol"

    return "neutral"
