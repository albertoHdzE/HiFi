"""
Technical indicator engine for HiFi (P2-E3).

Six indicators implemented in numpy/pandas with no external TA library.
This module is the Phase 2 baseline. Phase 8+ will migrate to a dedicated
venvs/ta/ MCP server running pandas-ta in an isolated environment (DJ-010).

All computations are pure functions with no I/O or side effects.

Indicators and primary sources:
- SMA(n): simple rolling mean. Standard.
- EMA(n): exponentially weighted mean with span n, adjust=False. Standard.
- RSI(n=14): Relative Strength Index. Wilder, J.W. (1978). New Concepts in
  Technical Trading Systems. Trend Research.
- MACD (12/26/9): Moving Average Convergence Divergence.
  Appel, G. (1979). The Moving Average Convergence-Divergence Trading Method.
  Advanced Version. Signalert Corporation.
- Bollinger Bands (20, ±2σ): Bollinger, J. (1992). Using Bollinger Bands.
  Stocks and Commodities Magazine, 10(2), 47-51.
- ATR(n=14): Average True Range. Wilder (1978), ibid.

Design decisions:
- adjusted_close is preferred as the price input when present in a bar.
  Close is used as fallback. This matters for return-based indicators when
  dividends or splits are present in the data window (see P1-E2 for why
  auto_adjust=False was chosen).
- EWM min_periods=n is set for RSI and ATR so early-period values (where
  the exponential weights have not yet converged to their steady-state) are
  returned as NaN, which _last() maps to None. This prevents the first few
  bars of a newly-started series from producing misleadingly confident values.
- All None returns are explicit contracts, not silent failures. An agent
  that receives None for RSI must interpret it as "insufficient data."
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from hifi.data.schemas import OHLCVBar
from hifi.engines.types import TechnicalIndicatorsResult

# ---------------------------------------------------------------------------
# Minimum bar counts for each indicator
# ---------------------------------------------------------------------------

# RSI and ATR: EWM with n=14, min_periods=14, plus 1 bar for diff/shift
_RSI_MIN_BARS: int = 15
_ATR_MIN_BARS: int = 15
# MACD: EMA(26) + EMA(9) of MACD; 26+9=35 bars for signal line to be meaningful
_MACD_MIN_BARS: int = 35


# ---------------------------------------------------------------------------
# P2-E3-T1: Helper — OHLCVBar list to OHLCV DataFrame
# ---------------------------------------------------------------------------


def _to_dataframe(bars: list[OHLCVBar], as_of_date: date) -> pd.DataFrame:
    """
    Convert a list of OHLCVBar objects to an OHLCV DataFrame.

    Filters to bars on or before as_of_date, sorts ascending by date,
    and adds a 'price' column: adjusted_close if present, otherwise close.

    Parameters
    ----------
    bars : list[OHLCVBar]
    as_of_date : date

    Returns
    -------
    pd.DataFrame
        Indexed by date. Columns: open, high, low, close, volume, price.
        Empty DataFrame when no bars pass the filter.
    """
    filtered = sorted(
        [b for b in bars if b.date <= as_of_date],
        key=lambda b: b.date,
    )
    if not filtered:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "adjusted_close", "price"]
        )

    df = pd.DataFrame(
        {
            "date": [b.date for b in filtered],
            "open": [b.open for b in filtered],
            "high": [b.high for b in filtered],
            "low": [b.low for b in filtered],
            "close": [b.close for b in filtered],
            "volume": [b.volume for b in filtered],
            "adjusted_close": [b.adjusted_close for b in filtered],
        }
    )
    df = df.set_index("date")

    # Price column: prefer adjusted_close (dividend-adjusted) if available
    df["price"] = df["adjusted_close"].where(
        df["adjusted_close"].notna(), df["close"]
    )
    return df


# ---------------------------------------------------------------------------
# P2-E3-T2: Helper — extract last non-NaN scalar
# ---------------------------------------------------------------------------


def _last(series: pd.Series) -> float | None:
    """
    Return the last non-NaN value from a pandas Series as a float.

    Returns None if the series is empty or all values are NaN.
    """
    valid = series.dropna()
    if valid.empty:
        return None
    val = float(valid.iloc[-1])
    return None if math.isnan(val) else val


# ---------------------------------------------------------------------------
# Internal indicator functions
# ---------------------------------------------------------------------------


def _compute_sma(price: pd.Series, window: int) -> pd.Series:
    """SMA(window). NaN for the first window-1 bars (rolling min_periods=window)."""
    return price.rolling(window=window).mean()


def _compute_ema(price: pd.Series, window: int) -> pd.Series:
    """EMA(window) with Pandas EWM span convention (adjust=False)."""
    return price.ewm(span=window, adjust=False).mean()


def _compute_rsi(price: pd.Series, n: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothed average (Wilder 1978).

    EWM with alpha=1/n and min_periods=n ensures the first n-1 bars
    return NaN; only bars with n or more observations in the exponential
    window contribute a meaningful estimate.

    RSI is in [0, 100] for a valid series:
    - Approaches 100 for a monotonically rising series (avg_loss → 0).
    - Approaches 0 for a monotonically falling series (avg_gain → 0).
    - Division by zero (avg_loss == 0) is guarded by replacing 0 with NaN,
      which propagates to a NaN RSI that _last() maps to None.
    """
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    # Handle the division by zero cases explicitly with numpy:
    #   avg_loss == 0, avg_gain  > 0 → RS = inf → RSI = 100 (fully bullish)
    #   avg_loss == 0, avg_gain == 0 → RS = NaN → RSI = NaN (no movement)
    #   avg_loss  > 0               → RS = avg_gain / avg_loss (standard)
    import numpy as _np
    g = avg_gain.to_numpy()
    loss_vals = avg_loss.to_numpy()
    with _np.errstate(divide="ignore", invalid="ignore"):
        rs_vals = _np.where(
            loss_vals == 0.0,
            _np.where(g == 0.0, _np.nan, _np.inf),
            g / loss_vals,
        )
    rs = pd.Series(rs_vals, index=price.index)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd(
    price: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD, signal line, histogram (Appel 1979).

    MACD   = EMA(12) - EMA(26)
    Signal = EMA(9) of MACD
    Hist   = MACD - Signal
    """
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _compute_bollinger(
    price: pd.Series, window: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: mid ± 2 standard deviations (Bollinger 1992).

    Uses sample std (ddof=1) for consistency with most TA implementations.
    Returns NaN for bars with fewer than window observations.
    """
    mid = price.rolling(window=window).mean()
    std = price.rolling(window=window).std(ddof=1)
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    return upper, mid, lower


def _compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Average True Range (Wilder 1978).

    TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = EWM(alpha=1/n, min_periods=n) of TR

    The first bar has no prev_close; its True Range equals High-Low since
    pandas max(axis=1) skips NaN by default.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["price"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------------------
# P2-E3-T3: Public interface
# ---------------------------------------------------------------------------


def compute_technical_indicators(
    bars: list[OHLCVBar],
    as_of_date: date,
    window: int = 20,
) -> TechnicalIndicatorsResult:
    """
    Compute six technical indicators from OHLCVBars as of a given date.

    Uses adjusted_close when available; falls back to close. All indicators
    return None when the series length is below the indicator's minimum
    required window.

    Parameters
    ----------
    bars : list[OHLCVBar]
        Price bars for the ticker. May contain bars after as_of_date;
        these are excluded.
    as_of_date : date
        Latest date to include. Ensures point-in-time correctness.
    window : int, default 20
        Window length for SMA, EMA, and Bollinger Bands.
        RSI (14), MACD (12/26/9), and ATR (14) use standard fixed windows.

    Returns
    -------
    TechnicalIndicatorsResult
        Fields are None when insufficient data is available.
    """
    df = _to_dataframe(bars, as_of_date)
    n = len(df)
    if n < 2:
        return TechnicalIndicatorsResult()

    price = df["price"]

    # SMA and EMA: require at least `window` bars
    if n >= window:
        sma = _last(_compute_sma(price, window))
        ema = _last(_compute_ema(price, window))
        bb_upper_s, bb_mid_s, bb_lower_s = _compute_bollinger(price, window)
        bb_upper = _last(bb_upper_s)
        bb_mid = _last(bb_mid_s)
        bb_lower = _last(bb_lower_s)
    else:
        sma = ema = bb_upper = bb_mid = bb_lower = None

    # RSI: Wilder (1978) standard period = 14; need 15 bars for first valid value
    rsi = _last(_compute_rsi(price)) if n >= _RSI_MIN_BARS else None

    # MACD: 35 bars for both EMA(26) and EMA(9) of MACD to be meaningful
    if n >= _MACD_MIN_BARS:
        macd_s, signal_s, hist_s = _compute_macd(price)
        macd = _last(macd_s)
        macd_signal = _last(signal_s)
        macd_hist = _last(hist_s)
    else:
        macd = macd_signal = macd_hist = None

    # ATR: Wilder (1978) standard period = 14; need 15 bars (14 + initial shift)
    atr = _last(_compute_atr(df)) if n >= _ATR_MIN_BARS else None

    return TechnicalIndicatorsResult(
        sma=sma,
        ema=ema,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        bb_upper=bb_upper,
        bb_mid=bb_mid,
        bb_lower=bb_lower,
        atr=atr,
    )
