"""
Risk metrics engine for HiFi (P2-E4).

Adapter over QuantStats that computes portfolio and risk metrics from
OHLCVDataset inputs and returns typed RiskMetricsResult objects.

Library choice (DJ-011):
QuantStats was chosen over Pyfolio (deprecated since Quantopian's 2020
shutdown, known pandas incompatibilities) and custom numpy (would diverge
from community norms on annualisation conventions, Sharpe denominator, and
drawdown sign conventions). QuantStats is actively maintained, provides
standard metrics via a clean pandas API, and generates professional-grade
tear sheets reused directly in Phase 10 (Evaluation & Backtesting).

Sign conventions:
- max_drawdown_252d is stored as a POSITIVE float. A 30% drawdown → 0.30.
  QuantStats returns a negative value; we apply abs().
- var_95_20d is stored as a POSITIVE float representing the magnitude of
  the loss exceeded with 5% probability over 20 bars. QuantStats returns
  a positive VaR by convention (parametric, Gaussian assumption).
- hist_vol_* are annualised (252 trading-day convention).
- sharpe_252d uses the risk_free_rate passed to the engine; when 0.0 is
  used, the caller should be aware this is the excess-return-free Sharpe.

Beta computation:
beta = Cov(r_asset, r_bench) / Var(r_bench)
computed directly in numpy from aligned daily return series. Requires a
benchmark OHLCVDataset (typically SPY). Returns None when no benchmark
is provided or when insufficient aligned data is available.

Phase 2 note: All metrics use trailing windows ending at as_of_date.
The `window` parameter (default 252) sets the lookback for Sharpe and
max drawdown. Volatility is computed separately at 20d, 60d, and 252d
to give agents a multi-horizon view of risk.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np
import pandas as pd

from hifi.data.schemas import OHLCVBar, OHLCVDataset
from hifi.engines.types import RiskMetricsResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P2-E4-T1: Price series helper
# ---------------------------------------------------------------------------


def _to_price_series(
    bars: list[OHLCVBar], as_of_date: date, window: int
) -> pd.Series:
    """
    Convert OHLCVBars to a pandas Series of prices.

    Filters to bars on or before as_of_date, sorts ascending, slices to the
    most recent `window` bars. Uses adjusted_close when present; falls back
    to close.

    Returns an empty Series when no bars pass the filter.
    """
    filtered = sorted(
        [b for b in bars if b.date <= as_of_date], key=lambda b: b.date
    )
    if not filtered:
        return pd.Series(dtype=float)

    filtered = filtered[-window:]
    prices = pd.Series(
        [
            b.adjusted_close if b.adjusted_close is not None else b.close
            for b in filtered
        ],
        index=pd.DatetimeIndex([pd.Timestamp(b.date) for b in filtered]),
        dtype=float,
    )
    return prices


def _safe_float(val) -> float | None:
    """
    Coerce any scalar value to a clean float or None.

    Returns None for NaN, infinity, or non-numeric values.
    QuantStats functions may return numpy scalars, pandas scalars, or
    plain Python floats; this function normalises them all.
    """
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# P2-E4-T2: Historical volatility
# ---------------------------------------------------------------------------


def compute_hist_vol(
    bars: list[OHLCVBar],
    as_of_date: date,
    windows: list[int] | None = None,
) -> dict[int, float | None]:
    """
    Compute annualised historical volatility at multiple trailing windows.

    Parameters
    ----------
    bars : list[OHLCVBar]
    as_of_date : date
    windows : list[int], default [20, 60, 252]

    Returns
    -------
    dict[int, float | None]
        Mapping from window size to annualised volatility (or None when
        insufficient data is available for that window).
    """
    import quantstats as qs

    if windows is None:
        windows = [20, 60, 252]

    # Fetch enough history to cover the largest window plus one (for returns)
    max_window = max(windows) + 1
    all_prices = _to_price_series(bars, as_of_date, max_window * 2)

    results: dict[int, float | None] = {}
    for w in windows:
        if len(all_prices) < w + 1:
            results[w] = None
            continue
        prices_slice = all_prices.iloc[-(w + 1):]
        returns = prices_slice.pct_change().dropna()
        if returns.empty:
            results[w] = None
            continue
        try:
            vol = qs.stats.volatility(
                returns, periods=252, annualize=True, prepare_returns=False
            )
            results[w] = _safe_float(vol)
        except Exception:
            logger.exception("volatility computation failed for window=%d", w)
            results[w] = None

    return results


# ---------------------------------------------------------------------------
# P2-E4-T3: Beta
# ---------------------------------------------------------------------------


def compute_beta(
    stock_bars: list[OHLCVBar],
    benchmark_bars: list[OHLCVBar],
    as_of_date: date,
    window: int = 252,
) -> float | None:
    """
    Compute CAPM beta: Cov(r_asset, r_bench) / Var(r_bench).

    Parameters
    ----------
    stock_bars : list[OHLCVBar]
    benchmark_bars : list[OHLCVBar]
    as_of_date : date
    window : int, default 252

    Returns
    -------
    float | None
        Beta or None when insufficient aligned data is available,
        or when benchmark variance is zero.
    """
    asset_prices = _to_price_series(stock_bars, as_of_date, window + 1)
    bench_prices = _to_price_series(benchmark_bars, as_of_date, window + 1)

    if asset_prices.empty or bench_prices.empty:
        return None

    r_a = asset_prices.pct_change().dropna()
    r_b = bench_prices.pct_change().dropna()

    common_idx = r_a.index.intersection(r_b.index)
    if len(common_idx) < 2:
        return None

    a = r_a.loc[common_idx].to_numpy()
    b = r_b.loc[common_idx].to_numpy()
    bench_var = float(np.var(b, ddof=1))

    if bench_var == 0.0 or math.isnan(bench_var):
        return None

    cov = float(np.cov(a, b)[0, 1])
    return _safe_float(cov / bench_var)


# ---------------------------------------------------------------------------
# P2-E4-T4: Max drawdown
# ---------------------------------------------------------------------------


def compute_max_drawdown(
    bars: list[OHLCVBar], as_of_date: date, window: int = 252
) -> float | None:
    """
    Compute maximum drawdown over the trailing `window` bars.

    Uses QuantStats max_drawdown (which expects prices, not returns) and
    returns the drawdown as a positive float: a 30% peak-to-trough decline
    is stored as 0.30, not -0.30.

    Parameters
    ----------
    bars : list[OHLCVBar]
    as_of_date : date
    window : int, default 252

    Returns
    -------
    float | None
        Maximum drawdown [0, 1] or None when insufficient data.
    """
    import quantstats as qs

    prices = _to_price_series(bars, as_of_date, window)
    if len(prices) < 2:
        return None
    try:
        dd = qs.stats.max_drawdown(prices)
        dd_f = _safe_float(dd)
        return None if dd_f is None else abs(dd_f)
    except Exception:
        logger.exception("max_drawdown computation failed")
        return None


# ---------------------------------------------------------------------------
# P2-E4-T5: Sharpe ratio
# ---------------------------------------------------------------------------


def compute_sharpe(
    bars: list[OHLCVBar],
    as_of_date: date,
    risk_free_rate: float = 0.0,
    window: int = 252,
) -> float | None:
    """
    Compute the annualised Sharpe ratio over the trailing `window` bars.

    Parameters
    ----------
    bars : list[OHLCVBar]
    as_of_date : date
    risk_free_rate : float, default 0.0
        Annualised risk-free rate (decimal, e.g. 0.04 for 4%).
        When 0.0, the result is the annualised return-to-volatility ratio.
    window : int, default 252

    Returns
    -------
    float | None
        Annualised Sharpe ratio or None when std(returns) == 0 or
        insufficient data.
    """
    import quantstats as qs

    prices = _to_price_series(bars, as_of_date, window + 1)
    if len(prices) < 2:
        return None

    returns = prices.pct_change().dropna()
    if returns.empty or returns.std() == 0.0:
        return None

    try:
        sr = qs.stats.sharpe(
            returns, rf=risk_free_rate, periods=252, annualize=True,
        )
        return _safe_float(sr)
    except Exception:
        logger.exception("sharpe computation failed")
        return None


# ---------------------------------------------------------------------------
# P2-E4-T6: Value at Risk
# ---------------------------------------------------------------------------


def compute_var(
    bars: list[OHLCVBar],
    as_of_date: date,
    confidence: float = 0.95,
    window: int = 20,
) -> float | None:
    """
    Compute daily parametric VaR at the given confidence level.

    Uses the parametric (Gaussian) VaR formula via QuantStats:
        VaR = -(mu - z * sigma)
    where z is the z-score for (1 - confidence).

    The result is a positive float representing the magnitude of loss
    exceeded with (1 - confidence) probability over `window` daily returns.

    Parameters
    ----------
    bars : list[OHLCVBar]
    as_of_date : date
    confidence : float, default 0.95
    window : int, default 20

    Returns
    -------
    float | None
    """
    import quantstats as qs

    prices = _to_price_series(bars, as_of_date, window + 1)
    if len(prices) < window + 1:
        return None

    returns = prices.pct_change().dropna()
    if returns.empty:
        return None

    try:
        var = qs.stats.value_at_risk(
            returns, confidence=confidence, prepare_returns=False
        )
        # QuantStats returns a negative number (convention: loss is negative).
        # Our schema stores VaR as a positive magnitude.
        val = _safe_float(var)
        return None if val is None else abs(val)
    except Exception:
        logger.exception("value_at_risk computation failed")
        return None


# ---------------------------------------------------------------------------
# P2-E4-T7: Full risk metrics
# ---------------------------------------------------------------------------


def compute_risk_metrics(
    dataset: OHLCVDataset,
    as_of_date: date,
    benchmark: OHLCVDataset | None = None,
    risk_free_rate: float = 0.0,
    window: int = 252,
) -> RiskMetricsResult:
    """
    Compute the full suite of portfolio risk metrics for one ticker.

    Aggregates all individual risk functions into a single RiskMetricsResult.
    Each metric is computed independently; a failure in one does not affect
    the others.

    Parameters
    ----------
    dataset : OHLCVDataset
        Price data for the target ticker.
    as_of_date : date
        Latest date to include.
    benchmark : OHLCVDataset | None
        Benchmark price data (typically SPY) for beta computation.
        When None, beta is None.
    risk_free_rate : float, default 0.0
        Annualised risk-free rate for Sharpe computation.
    window : int, default 252
        Trailing window in trading days for Sharpe, max drawdown, and
        252d volatility.

    Returns
    -------
    RiskMetricsResult
    """
    if not dataset.bars:
        return RiskMetricsResult()

    vols = compute_hist_vol(dataset.bars, as_of_date)

    bench_bars = benchmark.bars if benchmark is not None else None
    beta = (
        compute_beta(dataset.bars, bench_bars, as_of_date, window)
        if bench_bars is not None
        else None
    )

    return RiskMetricsResult(
        hist_vol_20d=vols.get(20),
        hist_vol_60d=vols.get(60),
        hist_vol_252d=vols.get(252),
        beta=beta,
        max_drawdown_252d=compute_max_drawdown(dataset.bars, as_of_date, window),
        sharpe_252d=compute_sharpe(
            dataset.bars, as_of_date, risk_free_rate, window
        ),
        var_95_20d=compute_var(dataset.bars, as_of_date, window=20),
    )
