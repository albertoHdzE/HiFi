"""
QuantStats portfolio tear sheets for Phase 10 method comparison (P10-E1, DJ-045, DJ-050).

Converts labeled MethodDecisionRecords to a daily strategy returns series and
computes QuantStats-backed portfolio metrics per aggregation method.

Strategy construction (DJ-045):
  For each (ticker, analysis_date, method, decision) quadruple, assign a position:
    +1 if decision == "Buy"
    -1 if decision == "Sell"
     0 if decision == "Hold"

  For each trading day t in [t0, t0+horizon_days), the strategy return is:
    strategy_return(t) = position * actual_daily_return(t)

  The portfolio return on day t is the equal-weight mean of position returns
  across all tickers active on that day. Days between quarter-end windows are
  filled with 0.0 (out of market).

Documented limitations (DJ-045):
  - Quarterly rebalancing ignores intra-period signal updates.
  - Equal weighting ignores position sizing, transaction costs, and slippage.
  - With 20 quarter-ends and 15 tickers, the total history spans ~14,400 daily
    return observations. Annual return estimates will have wide confidence intervals.
  - The CAGR and Sharpe annualisation assumes 252 trading days/year. The sparse
    quarterly signal structure makes this an approximation, not a precise estimate.

Output artifacts (DJ-050):
  TearsheetSummary objects are saved as JSON in data/tearsheets/.
  HTML reports are deferred to Phase 14 (Paper Trading analytics).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import quantstats as qs
from pydantic import BaseModel

from hifi.collective.schemas import MethodDecisionRecord

_METHODS = frozenset(
    {"confidence_weighted", "majority", "performance_weighted", "contrarian_adjusted"}
)


class TearsheetSummary(BaseModel):
    """
    Portfolio-level performance summary for one aggregation method (DJ-050).

    All float metrics are rounded to 4 decimal places. None indicates the metric
    could not be computed (e.g., zero-variance returns → undefined Sharpe).

    sharpe_annual: annualised Sharpe ratio (rf=0, periods=252)
    sortino_annual: annualised Sortino ratio (rf=0, periods=252)
    max_drawdown: maximum peak-to-trough drawdown (negative float, e.g. -0.15)
    calmar: abs(CAGR) / abs(max_drawdown); None if max_drawdown == 0
    cagr: compound annual growth rate
    win_rate: fraction of trading days with positive return (signal periods only)
    avg_return_per_period: mean realised return per 60-day signal period
    n_periods: number of (ticker, analysis_date) signal periods contributing
    generated_at: ISO 8601 timestamp
    """

    method_name: str
    tickers: list[str]
    n_periods: int
    sharpe_annual: float | None
    sortino_annual: float | None
    max_drawdown: float | None
    calmar: float | None
    cagr: float | None
    win_rate: float | None
    avg_return_per_period: float
    generated_at: str


# ---------------------------------------------------------------------------
# Strategy returns construction
# ---------------------------------------------------------------------------


def _load_daily_returns(
    ohlcv_map: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """
    Convert raw OHLCV DataFrames to daily return Series per ticker.

    Accepts DataFrames with either 'adjusted_close'/'close' (HiFi format) or
    'Adj Close'/'Close' (raw yfinance format). The DatetimeIndex must already be set.
    """
    daily: dict[str, pd.Series] = {}
    for ticker, df in ohlcv_map.items():
        col = next(
            (c for c in ("adjusted_close", "Adj Close", "close", "Close") if c in df.columns),
            None,
        )
        if col is None:
            raise ValueError(f"No close price column found for {ticker}")
        daily[ticker] = df[col].pct_change()
    return daily


def build_strategy_returns(
    method_records: list[MethodDecisionRecord],
    ohlcv_map: dict[str, pd.DataFrame],
    horizon_days: int = 60,
) -> pd.Series:
    """
    Convert labeled MethodDecisionRecords to a daily strategy returns Series.

    Parameters
    ----------
    method_records : list[MethodDecisionRecord]
        Records for one method. May contain unlabeled records (outcome_correct=None);
        all records contribute to the strategy regardless of label status.
        Only records matching horizon_days are used.
    ohlcv_map : dict[str, pd.DataFrame]
        {ticker: OHLCV DataFrame with DatetimeIndex}. Must cover at least the
        date range of the method_records.
    horizon_days : int
        Number of trading days each signal is held. Must match the records' horizon.

    Returns
    -------
    pd.Series
        Daily portfolio returns with DatetimeIndex. 0.0 on days outside all signal
        windows (gaps between quarter-end periods). Empty Series when ohlcv_map is
        empty or method_records is empty.

    Design (DJ-045):
        Position overlap: when two quarter-end windows for the same ticker overlap
        (rare with 60-day horizon and quarterly rebalancing), the later window
        takes priority for the overlapping days.
    """
    if not method_records or not ohlcv_map:
        return pd.Series(dtype=float)

    filtered = [r for r in method_records if r.horizon_days == horizon_days]
    if not filtered:
        return pd.Series(dtype=float)

    daily_returns = _load_daily_returns(ohlcv_map)

    def _position(decision: str) -> float:
        if decision == "Buy":
            return 1.0
        if decision == "Sell":
            return -1.0
        return 0.0

    # Build a per-ticker daily strategy return Series, then average across tickers
    # Each record defines a window of horizon_days trading days starting from
    # the first available trading day on/after analysis_date.
    ticker_series: list[pd.Series] = []

    for ticker in {r.ticker for r in filtered}:
        ticker_records = [r for r in filtered if r.ticker == ticker]
        if ticker not in daily_returns:
            continue

        dr = daily_returns[ticker].sort_index()
        # Build position series: default 0.0 (out of market)
        position = pd.Series(0.0, index=dr.index)

        # Apply windows; later windows override earlier ones (overlap rule)
        for record in sorted(ticker_records, key=lambda r: r.analysis_date):
            t0_target = pd.Timestamp(record.analysis_date)
            # Find first trading day on or after analysis_date
            start_candidates = dr.index[dr.index >= t0_target]
            if start_candidates.empty:
                continue
            t0_idx_loc = dr.index.get_loc(start_candidates[0])
            t1_idx_loc = t0_idx_loc + horizon_days
            if t1_idx_loc > len(dr):
                t1_idx_loc = len(dr)
            window_dates = dr.index[t0_idx_loc:t1_idx_loc]
            position[window_dates] = _position(record.decision)

        # Strategy return = position × actual daily return.
        # fillna(0.0): first day of pct_change is NaN (no prior price); treat as 0
        # since position=0 (Hold) or as a missed day for directional positions.
        strat = (position * dr).fillna(0.0)
        ticker_series.append(strat)

    if not ticker_series:
        return pd.Series(dtype=float)

    # Equal-weight portfolio: mean across tickers per day
    portfolio = pd.concat(ticker_series, axis=1).mean(axis=1)
    return portfolio.sort_index()


# ---------------------------------------------------------------------------
# Tear sheet computation
# ---------------------------------------------------------------------------


def _safe_stat(fn, *args, **kwargs) -> float | None:
    """Call a QuantStats stat function; return None on any exception or non-finite result."""
    import math
    try:
        v = float(fn(*args, **kwargs))
        return round(v, 4) if math.isfinite(v) else None
    except Exception:
        return None


def compute_tearsheet(
    method_records: list[MethodDecisionRecord],
    ohlcv_map: dict[str, pd.DataFrame],
    method_name: str,
    horizon_days: int = 60,
) -> TearsheetSummary:
    """
    Compute a TearsheetSummary for one aggregation method.

    Parameters
    ----------
    method_records : list[MethodDecisionRecord]
        All labeled records. Records for other methods are ignored via filtering.
    ohlcv_map : dict[str, pd.DataFrame]
        {ticker: OHLCV DataFrame with DatetimeIndex}.
    method_name : str
        One of the four canonical method keys.
    horizon_days : int
        Evaluation horizon. Passed to build_strategy_returns().

    Returns
    -------
    TearsheetSummary
        Portfolio metrics. Float fields are None when computation is undefined
        (e.g., zero-variance returns, insufficient history for calmar).
    """
    if method_name not in _METHODS:
        raise ValueError(
            f"method_name must be one of {sorted(_METHODS)}, got {method_name!r}"
        )

    filtered = [r for r in method_records if r.method_name == method_name]
    if not filtered:
        raise ValueError(f"No records found for method {method_name!r}")

    tickers = sorted({r.ticker for r in filtered})
    n_periods = len({(r.ticker, r.analysis_date) for r in filtered})

    returns = build_strategy_returns(filtered, ohlcv_map, horizon_days=horizon_days)

    # Average realised return per 60-day signal period (simple, not annualised)
    # = sum of strategy returns within all signal windows / number of signal periods
    n_signal_periods = len(filtered)
    avg_return_per_period = 0.0
    if n_signal_periods > 0 and not returns.empty:
        avg_return_per_period = round(float(returns.sum()) / n_signal_periods, 6)

    # Guard: need non-trivial variance for Sharpe/Sortino
    nonzero = returns[returns != 0.0]
    if nonzero.empty or nonzero.std() == 0:
        return TearsheetSummary(
            method_name=method_name,
            tickers=tickers,
            n_periods=n_periods,
            sharpe_annual=None,
            sortino_annual=None,
            max_drawdown=None,
            calmar=None,
            cagr=None,
            win_rate=None,
            avg_return_per_period=avg_return_per_period,
            generated_at=datetime.now(tz=UTC).isoformat(),
        )

    # Drop leading NaN from pct_change
    returns_clean = returns.dropna()

    sharpe = _safe_stat(qs.stats.sharpe, returns_clean, periods=252)
    sortino = _safe_stat(qs.stats.sortino, returns_clean, periods=252)
    max_dd = _safe_stat(qs.stats.max_drawdown, returns_clean)
    cagr = _safe_stat(qs.stats.cagr, returns_clean)

    # Calmar = CAGR / |max_drawdown| — undefined when max_drawdown == 0
    calmar: float | None = None
    if cagr is not None and max_dd is not None and max_dd != 0.0:
        import math
        raw_calmar = cagr / abs(max_dd)
        calmar = round(raw_calmar, 4) if math.isfinite(raw_calmar) else None

    win_rate = _safe_stat(qs.stats.win_rate, returns_clean)

    return TearsheetSummary(
        method_name=method_name,
        tickers=tickers,
        n_periods=n_periods,
        sharpe_annual=sharpe,
        sortino_annual=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        cagr=cagr,
        win_rate=win_rate,
        avg_return_per_period=avg_return_per_period,
        generated_at=datetime.now(tz=UTC).isoformat(),
    )


def compute_all_tearsheets(
    method_records: list[MethodDecisionRecord],
    ohlcv_map: dict[str, pd.DataFrame],
    horizon_days: int = 60,
) -> dict[str, TearsheetSummary]:
    """
    Compute TearsheetSummary for all four canonical methods.

    Returns {method_name: TearsheetSummary}. Methods absent from method_records
    are omitted from the result.
    """
    methods_present = {r.method_name for r in method_records}
    return {
        m: compute_tearsheet(method_records, ohlcv_map, m, horizon_days)
        for m in sorted(_METHODS)
        if m in methods_present
    }
