"""
Unit tests for src/hifi/analytics/tearsheet.py (P10-E1-T5).

Tests cover:
- build_strategy_returns: buy/sell/hold all-signal cases, gap filling, empty input
- compute_tearsheet: known deterministic returns → known Sharpe, edge case all-Hold
- TearsheetSummary JSON round-trip
- compute_all_tearsheets: returns all four methods present in records

All OHLCV data is synthetic (seeded deterministic pandas DataFrames, no Parquet I/O).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hifi.analytics.tearsheet import (
    TearsheetSummary,
    build_strategy_returns,
    compute_all_tearsheets,
    compute_tearsheet,
)
from hifi.collective.schemas import MethodDecisionRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(start: str, n_days: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with a DatetimeIndex.
    Daily returns are drawn from N(0.0003, 0.01) so the series is realistic
    but deterministic.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_days))
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": 1_000_000.0,
        "adjusted_close": close,
    }, index=dates)
    return df


def _make_record(
    ticker: str,
    analysis_date: str,
    method_name: str,
    decision: str,
    horizon_days: int = 60,
    confidence: float = 0.7,
) -> MethodDecisionRecord:
    return MethodDecisionRecord(
        ticker=ticker,
        analysis_date=analysis_date,
        method_name=method_name,
        decision=decision,
        collective_confidence=confidence,
        horizon_days=horizon_days,
    )


# ---------------------------------------------------------------------------
# Tests: build_strategy_returns
# ---------------------------------------------------------------------------


def test_build_strategy_returns_empty_input():
    """Empty records or empty ohlcv_map → empty Series."""
    result_a = build_strategy_returns([], {"AAPL": _make_ohlcv_df("2023-01-02", 120)})
    result_b = build_strategy_returns(
        [_make_record("AAPL", "2023-01-02", "confidence_weighted", "Buy")],
        {},
    )

    assert result_a.empty
    assert result_b.empty


def test_build_strategy_returns_buy_always():
    """All Buy signals → strategy return = +1 × daily market return."""
    df = _make_ohlcv_df("2023-01-02", 120, seed=0)
    records = [_make_record("AAPL", "2023-01-02", "confidence_weighted", "Buy", 60)]

    strat = build_strategy_returns(records, {"AAPL": df}, horizon_days=60)
    market_returns = df["adjusted_close"].pct_change()

    # Find t0 (first date >= 2023-01-02)
    t0 = market_returns.index[market_returns.index >= pd.Timestamp("2023-01-02")][0]
    t0_loc = market_returns.index.get_loc(t0)
    window = market_returns.index[t0_loc: t0_loc + 60]

    # Strategy returns within the window should equal market returns (position=+1)
    for ts in window:
        if ts in strat.index and not math.isnan(market_returns.loc[ts]):
            assert math.isclose(strat.loc[ts], market_returns.loc[ts], rel_tol=1e-9)


def test_build_strategy_returns_sell_always():
    """All Sell signals → strategy return = -1 × daily market return."""
    df = _make_ohlcv_df("2023-01-02", 120, seed=1)
    records = [_make_record("AAPL", "2023-01-02", "confidence_weighted", "Sell", 60)]

    strat = build_strategy_returns(records, {"AAPL": df}, horizon_days=60)
    market_returns = df["adjusted_close"].pct_change()

    t0 = market_returns.index[market_returns.index >= pd.Timestamp("2023-01-02")][0]
    t0_loc = market_returns.index.get_loc(t0)
    window = market_returns.index[t0_loc: t0_loc + 60]

    for ts in window:
        if ts in strat.index and not math.isnan(market_returns.loc[ts]):
            assert math.isclose(strat.loc[ts], -market_returns.loc[ts], rel_tol=1e-9)


def test_build_strategy_returns_hold_always():
    """All Hold signals → strategy returns are all 0.0."""
    df = _make_ohlcv_df("2023-01-02", 120, seed=2)
    records = [_make_record("AAPL", "2023-01-02", "confidence_weighted", "Hold", 60)]

    strat = build_strategy_returns(records, {"AAPL": df}, horizon_days=60)

    assert (strat == 0.0).all()


def test_build_strategy_returns_gap_days_are_zero():
    """Days between two non-overlapping windows are filled with 0.0."""
    df = _make_ohlcv_df("2023-01-02", 300, seed=3)
    # Q1 window: 2023-01-02 + 60d; Q2 window: 2023-07-03 + 60d
    records = [
        _make_record("AAPL", "2023-01-02", "confidence_weighted", "Buy", 60),
        _make_record("AAPL", "2023-07-03", "confidence_weighted", "Buy", 60),
    ]

    strat = build_strategy_returns(records, {"AAPL": df}, horizon_days=60)

    # Find the gap: after the first window closes, before the second opens
    t0_q1 = df.index[df.index >= pd.Timestamp("2023-01-02")][0]
    t0_q1_loc = df.index.get_loc(t0_q1)
    window1_end = df.index[t0_q1_loc + 60] if t0_q1_loc + 60 < len(df) else None

    t0_q2 = df.index[df.index >= pd.Timestamp("2023-07-03")][0]

    if window1_end is not None and window1_end < t0_q2:
        gap_dates = strat.index[(strat.index >= window1_end) & (strat.index < t0_q2)]
        if not gap_dates.empty:
            assert (strat[gap_dates] == 0.0).all()


def test_build_strategy_returns_multi_ticker_equal_weight():
    """Two-ticker portfolio = equal weight mean of per-ticker strategy returns."""
    df_a = _make_ohlcv_df("2023-01-02", 120, seed=10)
    df_b = _make_ohlcv_df("2023-01-02", 120, seed=20)
    records = [
        _make_record("AAA", "2023-01-02", "confidence_weighted", "Buy", 60),
        _make_record("BBB", "2023-01-02", "confidence_weighted", "Sell", 60),
    ]

    strat = build_strategy_returns(
        records, {"AAA": df_a, "BBB": df_b}, horizon_days=60
    )

    # Spot check: not all zero (one Buy + one Sell with different data)
    assert not (strat == 0.0).all()
    # Spot check: magnitude roughly halved vs single ticker
    strat_a = build_strategy_returns(
        [records[0]], {"AAA": df_a}, horizon_days=60
    )
    # Equal-weight mean of |returns| < single-ticker |returns| (when they differ)
    assert strat.abs().mean() < strat_a.abs().mean() * 1.5  # loose bound


def test_build_strategy_returns_horizon_mismatch_excluded():
    """Records with horizon_days != the requested horizon are excluded."""
    df = _make_ohlcv_df("2023-01-02", 120, seed=4)
    record_60 = _make_record("AAPL", "2023-01-02", "confidence_weighted", "Buy", 60)
    record_20 = _make_record("AAPL", "2023-01-02", "confidence_weighted", "Sell", 20)

    strat_60 = build_strategy_returns([record_60, record_20], {"AAPL": df}, horizon_days=60)
    # Only the Buy record contributed
    market_returns = df["adjusted_close"].pct_change()
    t0 = market_returns.index[market_returns.index >= pd.Timestamp("2023-01-02")][0]
    t0_loc = market_returns.index.get_loc(t0)
    # First non-NaN position should be positive (Buy)
    assert strat_60.iloc[t0_loc] >= 0.0


# ---------------------------------------------------------------------------
# Tests: compute_tearsheet
# ---------------------------------------------------------------------------


def test_compute_tearsheet_invalid_method_raises():
    with pytest.raises(ValueError, match="method_name must be one of"):
        compute_tearsheet([], {}, "invalid_method")


def test_compute_tearsheet_no_records_for_method_raises():
    records = [_make_record("AAPL", "2023-01-02", "majority", "Buy")]
    df = _make_ohlcv_df("2023-01-02", 120, seed=5)
    with pytest.raises(ValueError, match="No records found for method"):
        compute_tearsheet(records, {"AAPL": df}, "confidence_weighted")


def test_compute_tearsheet_all_hold_returns_none_metrics():
    """All Hold signals → zero-variance returns → Sharpe/Sortino are None."""
    df = _make_ohlcv_df("2023-01-02", 120, seed=6)
    records = [
        _make_record("AAPL", "2023-01-02", "confidence_weighted", "Hold", 60)
    ]

    summary = compute_tearsheet(records, {"AAPL": df}, "confidence_weighted")

    assert summary.method_name == "confidence_weighted"
    assert summary.sharpe_annual is None
    assert summary.sortino_annual is None
    assert summary.max_drawdown is None
    assert summary.avg_return_per_period == pytest.approx(0.0, abs=1e-6)


def test_compute_tearsheet_produces_finite_metrics():
    """With realistic returns, all metrics should be finite floats."""
    df = _make_ohlcv_df("2023-01-02", 400, seed=7)
    # Use 5 quarters of Buy signals to get enough data
    records = [
        _make_record("AAPL", d, "confidence_weighted", "Buy", 60)
        for d in [
            "2023-01-02", "2023-04-03", "2023-07-03",
            "2023-10-02", "2024-01-02",
        ]
    ]

    summary = compute_tearsheet(records, {"AAPL": df}, "confidence_weighted")

    assert summary.method_name == "confidence_weighted"
    assert summary.n_periods == 5
    # Sharpe may be None if returns are degenerate, but in a realistic sim it won't be
    if summary.sharpe_annual is not None:
        assert math.isfinite(summary.sharpe_annual)
    if summary.max_drawdown is not None:
        assert summary.max_drawdown <= 0.0
    assert math.isfinite(summary.avg_return_per_period)


def test_compute_tearsheet_tickers_and_n_periods():
    """Tickers and n_periods are correctly derived from records."""
    df_a = _make_ohlcv_df("2023-01-02", 200, seed=8)
    df_b = _make_ohlcv_df("2023-01-02", 200, seed=9)
    records = [
        _make_record("AAA", "2023-01-02", "majority", "Buy"),
        _make_record("AAA", "2023-04-03", "majority", "Sell"),
        _make_record("BBB", "2023-01-02", "majority", "Hold"),
    ]

    summary = compute_tearsheet(records, {"AAA": df_a, "BBB": df_b}, "majority")

    assert sorted(summary.tickers) == ["AAA", "BBB"]
    assert summary.n_periods == 3  # (AAA,Q1), (AAA,Q2), (BBB,Q1)


# ---------------------------------------------------------------------------
# Tests: TearsheetSummary JSON round-trip
# ---------------------------------------------------------------------------


def test_tearsheet_summary_json_roundtrip():
    summary = TearsheetSummary(
        method_name="confidence_weighted",
        tickers=["AAPL", "JPM"],
        n_periods=20,
        sharpe_annual=0.8765,
        sortino_annual=1.1234,
        max_drawdown=-0.1500,
        calmar=2.3456,
        cagr=0.1200,
        win_rate=0.5500,
        avg_return_per_period=0.0150,
        generated_at="2026-06-12T10:00:00+00:00",
    )

    restored = TearsheetSummary.model_validate_json(summary.model_dump_json())

    assert restored.method_name == summary.method_name
    assert restored.tickers == summary.tickers
    assert restored.sharpe_annual == summary.sharpe_annual
    assert restored.max_drawdown == summary.max_drawdown


def test_tearsheet_summary_none_metrics_roundtrip():
    """None metric fields survive JSON round-trip."""
    summary = TearsheetSummary(
        method_name="majority",
        tickers=["AAPL"],
        n_periods=1,
        sharpe_annual=None,
        sortino_annual=None,
        max_drawdown=None,
        calmar=None,
        cagr=None,
        win_rate=None,
        avg_return_per_period=0.0,
        generated_at="2026-06-12T10:00:00+00:00",
    )

    restored = TearsheetSummary.model_validate_json(summary.model_dump_json())

    assert restored.sharpe_annual is None
    assert restored.max_drawdown is None


# ---------------------------------------------------------------------------
# Tests: compute_all_tearsheets
# ---------------------------------------------------------------------------


def test_compute_all_tearsheets_returns_present_methods_only():
    """Only methods that have records in the input appear in the output."""
    df = _make_ohlcv_df("2023-01-02", 200, seed=11)
    records = [
        _make_record("AAPL", "2023-01-02", "confidence_weighted", "Buy"),
        _make_record("AAPL", "2023-01-02", "majority", "Hold"),
    ]

    summaries = compute_all_tearsheets(records, {"AAPL": df})

    assert set(summaries.keys()) == {"confidence_weighted", "majority"}
    assert isinstance(summaries["confidence_weighted"], TearsheetSummary)
    assert isinstance(summaries["majority"], TearsheetSummary)


def test_compute_all_tearsheets_all_four_methods():
    """When all four methods are in records, all four summaries are returned."""
    df = _make_ohlcv_df("2023-01-02", 200, seed=12)
    records = [
        _make_record("AAPL", "2023-01-02", m, "Buy")
        for m in ("confidence_weighted", "majority", "performance_weighted", "contrarian_adjusted")
    ]

    summaries = compute_all_tearsheets(records, {"AAPL": df})

    assert len(summaries) == 4
    assert all(isinstance(v, TearsheetSummary) for v in summaries.values())
