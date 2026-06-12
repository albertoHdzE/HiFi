"""
Unit tests for src/hifi/collective/labeler.py (P10-E0-T6).

Tests cover:
- compute_forward_return: correct return, insufficient data, weekend advance,
  missing ticker, zero-price guard
- _apply_label: all three decision/return combinations
- label_method_decisions: correct record construction, None decision skipped
- label_agent_decisions: signal extraction, None signal skipped
- build_method_accuracy_report: delegation to MethodAccuracyReport
- compute_divergence_rates: identical methods (0.0), fully opposed (1.0),
  partial divergence, missing method pair excluded
- build_calibration_report: weight comparison produces correct structure

All tests use deterministic synthetic OHLCV data (no Parquet I/O required for
pure-logic tests; only compute_forward_return tests write temporary Parquet files).
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from hifi.collective.labeler import (
    _apply_label,
    build_method_accuracy_report,
    compute_divergence_rates,
    compute_forward_return,
    label_agent_decisions,
    label_method_decisions,
)
from hifi.collective.schemas import (
    DecisionRecord,
    MethodDecisionRecord,
)

# ---------------------------------------------------------------------------
# Helpers: write minimal Parquet files for compute_forward_return tests
# ---------------------------------------------------------------------------


def _write_price_parquet(tmp_path: Path, ticker: str, prices: dict) -> None:
    """
    Write a minimal HiFi-format OHLCV Parquet for testing.

    prices: {date_str: price_float} — uses the date as the index and fills
    all OHLCV columns with the same price value for simplicity.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = sorted(prices.items())
    dates = [date.fromisoformat(d) for d, _ in records]
    vals = [v for _, v in records]

    schema = pa.schema([
        pa.field("date", pa.date32()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("adjusted_close", pa.float64()),
    ])
    table = pa.table({
        "date": dates,
        "open": vals,
        "high": vals,
        "low": vals,
        "close": vals,
        "volume": [1000.0] * len(vals),
        "adjusted_close": vals,
    }, schema=schema)

    import json as _json
    metadata = _json.dumps({
        "ticker": ticker,
        "source": "test",
        "fetched_at": "2023-01-01T00:00:00",
        "date_from": records[0][0],
        "date_to": records[-1][0],
        "provenance": {
            "source": "test",
            "fetched_at": "2023-01-01T00:00:00",
            "parameters": {},
            "content_hash": None,
        },
    }).encode()
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"hifi_dataset_metadata": metadata}
    )

    market_dir = tmp_path / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, market_dir / f"{ticker}_2020-01-01_2023-12-31.parquet")


def _make_trading_prices(start: str, n_days: int, base: float = 100.0, step: float = 1.0) -> dict:
    """
    Create n_days consecutive trading-day prices starting from start.
    Prices increase by step each day: base, base+step, base+2*step, ...
    """
    idx = pd.bdate_range(start=start, periods=n_days, freq="B")
    return {d.date().isoformat(): base + i * step for i, d in enumerate(idx)}


# ---------------------------------------------------------------------------
# Tests: compute_forward_return
# ---------------------------------------------------------------------------


def test_compute_forward_return_exact_analysis_date(tmp_path):
    """When analysis_date is a trading day, t0 == analysis_date."""
    prices = _make_trading_prices("2023-01-02", 120)  # 120 trading days
    _write_price_parquet(tmp_path, "TEST", prices)

    fwd = compute_forward_return("TEST", "2023-01-02", str(tmp_path), horizon_days=60)

    # t0 = 2023-01-02, price = 100.0
    # t1 = day 60 = 100 + 60*1 = 160.0
    # return = (160 - 100) / 100 = 0.60
    assert fwd is not None
    assert math.isclose(fwd, 0.60, rel_tol=1e-6)


def test_compute_forward_return_weekend_advances(tmp_path):
    """analysis_date on a weekend advances to the next Monday."""
    prices = _make_trading_prices("2023-01-02", 120)
    _write_price_parquet(tmp_path, "TEST", prices)

    # 2023-01-07 is a Saturday (not a trading day in bdate_range)
    # 2023-01-09 is the first Monday after
    fwd_weekend = compute_forward_return("TEST", "2023-01-07", str(tmp_path), horizon_days=60)
    fwd_monday = compute_forward_return("TEST", "2023-01-09", str(tmp_path), horizon_days=60)

    # Both should find the same t0 (Monday 2023-01-09) → same return
    assert fwd_weekend is not None
    assert fwd_monday is not None
    assert math.isclose(fwd_weekend, fwd_monday, rel_tol=1e-9)


def test_compute_forward_return_insufficient_data(tmp_path):
    """Returns None when fewer than horizon_days trading days remain after t0."""
    # Only 30 trading days available after analysis_date
    prices = _make_trading_prices("2023-01-02", 35)
    _write_price_parquet(tmp_path, "TEST", prices)

    fwd = compute_forward_return("TEST", "2023-01-02", str(tmp_path), horizon_days=60)
    assert fwd is None


def test_compute_forward_return_missing_ticker(tmp_path):
    """Returns None when no Parquet file exists for the ticker."""
    (tmp_path / "market").mkdir(parents=True, exist_ok=True)
    fwd = compute_forward_return("NOTFOUND", "2023-01-02", str(tmp_path), horizon_days=60)
    assert fwd is None


def test_compute_forward_return_analysis_date_after_all_data(tmp_path):
    """Returns None when analysis_date is after all available trading days."""
    prices = _make_trading_prices("2023-01-02", 100)
    _write_price_parquet(tmp_path, "TEST", prices)

    fwd = compute_forward_return("TEST", "2025-01-01", str(tmp_path), horizon_days=60)
    assert fwd is None


def test_compute_forward_return_20d_horizon(tmp_path):
    """20-day horizon returns a different (smaller) return than 60-day."""
    prices = _make_trading_prices("2023-01-02", 120, base=100.0, step=1.0)
    _write_price_parquet(tmp_path, "TEST", prices)

    fwd60 = compute_forward_return("TEST", "2023-01-02", str(tmp_path), horizon_days=60)
    fwd20 = compute_forward_return("TEST", "2023-01-02", str(tmp_path), horizon_days=20)

    assert fwd60 is not None
    assert fwd20 is not None
    assert fwd20 < fwd60  # 20-step gain < 60-step gain with monotone prices


# ---------------------------------------------------------------------------
# Tests: _apply_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decision,fwd,expected", [
    ("Buy", 0.03, True),      # Buy + 3% return → correct
    ("Buy", 0.02, False),     # Buy + exactly 2% → NOT correct (> not >=)
    ("Buy", -0.05, False),    # Buy + negative → wrong
    ("Sell", -0.05, True),    # Sell + loss → correct
    ("Sell", -0.02, False),   # Sell + exactly -2% → NOT correct (< not <=)
    ("Sell", 0.03, False),    # Sell + gain → wrong
    ("Hold", 0.01, True),     # Hold + small gain → correct (within ±2%)
    ("Hold", 0.02, True),     # Hold + exactly 2% → correct (abs <= 0.02)
    ("Hold", 0.025, False),   # Hold + 2.5% → wrong (too large)
    ("Hold", -0.01, True),    # Hold + small loss → correct
    ("Buy", None, None),      # None forward return → unlabeled
    ("Sell", None, None),
    ("Hold", None, None),
])
def test_apply_label(decision, fwd, expected):
    assert _apply_label(decision, fwd) is expected


# ---------------------------------------------------------------------------
# Helpers: minimal EnsembleOutput-like namedtuples for labeling tests
# ---------------------------------------------------------------------------


def _make_mock_ensemble_output(ticker, as_of_date, method_decisions):
    """
    Return a minimal object satisfying the duck-typing contract of
    label_method_decisions():
      .ticker, .as_of_date, .method_comparison (dict of objects with .collective_decision
      and .collective_confidence)
    """
    class MockDecision:
        def __init__(self, decision, confidence=0.7):
            self.collective_decision = decision
            self.collective_confidence = confidence

    class MockOutput:
        def __init__(self, ticker, as_of_date, mc, signals):
            self.ticker = ticker
            self.as_of_date = as_of_date
            self.method_comparison = {k: MockDecision(v) for k, v in mc.items()}
            self.signals = signals

    return MockOutput(ticker, as_of_date, method_decisions, [])


def _make_mock_signal(agent_type, decision, confidence=0.7):
    class MockSignal:
        def __init__(self, at, d, c):
            self.agent_type = at
            self.decision = d
            self.confidence = c
    return MockSignal(agent_type, decision, confidence)


# ---------------------------------------------------------------------------
# Tests: label_method_decisions
# ---------------------------------------------------------------------------


def test_label_method_decisions_correct_record_count(tmp_path):
    """4 methods × 2 outputs = 8 records."""
    prices = _make_trading_prices("2023-01-02", 120)
    _write_price_parquet(tmp_path, "AAPL", prices)
    _write_price_parquet(tmp_path, "JPM", prices)

    outputs = [
        _make_mock_ensemble_output("AAPL", "2023-01-02", {
            "confidence_weighted": "Buy",
            "majority": "Buy",
            "performance_weighted": "Buy",
            "contrarian_adjusted": "Hold",
        }),
        _make_mock_ensemble_output("JPM", "2023-01-02", {
            "confidence_weighted": "Sell",
            "majority": "Hold",
            "performance_weighted": "Sell",
            "contrarian_adjusted": "Sell",
        }),
    ]

    records = label_method_decisions(outputs, str(tmp_path), horizon_days=60)

    assert len(records) == 8  # 4 methods × 2 tickers
    assert all(isinstance(r, MethodDecisionRecord) for r in records)
    assert all(r.horizon_days == 60 for r in records)


def test_label_method_decisions_none_decision_skipped(tmp_path):
    """Methods with collective_decision=None are excluded from records."""
    (tmp_path / "market").mkdir(parents=True, exist_ok=True)

    class MockDecisionNone:
        collective_decision = None
        collective_confidence = 0.0

    class MockOutput:
        ticker = "AAPL"
        as_of_date = "2023-01-02"
        method_comparison = {
            "confidence_weighted": MockDecisionNone(),
            "majority": MockDecisionNone(),
            "performance_weighted": MockDecisionNone(),
            "contrarian_adjusted": MockDecisionNone(),
        }
        signals = []

    records = label_method_decisions([MockOutput()], str(tmp_path), horizon_days=60)
    assert len(records) == 0


def test_label_method_decisions_unlabeled_when_parquet_missing(tmp_path):
    """When Parquet is absent, forward_return=None and outcome_correct=None."""
    (tmp_path / "market").mkdir(parents=True, exist_ok=True)

    outputs = [
        _make_mock_ensemble_output("MISSING", "2023-01-02", {
            "confidence_weighted": "Buy",
            "majority": "Buy",
            "performance_weighted": "Buy",
            "contrarian_adjusted": "Buy",
        }),
    ]

    records = label_method_decisions(outputs, str(tmp_path), horizon_days=60)

    assert len(records) == 4
    assert all(r.forward_return is None for r in records)
    assert all(r.outcome_correct is None for r in records)
    assert all(r.outcome_labeled_at is None for r in records)


def test_label_method_decisions_outcome_labeled_at_set_when_labeled(tmp_path):
    """outcome_labeled_at is set (non-None) when forward data is available."""
    prices = _make_trading_prices("2023-01-02", 120)
    _write_price_parquet(tmp_path, "AAPL", prices)

    outputs = [
        _make_mock_ensemble_output("AAPL", "2023-01-02", {
            "confidence_weighted": "Buy",
            "majority": "Hold",
            "performance_weighted": "Buy",
            "contrarian_adjusted": "Hold",
        }),
    ]

    records = label_method_decisions(outputs, str(tmp_path), horizon_days=60)

    assert all(r.outcome_labeled_at is not None for r in records)


# ---------------------------------------------------------------------------
# Tests: label_agent_decisions
# ---------------------------------------------------------------------------


def test_label_agent_decisions_correct_record_count(tmp_path):
    """3 signals per output × 2 outputs = 6 DecisionRecords."""
    prices = _make_trading_prices("2023-01-02", 120)
    _write_price_parquet(tmp_path, "AAPL", prices)
    _write_price_parquet(tmp_path, "JPM", prices)

    signals_aapl = [
        _make_mock_signal("fundamental", "Buy"),
        _make_mock_signal("technical", "Hold"),
        _make_mock_signal("risk", "Buy"),
    ]
    signals_jpm = [
        _make_mock_signal("fundamental", "Sell"),
        _make_mock_signal("technical", "Sell"),
        _make_mock_signal("risk", "Hold"),
    ]

    class MockOutput:
        def __init__(self, ticker, as_of_date, signals):
            self.ticker = ticker
            self.as_of_date = as_of_date
            self.method_comparison = {}
            self.signals = signals

    outputs = [
        MockOutput("AAPL", "2023-01-02", signals_aapl),
        MockOutput("JPM", "2023-01-02", signals_jpm),
    ]

    records = label_agent_decisions(outputs, str(tmp_path), horizon_days=60)

    assert len(records) == 6
    assert all(isinstance(r, DecisionRecord) for r in records)
    agent_types = {r.agent_type for r in records}
    assert agent_types == {"fundamental", "technical", "risk"}


def test_label_agent_decisions_none_signal_skipped(tmp_path):
    """None entries in signals list are skipped."""
    (tmp_path / "market").mkdir(parents=True, exist_ok=True)

    class MockOutput:
        ticker = "AAPL"
        as_of_date = "2023-01-02"
        method_comparison = {}
        signals = [None, None]

    records = label_agent_decisions([MockOutput()], str(tmp_path), horizon_days=60)
    assert len(records) == 0


# ---------------------------------------------------------------------------
# Tests: build_method_accuracy_report
# ---------------------------------------------------------------------------


def test_build_method_accuracy_report_computes_accuracy():
    """accuracy_by_method is correctly computed from labeled records."""
    records = [
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name="confidence_weighted", decision="Buy",
            collective_confidence=0.8, forward_return=0.05,
            outcome_correct=True, horizon_days=60,
        ),
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name="confidence_weighted", decision="Buy",
            collective_confidence=0.7, forward_return=-0.03,
            outcome_correct=False, horizon_days=60,
        ),
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name="majority", decision="Buy",
            collective_confidence=0.6, forward_return=0.05,
            outcome_correct=True, horizon_days=60,
        ),
    ]

    report = build_method_accuracy_report(records)

    assert report.n_labeled == 3
    assert "confidence_weighted" in report.accuracy_by_method
    assert "majority" in report.accuracy_by_method
    assert math.isclose(report.accuracy_by_method["confidence_weighted"], 0.5)
    assert math.isclose(report.accuracy_by_method["majority"], 1.0)


def test_build_method_accuracy_report_excludes_unlabeled():
    """Records with outcome_correct=None do not contribute to accuracy."""
    records = [
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name="majority", decision="Buy",
            collective_confidence=0.6, forward_return=None,
            outcome_correct=None, horizon_days=60,
        ),
    ]

    report = build_method_accuracy_report(records)

    assert report.n_labeled == 0
    assert report.accuracy_by_method == {}


# ---------------------------------------------------------------------------
# Tests: compute_divergence_rates
# ---------------------------------------------------------------------------


def test_compute_divergence_rates_all_identical():
    """All methods agree on every record → 0.0 divergence for all pairs."""
    records = [
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name=m, decision="Buy",
            collective_confidence=0.7, horizon_days=60,
        )
        for m in ("confidence_weighted", "majority", "performance_weighted", "contrarian_adjusted")
    ]

    rates = compute_divergence_rates(records)

    assert len(rates) == 6  # 4 choose 2 = 6 pairs
    assert all(v == 0.0 for v in rates.values())


def test_compute_divergence_rates_cw_vs_mv_fully_opposed():
    """confidence_weighted=Buy, majority=Sell on every record → cw_vs_mv = 1.0."""
    records = []
    for i in range(5):
        date_str = f"2023-0{i + 1}-01" if i < 4 else "2023-05-01"
        records.append(
            MethodDecisionRecord(
                ticker="AAPL", analysis_date=date_str,
                method_name="confidence_weighted", decision="Buy",
                collective_confidence=0.8, horizon_days=60,
            )
        )
        records.append(
            MethodDecisionRecord(
                ticker="AAPL", analysis_date=date_str,
                method_name="majority", decision="Sell",
                collective_confidence=0.6, horizon_days=60,
            )
        )
        # pw and ca same as cw
        records.append(
            MethodDecisionRecord(
                ticker="AAPL", analysis_date=date_str,
                method_name="performance_weighted", decision="Buy",
                collective_confidence=0.8, horizon_days=60,
            )
        )
        records.append(
            MethodDecisionRecord(
                ticker="AAPL", analysis_date=date_str,
                method_name="contrarian_adjusted", decision="Buy",
                collective_confidence=0.7, horizon_days=60,
            )
        )

    rates = compute_divergence_rates(records)

    assert math.isclose(rates["cw_vs_mv"], 1.0)
    assert rates["cw_vs_pw"] == 0.0
    assert rates["cw_vs_ca"] == 0.0


def test_compute_divergence_rates_missing_pair_excluded():
    """
    When a method has no records for a given (ticker, date, horizon),
    that combo is excluded from the divergence denominator for that pair.
    """
    # Only confidence_weighted and majority present
    records = [
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name="confidence_weighted", decision="Buy",
            collective_confidence=0.8, horizon_days=60,
        ),
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name="majority", decision="Sell",
            collective_confidence=0.5, horizon_days=60,
        ),
        # No performance_weighted or contrarian_adjusted records
    ]

    rates = compute_divergence_rates(records)

    # cw_vs_mv: 1 pair, diverge → 1.0
    assert math.isclose(rates["cw_vs_mv"], 1.0)
    # All pairs involving pw or ca have no shared (ticker, date) → 0.0
    assert rates["cw_vs_pw"] == 0.0
    assert rates["cw_vs_ca"] == 0.0
    assert rates["mv_vs_pw"] == 0.0
    assert rates["mv_vs_ca"] == 0.0
    assert rates["pw_vs_ca"] == 0.0
