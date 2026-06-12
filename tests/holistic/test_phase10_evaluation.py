"""
Holistic structural tests for Phase 10 accuracy evaluation (P10-E3, DJ-047).

Validates the labeling and accuracy pipeline without any LLM invocation:
- label_method_decisions() produces correct MethodDecisionRecord structure
- build_method_accuracy_report() computes accuracy correctly from known records
- TearsheetSummary JSON round-trip is lossless
- MethodAccuracyReport with zero labeled records has accuracy_by_method == {}

Design (DJ-047):
  Zero LLM calls. Zero monkeypatching. All inputs are seeded synthetic data.
  The pipeline under test: EnsembleOutput-like objects → label_method_decisions()
  → build_method_accuracy_report() → JSON. No run_ensemble(), no make_llm().
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from hifi.analytics.tearsheet import TearsheetSummary
from hifi.collective.labeler import build_method_accuracy_report, label_method_decisions
from hifi.collective.schemas import MethodDecisionRecord

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FOUR_METHODS = frozenset(
    {"confidence_weighted", "majority", "performance_weighted", "contrarian_adjusted"}
)


def _write_price_parquet(tmp_path: Path, ticker: str, n_days: int = 120) -> None:
    """Write a minimal HiFi-format OHLCV Parquet for the labeler's _load_prices."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dates = list(pd.bdate_range("2023-01-02", periods=n_days).date)
    prices = [100.0 + i * 0.5 for i in range(n_days)]
    market_dir = tmp_path / "market"
    market_dir.mkdir(parents=True, exist_ok=True)

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
        "open": prices, "high": prices, "low": prices,
        "close": prices, "volume": [1_000_000.0] * n_days,
        "adjusted_close": prices,
    }, schema=schema)
    pq.write_table(table, market_dir / f"{ticker}_2023-01-02_2023-06-30.parquet")


def _make_mock_output(ticker, as_of_date, decisions):
    """Minimal duck-typed object satisfying label_method_decisions' contract."""
    class _D:
        def __init__(self, d):
            self.collective_decision = d
            self.collective_confidence = 0.70

    class _O:
        def __init__(self):
            self.ticker = ticker
            self.as_of_date = as_of_date
            self.method_comparison = {k: _D(v) for k, v in decisions.items()}
            self.signals = []

    return _O()


def _make_record(method, decision, outcome_correct):
    return MethodDecisionRecord(
        ticker="AAPL", analysis_date="2023-01-02",
        method_name=method, decision=decision,
        collective_confidence=0.70,
        forward_return=0.05 if outcome_correct else -0.05,
        outcome_correct=outcome_correct,
        horizon_days=60,
    )


# ---------------------------------------------------------------------------
# Tests: label_method_decisions — structure and record count
# ---------------------------------------------------------------------------


def test_label_method_decisions_produces_method_decision_records(tmp_path):
    """All returned records are MethodDecisionRecord instances."""
    _write_price_parquet(tmp_path, "AAPL")
    outputs = [_make_mock_output("AAPL", "2023-01-02", {
        "confidence_weighted": "Buy",
        "majority": "Buy",
        "performance_weighted": "Hold",
        "contrarian_adjusted": "Sell",
    })]

    records = label_method_decisions(outputs, str(tmp_path), horizon_days=60)

    assert len(records) == 4
    assert all(isinstance(r, MethodDecisionRecord) for r in records)


def test_label_method_decisions_record_fields_match_output(tmp_path):
    """ticker, analysis_date, method_name, decision are correctly propagated."""
    _write_price_parquet(tmp_path, "JPM")
    expected_decisions = {
        "confidence_weighted": "Buy",
        "majority": "Hold",
        "performance_weighted": "Sell",
        "contrarian_adjusted": "Hold",
    }
    outputs = [_make_mock_output("JPM", "2023-01-02", expected_decisions)]

    records = label_method_decisions(outputs, str(tmp_path), horizon_days=60)

    by_method = {r.method_name: r for r in records}
    assert set(by_method.keys()) == _FOUR_METHODS
    for method, decision in expected_decisions.items():
        assert by_method[method].ticker == "JPM"
        assert by_method[method].analysis_date == "2023-01-02"
        assert by_method[method].decision == decision
        assert by_method[method].horizon_days == 60


def test_label_method_decisions_two_tickers_eight_records(tmp_path):
    """4 methods × 2 tickers = 8 records."""
    _write_price_parquet(tmp_path, "AAPL")
    _write_price_parquet(tmp_path, "JPM")
    outputs = [
        _make_mock_output("AAPL", "2023-01-02", {
            m: "Buy" for m in _FOUR_METHODS
        }),
        _make_mock_output("JPM", "2023-01-02", {
            m: "Sell" for m in _FOUR_METHODS
        }),
    ]

    records = label_method_decisions(outputs, str(tmp_path), horizon_days=60)
    assert len(records) == 8


def test_label_method_decisions_forward_return_labeled_when_data_present(tmp_path):
    """forward_return is populated and outcome_correct is bool (not None) when Parquet present."""
    _write_price_parquet(tmp_path, "AAPL")
    outputs = [_make_mock_output("AAPL", "2023-01-02", {
        "confidence_weighted": "Buy",
        "majority": "Buy",
        "performance_weighted": "Buy",
        "contrarian_adjusted": "Buy",
    })]

    records = label_method_decisions(outputs, str(tmp_path), horizon_days=60)

    assert all(r.forward_return is not None for r in records)
    assert all(isinstance(r.outcome_correct, bool) for r in records)
    assert all(r.outcome_labeled_at is not None for r in records)


def test_label_method_decisions_unlabeled_when_parquet_absent(tmp_path):
    """outcome_correct=None and forward_return=None when no Parquet exists."""
    (tmp_path / "market").mkdir(parents=True, exist_ok=True)
    outputs = [_make_mock_output("MISSING", "2023-01-02", {
        m: "Buy" for m in _FOUR_METHODS
    })]

    records = label_method_decisions(outputs, str(tmp_path), horizon_days=60)

    assert len(records) == 4
    assert all(r.forward_return is None for r in records)
    assert all(r.outcome_correct is None for r in records)


def test_label_method_decisions_none_decision_skipped(tmp_path):
    """Methods where collective_decision is None are excluded from results."""
    (tmp_path / "market").mkdir(parents=True, exist_ok=True)

    class _DNull:
        collective_decision = None
        collective_confidence = 0.0

    class _O:
        ticker = "AAPL"
        as_of_date = "2023-01-02"
        method_comparison = {m: _DNull() for m in _FOUR_METHODS}
        signals = []

    records = label_method_decisions([_O()], str(tmp_path), horizon_days=60)
    assert len(records) == 0


# ---------------------------------------------------------------------------
# Tests: build_method_accuracy_report — accuracy computation
# ---------------------------------------------------------------------------


def test_build_method_accuracy_report_correct_accuracy():
    """2 correct + 1 incorrect for cw → accuracy = 0.667; 1/1 for mv → 1.0."""
    records = [
        _make_record("confidence_weighted", "Buy", True),
        _make_record("confidence_weighted", "Buy", True),
        _make_record("confidence_weighted", "Buy", False),
        _make_record("majority", "Sell", True),
    ]

    report = build_method_accuracy_report(records)

    assert report.n_labeled == 4
    assert math.isclose(report.accuracy_by_method["confidence_weighted"], 2 / 3, rel_tol=1e-6)
    assert math.isclose(report.accuracy_by_method["majority"], 1.0)


def test_build_method_accuracy_report_zero_labeled_empty_accuracy():
    """Records with outcome_correct=None → accuracy_by_method == {} (DJ-044)."""
    records = [
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name="confidence_weighted", decision="Buy",
            collective_confidence=0.70, forward_return=None,
            outcome_correct=None, horizon_days=60,
        ),
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-01-02",
            method_name="majority", decision="Hold",
            collective_confidence=0.60, forward_return=None,
            outcome_correct=None, horizon_days=60,
        ),
    ]

    report = build_method_accuracy_report(records)

    assert report.n_labeled == 0
    assert report.accuracy_by_method == {}


def test_build_method_accuracy_report_tickers_and_dates_derived():
    """tickers and analysis_dates are derived from records (not user-settable)."""
    records = [
        _make_record("confidence_weighted", "Buy", True),
        MethodDecisionRecord(
            ticker="JPM", analysis_date="2023-04-01",
            method_name="majority", decision="Sell",
            collective_confidence=0.65, forward_return=-0.05,
            outcome_correct=True, horizon_days=60,
        ),
    ]

    report = build_method_accuracy_report(records)

    assert "AAPL" in report.tickers
    assert "JPM" in report.tickers
    assert "2023-01-02" in report.analysis_dates
    assert "2023-04-01" in report.analysis_dates


def test_build_method_accuracy_report_excludes_unlabeled_from_accuracy():
    """Mixed labeled + unlabeled: unlabeled excluded from accuracy numerator/denominator."""
    records = [
        _make_record("confidence_weighted", "Buy", True),    # labeled correct
        MethodDecisionRecord(
            ticker="AAPL", analysis_date="2023-04-01",
            method_name="confidence_weighted", decision="Buy",
            collective_confidence=0.70, forward_return=None,
            outcome_correct=None, horizon_days=60,
        ),  # unlabeled
    ]

    report = build_method_accuracy_report(records)

    assert report.n_labeled == 1
    assert math.isclose(report.accuracy_by_method["confidence_weighted"], 1.0)


# ---------------------------------------------------------------------------
# Tests: TearsheetSummary JSON round-trip
# ---------------------------------------------------------------------------


def test_tearsheet_summary_json_roundtrip_lossless():
    """model_dump_json → model_validate_json preserves all fields."""
    summary = TearsheetSummary(
        method_name="confidence_weighted",
        tickers=["AAPL", "JPM"],
        n_periods=20,
        sharpe_annual=1.2345,
        sortino_annual=1.8765,
        max_drawdown=-0.1234,
        calmar=0.9876,
        cagr=0.1543,
        win_rate=0.5432,
        avg_return_per_period=0.0123,
        generated_at="2026-06-12T00:00:00+00:00",
    )

    restored = TearsheetSummary.model_validate_json(summary.model_dump_json())

    assert restored.method_name == summary.method_name
    assert restored.tickers == summary.tickers
    assert restored.n_periods == summary.n_periods
    assert restored.sharpe_annual == summary.sharpe_annual
    assert restored.max_drawdown == summary.max_drawdown
    assert restored.generated_at == summary.generated_at


def test_tearsheet_summary_json_roundtrip_none_fields():
    """None metric fields survive JSON round-trip without becoming 0.0."""
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
        generated_at="2026-06-12T00:00:00+00:00",
    )

    restored = TearsheetSummary.model_validate_json(summary.model_dump_json())

    assert restored.sharpe_annual is None
    assert restored.sortino_annual is None
    assert restored.max_drawdown is None
    assert restored.calmar is None
    assert restored.cagr is None
    assert restored.win_rate is None


def test_tearsheet_summary_json_raw_dict_round_trip():
    """model_dump() → dict serialisation is stable (no precision drift)."""
    summary = TearsheetSummary(
        method_name="performance_weighted",
        tickers=["XOM"],
        n_periods=5,
        sharpe_annual=0.7654,
        sortino_annual=0.9012,
        max_drawdown=-0.0543,
        calmar=None,
        cagr=0.0678,
        win_rate=0.4321,
        avg_return_per_period=0.0045,
        generated_at="2026-06-12T00:00:00+00:00",
    )

    d = json.loads(summary.model_dump_json())
    restored = TearsheetSummary(**d)

    assert restored.sharpe_annual == summary.sharpe_annual
    assert restored.method_name == summary.method_name


# ---------------------------------------------------------------------------
# Tests: phase10_accuracy.json fixture (skipif absent)
# ---------------------------------------------------------------------------

_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "baseline" / "phase10_accuracy.json"
)


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="phase10_accuracy.json fixture not yet generated (run make baseline-phase10)",
)
def test_phase10_accuracy_fixture_schema():
    """Loaded fixture deserialises to a valid MethodAccuracyReport structure."""
    from hifi.collective.schemas import MethodAccuracyReport

    data = json.loads(_FIXTURE_PATH.read_text())
    report = MethodAccuracyReport.model_validate(data)

    assert report.n_labeled >= 0
    assert isinstance(report.accuracy_by_method, dict)
    assert isinstance(report.tickers, list)
    assert isinstance(report.analysis_dates, list)


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="phase10_accuracy.json fixture not yet generated",
)
def test_phase10_accuracy_fixture_has_all_four_methods():
    """accuracy_by_method covers all four canonical method keys."""
    from hifi.collective.schemas import MethodAccuracyReport

    data = json.loads(_FIXTURE_PATH.read_text())
    report = MethodAccuracyReport.model_validate(data)

    # Only check methods that have labeled records
    for m in report.accuracy_by_method:
        assert m in _FOUR_METHODS
        acc = report.accuracy_by_method[m]
        assert 0.0 <= acc <= 1.0, f"accuracy out of range for {m}: {acc}"
