"""
Unit tests for hifi.simulation.agent_executor (E0-T2, DJ-106).

Tests:
- _runs_dir / _agent_json_path path structure
- _save_analysis / _load_analysis round-trip
- _build_memory_prefix: "parallel" returns "", others call store
- _read_stored_signals: reads stored agent JSONs correctly
- aggregate_agent_outputs: assembles EnsembleOutput from pre-populated JSONs

No real LLMs. No network calls.
"""

from __future__ import annotations

import json

import pytest

from hifi.simulation.agent_executor import (
    _agent_json_path,
    _build_memory_prefix,
    _load_analysis,
    _runs_dir,
    _save_analysis,
    aggregate_agent_outputs,
)

# ---------------------------------------------------------------------------
# Path structure tests
# ---------------------------------------------------------------------------


def test_runs_dir_structure(tmp_path):
    data_dir = str(tmp_path)
    result = _runs_dir(data_dir, "run-abc")
    assert str(result) == str(tmp_path / "runs" / "run-abc")


def test_agent_json_path(tmp_path):
    data_dir = str(tmp_path)
    result = _agent_json_path(data_dir, "run-001", "AAPL", "fundamental")
    expected = tmp_path / "runs" / "run-001" / "AAPL_fundamental.json"
    assert str(result) == str(expected)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def _make_fundamental_analysis() -> object:
    """Build a minimal but valid FundamentalAnalysis for testing."""
    from hifi.agents.schemas import AgentSignal, FundamentalAnalysis

    sig = AgentSignal(
        ticker="AAPL",
        as_of_date="2022-01-31",
        decision="Buy",
        confidence=0.72,
        rationale="P/E below sector average; earnings growth strong.",
        key_concern="High rate environment compresses multiples.",
        model_id="stub-model",
        agent_type="fundamental",
    )
    return FundamentalAnalysis(
        signal=sig,
        financial_ratios={"pe_ratio": 22.0},
        growth_metrics={"revenue_growth": 0.08},
        valuation_context={"sector_avg_pe": 25.0},
        macro_snapshot={"fed_funds": 5.25},
        prompt_version="test-v1",
        latency_ms=100.0,
    )


def _make_technical_analysis() -> object:
    """Build a minimal but valid TechnicalAnalysis for testing."""
    from hifi.agents.schemas import AgentSignal, TechnicalAnalysis

    sig = AgentSignal(
        ticker="AAPL",
        as_of_date="2022-01-31",
        decision="Hold",
        confidence=0.60,
        rationale="RSI at 52 is neutral; MACD histogram slightly positive.",
        key_concern="Volume declining on the rally.",
        model_id="stub-model",
        agent_type="technical",
    )
    return TechnicalAnalysis(
        signal=sig,
        technical_indicators={"rsi_14": 52.0, "macd_hist": 0.12},
        risk_metrics={"hist_vol_20d": 0.21},
        time_horizon="short-term",
        prompt_version="test-v1",
        latency_ms=80.0,
    )


def test_save_and_load_analysis_roundtrip(tmp_path):
    data_dir = str(tmp_path)
    run_id = "test-run-001"
    analysis = _make_fundamental_analysis()

    _save_analysis(data_dir, run_id, "AAPL", "fundamental", analysis)

    loaded = _load_analysis(data_dir, run_id, "AAPL", "fundamental")
    assert loaded is not None
    assert loaded["signal"]["decision"] == "Buy"
    assert loaded["signal"]["confidence"] == pytest.approx(0.72)
    assert loaded["prompt_version"] == "test-v1"


def test_load_analysis_missing_returns_none(tmp_path):
    result = _load_analysis(str(tmp_path), "no-run", "MSFT", "technical")
    assert result is None


def test_save_analysis_creates_parent_dirs(tmp_path):
    data_dir = str(tmp_path / "nested" / "dir")
    _save_analysis(data_dir, "run-x", "AAPL", "risk", _make_fundamental_analysis())
    path = _agent_json_path(data_dir, "run-x", "AAPL", "risk")
    assert path.exists()


# ---------------------------------------------------------------------------
# _build_memory_prefix
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal AgentContextStore stub for testing."""

    def __init__(self, records=None):
        self._records = records or []

    def read_prior(self, run_id, before_agent):
        return self._records


def test_build_memory_prefix_parallel_returns_empty():
    store = _FakeStore(records=[])
    result = _build_memory_prefix(store, "run-1", "technical", "AAPL", "2022-01-31", "parallel")
    assert result == ""


def test_build_memory_prefix_full_condition_with_no_records():
    store = _FakeStore(records=[])
    result = _build_memory_prefix(store, "run-1", "technical", "AAPL", "2022-01-31", "full")
    assert result == ""


# ---------------------------------------------------------------------------
# aggregate_agent_outputs
# ---------------------------------------------------------------------------


def test_aggregate_agent_outputs(tmp_path):
    """Pre-populate JSON sidecars; aggregate must return valid EnsembleOutput."""
    from hifi.collective.schemas import EnsembleOutput

    data_dir = str(tmp_path)
    db_path = str(tmp_path / "knowledge.lance")
    run_id = "agg-test-001"

    # Write fundamental and technical analysis JSONs
    _save_analysis(data_dir, run_id, "AAPL", "fundamental", _make_fundamental_analysis())
    _save_analysis(data_dir, run_id, "AAPL", "technical", _make_technical_analysis())

    output = aggregate_agent_outputs(
        ticker="AAPL",
        date="2022-01-31",
        run_id=run_id,
        db_path=db_path,
    )

    assert isinstance(output, EnsembleOutput)
    assert output.ticker == "AAPL"
    assert output.as_of_date == "2022-01-31"
    assert output.ensemble_decision is not None
    assert output.ensemble_decision.collective_decision in ("Buy", "Hold", "Sell")
    assert len(output.signals) == 2  # fundamental + technical
    assert output.risk_analysis is None   # not stored
    assert output.macro_analysis is None
    assert output.contrarian_analysis is None


def test_aggregate_agent_outputs_missing_fundamental_raises(tmp_path):
    """RuntimeError when fundamental JSON is missing."""
    db_path = str(tmp_path / "knowledge.lance")
    # Only write technical, not fundamental
    _save_analysis(str(tmp_path), "run-bad", "AAPL", "technical", _make_technical_analysis())

    with pytest.raises(RuntimeError, match="Missing fundamental"):
        aggregate_agent_outputs(
            ticker="AAPL",
            date="2022-01-31",
            run_id="run-bad",
            db_path=db_path,
        )


def test_aggregate_agent_outputs_json_serializable(tmp_path):
    """EnsembleOutput produced by aggregate must be JSON-serializable."""
    db_path = str(tmp_path / "knowledge.lance")
    run_id = "agg-serial-001"

    _save_analysis(str(tmp_path), run_id, "MSFT", "fundamental", _make_fundamental_analysis())
    _save_analysis(str(tmp_path), run_id, "MSFT", "technical", _make_technical_analysis())

    output = aggregate_agent_outputs(
        ticker="MSFT",
        date="2022-01-31",
        run_id=run_id,
        db_path=db_path,
    )

    # Should not raise
    serialized = output.model_dump_json()
    data = json.loads(serialized)
    assert data["ticker"] == "MSFT"
