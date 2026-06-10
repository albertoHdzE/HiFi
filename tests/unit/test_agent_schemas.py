"""Unit tests for agent output schemas (P3-E2)."""

import json

import pytest
from pydantic import ValidationError

from hifi.agents.schemas import AgentSignal, FundamentalAnalysis

# ---------------------------------------------------------------------------
# AgentSignal
# ---------------------------------------------------------------------------


def _valid_signal(**overrides) -> AgentSignal:
    defaults = dict(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Buy",
        confidence=0.75,
        rationale="P/E of 28.3 is reasonable given ROE of 0.24 and macro tailwinds.",
        key_concern="High debt/equity ratio of 1.8 limits financial flexibility.",
        data_gaps=[],
        call_ids=["abc123def456"],
        model_id="qwen2.5-coder-32b-instruct-mlx",
        agent_type="fundamental",
    )
    defaults.update(overrides)
    return AgentSignal(**defaults)


def test_agent_signal_valid_buy():
    sig = _valid_signal(decision="Buy")
    assert sig.decision == "Buy"
    assert sig.confidence == 0.75


def test_agent_signal_valid_hold():
    sig = _valid_signal(decision="Hold", confidence=0.5)
    assert sig.decision == "Hold"


def test_agent_signal_valid_sell():
    sig = _valid_signal(decision="Sell", confidence=0.3)
    assert sig.decision == "Sell"


def test_agent_signal_confidence_boundary_zero():
    sig = _valid_signal(confidence=0.0)
    assert sig.confidence == 0.0


def test_agent_signal_confidence_boundary_one():
    sig = _valid_signal(confidence=1.0)
    assert sig.confidence == 1.0


def test_agent_signal_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        _valid_signal(confidence=-0.01)


def test_agent_signal_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        _valid_signal(confidence=1.01)


def test_agent_signal_rejects_invalid_decision():
    with pytest.raises(ValidationError):
        _valid_signal(decision="Maybe")


def test_agent_signal_rejects_empty_rationale():
    with pytest.raises(ValidationError):
        _valid_signal(rationale="   ")


def test_agent_signal_rejects_empty_key_concern():
    with pytest.raises(ValidationError):
        _valid_signal(key_concern="")


def test_agent_signal_empty_data_gaps_and_call_ids():
    sig = _valid_signal(data_gaps=[], call_ids=[])
    assert sig.data_gaps == []
    assert sig.call_ids == []


def test_agent_signal_data_gaps_populated():
    sig = _valid_signal(data_gaps=["pe", "revenue_growth_yoy"])
    assert "pe" in sig.data_gaps


def test_agent_signal_json_serialisable():
    sig = _valid_signal()
    dumped = json.dumps(sig.model_dump())
    loaded = json.loads(dumped)
    assert loaded["decision"] == "Buy"
    assert loaded["confidence"] == 0.75


# ---------------------------------------------------------------------------
# FundamentalAnalysis
# ---------------------------------------------------------------------------


def _valid_analysis(**overrides) -> FundamentalAnalysis:
    sig = _valid_signal()
    defaults = dict(
        signal=sig,
        financial_ratios={"pe": 28.3, "roe": 0.24, "call_id": "abc123"},
        growth_metrics={"net_margin": 0.25, "call_id": "def456"},
        valuation_context={"pe_1y_percentile": 0.6, "call_id": "ghi789"},
        macro_snapshot={"fed_funds_rate": 4.75, "call_id": "jkl012"},
        prompt_version="fundamental_v1",
        latency_ms=4200.0,
    )
    defaults.update(overrides)
    return FundamentalAnalysis(**defaults)


def test_fundamental_analysis_valid():
    a = _valid_analysis()
    assert a.signal.decision == "Buy"
    assert a.prompt_version == "fundamental_v1"
    assert a.latency_ms == 4200.0


def test_fundamental_analysis_latency_optional():
    a = _valid_analysis(latency_ms=None)
    assert a.latency_ms is None


def test_fundamental_analysis_json_serialisable():
    a = _valid_analysis()
    dumped = json.dumps(a.model_dump())
    loaded = json.loads(dumped)
    assert loaded["signal"]["decision"] == "Buy"
    assert loaded["prompt_version"] == "fundamental_v1"


def test_fundamental_analysis_tool_results_flat():
    a = _valid_analysis()
    flat = a.tool_results_flat()
    assert flat["pe"] == 28.3
    assert flat["fed_funds_rate"] == 4.75
    assert flat["net_margin"] == 0.25


def test_fundamental_analysis_signal_none_is_allowed():
    # A failed analysis (parse error) stores signal=None
    a = FundamentalAnalysis(
        signal=None,
        financial_ratios={},
        growth_metrics={},
        valuation_context={},
        macro_snapshot={},
        prompt_version="fundamental_v1",
        latency_ms=None,
    )
    assert a.signal is None
