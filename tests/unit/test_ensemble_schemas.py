"""Unit tests for ensemble output schemas (P4-E2-T4 through T8)."""

import json

import pytest

from hifi.agents.schemas import AgentSignal, FundamentalAnalysis, TechnicalAnalysis
from hifi.collective.schemas import EnsembleDecision, EnsembleOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(decision="Buy", confidence=0.75, agent_type="fundamental") -> AgentSignal:
    return AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision=decision,
        confidence=confidence,
        rationale="P/E of 28.3 is reasonable.",
        key_concern="High rates compress multiples.",
        data_gaps=[],
        call_ids=["abc123"],
        model_id="test-model",
        agent_type=agent_type,
    )


def _make_fundamental_analysis(signal=None) -> FundamentalAnalysis:
    return FundamentalAnalysis(
        signal=signal or _make_signal(agent_type="fundamental"),
        financial_ratios={"pe": 28.3, "call_id": "abc"},
        growth_metrics={"net_margin": 0.25, "call_id": "def"},
        valuation_context={"pe_1y_percentile": 0.6, "call_id": "ghi"},
        macro_snapshot={"fed_funds_rate": 4.75, "call_id": "jkl"},
        prompt_version="fundamental_v1",
        latency_ms=1200.0,
    )


def _make_technical_analysis(signal=None, time_horizon="medium-term") -> TechnicalAnalysis:
    return TechnicalAnalysis(
        signal=signal or _make_signal(agent_type="technical"),
        technical_indicators={"rsi": 48.0, "call_id": "mno"},
        risk_metrics={"sharpe_252d": 0.82, "call_id": "pqr"},
        time_horizon=time_horizon,
        prompt_version="technical_v1",
        latency_ms=950.0,
    )


def _make_ensemble_decision(
    collective_decision="Buy",
    collective_confidence=0.6,
    n_valid=2,
    agreement=True,
    entropy=0.0,
    dispersion=0.0,
) -> EnsembleDecision:
    return EnsembleDecision(
        collective_decision=collective_decision,
        collective_confidence=collective_confidence,
        n_valid_signals=n_valid,
        agreement=agreement,
        disagreement_entropy=entropy,
        opinion_dispersion=dispersion,
        agent_decisions=["Buy", "Buy"],
        agent_confidences=[0.75, 0.60],
        winning_score=1.35,
        total_score=1.35,
    )


# ---------------------------------------------------------------------------
# TechnicalAnalysis tests (P4-E2-T4)
# ---------------------------------------------------------------------------


def test_technical_analysis_serialises_to_json():
    ta = _make_technical_analysis()
    dumped = ta.model_dump()
    json_str = json.dumps(dumped)
    loaded = json.loads(json_str)
    assert loaded["prompt_version"] == "technical_v1"
    assert loaded["time_horizon"] == "medium-term"


def test_technical_analysis_tool_results_flat():
    ta = _make_technical_analysis()
    flat = ta.tool_results_flat()
    assert "rsi" in flat
    assert "sharpe_252d" in flat


def test_technical_analysis_signal_none_allowed():
    ta = TechnicalAnalysis(
        signal=None,
        technical_indicators={},
        risk_metrics={},
        time_horizon=None,
        prompt_version="technical_v1",
    )
    assert ta.signal is None
    json.dumps(ta.model_dump())  # must not raise


# ---------------------------------------------------------------------------
# EnsembleDecision tests (P4-E2-T5, T6, T7)
# ---------------------------------------------------------------------------


def test_ensemble_decision_agreeing_agents(monkeypatch):
    """Two agreeing agents: agreement=True, entropy=0.0."""
    from hifi.collective.voting import confidence_weighted_vote

    sig1 = _make_signal("Buy", 0.75)
    sig2 = _make_signal("Buy", 0.60, agent_type="technical")
    decision = confidence_weighted_vote([sig1, sig2])

    assert decision.agreement is True
    assert decision.disagreement_entropy == pytest.approx(0.0)
    assert decision.collective_decision == "Buy"
    assert decision.n_valid_signals == 2


def test_ensemble_decision_disagreeing_agents():
    """Two disagreeing agents: agreement=False, entropy > 0."""
    from hifi.collective.voting import confidence_weighted_vote

    sig1 = _make_signal("Buy", 0.80)
    sig2 = _make_signal("Sell", 0.60, agent_type="technical")
    decision = confidence_weighted_vote([sig1, sig2])

    assert decision.agreement is False
    assert decision.disagreement_entropy > 0.0
    assert decision.collective_decision == "Buy"  # higher confidence wins


def test_ensemble_decision_zero_signals():
    """Zero valid signals: collective_decision=None, n_valid_signals=0."""
    from hifi.collective.voting import confidence_weighted_vote

    decision = confidence_weighted_vote([None, None])
    assert decision.collective_decision is None
    assert decision.n_valid_signals == 0


# ---------------------------------------------------------------------------
# EnsembleOutput tests (P4-E2-T8)
# ---------------------------------------------------------------------------


def test_ensemble_output_serialises_to_json():
    output = EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=_make_fundamental_analysis(),
        technical_analysis=_make_technical_analysis(),
        ensemble_decision=_make_ensemble_decision(),
        latency_ms=2200.0,
    )
    dumped = output.model_dump()
    json_str = json.dumps(dumped)
    loaded = json.loads(json_str)
    assert loaded["ticker"] == "AAPL"
    assert loaded["ensemble_decision"]["collective_decision"] == "Buy"
    assert loaded["technical_analysis"]["time_horizon"] == "medium-term"
