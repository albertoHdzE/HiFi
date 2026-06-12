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


def test_ensemble_output_phase8_fields_default_none():
    """Phase 8 fields default to None for backward compat (DJ-038)."""
    output = EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=_make_fundamental_analysis(),
        technical_analysis=_make_technical_analysis(),
        ensemble_decision=_make_ensemble_decision(),
        latency_ms=2200.0,
    )
    assert output.risk_analysis is None
    assert output.macro_analysis is None
    assert output.sentiment_analysis is None
    assert output.contrarian_analysis is None


def test_ensemble_output_with_phase8_fields_json_safe():
    """EnsembleOutput with all Phase 8 fields populated is JSON-safe."""
    from hifi.agents.schemas import (
        ContrarianAnalysis,
        MacroAnalysis,
        RiskAnalysis,
        SentimentAnalysis,
    )

    risk_sig = _make_signal("Hold", 0.60, "risk")
    ra = RiskAnalysis(
        signal=risk_sig,
        risk_assessment="Moderate volatility.",
        prompt_version="risk_v1",
    )

    macro_sig = _make_signal("Hold", 0.55, "macro")
    ma = MacroAnalysis(
        signal=macro_sig,
        regime_assessment="Late-cycle",
        rationale="Rates at 4.75%.",
        prompt_version="macro_v1",
    )

    sent_sig = _make_signal("Buy", 0.65, "sentiment")
    sa = SentimentAnalysis(
        signal=sent_sig,
        sentiment_summary="Cautiously optimistic MD&A tone.",
        notable_signals=["services growth highlighted"],
        prompt_version="sentiment_v1",
    )

    ca = ContrarianAnalysis(
        alternative_thesis="Bear case: services may decelerate.",
        risk_scenario="Consumer pullback reduces revenue 10%.",
        counterargument="Consensus ignores macro tightening risk.",
        confidence=0.40,
        prompt_version="contrarian_v1",
    )

    output = EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=_make_fundamental_analysis(),
        technical_analysis=_make_technical_analysis(),
        ensemble_decision=_make_ensemble_decision(),
        latency_ms=5000.0,
        risk_analysis=ra,
        macro_analysis=ma,
        sentiment_analysis=sa,
        contrarian_analysis=ca,
    )
    dumped = output.model_dump()
    json_str = json.dumps(dumped)
    loaded = json.loads(json_str)
    assert loaded["risk_analysis"]["risk_assessment"] == "Moderate volatility."
    assert loaded["macro_analysis"]["regime_assessment"] == "Late-cycle"
    assert loaded["sentiment_analysis"]["sentiment_summary"] == "Cautiously optimistic MD&A tone."
    assert loaded["contrarian_analysis"]["confidence"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Phase 9 schema extension tests (P9-E0)
# ---------------------------------------------------------------------------


def test_ensemble_decision_default_contrarian_fields():
    """Phase 9 contrarian fields have backward-compatible defaults."""
    d = _make_ensemble_decision()
    assert d.contrarian_confidence_discount == pytest.approx(1.0)
    assert d.review_flagged is False


def test_ensemble_decision_explicit_contrarian_fields_round_trip():
    """EnsembleDecision with non-default contrarian fields serialises correctly."""
    d = EnsembleDecision(
        collective_decision="Buy",
        collective_confidence=0.54,    # discounted: was 0.72, discount=0.75
        n_valid_signals=2,
        agreement=False,
        disagreement_entropy=1.0,
        opinion_dispersion=0.1,
        agent_decisions=["Buy", "Sell"],
        agent_confidences=[0.8, 0.6],
        winning_score=0.8,
        total_score=1.4,
        contrarian_confidence_discount=0.75,
        review_flagged=False,
    )
    dumped = json.loads(d.model_dump_json())
    assert dumped["contrarian_confidence_discount"] == pytest.approx(0.75)
    assert dumped["review_flagged"] is False


def test_ensemble_decision_review_flagged_true_round_trip():
    d = _make_ensemble_decision()
    d2 = EnsembleDecision(
        **{**d.model_dump(), "review_flagged": True, "contrarian_confidence_discount": 0.65}
    )
    assert d2.review_flagged is True
    assert json.loads(d2.model_dump_json())["review_flagged"] is True


def test_ensemble_decision_contrarian_discount_out_of_range_raises():
    """contrarian_confidence_discount must be in [0, 1]."""
    with pytest.raises(ValueError):
        EnsembleDecision(
            collective_decision="Buy",
            collective_confidence=0.8,
            n_valid_signals=1,
            agreement=True,
            disagreement_entropy=0.0,
            opinion_dispersion=0.0,
            agent_decisions=["Buy"],
            agent_confidences=[0.8],
            winning_score=0.8,
            total_score=0.8,
            contrarian_confidence_discount=1.5,  # invalid
        )


def test_ensemble_output_phase9_fields_default_empty():
    """Phase 9 EnsembleOutput fields default to empty list/dict/string."""
    output = EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=_make_fundamental_analysis(),
        technical_analysis=_make_technical_analysis(),
        ensemble_decision=_make_ensemble_decision(),
        latency_ms=2200.0,
    )
    assert output.signals == []
    assert output.aggregation_method == "confidence_weighted"
    assert output.method_comparison == {}


def test_ensemble_output_with_signals_and_method_comparison_json_safe():
    """EnsembleOutput with all Phase 9 fields populated is JSON-safe."""

    decision = _make_ensemble_decision()
    sig = _make_signal()
    method_comparison = {
        "majority": decision,
        "confidence_weighted": decision,
        "performance_weighted": decision,
        "contrarian_adjusted": decision,
    }
    output = EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=_make_fundamental_analysis(),
        technical_analysis=_make_technical_analysis(),
        ensemble_decision=decision,
        latency_ms=2200.0,
        signals=[sig],
        aggregation_method="confidence_weighted",
        method_comparison=method_comparison,
    )
    dumped = json.loads(output.model_dump_json())
    assert len(dumped["signals"]) == 1
    assert dumped["aggregation_method"] == "confidence_weighted"
    assert set(dumped["method_comparison"].keys()) == {
        "majority", "confidence_weighted", "performance_weighted", "contrarian_adjusted"
    }


def test_ensemble_output_json_roundtrip_lossless():
    """model_dump_json → model_validate_json is lossless for Phase 9 fields."""
    sig = _make_signal()
    decision = _make_ensemble_decision()
    output = EnsembleOutput(
        ticker="JPM",
        as_of_date="2022-12-31",
        fundamental_analysis=_make_fundamental_analysis(),
        technical_analysis=_make_technical_analysis(),
        ensemble_decision=decision,
        latency_ms=3100.0,
        signals=[sig],
        aggregation_method="confidence_weighted",
        method_comparison={"confidence_weighted": decision},
    )
    restored = EnsembleOutput.model_validate_json(output.model_dump_json())
    assert restored.ticker == "JPM"
    assert len(restored.signals) == 1
    assert restored.signals[0].agent_type == sig.agent_type
    assert "confidence_weighted" in restored.method_comparison


# ---------------------------------------------------------------------------
# DecisionRecord and AgentPerformanceHistory tests (P9-E0-T3)
# ---------------------------------------------------------------------------


def test_decision_record_valid():
    from hifi.collective.schemas import DecisionRecord

    r = DecisionRecord(
        ticker="AAPL",
        analysis_date="2022-03-31",
        agent_type="fundamental",
        decision="Buy",
        confidence=0.75,
    )
    assert r.horizon_days == 60  # default
    assert r.outcome_correct is None
    assert r.forward_return is None


def test_decision_record_invalid_decision_raises():
    from hifi.collective.schemas import DecisionRecord

    with pytest.raises(ValueError):
        DecisionRecord(
            ticker="AAPL",
            analysis_date="2022-03-31",
            agent_type="fundamental",
            decision="StrongBuy",  # invalid
            confidence=0.9,
        )


def test_decision_record_confidence_out_of_range_raises():
    from hifi.collective.schemas import DecisionRecord

    with pytest.raises(ValueError):
        DecisionRecord(
            ticker="AAPL",
            analysis_date="2022-03-31",
            agent_type="technical",
            decision="Hold",
            confidence=1.5,  # invalid
        )


def test_agent_performance_history_n_labeled_auto_computed():
    from hifi.collective.schemas import AgentPerformanceHistory, DecisionRecord

    records = [
        DecisionRecord(ticker="AAPL", analysis_date="2022-03-31",
                       agent_type="fundamental", decision="Buy", confidence=0.8,
                       outcome_correct=True),
        DecisionRecord(ticker="AAPL", analysis_date="2022-06-30",
                       agent_type="fundamental", decision="Hold", confidence=0.6,
                       outcome_correct=False),
        DecisionRecord(ticker="AAPL", analysis_date="2022-09-30",
                       agent_type="fundamental", decision="Sell", confidence=0.7),
    ]
    history = AgentPerformanceHistory(
        records=records,
        weights={"fundamental": 0.5},
        last_updated="2024-01-01",
        n_labeled=999,  # will be overridden by model_validator
    )
    assert history.n_labeled == 2   # only 2 records have outcome_correct not None
