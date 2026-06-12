"""
Holistic structural tests for Phase 8 agent population (P10-E3).

Phase 8 introduced four new agents (risk, macro, sentiment, contrarian) and
extended EnsembleOutput with optional fields for each. These tests validate the
structural correctness of the schema layer — correct defaults, optional field
handling, JSON round-trip, and backward compatibility with the Phase 4 two-agent
call convention.

Philosophical note (DJ-047):
  Tests that verify LLM behaviour (which decision did the model produce? was the
  rationale grounded?) require a live LLM invocation with real output to be
  meaningful. Injecting canned LLM responses via monkeypatching conflates two
  concerns — pipeline structure (testable deterministically) and model behaviour
  (testable only against real outputs). Behavioural assertions live exclusively
  in make baseline-phase8.

  These holistic tests are deliberately LLM-free: they construct EnsembleOutput
  objects directly from deterministic builders and assert structural invariants.
"""

from __future__ import annotations

from hifi.agents.schemas import (
    AgentSignal,
    ContrarianAnalysis,
    FundamentalAnalysis,
    MacroAnalysis,
    RiskAnalysis,
    SentimentAnalysis,
    TechnicalAnalysis,
)
from hifi.collective.schemas import EnsembleOutput
from hifi.collective.voting import run_all_methods

# ---------------------------------------------------------------------------
# EnsembleOutput builder
# ---------------------------------------------------------------------------


def _make_signal(
    agent_type: str,
    decision: str = "Buy",
    confidence: float = 0.70,
    ticker: str = "AAPL",
    as_of_date: str = "2023-03-31",
) -> AgentSignal:
    return AgentSignal(
        ticker=ticker,
        as_of_date=as_of_date,
        decision=decision,
        confidence=confidence,
        rationale=f"Synthetic {agent_type} rationale.",
        key_concern="Test concern.",
        model_id="test-model",
        agent_type=agent_type,
    )


def _make_fundamental_analysis(signal: AgentSignal) -> FundamentalAnalysis:
    return FundamentalAnalysis(
        signal=signal,
        financial_ratios={},
        growth_metrics={},
        valuation_context={},
        macro_snapshot={},
        prompt_version="test",
        latency_ms=0.0,
    )


def _make_technical_analysis(signal: AgentSignal) -> TechnicalAnalysis:
    return TechnicalAnalysis(
        signal=signal,
        technical_indicators={},
        risk_metrics={},
        prompt_version="test",
        latency_ms=0.0,
    )


def _make_risk_analysis(signal: AgentSignal) -> RiskAnalysis:
    return RiskAnalysis(
        signal=signal,
        risk_assessment="Moderate volatility regime.",
        recommended_position_size=0.05,
        risk_metrics={},
        prompt_version="test",
        latency_ms=0.0,
    )


def _make_macro_analysis(signal: AgentSignal) -> MacroAnalysis:
    return MacroAnalysis(
        signal=signal,
        regime_assessment="Late tightening cycle.",
        rationale="CPI decelerating.",
        macro_snapshot={},
        prompt_version="test",
        latency_ms=0.0,
    )


def _make_sentiment_analysis(signal: AgentSignal) -> SentimentAnalysis:
    return SentimentAnalysis(
        signal=signal,
        sentiment_summary="Insufficient Data — no filing passages retrieved.",
        notable_signals=[],
        prompt_version="test",
        latency_ms=0.0,
    )


def _make_contrarian_analysis(confidence: float = 0.65) -> ContrarianAnalysis:
    return ContrarianAnalysis(
        alternative_thesis="Rate cuts fully priced; real rates remain restrictive.",
        risk_scenario="Credit spread widening triggers 15% equity correction.",
        counterargument="Consensus underestimates duration of high-rate environment.",
        confidence=confidence,
        prompt_version="test",
        latency_ms=0.0,
    )


def _make_full_ensemble_output(
    ticker: str = "AAPL",
    as_of_date: str = "2023-03-31",
    include_phase8: bool = True,
    fund_decision: str = "Buy",
    tech_decision: str = "Hold",
) -> EnsembleOutput:
    """
    Construct a structurally complete EnsembleOutput without any LLM call.

    Signals are deterministically built from the supplied decision strings.
    All four aggregation methods are computed via run_all_methods(), matching
    the behaviour of ensemble_runner.py.
    """
    fund_sig = _make_signal("fundamental", fund_decision, 0.80, ticker, as_of_date)
    tech_sig = _make_signal("technical", tech_decision, 0.65, ticker, as_of_date)

    signals = [fund_sig, tech_sig]
    contrarian = None

    if include_phase8:
        risk_sig = _make_signal("risk", "Hold", 0.60, ticker, as_of_date)
        macro_sig = _make_signal("macro", "Buy", 0.55, ticker, as_of_date)
        # Sentiment fail-open: Hold / 0.0
        sent_sig = _make_signal("sentiment", "Hold", 0.00, ticker, as_of_date)
        signals.extend([risk_sig, macro_sig, sent_sig])
        contrarian = _make_contrarian_analysis()

    method_comparison = run_all_methods(
        signals=signals, contrarian=contrarian, weights={}
    )
    primary = method_comparison["confidence_weighted"]

    risk_analysis = _make_risk_analysis(
        _make_signal("risk", "Hold", 0.60, ticker, as_of_date)
    ) if include_phase8 else None
    macro_analysis = _make_macro_analysis(
        _make_signal("macro", "Buy", 0.55, ticker, as_of_date)
    ) if include_phase8 else None
    sentiment_analysis = _make_sentiment_analysis(
        _make_signal("sentiment", "Hold", 0.00, ticker, as_of_date)
    ) if include_phase8 else None

    return EnsembleOutput(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamental_analysis=_make_fundamental_analysis(fund_sig),
        technical_analysis=_make_technical_analysis(tech_sig),
        ensemble_decision=primary,
        latency_ms=0.0,
        risk_analysis=risk_analysis,
        macro_analysis=macro_analysis,
        sentiment_analysis=sentiment_analysis,
        contrarian_analysis=contrarian,
        signals=signals,
        aggregation_method="confidence_weighted",
        method_comparison=method_comparison,
    )


# ---------------------------------------------------------------------------
# Tests: EnsembleOutput Phase 8 optional fields
# ---------------------------------------------------------------------------


def test_phase8_optional_fields_are_none_without_agents():
    """
    EnsembleOutput constructed with only Phase 4 agents has None for all Phase 8
    optional fields. This is the backward-compat guarantee (DJ-038).
    """
    output = _make_full_ensemble_output(include_phase8=False)

    assert output.risk_analysis is None
    assert output.macro_analysis is None
    assert output.sentiment_analysis is None
    assert output.contrarian_analysis is None


def test_phase8_optional_fields_populated_with_all_agents():
    """When Phase 8 agents are included, all four optional fields are non-None."""
    output = _make_full_ensemble_output(include_phase8=True)

    assert output.risk_analysis is not None
    assert output.macro_analysis is not None
    assert output.sentiment_analysis is not None
    assert output.contrarian_analysis is not None


def test_contrarian_analysis_schema_validity():
    """ContrarianAnalysis validates non-empty text fields and confidence range."""
    c = _make_contrarian_analysis(confidence=0.70)

    assert c.alternative_thesis.strip()
    assert c.risk_scenario.strip()
    assert c.counterargument.strip()
    assert 0.0 <= c.confidence <= 1.0


def test_sentiment_fail_open_signal_is_hold_zero_confidence():
    """Sentiment fail-open signal: decision=Hold, confidence=0.0."""
    sent_sig = _make_signal("sentiment", "Hold", 0.00)
    sa = _make_sentiment_analysis(sent_sig)

    assert sa.signal is not None
    assert sa.signal.decision == "Hold"
    assert sa.signal.confidence == 0.0
    assert "Insufficient" in sa.sentiment_summary


# ---------------------------------------------------------------------------
# Tests: backward compat — Phase 4 subset
# ---------------------------------------------------------------------------


def test_phase4_subset_has_four_method_comparison_keys():
    """
    Even with only fundamental + technical agents, method_comparison has the
    four canonical keys. run_all_methods() always produces all four.
    """
    output = _make_full_ensemble_output(include_phase8=False)

    assert set(output.method_comparison.keys()) == {
        "majority",
        "confidence_weighted",
        "performance_weighted",
        "contrarian_adjusted",
    }


def test_phase4_subset_contrarian_adjusted_discount_is_one():
    """
    Without a ContrarianAnalysis, contrarian_adjusted has discount=1.0 and
    review_flagged=False — the neutral default (DJ-040).
    """
    output = _make_full_ensemble_output(include_phase8=False)
    ca = output.method_comparison["contrarian_adjusted"]

    assert ca.contrarian_confidence_discount == 1.0
    assert ca.review_flagged is False


def test_phase4_subset_signals_has_two_entries():
    """Phase 4 subset: signals list has exactly two entries (fund + tech)."""
    output = _make_full_ensemble_output(include_phase8=False)

    assert len(output.signals) == 2
    agent_types = {s.agent_type for s in output.signals}
    assert agent_types == {"fundamental", "technical"}


# ---------------------------------------------------------------------------
# Tests: Phase 8 method_comparison with all agents
# ---------------------------------------------------------------------------


def test_phase8_method_comparison_has_four_keys():
    """Full 6-agent ensemble still has exactly four method_comparison keys."""
    output = _make_full_ensemble_output(include_phase8=True)

    assert len(output.method_comparison) == 4
    assert set(output.method_comparison.keys()) == {
        "majority",
        "confidence_weighted",
        "performance_weighted",
        "contrarian_adjusted",
    }


def test_phase8_ensemble_decision_matches_cw_method():
    """ensemble_decision == method_comparison['confidence_weighted'] (same function)."""
    output = _make_full_ensemble_output(include_phase8=True)

    cw = output.method_comparison["confidence_weighted"]
    ed = output.ensemble_decision

    assert cw.collective_decision == ed.collective_decision
    assert cw.collective_confidence == ed.collective_confidence
    assert cw.n_valid_signals == ed.n_valid_signals


def test_phase8_contrarian_discount_applied():
    """With contrarian confidence=0.65, discount = 1 - 0.5*0.65 = 0.675."""
    output = _make_full_ensemble_output(include_phase8=True)
    ca = output.method_comparison["contrarian_adjusted"]

    import pytest
    assert ca.contrarian_confidence_discount == pytest.approx(0.675, rel=1e-5)


def test_phase8_signals_contains_five_voting_agents():
    """Phase 8 ensemble with sentiment fail-open: 5 signals (fund,tech,risk,macro,sent)."""
    output = _make_full_ensemble_output(include_phase8=True)

    assert len(output.signals) == 5
    types = {s.agent_type for s in output.signals}
    assert types == {"fundamental", "technical", "risk", "macro", "sentiment"}


# ---------------------------------------------------------------------------
# Tests: JSON round-trip
# ---------------------------------------------------------------------------


def test_ensemble_output_json_roundtrip_phase4_subset():
    """Phase 4 subset EnsembleOutput round-trips through JSON losslessly."""
    output = _make_full_ensemble_output(include_phase8=False)
    restored = EnsembleOutput.model_validate_json(output.model_dump_json())

    assert restored.ticker == output.ticker
    assert restored.risk_analysis is None
    assert len(restored.method_comparison) == 4
    assert len(restored.signals) == 2


def test_ensemble_output_json_roundtrip_full_phase8():
    """Full Phase 8 EnsembleOutput round-trips through JSON losslessly."""
    output = _make_full_ensemble_output(include_phase8=True)
    restored = EnsembleOutput.model_validate_json(output.model_dump_json())

    assert restored.ticker == output.ticker
    assert restored.risk_analysis is not None
    assert restored.macro_analysis is not None
    assert restored.sentiment_analysis is not None
    assert restored.contrarian_analysis is not None
    assert restored.contrarian_analysis.confidence == output.contrarian_analysis.confidence
    assert len(restored.signals) == len(output.signals)


def test_ensemble_output_aggregation_method_field_default():
    """aggregation_method defaults to 'confidence_weighted'."""
    output = _make_full_ensemble_output()

    assert output.aggregation_method == "confidence_weighted"
