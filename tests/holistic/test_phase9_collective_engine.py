"""
Holistic structural tests for the Phase 9 Collective Decision Engine (P10-E3).

Tests the aggregation layer (voting, method_comparison, schema) without any LLM
invocation. Signals are constructed deterministically from known decision/confidence
tuples. All four aggregation methods are exercised via run_all_methods() directly.

Philosophical note (DJ-047):
  The previous version of this file used monkeypatch.setattr(make_llm, ...) to inject
  canned LLM responses into run_ensemble(). This conflated pipeline structure (which
  these tests own) with model behaviour (which make baseline-phase9 owns). That debt
  is now resolved: these tests call run_all_methods() and related functions directly
  with seeded inputs, exercising the full aggregation pipeline without any LLM.

  Specific behaviours validated by the stub approach that now belong in baseline-*:
  - That run_ensemble() returns a non-empty signals list from live agent calls
  - That the Sentiment fail-open path activates under real conditions
  - That the Contrarian Agent produces a valid ContrarianAnalysis JSON under load

What these tests DO validate:
  - All four aggregation methods run and return the four canonical keys
  - Confidence-weighted, majority, performance-weighted, and contrarian-adjusted
    math is correct for known inputs
  - Contrarian discount formula (1 - 0.5*confidence) and review_flagged threshold
  - EnsembleOutput schema construction and JSON round-trip with all Phase 9 fields
  - Backward compatibility: no contrarian → discount=1.0, review_flagged=False
  - Performance-weighted with uniform weights equals confidence-weighted
"""

from __future__ import annotations

import pytest

from hifi.agents.schemas import AgentSignal, ContrarianAnalysis
from hifi.collective.schemas import EnsembleDecision, EnsembleOutput
from hifi.collective.voting import (
    confidence_weighted_vote,
    contrarian_adjusted_vote,
    majority_vote,
    performance_weighted_vote,
    run_all_methods,
)

# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------


def _sig(
    agent_type: str,
    decision: str,
    confidence: float,
    ticker: str = "AAPL",
    as_of_date: str = "2023-03-31",
) -> AgentSignal:
    return AgentSignal(
        ticker=ticker,
        as_of_date=as_of_date,
        decision=decision,
        confidence=confidence,
        rationale=f"Test rationale for {agent_type}.",
        key_concern="Test concern.",
        model_id="test-model",
        agent_type=agent_type,
    )


def _contrarian(confidence: float = 0.65) -> ContrarianAnalysis:
    return ContrarianAnalysis(
        alternative_thesis="Bear case: rates remain elevated longer than priced.",
        risk_scenario="Credit tightening triggers 15% correction.",
        counterargument="Consensus underestimates rate duration.",
        confidence=confidence,
        prompt_version="test",
    )


# ---------------------------------------------------------------------------
# Tests: run_all_methods — canonical keys and structure
# ---------------------------------------------------------------------------


def test_run_all_methods_has_four_canonical_keys():
    """run_all_methods always returns the four canonical method keys."""
    signals = [_sig("fundamental", "Buy", 0.80), _sig("technical", "Hold", 0.65)]
    result = run_all_methods(signals=signals, contrarian=None, weights={})

    assert set(result.keys()) == {
        "majority",
        "confidence_weighted",
        "performance_weighted",
        "contrarian_adjusted",
    }


def test_run_all_methods_values_are_ensemble_decisions():
    """All four values are EnsembleDecision instances."""
    signals = [_sig("fundamental", "Buy", 0.80), _sig("technical", "Hold", 0.65)]
    result = run_all_methods(signals=signals, contrarian=None, weights={})

    for key, decision in result.items():
        assert isinstance(decision, EnsembleDecision), (
            f"method_comparison[{key!r}] is not an EnsembleDecision"
        )


def test_run_all_methods_decisions_are_valid_options():
    """All four methods produce Buy/Hold/Sell (or None when no signals)."""
    signals = [_sig("fundamental", "Buy", 0.80), _sig("technical", "Sell", 0.70)]
    result = run_all_methods(signals=signals, contrarian=None, weights={})

    for key, decision in result.items():
        assert decision.collective_decision in {"Buy", "Hold", "Sell", None}, (
            f"{key!r} produced unexpected decision: {decision.collective_decision!r}"
        )


def test_run_all_methods_empty_signals_all_none():
    """Empty signal list → all four methods have collective_decision=None."""
    result = run_all_methods(signals=[], contrarian=None, weights={})

    for key, decision in result.items():
        assert decision.collective_decision is None, (
            f"{key!r} expected None, got {decision.collective_decision!r}"
        )


# ---------------------------------------------------------------------------
# Tests: confidence_weighted method correctness
# ---------------------------------------------------------------------------


def test_confidence_weighted_vote_math():
    """
    Known inputs → known output.
    fund=Buy/0.80, tech=Hold/0.65: Buy score=0.80, Hold score=0.65, Sell=0.0
    total=1.45; Buy wins; confidence = 0.80/1.45 ≈ 0.5517
    """
    signals = [_sig("fundamental", "Buy", 0.80), _sig("technical", "Hold", 0.65)]
    result = confidence_weighted_vote(signals)

    assert result.collective_decision == "Buy"
    assert result.collective_confidence == pytest.approx(0.80 / 1.45, rel=1e-5)
    assert result.n_valid_signals == 2
    assert result.agreement is False
    assert result.total_score == pytest.approx(1.45, rel=1e-5)


def test_confidence_weighted_vote_tie_returns_hold():
    """Equal scores for Buy and Sell → tie → Hold with confidence 0.0."""
    signals = [
        _sig("fundamental", "Buy", 0.70),
        _sig("technical", "Sell", 0.70),
    ]
    result = confidence_weighted_vote(signals)

    assert result.collective_decision == "Hold"
    assert result.collective_confidence == 0.0


# ---------------------------------------------------------------------------
# Tests: majority_vote method correctness
# ---------------------------------------------------------------------------


def test_majority_vote_plurality():
    """Two Buy + one Sell → majority=Buy (2/3 votes)."""
    signals = [
        _sig("fundamental", "Buy", 0.60),
        _sig("technical", "Buy", 0.70),
        _sig("risk", "Sell", 0.80),
    ]
    result = majority_vote(signals)

    assert result.collective_decision == "Buy"
    assert result.collective_confidence == pytest.approx(2 / 3, rel=1e-5)
    assert result.n_valid_signals == 3


def test_majority_vote_and_cw_can_differ():
    """
    High-confidence outlier can win cw but lose majority.
    One high-conf Sell (0.90) vs two low-conf Buy (0.30 each):
      cw: Sell=0.90, Buy=0.60 → Sell wins
      majority: Buy=2, Sell=1 → Buy wins
    """
    signals = [
        _sig("fundamental", "Buy", 0.30),
        _sig("technical", "Buy", 0.30),
        _sig("risk", "Sell", 0.90),
    ]
    cw = confidence_weighted_vote(signals)
    mv = majority_vote(signals)

    assert cw.collective_decision == "Sell"
    assert mv.collective_decision == "Buy"


# ---------------------------------------------------------------------------
# Tests: performance_weighted_vote with uniform weights
# ---------------------------------------------------------------------------


def test_performance_weighted_uniform_equals_confidence_weighted():
    """
    With equal weights for all agent types, performance_weighted is mathematically
    identical to confidence_weighted (same score formula, same winner).
    """
    signals = [
        _sig("fundamental", "Buy", 0.80),
        _sig("technical", "Hold", 0.65),
        _sig("risk", "Buy", 0.60),
    ]
    weights = {"fundamental": 0.25, "technical": 0.25, "risk": 0.25, "macro": 0.25}

    cw = confidence_weighted_vote(signals)
    pw = performance_weighted_vote(signals, weights)

    assert cw.collective_decision == pw.collective_decision


def test_performance_weighted_differentiates_with_non_uniform_weights():
    """
    Non-uniform weights: a low-confidence winner under equal weights can be
    overridden when a high-weight agent votes differently.
    """
    signals = [
        _sig("fundamental", "Buy", 0.50),
        _sig("technical", "Buy", 0.50),
        _sig("risk", "Sell", 0.40),
    ]
    # With equal weights, Buy wins. With risk weight >> others, Sell might win.
    weights_equal = {"fundamental": 0.25, "technical": 0.25, "risk": 0.25}
    weights_risk_dominant = {"fundamental": 0.10, "technical": 0.10, "risk": 2.00}

    pw_equal = performance_weighted_vote(signals, weights_equal)
    pw_risk = performance_weighted_vote(signals, weights_risk_dominant)

    assert pw_equal.collective_decision == "Buy"
    assert pw_risk.collective_decision == "Sell"


# ---------------------------------------------------------------------------
# Tests: contrarian_adjusted discount formula (DJ-040)
# ---------------------------------------------------------------------------


def test_contrarian_adjusted_discount_formula():
    """discount = 1 - 0.5 * contrarian.confidence (alpha=0.5, DJ-040)."""
    signals = [_sig("fundamental", "Buy", 0.80), _sig("technical", "Hold", 0.65)]
    ca_result = contrarian_adjusted_vote(signals, _contrarian(confidence=0.65))

    expected_discount = 1.0 - 0.5 * 0.65  # = 0.675
    assert ca_result.contrarian_confidence_discount == pytest.approx(expected_discount, rel=1e-5)


def test_contrarian_adjusted_review_not_flagged_below_theta():
    """confidence=0.65 < theta=0.70 → review_flagged=False."""
    signals = [_sig("fundamental", "Buy", 0.80)]
    ca_result = contrarian_adjusted_vote(signals, _contrarian(confidence=0.65))

    assert ca_result.review_flagged is False


def test_contrarian_adjusted_review_flagged_above_theta():
    """confidence=0.80 > theta=0.70 → review_flagged=True."""
    signals = [_sig("fundamental", "Buy", 0.80)]
    ca_result = contrarian_adjusted_vote(signals, _contrarian(confidence=0.80))

    assert ca_result.review_flagged is True


def test_contrarian_adjusted_direction_unchanged():
    """Discounting never changes the winning direction (only reduces confidence)."""
    signals = [_sig("fundamental", "Buy", 0.80), _sig("technical", "Hold", 0.65)]
    base = confidence_weighted_vote(signals)
    discounted = contrarian_adjusted_vote(signals, _contrarian(confidence=0.90))

    assert discounted.collective_decision == base.collective_decision
    assert discounted.collective_confidence < base.collective_confidence


def test_contrarian_none_gives_discount_one_no_flag():
    """No ContrarianAnalysis → discount=1.0, review_flagged=False (neutral default)."""
    signals = [_sig("fundamental", "Buy", 0.80)]
    ca_result = contrarian_adjusted_vote(signals, contrarian=None)

    assert ca_result.contrarian_confidence_discount == 1.0
    assert ca_result.review_flagged is False


# ---------------------------------------------------------------------------
# Tests: run_all_methods — confidence_weighted == ensemble_decision
# ---------------------------------------------------------------------------


def test_cw_method_key_equals_standalone_cw_vote():
    """
    method_comparison['confidence_weighted'] is computed by the same function as
    ensemble_decision in ensemble_runner.py — results must be identical.
    """
    signals = [
        _sig("fundamental", "Buy", 0.80),
        _sig("technical", "Hold", 0.65),
        _sig("risk", "Hold", 0.60),
    ]
    result = run_all_methods(signals=signals, contrarian=None, weights={})
    standalone = confidence_weighted_vote(signals)

    assert result["confidence_weighted"].collective_decision == standalone.collective_decision
    assert result["confidence_weighted"].collective_confidence == standalone.collective_confidence
    assert result["confidence_weighted"].n_valid_signals == standalone.n_valid_signals


# ---------------------------------------------------------------------------
# Tests: EnsembleOutput JSON round-trip with Phase 9 fields
# ---------------------------------------------------------------------------


def _build_ensemble_output(
    ticker: str = "AAPL",
    as_of_date: str = "2023-03-31",
) -> EnsembleOutput:
    """Construct a structurally valid EnsembleOutput with all Phase 9 fields."""
    from hifi.agents.schemas import FundamentalAnalysis, TechnicalAnalysis

    fund_sig = _sig("fundamental", "Buy", 0.80, ticker, as_of_date)
    tech_sig = _sig("technical", "Hold", 0.65, ticker, as_of_date)
    signals = [fund_sig, tech_sig]

    contrarian = _contrarian(0.65)
    method_comparison = run_all_methods(
        signals=signals, contrarian=contrarian, weights={}
    )

    return EnsembleOutput(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamental_analysis=FundamentalAnalysis(
            signal=fund_sig,
            financial_ratios={}, growth_metrics={}, valuation_context={},
            macro_snapshot={}, prompt_version="test", latency_ms=0.0,
        ),
        technical_analysis=TechnicalAnalysis(
            signal=tech_sig,
            technical_indicators={}, risk_metrics={},
            prompt_version="test", latency_ms=0.0,
        ),
        ensemble_decision=method_comparison["confidence_weighted"],
        latency_ms=0.0,
        signals=signals,
        aggregation_method="confidence_weighted",
        method_comparison=method_comparison,
    )


def test_ensemble_output_json_roundtrip_lossless():
    """model_dump_json → model_validate_json is lossless with all Phase 9 fields."""
    output = _build_ensemble_output()
    restored = EnsembleOutput.model_validate_json(output.model_dump_json())

    assert restored.ticker == output.ticker
    assert restored.aggregation_method == output.aggregation_method
    assert len(restored.signals) == len(output.signals)
    assert set(restored.method_comparison.keys()) == set(output.method_comparison.keys())
    assert (
        restored.method_comparison["confidence_weighted"].collective_decision
        == output.method_comparison["confidence_weighted"].collective_decision
    )


def test_ensemble_output_method_comparison_has_four_keys():
    """EnsembleOutput.method_comparison always has exactly 4 keys."""
    output = _build_ensemble_output()

    assert len(output.method_comparison) == 4


def test_ensemble_output_contrarian_adjusted_fields_in_json():
    """Contrarian integration fields survive JSON round-trip."""
    output = _build_ensemble_output()
    restored = EnsembleOutput.model_validate_json(output.model_dump_json())

    ca = restored.method_comparison["contrarian_adjusted"]
    assert ca.contrarian_confidence_discount == pytest.approx(0.675, rel=1e-5)
    assert ca.review_flagged is False
