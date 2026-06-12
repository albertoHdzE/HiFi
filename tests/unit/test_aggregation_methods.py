"""
Unit tests for majority_vote, performance_weighted_vote, contrarian_adjusted_vote,
and run_all_methods (P9-E1).

Scientific focus:
- majority_vote: equal-weight mode; collective_confidence = winning_count / n_valid
- performance_weighted_vote: w_i * c_i per option; falls back to cw_vote with uniform weights
- contrarian_adjusted_vote: discount = 1 - 0.5*c applied post-vote; direction unchanged
- run_all_methods: all four keys always present; cw key equals confidence_weighted_vote()
"""


import pytest

from hifi.agents.schemas import AgentSignal, ContrarianAnalysis
from hifi.collective.voting import (
    confidence_weighted_vote,
    contrarian_adjusted_vote,
    majority_vote,
    performance_weighted_vote,
    run_all_methods,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig(decision: str, confidence: float, agent_type: str = "fundamental") -> AgentSignal:
    return AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision=decision,
        confidence=confidence,
        rationale=f"Test rationale for {decision}.",
        key_concern="Test concern.",
        data_gaps=[],
        call_ids=["abc"],
        model_id="stub-model",
        agent_type=agent_type,
    )


def _contrarian(confidence: float) -> ContrarianAnalysis:
    return ContrarianAnalysis(
        alternative_thesis="Bear case: revenue growth may stall.",
        risk_scenario="Credit tightening triggers 15% equity correction.",
        counterargument="Consensus underestimates refinancing risk.",
        confidence=confidence,
        prompt_version="contrarian_v1",
    )


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------


def test_majority_vote_clear_winner():
    """2 Buy, 1 Sell → Buy wins; collective_confidence = 2/3."""
    s1 = _sig("Buy", 0.9)
    s2 = _sig("Buy", 0.5, "technical")
    s3 = _sig("Sell", 0.8, "risk")
    d = majority_vote([s1, s2, s3])

    assert d.collective_decision == "Buy"
    assert d.collective_confidence == pytest.approx(2 / 3)
    assert d.n_valid_signals == 3
    assert d.agreement is False


def test_majority_vote_unanimous():
    """All three agents vote Hold → agreement=True, cc=1.0."""
    s1 = _sig("Hold", 0.6)
    s2 = _sig("Hold", 0.7, "technical")
    s3 = _sig("Hold", 0.8, "risk")
    d = majority_vote([s1, s2, s3])

    assert d.collective_decision == "Hold"
    assert d.collective_confidence == pytest.approx(1.0)
    assert d.agreement is True


def test_majority_vote_two_way_tie_defaults_hold():
    """1 Buy, 1 Sell tie (equal votes) → Hold, cc=0.0."""
    s1 = _sig("Buy", 0.9)
    s2 = _sig("Sell", 0.4, "technical")
    d = majority_vote([s1, s2])

    assert d.collective_decision == "Hold"
    assert d.collective_confidence == pytest.approx(0.0)


def test_majority_vote_three_way_tie():
    """One vote each → Hold, cc=0.0."""
    s1 = _sig("Buy", 0.9)
    s2 = _sig("Hold", 0.5, "technical")
    s3 = _sig("Sell", 0.8, "risk")
    d = majority_vote([s1, s2, s3])

    assert d.collective_decision == "Hold"
    assert d.collective_confidence == pytest.approx(0.0)


def test_majority_vote_total_score_equals_n_valid():
    """total_score is the count of valid agents, not a confidence sum."""
    s1 = _sig("Buy", 0.9)
    s2 = _sig("Buy", 0.5, "technical")
    s3 = _sig("Sell", 0.3, "risk")
    d = majority_vote([s1, s2, s3])

    assert d.total_score == pytest.approx(3.0)  # count, not 0.9+0.5+0.3=1.7


def test_majority_vote_winning_score_equals_winning_count():
    """winning_score equals the count of the winning option, not confidence sum."""
    s1 = _sig("Buy", 0.9)
    s2 = _sig("Buy", 0.5, "technical")
    s3 = _sig("Sell", 0.3, "risk")
    d = majority_vote([s1, s2, s3])

    assert d.winning_score == pytest.approx(2.0)  # 2 votes for Buy


def test_majority_vote_no_valid_signals():
    """Empty and all-None: collective_decision=None, n_valid=0."""
    assert majority_vote([]).collective_decision is None
    assert majority_vote([None, None]).n_valid_signals == 0


def test_majority_vote_contrarian_fields_are_neutral_defaults():
    """majority_vote does not apply contrarian integration — fields are defaults."""
    d = majority_vote([_sig("Buy", 0.8)])
    assert d.contrarian_confidence_discount == pytest.approx(1.0)
    assert d.review_flagged is False


def test_majority_vote_diversity_metrics_match_cw_vote():
    """Entropy and dispersion computed identically to confidence_weighted_vote."""
    s1 = _sig("Buy", 0.8)
    s2 = _sig("Sell", 0.6, "technical")
    d_maj = majority_vote([s1, s2])
    d_cw = confidence_weighted_vote([s1, s2])

    assert d_maj.disagreement_entropy == pytest.approx(d_cw.disagreement_entropy)
    assert d_maj.opinion_dispersion == pytest.approx(d_cw.opinion_dispersion)
    assert d_maj.agreement == d_cw.agreement


# ---------------------------------------------------------------------------
# performance_weighted_vote
# ---------------------------------------------------------------------------


def test_performance_weighted_empty_weights_equals_cw_vote():
    """With empty weights, every agent gets weight=1.0 → same as cw_vote."""
    s1 = _sig("Buy", 0.8)
    s2 = _sig("Sell", 0.6, "technical")

    d_pw = performance_weighted_vote([s1, s2], weights={})
    d_cw = confidence_weighted_vote([s1, s2])

    assert d_pw.collective_decision == d_cw.collective_decision
    assert d_pw.collective_confidence == pytest.approx(d_cw.collective_confidence)


def test_performance_weighted_changes_winner():
    """
    With appropriate weights, performance_weighted can override cw_vote.

    Setup: fundamental says Sell/0.8, technical says Buy/0.6.
    cw_vote: Buy=0.6, Sell=0.8 → Sell wins.
    pw_vote with weights={fundamental: 0.1, technical: 0.9}:
      Buy = 0.9*0.6 = 0.54, Sell = 0.1*0.8 = 0.08 → Buy wins.
    """
    s1 = _sig("Sell", 0.8, "fundamental")
    s2 = _sig("Buy", 0.6, "technical")

    d_cw = confidence_weighted_vote([s1, s2])
    assert d_cw.collective_decision == "Sell"

    d_pw = performance_weighted_vote(
        [s1, s2], weights={"fundamental": 0.1, "technical": 0.9}
    )
    assert d_pw.collective_decision == "Buy"


def test_performance_weighted_unknown_type_fallback_to_1():
    """agent_type not in weights → weight=1.0 (equal contribution)."""
    s1 = _sig("Buy", 0.7, "macro")   # not in weights
    s2 = _sig("Sell", 0.5, "technical")
    weights = {"technical": 0.8}     # macro absent

    d = performance_weighted_vote([s1, s2], weights=weights)
    # macro weight=1.0 (fallback), Buy score = 1.0*0.7 = 0.70
    # technical weight=0.8, Sell score = 0.8*0.5 = 0.40
    assert d.collective_decision == "Buy"


def test_performance_weighted_no_valid_signals():
    d = performance_weighted_vote([None, None], weights={"fundamental": 0.9})
    assert d.collective_decision is None
    assert d.n_valid_signals == 0


def test_performance_weighted_contrarian_fields_are_neutral():
    d = performance_weighted_vote([_sig("Hold", 0.7)], weights={})
    assert d.contrarian_confidence_discount == pytest.approx(1.0)
    assert d.review_flagged is False


def test_performance_weighted_uniform_weights_same_direction_as_cw():
    """All equal weights preserve direction (only magnitude may differ slightly)."""
    s1 = _sig("Buy", 0.9)
    s2 = _sig("Sell", 0.4, "technical")
    uniform = {"fundamental": 1.0, "technical": 1.0}

    d_pw = performance_weighted_vote([s1, s2], weights=uniform)
    d_cw = confidence_weighted_vote([s1, s2])

    assert d_pw.collective_decision == d_cw.collective_decision


# ---------------------------------------------------------------------------
# contrarian_adjusted_vote
# ---------------------------------------------------------------------------


def test_contrarian_adjusted_no_contrarian_equals_cw_vote():
    """When contrarian=None, result is identical to confidence_weighted_vote."""
    s1 = _sig("Buy", 0.8)
    s2 = _sig("Sell", 0.5, "technical")

    d_ca = contrarian_adjusted_vote([s1, s2], contrarian=None)
    d_cw = confidence_weighted_vote([s1, s2])

    assert d_ca.collective_decision == d_cw.collective_decision
    assert d_ca.collective_confidence == pytest.approx(d_cw.collective_confidence)
    assert d_ca.contrarian_confidence_discount == pytest.approx(1.0)
    assert d_ca.review_flagged is False


def test_contrarian_adjusted_discount_applied_correctly():
    """
    contrarian.confidence = 0.5 → discount = 1 - 0.5*0.5 = 0.75
    discounted_cc = base_cc * 0.75
    """
    s1 = _sig("Buy", 0.8)
    base = confidence_weighted_vote([s1])
    c = _contrarian(confidence=0.5)

    d = contrarian_adjusted_vote([s1], contrarian=c)

    expected_discount = 1.0 - 0.5 * 0.5
    assert d.contrarian_confidence_discount == pytest.approx(expected_discount)
    assert d.collective_confidence == pytest.approx(
        base.collective_confidence * expected_discount
    )


def test_contrarian_adjusted_review_flagged_above_threshold():
    """contrarian.confidence = 0.8 > 0.70 → review_flagged = True."""
    s1 = _sig("Buy", 0.8)
    d = contrarian_adjusted_vote([s1], contrarian=_contrarian(confidence=0.8))

    assert d.review_flagged is True


def test_contrarian_adjusted_review_not_flagged_at_boundary():
    """contrarian.confidence = 0.70 is NOT > 0.70 → review_flagged = False."""
    s1 = _sig("Buy", 0.8)
    d = contrarian_adjusted_vote([s1], contrarian=_contrarian(confidence=0.70))

    assert d.review_flagged is False


def test_contrarian_adjusted_review_flagged_just_above_boundary():
    """contrarian.confidence = 0.701 > 0.70 → review_flagged = True."""
    s1 = _sig("Buy", 0.8)
    d = contrarian_adjusted_vote([s1], contrarian=_contrarian(confidence=0.701))

    assert d.review_flagged is True


def test_contrarian_adjusted_winning_direction_unchanged():
    """Discount compresses confidence but never changes Buy/Hold/Sell direction."""
    s1 = _sig("Buy", 0.9)
    s2 = _sig("Hold", 0.3, "technical")
    base = confidence_weighted_vote([s1, s2])
    d = contrarian_adjusted_vote([s1, s2], contrarian=_contrarian(confidence=0.99))

    # Even with maximum contrarian confidence, direction is preserved
    assert d.collective_decision == base.collective_decision


def test_contrarian_adjusted_discount_factor_stored_separately():
    """
    contrarian_confidence_discount stores the factor (1 - α*c),
    not the discounted confidence itself. This preserves the ability
    to reconstruct the undiscounted confidence in Phase 10 analysis.
    """
    s1 = _sig("Buy", 0.80)
    c = _contrarian(confidence=0.6)
    d = contrarian_adjusted_vote([s1], contrarian=c)

    expected_factor = 1.0 - 0.5 * 0.6   # = 0.70
    base = confidence_weighted_vote([s1])
    reconstructed_base_cc = d.collective_confidence / expected_factor

    assert d.contrarian_confidence_discount == pytest.approx(expected_factor)
    assert reconstructed_base_cc == pytest.approx(base.collective_confidence)


def test_contrarian_adjusted_no_valid_signals():
    d = contrarian_adjusted_vote([None], contrarian=_contrarian(0.8))
    assert d.collective_decision is None


# ---------------------------------------------------------------------------
# run_all_methods
# ---------------------------------------------------------------------------


def test_run_all_methods_returns_four_keys():
    s1 = _sig("Buy", 0.8)
    s2 = _sig("Hold", 0.6, "technical")
    result = run_all_methods([s1, s2], contrarian=None, weights={})

    assert set(result.keys()) == {
        "majority",
        "confidence_weighted",
        "performance_weighted",
        "contrarian_adjusted",
    }


def test_run_all_methods_cw_key_equals_confidence_weighted_vote():
    """method_comparison["confidence_weighted"] must equal confidence_weighted_vote()."""
    s1 = _sig("Buy", 0.8)
    s2 = _sig("Sell", 0.5, "technical")

    result = run_all_methods([s1, s2], contrarian=None, weights={})
    expected = confidence_weighted_vote([s1, s2])

    assert result["confidence_weighted"].collective_decision == expected.collective_decision
    assert result["confidence_weighted"].collective_confidence == pytest.approx(
        expected.collective_confidence
    )


def test_run_all_methods_ca_applies_discount_when_contrarian_given():
    """contrarian_adjusted key has discount < 1.0 when contrarian is provided."""
    s1 = _sig("Buy", 0.8)
    result = run_all_methods(
        [s1], contrarian=_contrarian(confidence=0.6), weights={}
    )

    assert result["contrarian_adjusted"].contrarian_confidence_discount < 1.0
    assert result["contrarian_adjusted"].contrarian_confidence_discount == pytest.approx(0.70)


def test_run_all_methods_no_contrarian_empty_weights_pw_equals_cw():
    """With no contrarian and uniform weights, performance_weighted == confidence_weighted."""
    s1 = _sig("Buy", 0.9)
    s2 = _sig("Hold", 0.4, "technical")
    result = run_all_methods([s1, s2], contrarian=None, weights={})

    pw_decision = result["performance_weighted"].collective_decision
    cw_decision = result["confidence_weighted"].collective_decision
    assert pw_decision == cw_decision
    assert result["performance_weighted"].collective_confidence == pytest.approx(
        result["confidence_weighted"].collective_confidence
    )


def test_run_all_methods_with_none_signals():
    """None signals are handled gracefully by all methods."""
    s1 = _sig("Buy", 0.8)
    result = run_all_methods([s1, None], contrarian=None, weights={})

    for key, decision in result.items():
        assert decision.n_valid_signals == 1, f"{key}: expected 1 valid signal"
