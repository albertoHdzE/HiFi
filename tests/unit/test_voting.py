"""Unit tests for confidence_weighted_vote (P4-E3-T6 through T9)."""


import pytest

from hifi.agents.schemas import AgentSignal
from hifi.collective.voting import confidence_weighted_vote


def _sig(decision: str, confidence: float, agent_type: str = "fundamental") -> AgentSignal:
    return AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision=decision,
        confidence=confidence,
        rationale="Test rationale with RSI of 48.0.",
        key_concern="Test concern.",
        data_gaps=[],
        call_ids=["abc"],
        model_id="test-model",
        agent_type=agent_type,
    )


# ---------------------------------------------------------------------------
# Unanimous cases
# ---------------------------------------------------------------------------


def test_unanimous_buy():
    s1 = _sig("Buy", 0.8)
    s2 = _sig("Buy", 0.6, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.collective_decision == "Buy"
    assert d.agreement is True
    assert d.disagreement_entropy == pytest.approx(0.0)
    assert d.n_valid_signals == 2
    assert d.collective_confidence == pytest.approx(1.0)


def test_unanimous_sell():
    s1 = _sig("Sell", 0.9)
    s2 = _sig("Sell", 0.7, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.collective_decision == "Sell"
    assert d.agreement is True
    assert d.disagreement_entropy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Disagreement cases: confidence-weighted winner
# ---------------------------------------------------------------------------


def test_disagree_buy_wins():
    s1 = _sig("Buy", 0.80)
    s2 = _sig("Sell", 0.60, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.collective_decision == "Buy"
    assert d.agreement is False
    assert d.collective_confidence == pytest.approx(0.80 / 1.40)


def test_disagree_sell_wins():
    s1 = _sig("Buy", 0.40)
    s2 = _sig("Sell", 0.90, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.collective_decision == "Sell"
    assert d.collective_confidence == pytest.approx(0.90 / 1.30)


def test_disagree_hold_wins():
    s1 = _sig("Hold", 0.70)
    s2 = _sig("Sell", 0.50, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.collective_decision == "Hold"
    assert d.agreement is False


# ---------------------------------------------------------------------------
# Tie defaults to Hold, collective_confidence = 0.0
# ---------------------------------------------------------------------------


def test_tie_defaults_to_hold():
    # Buy(0.5) vs Sell(0.5): equal confidence, equal scores -> tie
    s1 = _sig("Buy", 0.5)
    s2 = _sig("Sell", 0.5, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.collective_decision == "Hold"
    assert d.collective_confidence == pytest.approx(0.0)


def test_three_way_tie_defaults_to_hold():
    # Three agents each voting differently, equal confidence
    s1 = _sig("Buy", 0.5)
    s2 = _sig("Hold", 0.5, "technical")
    s3 = _sig("Sell", 0.5, "fundamental")
    d = confidence_weighted_vote([s1, s2, s3])
    assert d.collective_decision == "Hold"
    assert d.collective_confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Zero signals
# ---------------------------------------------------------------------------


def test_zero_signals_returns_none_decision():
    d = confidence_weighted_vote([])
    assert d.collective_decision is None
    assert d.n_valid_signals == 0
    assert d.total_score == pytest.approx(0.0)


def test_all_none_signals_returns_none_decision():
    d = confidence_weighted_vote([None, None])
    assert d.collective_decision is None
    assert d.n_valid_signals == 0


# ---------------------------------------------------------------------------
# Single valid signal among Nones
# ---------------------------------------------------------------------------


def test_single_valid_signal_wins():
    s1 = _sig("Buy", 0.7)
    d = confidence_weighted_vote([s1, None])
    assert d.collective_decision == "Buy"
    assert d.n_valid_signals == 1
    assert d.agreement is True


# ---------------------------------------------------------------------------
# Entropy correctness (P4-E3-T10)
# ---------------------------------------------------------------------------


def test_entropy_zero_unanimous():
    s1 = _sig("Hold", 0.6)
    s2 = _sig("Hold", 0.8, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.disagreement_entropy == pytest.approx(0.0)


def test_entropy_one_even_split():
    # Buy + Sell: p_Buy = 0.5, p_Sell = 0.5 -> H = 1.0
    s1 = _sig("Buy", 0.6)
    s2 = _sig("Sell", 0.8, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.disagreement_entropy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Opinion dispersion (P4-E3-T11)
# ---------------------------------------------------------------------------


def test_dispersion_zero_equal_confidence():
    s1 = _sig("Buy", 0.7)
    s2 = _sig("Buy", 0.7, "technical")
    d = confidence_weighted_vote([s1, s2])
    assert d.opinion_dispersion == pytest.approx(0.0)


def test_dispersion_formula_two_agents():
    # |c1 - c2| / 2
    s1 = _sig("Buy", 0.8)
    s2 = _sig("Sell", 0.4, "technical")
    d = confidence_weighted_vote([s1, s2])
    expected = abs(0.8 - 0.4) / 2
    assert d.opinion_dispersion == pytest.approx(expected)
