"""
Unit tests for herding_coefficient, consensus_stability, and compute_rolling_metrics
(P9-E2, David §5.6.3, §5.6.4).

Scientific grounding:
- κ (herding): mean fraction of agents voting with the plurality per period.
  Near 1.0 = systematic herding. Near 1/3 (for 3 options) = independence.
- S (consensus stability): fraction of consecutive-equal decisions in W periods.
  S=1.0: never changed. S=0.0: changed every period.
- Both return None when fewer than W records exist — epistemic honesty.
"""

import pytest

from hifi.collective.metrics import (
    compute_rolling_metrics,
    consensus_stability,
    herding_coefficient,
)

# ---------------------------------------------------------------------------
# herding_coefficient (David §5.6.3)
# ---------------------------------------------------------------------------


def test_herding_unanimous_all_periods():
    """All agents agree every period → a_t = 1.0 → κ = 1.0."""
    # 5 periods, 3 agents each voting the same
    votes = [["Buy", "Buy", "Buy"]] * 5
    result = herding_coefficient(votes, w=5)
    assert result == pytest.approx(1.0)


def test_herding_two_agents_50_50_every_period():
    """2 agents, always 1 Buy + 1 Sell → a_t = 0.5 → κ = 0.5."""
    votes = [["Buy", "Sell"]] * 5
    result = herding_coefficient(votes, w=5)
    assert result == pytest.approx(0.5)


def test_herding_three_agents_two_agree():
    """3 agents, 2 agree: a_t = 2/3 every period → κ = 2/3."""
    votes = [["Buy", "Buy", "Sell"]] * 5
    result = herding_coefficient(votes, w=5)
    assert result == pytest.approx(2 / 3)


def test_herding_insufficient_history_returns_none():
    """len < W → None (insufficient history)."""
    votes = [["Buy", "Sell"]] * 4
    assert herding_coefficient(votes, w=5) is None


def test_herding_exactly_w_records_returns_float():
    """len == W is sufficient — should return float, not None."""
    votes = [["Buy", "Buy"]] * 5
    result = herding_coefficient(votes, w=5)
    assert result is not None
    assert isinstance(result, float)


def test_herding_w_minus_one_returns_none():
    """len == W-1 is insufficient — must return None."""
    votes = [["Hold"]] * 4
    assert herding_coefficient(votes, w=5) is None


def test_herding_uses_last_w_periods():
    """Only the last W periods matter; earlier history is ignored."""
    # First 10 periods: perfectly herded (κ would be 1.0 over those)
    # Last 5 periods: 50/50 split (κ = 0.5)
    old_periods = [["Buy", "Buy"]] * 10
    new_periods = [["Buy", "Sell"]] * 5
    votes = old_periods + new_periods

    result = herding_coefficient(votes, w=5)
    assert result == pytest.approx(0.5)


def test_herding_w10_and_w20_from_20_period_sequence():
    """20-period sequence: W=10 and W=20 both produce floats."""
    votes = [["Buy", "Hold", "Sell"]] * 20  # 3-way split, a_t = 1/3

    k10 = herding_coefficient(votes, w=10)
    k20 = herding_coefficient(votes, w=20)

    assert k10 is not None
    assert k20 is not None
    assert k10 == pytest.approx(1 / 3)
    assert k20 == pytest.approx(1 / 3)


def test_herding_w20_insufficient_for_shorter_sequence():
    """7 periods: W=5 → float, w=10 → None, w=20 → None."""
    votes = [["Buy", "Buy"]] * 7

    assert herding_coefficient(votes, w=5) is not None
    assert herding_coefficient(votes, w=10) is None
    assert herding_coefficient(votes, w=20) is None


def test_herding_empty_sequence_returns_none():
    assert herding_coefficient([], w=5) is None


def test_herding_single_agent_per_period_always_1():
    """Single agent always agrees with itself → κ = 1.0."""
    votes = [["Hold"]] * 5
    assert herding_coefficient(votes, w=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# consensus_stability (David §5.6.4)
# ---------------------------------------------------------------------------


def test_stability_same_decision_always():
    """Decision never changes → S = 1.0."""
    decisions = ["Buy"] * 5
    assert consensus_stability(decisions, w=5) == pytest.approx(1.0)


def test_stability_alternating_every_period():
    """Decision alternates every period → S = 0.0."""
    decisions = ["Buy", "Sell", "Buy", "Sell", "Buy"]
    assert consensus_stability(decisions, w=5) == pytest.approx(0.0)


def test_stability_three_stable_one_change():
    """5 periods: 3 stable pairs, 1 change → S = 3/4."""
    decisions = ["Buy", "Buy", "Buy", "Sell", "Buy"]
    # Pairs: (Buy,Buy)=1, (Buy,Buy)=1, (Buy,Sell)=0, (Sell,Buy)=0
    # Wait: that's 2 stable out of 4 pairs → S = 2/4 = 0.5
    # Let me recalculate: 5 decisions → 4 consecutive pairs
    # ["Buy", "Buy", "Buy", "Sell", "Buy"]
    # t=0: Buy==Buy → stable
    # t=1: Buy==Buy → stable
    # t=2: Buy==Sell → change
    # t=3: Sell==Buy → change
    # 2 stable / 4 = 0.5
    assert consensus_stability(decisions, w=5) == pytest.approx(0.5)


def test_stability_four_stable_one_change():
    """["Buy", "Buy", "Buy", "Buy", "Sell"]: 3 stable, 1 change → 3/4."""
    decisions = ["Buy", "Buy", "Buy", "Buy", "Sell"]
    # Pairs: (B,B)=1, (B,B)=1, (B,B)=1, (B,S)=0 → 3/4
    assert consensus_stability(decisions, w=5) == pytest.approx(3 / 4)


def test_stability_insufficient_history_returns_none():
    decisions = ["Buy", "Sell", "Hold"]
    assert consensus_stability(decisions, w=5) is None


def test_stability_exactly_w_records_returns_float():
    decisions = ["Hold"] * 5
    result = consensus_stability(decisions, w=5)
    assert result is not None
    assert isinstance(result, float)


def test_stability_w_minus_one_returns_none():
    decisions = ["Buy"] * 4
    assert consensus_stability(decisions, w=5) is None


def test_stability_w1_returns_none():
    """W=1 is undefined (W-1=0 denominator) — guard returns None."""
    assert consensus_stability(["Buy"], w=1) is None


def test_stability_w_lt_2_returns_none():
    assert consensus_stability(["Buy", "Hold"], w=0) is None


def test_stability_uses_last_w_decisions():
    """Only the last W decisions matter; earlier history ignored."""
    old = ["Buy"] * 10   # stable
    new = ["Buy", "Sell", "Buy", "Sell", "Buy"]  # alternating → S=0.0
    decisions = old + new

    result = consensus_stability(decisions, w=5)
    assert result == pytest.approx(0.0)


def test_stability_empty_returns_none():
    assert consensus_stability([], w=5) is None


# ---------------------------------------------------------------------------
# compute_rolling_metrics
# ---------------------------------------------------------------------------


def test_compute_rolling_metrics_returns_correct_keys():
    """Result has exactly 6 keys: kappa_W{5,10,20} and stability_W{5,10,20}."""
    votes = [["Buy"]] * 3
    decisions = ["Buy"] * 3
    result = compute_rolling_metrics(votes, decisions)

    expected_keys = {
        "kappa_W5", "kappa_W10", "kappa_W20",
        "stability_W5", "stability_W10", "stability_W20",
    }
    assert set(result.keys()) == expected_keys


def test_compute_rolling_metrics_all_float_for_20_period_sequence():
    """20-period sequence: all windows produce float values."""
    votes = [["Buy", "Hold"]] * 20
    decisions = ["Buy"] * 20
    result = compute_rolling_metrics(votes, decisions)

    for key, val in result.items():
        assert val is not None, f"{key} should be float for 20-period history"
        assert isinstance(val, float)


def test_compute_rolling_metrics_partial_none_for_short_sequence():
    """7-period sequence: W=5 → float; W=10 → None; W=20 → None."""
    votes = [["Buy", "Buy"]] * 7
    decisions = ["Buy"] * 7
    result = compute_rolling_metrics(votes, decisions)

    assert result["kappa_W5"] is not None
    assert result["stability_W5"] is not None
    assert result["kappa_W10"] is None
    assert result["stability_W10"] is None
    assert result["kappa_W20"] is None
    assert result["stability_W20"] is None


def test_compute_rolling_metrics_empty_all_none():
    """Empty sequences: all values are None."""
    result = compute_rolling_metrics([], [])

    for val in result.values():
        assert val is None


def test_compute_rolling_metrics_correctness_unanimous_stable():
    """
    20 periods of unanimous vote and stable decision:
    κ = 1.0 (every agent agrees), S = 1.0 (never changes).
    """
    votes = [["Buy", "Buy", "Buy"]] * 20
    decisions = ["Buy"] * 20
    result = compute_rolling_metrics(votes, decisions)

    assert result["kappa_W5"] == pytest.approx(1.0)
    assert result["kappa_W10"] == pytest.approx(1.0)
    assert result["kappa_W20"] == pytest.approx(1.0)
    assert result["stability_W5"] == pytest.approx(1.0)
    assert result["stability_W10"] == pytest.approx(1.0)
    assert result["stability_W20"] == pytest.approx(1.0)
