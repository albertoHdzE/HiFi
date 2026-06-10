"""Unit tests for standalone diversity metric functions (P4-E3-T10 through T12)."""

import math

import pytest

from hifi.collective.metrics import (
    disagreement_entropy,
    opinion_dispersion,
    pairwise_diversity,
)

# ---------------------------------------------------------------------------
# disagreement_entropy (David §5.6.1)
# ---------------------------------------------------------------------------


def test_entropy_empty_returns_zero():
    assert disagreement_entropy([]) == pytest.approx(0.0)


def test_entropy_unanimous_returns_zero():
    assert disagreement_entropy(["Buy", "Buy", "Buy"]) == pytest.approx(0.0)


def test_entropy_even_two_way_split():
    # p_Buy = 0.5, p_Sell = 0.5 -> H = 1.0
    assert disagreement_entropy(["Buy", "Sell"]) == pytest.approx(1.0)


def test_entropy_even_three_way_split():
    # p = 1/3 each -> H = log2(3) ~= 1.585
    result = disagreement_entropy(["Buy", "Hold", "Sell"])
    assert result == pytest.approx(math.log2(3), rel=1e-6)


def test_entropy_maximum_is_log2_3():
    result = disagreement_entropy(["Buy", "Hold", "Sell"])
    assert result <= math.log2(3) + 1e-9


def test_entropy_single_element_returns_zero():
    assert disagreement_entropy(["Hold"]) == pytest.approx(0.0)


def test_entropy_uneven_split():
    # Buy(3), Sell(1): p_Buy=0.75, p_Sell=0.25
    # H = -(0.75*log2(0.75) + 0.25*log2(0.25))
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert disagreement_entropy(["Buy", "Buy", "Buy", "Sell"]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# opinion_dispersion (David §5.6.2)
# ---------------------------------------------------------------------------


def test_dispersion_empty_returns_zero():
    assert opinion_dispersion([]) == pytest.approx(0.0)


def test_dispersion_single_element_returns_zero():
    assert opinion_dispersion([0.8]) == pytest.approx(0.0)


def test_dispersion_equal_confidences_returns_zero():
    assert opinion_dispersion([0.7, 0.7, 0.7]) == pytest.approx(0.0)


def test_dispersion_two_agents_formula():
    # D = |c1 - c2| / 2
    result = opinion_dispersion([0.8, 0.4])
    assert result == pytest.approx(0.2)


def test_dispersion_three_agents():
    # mean = 0.6; deviations: 0.2, 0.0, 0.2; D = 0.4/3
    result = opinion_dispersion([0.8, 0.6, 0.4])
    assert result == pytest.approx(0.4 / 3, rel=1e-6)


# ---------------------------------------------------------------------------
# pairwise_diversity (David §5.6.5 categorical)
# ---------------------------------------------------------------------------


def test_pairwise_diversity_empty_returns_zero():
    assert pairwise_diversity([]) == pytest.approx(0.0)


def test_pairwise_diversity_single_returns_zero():
    assert pairwise_diversity(["Buy"]) == pytest.approx(0.0)


def test_pairwise_diversity_all_agree_returns_zero():
    assert pairwise_diversity(["Buy", "Buy"]) == pytest.approx(0.0)


def test_pairwise_diversity_all_disagree_two_agents():
    # 1 pair, both disagree -> diversity = 1.0
    assert pairwise_diversity(["Buy", "Sell"]) == pytest.approx(1.0)


def test_pairwise_diversity_three_agents_all_same():
    assert pairwise_diversity(["Hold", "Hold", "Hold"]) == pytest.approx(0.0)


def test_pairwise_diversity_three_agents_two_disagree():
    # Pairs: (Buy,Buy)=agree, (Buy,Sell)=disagree, (Buy,Sell)=disagree
    # 2 disagree / 3 total pairs = 2/3
    result = pairwise_diversity(["Buy", "Buy", "Sell"])
    assert result == pytest.approx(2 / 3, rel=1e-6)


def test_pairwise_diversity_three_agents_all_different():
    # 3 pairs, all disagree -> diversity = 1.0
    assert pairwise_diversity(["Buy", "Hold", "Sell"]) == pytest.approx(1.0)
