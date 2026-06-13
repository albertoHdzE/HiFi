"""
Holistic evaluation pipeline tests for Phase 11 (P11-E4-T5, DJ-058).

Tests the three-tier evaluation logic with synthetic EnsembleOutput objects.
No LLM, no live services, no monkeypatching of business logic.

The tests verify that the comparison logic (diversity delta, GR delta) is
correctly wired to FineTuneEvaluationResult fields.
"""

from __future__ import annotations

from datetime import UTC, datetime

from hifi.models.training_data import FineTuneEvaluationResult

# ---------------------------------------------------------------------------
# Synthetic evaluation helpers
# ---------------------------------------------------------------------------


def _make_result(
    ticker: str,
    base_tech_gr: float,
    ft_tech_gr: float,
    base_diversity: float,
    ft_diversity: float,
) -> FineTuneEvaluationResult:
    """Construct a FineTuneEvaluationResult with minimal required fields."""
    return FineTuneEvaluationResult(
        ticker=ticker,
        analysis_date="2023-03-31",
        base_technical_gr=base_tech_gr,
        base_fundamental_gr=1.0,
        finetuned_technical_gr=ft_tech_gr,
        finetuned_fundamental_gr=1.0,
        base_pairwise_diversity=base_diversity,
        finetuned_pairwise_diversity=ft_diversity,
        base_disagreement_entropy=0.6,
        finetuned_disagreement_entropy=0.6,
        generated_at=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# Tier 1: Individual quality (GR improvement)
# ---------------------------------------------------------------------------


def test_tier1_gr_improvement_computation() -> None:
    """
    Tier 1: GR improvement >= 0.05 is correctly flagged.

    Simulates the 'before vs after' comparison that run_phase11_evaluation.py
    performs after running the ensemble with base and fine-tuned models.
    """
    result = _make_result("AAPL", base_tech_gr=0.667, ft_tech_gr=0.720, base_diversity=0.5, ft_diversity=0.48)  # noqa: E501
    assert result.gr_improved_technical is True
    assert result.finetuned_technical_gr - result.base_technical_gr >= 0.05


def test_tier1_gr_no_improvement() -> None:
    """Tier 1: No flag when improvement < 0.05."""
    result = _make_result("JPM", base_tech_gr=0.667, ft_tech_gr=0.700, base_diversity=0.5, ft_diversity=0.49)  # noqa: E501
    assert result.gr_improved_technical is False


# ---------------------------------------------------------------------------
# Tier 3: Diversity impact (OQ-M02)
# ---------------------------------------------------------------------------


def test_tier3_diversity_preserved() -> None:
    """
    Tier 3: Diversity is preserved when finetuned_pairwise_diversity >= 0.9 * base.

    This is the primary risk-control check for fine-tuning (David §5.3, §10.3):
    if diversity degrades by more than 10%, the ensemble benefit erodes.
    """
    # 0.49 >= 0.9 * 0.5 = 0.45 -> preserved
    result = _make_result("XOM", base_tech_gr=0.7, ft_tech_gr=0.75, base_diversity=0.5, ft_diversity=0.49)  # noqa: E501
    assert result.diversity_preserved is True


def test_tier3_diversity_degraded() -> None:
    """Tier 3: Flag diversity degradation when finetuned < 0.9 * base."""
    # 0.40 < 0.9 * 0.5 = 0.45 -> degraded
    result = _make_result("AAPL", base_tech_gr=0.7, ft_tech_gr=0.75, base_diversity=0.5, ft_diversity=0.40)  # noqa: E501
    assert result.diversity_preserved is False


# ---------------------------------------------------------------------------
# Pipeline structure: multi-ticker comparison
# ---------------------------------------------------------------------------


def test_three_tier_logic_wired_multi_ticker() -> None:
    """
    Three tiers produce consistent outputs across multiple tickers.

    Simulates the aggregate evaluation that run_phase11_evaluation.py performs
    for AAPL, JPM, XOM and derives OQ-M01/OQ-M02 answers.
    """
    scenarios = [
        ("AAPL", 0.667, 0.720, 0.50, 0.48),
        ("JPM",  0.650, 0.710, 0.45, 0.44),
        ("XOM",  0.700, 0.750, 0.55, 0.52),
    ]

    results = [_make_result(*s) for s in scenarios]

    # All results are valid FineTuneEvaluationResult instances
    for r in results:
        assert isinstance(r, FineTuneEvaluationResult)
        assert r.ticker in ("AAPL", "JPM", "XOM")

    # OQ-M01: at least one ticker shows GR improvement
    any_gr_improved = any(r.gr_improved_technical for r in results)
    assert any_gr_improved, "Expected at least one ticker to show GR improvement"

    # OQ-M02: diversity metric is present and bounded
    for r in results:
        assert 0.0 <= r.finetuned_pairwise_diversity <= 1.0
        assert 0.0 <= r.base_pairwise_diversity <= 1.0


def test_evaluation_result_serialisable() -> None:
    """Each FineTuneEvaluationResult is JSON-serialisable (needed for fixture writing)."""
    import json
    result = _make_result("AAPL", 0.667, 0.720, 0.5, 0.48)
    serialised = result.model_dump_json()
    parsed = json.loads(serialised)
    assert parsed["ticker"] == "AAPL"
    assert isinstance(parsed["gr_improved_technical"], bool)
    assert isinstance(parsed["diversity_preserved"], bool)
