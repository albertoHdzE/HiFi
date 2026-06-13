"""
Unit tests for FineTuneEvaluationResult schema (P11-E4-T4, DJ-058).

Tests schema validation, derived field computation, and JSON round-trips.
No LLM, no live services required.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hifi.models.training_data import FineTuneEvaluationResult

_NOW = datetime.now(UTC).isoformat()

_BASE = dict(
    ticker="AAPL",
    analysis_date="2023-03-31",
    base_technical_gr=0.667,
    base_fundamental_gr=1.000,
    finetuned_technical_gr=0.720,
    finetuned_fundamental_gr=1.000,
    base_pairwise_diversity=0.5,
    finetuned_pairwise_diversity=0.48,
    base_disagreement_entropy=0.6,
    finetuned_disagreement_entropy=0.58,
    generated_at=_NOW,
)


def test_schema_construction_valid() -> None:
    """FineTuneEvaluationResult constructs from valid inputs."""
    result = FineTuneEvaluationResult(**_BASE)
    assert result.ticker == "AAPL"
    assert result.analysis_date == "2023-03-31"


def test_json_roundtrip() -> None:
    """Schema round-trips through JSON without data loss."""
    result = FineTuneEvaluationResult(**_BASE)
    restored = FineTuneEvaluationResult.model_validate_json(result.model_dump_json())
    assert restored.ticker == result.ticker
    assert restored.finetuned_technical_gr == result.finetuned_technical_gr
    assert restored.diversity_preserved == result.diversity_preserved


def test_diversity_preserved_boundary_true() -> None:
    """diversity_preserved=True at exactly 0.9x threshold."""
    result = FineTuneEvaluationResult(
        **{**_BASE, "base_pairwise_diversity": 0.5, "finetuned_pairwise_diversity": 0.45}  # 0.45 = 0.9 * 0.5  # noqa: E501
    )
    assert result.diversity_preserved is True


def test_diversity_preserved_boundary_false() -> None:
    """diversity_preserved=False when just below 0.9x threshold."""
    result = FineTuneEvaluationResult(
        **{**_BASE, "base_pairwise_diversity": 0.5, "finetuned_pairwise_diversity": 0.449}
    )
    assert result.diversity_preserved is False


def test_gr_improved_technical_threshold_met() -> None:
    """gr_improved_technical=True when delta clearly exceeds 0.05."""
    result = FineTuneEvaluationResult(
        **{**_BASE, "base_technical_gr": 0.600, "finetuned_technical_gr": 0.660}  # delta=0.06
    )
    assert result.gr_improved_technical is True


def test_gr_improved_technical_threshold_not_met() -> None:
    """gr_improved_technical=False when delta < 0.05."""
    result = FineTuneEvaluationResult(
        **{**_BASE, "base_technical_gr": 0.667, "finetuned_technical_gr": 0.716}
    )
    assert result.gr_improved_technical is False


def test_gr_improved_fundamental() -> None:
    """gr_improved_fundamental computed correctly."""
    result_improved = FineTuneEvaluationResult(
        **{**_BASE, "base_fundamental_gr": 0.800, "finetuned_fundamental_gr": 0.860}
    )
    result_not = FineTuneEvaluationResult(
        **{**_BASE, "base_fundamental_gr": 0.800, "finetuned_fundamental_gr": 0.840}
    )
    assert result_improved.gr_improved_fundamental is True
    assert result_not.gr_improved_fundamental is False


def test_gr_bounds_validation() -> None:
    """GR fields must be in [0, 1]."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FineTuneEvaluationResult(**{**_BASE, "finetuned_technical_gr": 1.1})
    with pytest.raises(ValidationError):
        FineTuneEvaluationResult(**{**_BASE, "base_fundamental_gr": -0.1})


def test_diversity_zero_base() -> None:
    """diversity_preserved=True when both base and finetuned are zero."""
    result = FineTuneEvaluationResult(
        **{**_BASE, "base_pairwise_diversity": 0.0, "finetuned_pairwise_diversity": 0.0}
    )
    assert result.diversity_preserved is True
