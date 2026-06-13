"""
Fixture validation test for the Phase 11 fine-tuning evaluation baseline (P11-E6-T1).

Skipped when tests/fixtures/baseline/phase11_evaluation.json is absent.
Generate it with: make baseline-phase11
"""

from __future__ import annotations

from pathlib import Path

import pytest

_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "baseline" / "phase11_evaluation.json"
)
_FIXTURE_EXISTS = _FIXTURE.exists()


@pytest.mark.skipif(not _FIXTURE_EXISTS, reason="phase11_evaluation.json not yet generated")
def test_phase11_fixture_structure() -> None:
    """Fixture has expected top-level structure."""
    import json
    data = json.loads(_FIXTURE.read_text())
    assert "metadata" in data
    assert "results" in data
    assert len(data["results"]) > 0


@pytest.mark.skipif(not _FIXTURE_EXISTS, reason="phase11_evaluation.json not yet generated")
def test_phase11_fixture_results_are_valid() -> None:
    """Each result in the fixture validates as FineTuneEvaluationResult."""
    import json

    from hifi.models.training_data import FineTuneEvaluationResult

    data = json.loads(_FIXTURE.read_text())
    for record in data["results"]:
        result = FineTuneEvaluationResult.model_validate(record)
        assert result.ticker in data["metadata"].get("tickers", [result.ticker])
        assert 0.0 <= result.base_technical_gr <= 1.0
        assert 0.0 <= result.finetuned_technical_gr <= 1.0


@pytest.mark.skipif(not _FIXTURE_EXISTS, reason="phase11_evaluation.json not yet generated")
def test_phase11_fixture_diversity_metrics_present() -> None:
    """Diversity metrics are present in each result (needed for OQ-M02 answer)."""
    import json
    data = json.loads(_FIXTURE.read_text())
    for record in data["results"]:
        assert "base_pairwise_diversity" in record
        assert "finetuned_pairwise_diversity" in record
        assert "diversity_preserved" in record


@pytest.mark.skipif(not _FIXTURE_EXISTS, reason="phase11_evaluation.json not yet generated")
def test_phase11_fixture_gr_metrics_present() -> None:
    """GR metrics are present for both agents (needed for OQ-M01 answer)."""
    import json
    data = json.loads(_FIXTURE.read_text())
    for record in data["results"]:
        assert "base_technical_gr" in record
        assert "finetuned_technical_gr" in record
        assert "gr_improved_technical" in record
