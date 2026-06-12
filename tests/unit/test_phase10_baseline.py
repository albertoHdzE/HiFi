"""
Fixture validation test for the Phase 10 accuracy baseline (P10-E5-T2).

Skipped when tests/fixtures/baseline/phase10_accuracy.json is absent.
Generate it with: make baseline-phase10
"""

from __future__ import annotations

from pathlib import Path

import pytest

_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "baseline" / "phase10_accuracy.json"
)
_FIXTURE_EXISTS = _FIXTURE.exists()


@pytest.mark.skipif(not _FIXTURE_EXISTS, reason="phase10_accuracy.json not yet generated")
def test_phase10_fixture_is_valid_method_accuracy_report():
    from hifi.collective.schemas import MethodAccuracyReport
    report = MethodAccuracyReport.model_validate_json(_FIXTURE.read_text())

    assert report.n_labeled >= 1, "Expected at least one labeled record"
    assert len(report.accuracy_by_method) > 0, "accuracy_by_method must be non-empty"


@pytest.mark.skipif(not _FIXTURE_EXISTS, reason="phase10_accuracy.json not yet generated")
def test_phase10_fixture_accuracy_values_in_range():
    from hifi.collective.schemas import MethodAccuracyReport
    report = MethodAccuracyReport.model_validate_json(_FIXTURE.read_text())

    for method_name, acc in report.accuracy_by_method.items():
        assert 0.0 <= acc <= 1.0, (
            f"accuracy_by_method[{method_name!r}] = {acc} is outside [0, 1]"
        )


@pytest.mark.skipif(not _FIXTURE_EXISTS, reason="phase10_accuracy.json not yet generated")
def test_phase10_fixture_tickers_and_dates_present():
    from hifi.collective.schemas import MethodAccuracyReport
    report = MethodAccuracyReport.model_validate_json(_FIXTURE.read_text())

    assert len(report.tickers) > 0
    assert len(report.analysis_dates) > 0
