"""
Unit tests for the Phase 4 ensemble baseline fixture (P4-E5-T2 through T6).

These tests validate the structure and content of the baseline fixture produced
by scripts/run_phase4_ensemble.py. They are skipped automatically when the
fixture does not exist.

To generate the fixture:
    uv run python scripts/run_phase4_ensemble.py
"""

import json
import math
from pathlib import Path

import pytest

_FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "baseline" / "phase4_ensemble.json"
)

pytestmark = pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason="phase4_ensemble.json not generated yet -- run scripts/run_phase4_ensemble.py",
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_baseline_fixture_exists():
    assert _FIXTURE_PATH.exists()


def test_baseline_has_required_top_level_keys(baseline):
    for key in ("metadata", "outputs", "metrics"):
        assert key in baseline


def test_metadata_has_required_fields(baseline):
    meta = baseline["metadata"]
    for field in ("phase", "models", "prompt_versions", "data_as_of", "run_date", "hifi_commit"):
        assert field in meta, f"metadata missing field: {field}"


def test_all_three_tickers_present(baseline):
    outputs = baseline["outputs"]
    for ticker in ("AAPL", "JPM", "XOM"):
        assert ticker in outputs, f"{ticker} missing from outputs"


def test_each_ensemble_decision_is_valid_or_none(baseline):
    for ticker, output in baseline["outputs"].items():
        decision = output["ensemble_decision"]["collective_decision"]
        assert decision in ("Buy", "Hold", "Sell", None), (
            f"{ticker}: invalid collective_decision {decision!r}"
        )


def test_disagreement_entropy_in_valid_range(baseline):
    max_entropy = math.log2(3)
    for ticker, output in baseline["outputs"].items():
        entropy = output["ensemble_decision"]["disagreement_entropy"]
        assert 0.0 <= entropy <= max_entropy + 1e-9, (
            f"{ticker}: entropy {entropy} out of [0, log2(3)]"
        )


def test_pairwise_diversity_in_valid_range(baseline):
    diversity = baseline["metrics"]["pairwise_diversity"]
    assert 0.0 <= diversity <= 1.0, f"pairwise_diversity {diversity} out of [0, 1]"


def test_fundamental_compliance_meets_threshold(baseline):
    rate = baseline["metrics"]["fundamental_compliance_rate"]
    assert rate >= 0.90, f"fundamental_compliance_rate {rate} below 0.90"


def test_technical_compliance_meets_threshold(baseline):
    rate = baseline["metrics"]["technical_compliance_rate"]
    assert rate >= 0.90, f"technical_compliance_rate {rate} below 0.90"


def test_metrics_has_required_fields(baseline):
    metrics = baseline["metrics"]
    for field in (
        "fundamental_compliance_rate",
        "technical_compliance_rate",
        "ensemble_agreement_rate",
        "mean_disagreement_entropy",
        "mean_opinion_dispersion",
        "pairwise_diversity",
        "mean_total_latency_ms",
        "n_tickers",
    ):
        assert field in metrics, f"metrics missing field: {field}"
