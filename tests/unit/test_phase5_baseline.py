"""
Phase 5 baseline fixture tests (P5-E6-T4).

Skipped when tests/fixtures/baseline/phase5_verification.json is absent
(before scripts/archive/run_phase5_verification.py has been executed). After the
script runs, these tests assert structural correctness and quality gates.
"""

from __future__ import annotations

import json
import os

import pytest

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "baseline", "phase5_verification.json"
)
_FIXTURE_EXISTS = os.path.isfile(_FIXTURE_PATH)
_SKIP_REASON = (
    "phase5_verification.json not yet generated. "
    "Run: uv run python scripts/archive/run_phase5_verification.py"
)

pytestmark = pytest.mark.skipif(not _FIXTURE_EXISTS, reason=_SKIP_REASON)


@pytest.fixture(scope="module")
def fixture_data():
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_fixture_has_required_top_level_keys(fixture_data):
    assert "metadata" in fixture_data
    assert "reports" in fixture_data
    assert "metrics" in fixture_data


def test_fixture_metadata_fields(fixture_data):
    meta = fixture_data["metadata"]
    assert meta["phase"] == "5"
    assert "verified_from" in meta
    assert "run_date" in meta


def test_fixture_has_three_tickers(fixture_data):
    reports = fixture_data["reports"]
    assert len(reports) == 3
    for ticker in ("AAPL", "JPM", "XOM"):
        assert ticker in reports


def test_fixture_metrics_keys(fixture_data):
    metrics = fixture_data["metrics"]
    for domain in ("fundamental", "technical", "ensemble"):
        assert domain in metrics


# ---------------------------------------------------------------------------
# Per-ticker report structure
# ---------------------------------------------------------------------------


def test_each_ticker_has_ensemble_report(fixture_data):
    for ticker, ens_report in fixture_data["reports"].items():
        assert "fundamental_report" in ens_report, f"{ticker} missing fundamental_report"
        assert "technical_report" in ens_report, f"{ticker} missing technical_report"
        assert "contradictions" in ens_report, f"{ticker} missing contradictions"
        assert "triggered_by_disagreement" in ens_report, f"{ticker} missing trigger flag"


def test_agent_report_metric_bounds(fixture_data):
    for ticker, ens_report in fixture_data["reports"].items():
        for role in ("fundamental_report", "technical_report"):
            r = ens_report[role]
            hr = r["hallucination_rate"]
            gr = r["grounding_rate"]
            assert 0.0 <= hr <= 1.0, f"{ticker}/{role} HR={hr} out of [0,1]"
            assert 0.0 <= gr <= 1.0, f"{ticker}/{role} GR={gr} out of [0,1]"
            assert r["n_claims"] == r["n_verified"] + r["n_hallucinated"] + r["n_unresolvable"], (
                f"{ticker}/{role} claim counts inconsistent"
            )


def test_ensemble_hr_matches_agent_counts(fixture_data):
    """Ensemble HR consistent with per-agent claim counts."""
    for ticker, ens_report in fixture_data["reports"].items():
        fr = ens_report["fundamental_report"]
        tr = ens_report["technical_report"]
        total_h = fr["n_hallucinated"] + tr["n_hallucinated"]
        f_res = fr["n_claims"] - fr["n_unresolvable"]
        t_res = tr["n_claims"] - tr["n_unresolvable"]
        total_res = f_res + t_res
        expected_ehr = total_h / total_res if total_res > 0 else 0.0
        actual_ehr = ens_report["ensemble_hallucination_rate"]
        assert abs(actual_ehr - expected_ehr) < 1e-4, (
            f"{ticker} ensemble HR mismatch: got {actual_ehr}, expected {expected_ehr}"
        )


# ---------------------------------------------------------------------------
# Alias table coverage gate (P5-E2 coverage goal: unresolvable_rate < 10%)
# ---------------------------------------------------------------------------


def test_alias_table_coverage_meets_threshold(fixture_data):
    """
    Alias table coverage > 0.90 across fundamental and technical agents.

    If this fails, the alias table in extractor.py needs extension.
    See DJ-019: coverage measured at P5-E6-T3.
    """
    for domain in ("fundamental", "technical"):
        cov = fixture_data["metrics"][domain].get("alias_table_coverage", 1.0)
        assert cov >= 0.90, (
            f"{domain} alias_table_coverage={cov:.3f} < 0.90. "
            "Extend FIELD_ALIAS_TABLE in src/hifi/verification/extractor.py."
        )
