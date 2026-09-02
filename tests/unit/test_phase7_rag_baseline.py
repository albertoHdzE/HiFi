"""
Phase 7 RAG baseline fixture tests (P7-E8).

Skipped when tests/fixtures/baseline/phase7_rag_baseline.json is absent
(before scripts/archive/run_phase7_rag_baseline.py has been executed). After the
script runs, these tests assert structural correctness and quality gates.

To generate the fixture (requires LM Studio + SEC fixtures):
    uv run python scripts/record_sec_fixtures.py   # once, needs internet
    uv run python scripts/archive/run_phase7_rag_baseline.py
"""

from __future__ import annotations

import json
import os

import pytest

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "baseline", "phase7_rag_baseline.json"
)
_FIXTURE_EXISTS = os.path.isfile(_FIXTURE_PATH)
_SKIP_REASON = (
    "phase7_rag_baseline.json not yet generated. "
    "Run: uv run python scripts/archive/run_phase7_rag_baseline.py"
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
    for key in ("metadata", "outputs", "verification", "metrics", "delta_vs_phase5"):
        assert key in fixture_data, f"Missing top-level key: {key}"


def test_fixture_metadata(fixture_data):
    meta = fixture_data["metadata"]
    assert meta["phase"] == "7"
    assert meta["rag_enabled"] is True
    assert "data_as_of" in meta
    assert "run_date" in meta
    assert "hifi_commit" in meta


def test_fixture_has_three_tickers(fixture_data):
    for section in ("outputs", "verification"):
        for ticker in ("AAPL", "JPM", "XOM"):
            assert ticker in fixture_data[section], (
                f"Missing {ticker} in {section}"
            )


# ---------------------------------------------------------------------------
# RAG-specific assertions
# ---------------------------------------------------------------------------


def test_fundamental_prompts_are_v2_when_rag_active(fixture_data):
    """Agents that received passages should have used v2 prompts."""
    for ticker, output in fixture_data["outputs"].items():
        pv = output["fundamental_analysis"]["prompt_version"]
        assert pv == "fundamental_v2", (
            f"{ticker} fundamental prompt_version expected 'fundamental_v2', got '{pv}'"
        )


def test_technical_prompts_are_v2_when_rag_active(fixture_data):
    for ticker, output in fixture_data["outputs"].items():
        pv = output["technical_analysis"]["prompt_version"]
        assert pv == "technical_v2", (
            f"{ticker} technical prompt_version expected 'technical_v2', got '{pv}'"
        )


# ---------------------------------------------------------------------------
# Metrics structure
# ---------------------------------------------------------------------------


def test_metrics_has_fundamental_technical_ensemble(fixture_data):
    metrics = fixture_data["metrics"]
    for section in ("fundamental", "technical", "ensemble"):
        assert section in metrics, f"Missing metrics section: {section}"


def test_hallucination_rates_are_valid_fractions(fixture_data):
    metrics = fixture_data["metrics"]
    for agent in ("fundamental", "technical"):
        hr = metrics[agent]["mean_hallucination_rate"]
        assert 0.0 <= hr <= 1.0, f"{agent} HR={hr} out of range"
    ehr = metrics["ensemble"]["mean_ensemble_hallucination_rate"]
    assert 0.0 <= ehr <= 1.0, f"Ensemble HR={ehr} out of range"


def test_grounding_rates_are_valid_fractions(fixture_data):
    metrics = fixture_data["metrics"]
    for agent in ("fundamental", "technical"):
        gr = metrics[agent]["mean_grounding_rate"]
        assert 0.0 <= gr <= 1.0, f"{agent} GR={gr} out of range"


# ---------------------------------------------------------------------------
# Delta vs Phase 5
# ---------------------------------------------------------------------------


def test_delta_vs_phase5_present(fixture_data):
    delta = fixture_data["delta_vs_phase5"]
    for key in ("fundamental_hr", "technical_hr", "fundamental_gr", "technical_gr"):
        assert key in delta, f"Missing delta key: {key}"


# ---------------------------------------------------------------------------
# Output JSON integrity
# ---------------------------------------------------------------------------


def test_all_outputs_json_safe(fixture_data):
    """Re-serialising must not raise."""
    for ticker, output in fixture_data["outputs"].items():
        try:
            json.dumps(output)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"{ticker} output not JSON-safe: {exc}")
