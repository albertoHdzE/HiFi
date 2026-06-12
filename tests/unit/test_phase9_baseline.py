"""
Phase 9 collective engine baseline fixture tests (P9-E6).

Skipped when tests/fixtures/baseline/phase9_collective.json is absent
(before scripts/run_phase9_baseline.py has been executed). After the script
runs, these tests assert structural correctness and quality gates on the
method_comparison output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

_FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "baseline" / "phase9_collective.json"
)
_METHODS = (
    "confidence_weighted",
    "majority",
    "performance_weighted",
    "contrarian_adjusted",
)

pytestmark = pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason=(
        "phase9_collective.json not yet generated. "
        "Run: uv run python scripts/run_phase9_baseline.py"
    ),
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------


def test_fixture_has_required_keys(baseline):
    for key in ("metadata", "outputs", "rolling_metrics"):
        assert key in baseline, f"missing top-level key: {key}"


def test_metadata_fields(baseline):
    meta = baseline["metadata"]
    for field in ("phase", "as_of", "run_date", "hifi_commit"):
        assert field in meta, f"metadata missing field: {field}"
    assert meta["phase"] == "9"
    assert meta["as_of"] == "2023-03-31"


def test_three_tickers_present(baseline):
    outputs = baseline["outputs"]
    for ticker in ("AAPL", "JPM", "XOM"):
        assert ticker in outputs, f"{ticker} missing from outputs"


# ---------------------------------------------------------------------------
# Per-ticker: ensemble_decision
# ---------------------------------------------------------------------------


def test_ensemble_decision_valid_or_none(baseline):
    for ticker, output in baseline["outputs"].items():
        decision = output["ensemble_decision"]["collective_decision"]
        assert decision in ("Buy", "Hold", "Sell", None), (
            f"{ticker}: invalid collective_decision {decision!r}"
        )


def test_collective_confidence_in_range(baseline):
    for ticker, output in baseline["outputs"].items():
        cc = output["ensemble_decision"]["collective_confidence"]
        assert 0.0 <= cc <= 1.0, f"{ticker}: collective_confidence {cc} out of [0,1]"


def test_disagreement_entropy_in_range(baseline):
    max_entropy = math.log2(3)
    for ticker, output in baseline["outputs"].items():
        h = output["ensemble_decision"]["disagreement_entropy"]
        assert 0.0 <= h <= max_entropy + 1e-9, (
            f"{ticker}: disagreement_entropy {h} out of [0, log2(3)]"
        )


def test_contrarian_confidence_discount_in_range(baseline):
    for ticker, output in baseline["outputs"].items():
        discount = output["ensemble_decision"]["contrarian_confidence_discount"]
        assert 0.0 <= discount <= 1.0, (
            f"{ticker}: contrarian_confidence_discount {discount} out of [0,1]"
        )


def test_review_flagged_is_bool(baseline):
    for ticker, output in baseline["outputs"].items():
        flagged = output["ensemble_decision"]["review_flagged"]
        assert isinstance(flagged, bool), (
            f"{ticker}: review_flagged is {type(flagged).__name__}, expected bool"
        )


# ---------------------------------------------------------------------------
# Per-ticker: method_comparison
# ---------------------------------------------------------------------------


def test_method_comparison_has_all_four_methods(baseline):
    for ticker, output in baseline["outputs"].items():
        mc = output.get("method_comparison", {})
        for method in _METHODS:
            assert method in mc, (
                f"{ticker}: method_comparison missing key {method!r}"
            )


def test_method_comparison_decisions_valid(baseline):
    for ticker, output in baseline["outputs"].items():
        for method in _METHODS:
            mc_decision = (
                output["method_comparison"][method]["collective_decision"]
            )
            assert mc_decision in ("Buy", "Hold", "Sell", None), (
                f"{ticker}/{method}: invalid decision {mc_decision!r}"
            )


def test_method_comparison_confidence_in_range(baseline):
    for ticker, output in baseline["outputs"].items():
        for method in _METHODS:
            cc = output["method_comparison"][method]["collective_confidence"]
            assert 0.0 <= cc <= 1.0, (
                f"{ticker}/{method}: collective_confidence {cc} out of [0,1]"
            )


def test_confidence_weighted_matches_ensemble_decision(baseline):
    """method_comparison['confidence_weighted'] must equal ensemble_decision."""
    for ticker, output in baseline["outputs"].items():
        ed = output["ensemble_decision"]
        cw = output["method_comparison"]["confidence_weighted"]
        assert ed["collective_decision"] == cw["collective_decision"], (
            f"{ticker}: ensemble_decision vs confidence_weighted decision mismatch"
        )
        assert abs(ed["collective_confidence"] - cw["collective_confidence"]) < 1e-9, (
            f"{ticker}: ensemble_decision vs confidence_weighted confidence mismatch"
        )


# ---------------------------------------------------------------------------
# Per-ticker: signals
# ---------------------------------------------------------------------------


def test_signals_list_non_empty(baseline):
    for ticker, output in baseline["outputs"].items():
        signals = output.get("signals", [])
        assert len(signals) >= 1, f"{ticker}: signals list is empty"


def test_signals_have_required_fields(baseline):
    for ticker, output in baseline["outputs"].items():
        for i, sig in enumerate(output.get("signals", [])):
            for field in ("agent_type", "decision", "confidence"):
                assert field in sig, (
                    f"{ticker} signal[{i}]: missing field {field!r}"
                )
            assert sig["decision"] in ("Buy", "Hold", "Sell"), (
                f"{ticker} signal[{i}]: invalid decision {sig['decision']!r}"
            )
            assert 0.0 <= sig["confidence"] <= 1.0, (
                f"{ticker} signal[{i}]: confidence {sig['confidence']} out of [0,1]"
            )
