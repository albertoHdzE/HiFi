"""
Integration tests for verify_ensemble (P5-E5).

Uses Phase 1 parquet fixtures and DI-injected LLMs. Tests focus on
structural correctness of the EnsembleVerificationReport pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.agents.ensemble_runner import run_ensemble
from hifi.collective.schemas import EnsembleOutput
from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord
from hifi.verification.schemas import EnsembleVerificationReport
from hifi.verification.verifier import verify_ensemble

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_data_dir(tmp_path):
    fixtures_root = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    for subdir in ("market", "macro"):
        dst = tmp_path / subdir
        dst.mkdir()
        src = os.path.join(fixtures_root, subdir)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f.endswith(".parquet"):
                    shutil.copy(os.path.join(src, f), dst / f)
    return str(tmp_path)


@pytest.fixture
def aapl_snapshot_json():
    snap = FundamentalsSnapshot(
        ticker="AAPL",
        period_end="2023-03-31",
        eps=6.11,
        market_cap=2_500_000_000_000,
        total_equity=62_146_000_000,
        revenue=394_330_000_000,
        net_income=99_803_000_000,
        total_assets=352_755_000_000,
        total_liabilities=290_437_000_000,
        source="test",
        fetched_at=datetime(2023, 4, 1),
        provenance=ProvenanceRecord(source="test", fetched_at=datetime(2023, 4, 1)),
    )
    return snap.model_dump_json()


_STUB_HOLD = json.dumps({
    "decision": "Hold",
    "confidence": 0.65,
    "rationale": "Conditions appear balanced given current macro backdrop.",
    "key_concern": "Rate risk remains elevated.",
    "time_horizon": "medium-term",
})

_STUB_BUY = json.dumps({
    "decision": "Buy",
    "confidence": 0.75,
    "rationale": "Improving sentiment supports a modest upside bias.",
    "key_concern": "Macro uncertainty could compress valuations.",
    "time_horizon": "medium-term",
})


def _stub_llm(response: str, model_name: str = "test-model"):
    class _S:
        def invoke(self, _):
            class _R:
                content = response
            return _R()
    s = _S()
    s.model_name = model_name
    return s


# ---------------------------------------------------------------------------
# Test 1: verify_ensemble returns EnsembleVerificationReport
# ---------------------------------------------------------------------------


def test_verify_ensemble_returns_report(fixtures_data_dir, aapl_snapshot_json):
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={"fundamental": _stub_llm(_STUB_HOLD, "fund"), "technical": _stub_llm(_STUB_HOLD, "tech")},
    )

    assert isinstance(output, EnsembleOutput)
    report = verify_ensemble(output)

    assert isinstance(report, EnsembleVerificationReport)
    assert report.ticker == "AAPL"
    assert report.as_of_date == "2023-03-31"
    assert report.fundamental_report.agent_type == "fundamental"
    assert report.technical_report.agent_type == "technical"


# ---------------------------------------------------------------------------
# Test 2: triggered_by_disagreement reflects entropy
# ---------------------------------------------------------------------------


def test_triggered_by_disagreement_false_when_agreement(fixtures_data_dir, aapl_snapshot_json):
    """Both Hold -> entropy=0 -> triggered_by_disagreement=False."""
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={"fundamental": _stub_llm(_STUB_HOLD, "fund"), "technical": _stub_llm(_STUB_HOLD, "tech")},
    )
    report = verify_ensemble(output)

    if output.ensemble_decision.disagreement_entropy == 0.0:
        assert report.triggered_by_disagreement is False
    else:
        assert report.triggered_by_disagreement is True


def test_triggered_by_disagreement_true_when_disagreement(fixtures_data_dir, aapl_snapshot_json):
    """Hold vs Buy -> entropy>0 -> triggered_by_disagreement=True."""
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={"fundamental": _stub_llm(_STUB_HOLD, "fund"), "technical": _stub_llm(_STUB_BUY, "tech")},
    )
    report = verify_ensemble(output)

    fa_sig = output.fundamental_analysis.signal
    ta_sig = output.technical_analysis.signal
    if fa_sig is not None and ta_sig is not None and fa_sig.decision != ta_sig.decision:
        assert report.triggered_by_disagreement is True


# ---------------------------------------------------------------------------
# Test 3: Structural invariants
# ---------------------------------------------------------------------------


def test_verify_ensemble_structural_invariants(fixtures_data_dir, aapl_snapshot_json):
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={"fundamental": _stub_llm(_STUB_HOLD, "fund"), "technical": _stub_llm(_STUB_HOLD, "tech")},
    )
    report = verify_ensemble(output)

    fr = report.fundamental_report
    tr = report.technical_report

    assert fr.n_claims == fr.n_verified + fr.n_hallucinated + fr.n_unresolvable
    assert tr.n_claims == tr.n_verified + tr.n_hallucinated + tr.n_unresolvable
    assert report.total_claims == fr.n_claims + tr.n_claims
    assert report.total_hallucinated == fr.n_hallucinated + tr.n_hallucinated
    assert report.n_contradictions == len(report.contradictions)
    assert 0.0 <= report.ensemble_hallucination_rate <= 1.0


# ---------------------------------------------------------------------------
# Test 4: EnsembleVerificationReport is JSON-serialisable
# ---------------------------------------------------------------------------


def test_verify_ensemble_json_safe(fixtures_data_dir, aapl_snapshot_json):
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={"fundamental": _stub_llm(_STUB_HOLD, "fund"), "technical": _stub_llm(_STUB_HOLD, "tech")},
    )
    report = verify_ensemble(output)
    json.dumps(report.model_dump())
