"""
Integration tests for verify_agent (P5-E3).

Uses Phase 1 parquet fixtures (real MCP tool results) and monkeypatched LLMs.
Tests focus on structural and pipeline correctness rather than specific
claim values, since the exact MCP outputs for each ticker/date depend on
the fixture data rather than being hand-known.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.agents.fundamental_agent import run_analysis
from hifi.agents.schemas import FundamentalAnalysis, TechnicalAnalysis
from hifi.agents.technical_agent import run_technical_analysis
from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord
from hifi.verification.schemas import AgentVerificationReport
from hifi.verification.verifier import verify_agent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_data_dir(tmp_path):
    """Phase 1 parquet fixtures copied to a temp data dir."""
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


# Stub LLM that returns a rationale with no known 'field of value' patterns
# (all would be unresolvable). This guarantees HR=0 so structural tests
# can assert on stable properties without knowing exact MCP output values.
_NO_CLAIM_RATIONALE = (
    "The company shows solid fundamentals and appears fairly valued "
    "given current market conditions. Macro headwinds remain a concern."
)

_STUB_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.65,
    "rationale": _NO_CLAIM_RATIONALE,
    "key_concern": "Rate risk from elevated short-term rates.",
    "time_horizon": "medium-term",
})


def _stub_llm(model_name: str = "test-model"):
    class _Stub:
        def invoke(self, _messages):
            class _R:
                content = _STUB_RESPONSE
            return _R()
    s = _Stub()
    s.model_name = model_name
    return s


# ---------------------------------------------------------------------------
# Test 1: verify_agent on FundamentalAnalysis returns AgentVerificationReport
# ---------------------------------------------------------------------------


def test_verify_agent_fundamental_returns_report(fixtures_data_dir, aapl_snapshot_json):
    analysis = run_analysis(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_llm=_stub_llm("fund-model"),
    )

    assert isinstance(analysis, FundamentalAnalysis)
    report = verify_agent(analysis)

    assert isinstance(report, AgentVerificationReport)
    assert report.ticker == "AAPL"
    assert report.as_of_date == "2023-03-31"
    assert report.agent_type == "fundamental"
    assert report.prompt_version == "fundamental_v1"


def test_verify_agent_fundamental_metrics_in_range(fixtures_data_dir, aapl_snapshot_json):
    analysis = run_analysis(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_llm=_stub_llm("fund-model"),
    )

    report = verify_agent(analysis)

    # Structural invariants hold regardless of claim content.
    assert 0.0 <= report.hallucination_rate <= 1.0
    assert 0.0 <= report.grounding_rate <= 1.0
    assert report.n_claims == report.n_verified + report.n_hallucinated + report.n_unresolvable
    # No-claim rationale -> all unresolvable (or no claims); HR = 0.0
    assert report.hallucination_rate == 0.0


# ---------------------------------------------------------------------------
# Test 2: verify_agent on TechnicalAnalysis
# ---------------------------------------------------------------------------


def test_verify_agent_technical_returns_report(fixtures_data_dir):
    analysis = run_technical_analysis(
        ticker="AAPL",
        as_of_date="2023-03-31",
        data_dir=fixtures_data_dir,
        _test_llm=_stub_llm("tech-model"),
    )

    assert isinstance(analysis, TechnicalAnalysis)
    report = verify_agent(analysis)

    assert isinstance(report, AgentVerificationReport)
    assert report.agent_type == "technical"
    assert 0.0 <= report.hallucination_rate <= 1.0
    assert report.n_claims == report.n_verified + report.n_hallucinated + report.n_unresolvable


# ---------------------------------------------------------------------------
# Test 3: verify_agent on signal=None analysis returns empty report
# ---------------------------------------------------------------------------


def test_verify_agent_none_signal_empty_report():
    analysis = FundamentalAnalysis(
        signal=None,
        financial_ratios={},
        growth_metrics={},
        valuation_context={},
        macro_snapshot={},
        prompt_version="fundamental_v1",
    )
    report = verify_agent(analysis)
    assert report.n_claims == 0
    assert report.results == []
    assert report.hallucination_rate == 0.0


# ---------------------------------------------------------------------------
# Test 4: verify_agent report is JSON-serialisable
# ---------------------------------------------------------------------------


def test_verify_agent_report_json_safe(fixtures_data_dir, aapl_snapshot_json):
    analysis = run_analysis(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_llm=_stub_llm("fund-model"),
    )

    report = verify_agent(analysis)
    # Must not raise
    json.dumps(report.model_dump())
