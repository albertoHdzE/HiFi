"""
Holistic test for the Phase 3 agent pipeline (P3-E6).

Uses monkeypatched MCP tools and a stub LLM so the test is fully deterministic
and requires no live LM Studio instance.

What this test validates:
1. The full graph runs end-to-end (load_snapshot -> call_mcp_tools -> generate -> parse)
2. FundamentalAnalysis is structurally valid (all required fields, JSON-safe)
3. signal.call_ids is non-empty (audit trail used)
4. signal.confidence is in [0, 1]
5. signal.decision is a valid enum value
6. Phase 2 engine pipeline holistic test still passes (regression guard)
"""

import json
import os
import shutil

import pytest

from hifi.agents.fundamental_agent import (
    FundamentalistState,
    call_mcp_tools_node,
    parse_output_node,
    run_analysis,
)
from hifi.agents.schemas import AgentSignal, FundamentalAnalysis

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
        for f in os.listdir(src):
            if f.endswith(".parquet"):
                shutil.copy(os.path.join(src, f), dst / f)
    return str(tmp_path)


@pytest.fixture
def aapl_snapshot_json():
    """Minimal valid FundamentalsSnapshot JSON for AAPL."""
    from datetime import datetime

    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

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


_STUB_SIGNAL_JSON = json.dumps({
    "decision": "Hold",
    "confidence": 0.65,
    "rationale": (
        "P/E of 28.3 is within the 1-year range (pe_1y_percentile 0.60). "
        "ROE of 0.24 is strong."
    ),
    "key_concern": "High fed funds rate of 4.75 compresses growth multiples.",
})


# ---------------------------------------------------------------------------
# Individual node tests (deterministic)
# ---------------------------------------------------------------------------


def test_call_mcp_tools_node_returns_four_results(fixtures_data_dir, aapl_snapshot_json):
    state: FundamentalistState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "snapshot_json": aapl_snapshot_json,
        "data_dir": fixtures_data_dir,
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "error": None,
        "start_time": 0.0,
    }
    update = call_mcp_tools_node(state)
    results = update["tool_results"]
    assert set(results.keys()) == {
        "financial_ratios", "growth_metrics", "valuation_context", "macro_snapshot"
    }


def test_call_mcp_tools_node_results_have_call_ids(fixtures_data_dir, aapl_snapshot_json):
    state: FundamentalistState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "snapshot_json": aapl_snapshot_json,
        "data_dir": fixtures_data_dir,
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "error": None,
        "start_time": 0.0,
    }
    update = call_mcp_tools_node(state)
    for key, result in update["tool_results"].items():
        if "error" not in result:
            assert "call_id" in result, f"{key} missing call_id"


def test_parse_output_node_produces_valid_signal(monkeypatch, aapl_snapshot_json):
    """parse_output_node produces AgentSignal from a pre-set valid JSON response."""
    class _StubLLM:
        model_name = "qwen2.5-coder-32b-instruct-mlx"
        def invoke(self, messages):
            raise AssertionError("LLM should not be called when first parse succeeds")

    import hifi.agents.fundamental_agent as fa
    monkeypatch.setattr(fa, "make_llm", lambda *args, **kwargs: _StubLLM())

    state: FundamentalistState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "snapshot_json": aapl_snapshot_json,
        "data_dir": "data",
        "tool_results": {
            "financial_ratios": {"pe": 28.3, "roe": 0.24, "call_id": "abc"},
            "growth_metrics": {"net_margin": 0.25, "call_id": "def"},
            "valuation_context": {"pe_1y_percentile": 0.6, "call_id": "ghi"},
            "macro_snapshot": {"fed_funds_rate": 4.75, "call_id": "jkl"},
        },
        "llm_response": _STUB_SIGNAL_JSON,
        "signal": None,
        "error": None,
        "start_time": 0.0,
    }
    result = parse_output_node(state)
    sig = result.get("signal")
    assert isinstance(sig, AgentSignal)
    assert sig.decision in ("Buy", "Hold", "Sell")
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.ticker == "AAPL"


# ---------------------------------------------------------------------------
# Full pipeline test (monkeypatched LLM, real MCP via fixtures)
# ---------------------------------------------------------------------------


def test_full_agent_pipeline_aapl(monkeypatch, fixtures_data_dir, aapl_snapshot_json):
    """
    End-to-end agent run for AAPL Q1 2023.
    MCP server uses Phase 1 fixtures. LLM is monkeypatched to return a valid response.
    """
    class _StubLLM:
        model_name = "qwen2.5-coder-32b-instruct-mlx"
        call_count = 0

        def invoke(self, messages):
            self.call_count += 1
            class _Resp:
                content = _STUB_SIGNAL_JSON
            return _Resp()

    stub = _StubLLM()
    import hifi.agents.fundamental_agent as fa
    monkeypatch.setattr(fa, "make_llm", lambda *args, **kwargs: stub)

    analysis = run_analysis(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
    )

    # Structural validity
    assert isinstance(analysis, FundamentalAnalysis)
    assert analysis.signal is not None
    assert analysis.signal.decision in ("Buy", "Hold", "Sell")
    assert 0.0 <= analysis.signal.confidence <= 1.0
    assert analysis.signal.ticker == "AAPL"
    assert analysis.prompt_version == "fundamental_v1"
    assert analysis.latency_ms is not None and analysis.latency_ms > 0

    # Audit trail: call_ids should be populated from MCP results
    assert len(analysis.signal.call_ids) > 0

    # JSON safety: full analysis must serialise without error
    dumped = json.dumps(analysis.model_dump())
    loaded = json.loads(dumped)
    assert loaded["signal"]["decision"] in ("Buy", "Hold", "Sell")


def test_full_agent_pipeline_json_safe(monkeypatch, fixtures_data_dir, aapl_snapshot_json):
    """FundamentalAnalysis.model_dump() produces only JSON-safe values (no NaN)."""
    class _StubLLM:
        model_name = "model"
        def invoke(self, m):
            class R:
                content = _STUB_SIGNAL_JSON
            return R()

    import hifi.agents.fundamental_agent as fa
    monkeypatch.setattr(fa, "make_llm", lambda *args, **kwargs: _StubLLM())

    analysis = run_analysis(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
    )
    # Should not raise
    json.dumps(analysis.model_dump())


# ---------------------------------------------------------------------------
# Phase 2 regression guard
# ---------------------------------------------------------------------------


def test_phase2_engine_pipeline_still_passes(fixtures_data_dir, aapl_snapshot_json):
    """
    Regression guard: Phase 2 MCP tools still return valid results.
    This is the same assertion as test_phase2_engine_pipeline.py.
    """
    from hifi.agents.mcp_client import call_tool

    result = call_tool(
        "get_technical_indicators",
        {"ticker": "AAPL", "date": "2023-03-31", "window": 20},
        data_dir=fixtures_data_dir,
    )
    assert "call_id" in result
    rsi = result.get("rsi")
    if rsi is not None:
        assert 0.0 <= rsi <= 100.0
