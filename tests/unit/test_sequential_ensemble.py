"""
Unit tests for run_sequential_ensemble and the sequential ensemble graph (E3-T2/T3/T4, DJ-089b).

Tests verify:
- Agent 1 (fundamental) receives no prior context block.
- Agent 2 (technical) receives Agent 1's summary in its memory_prefix.
- Contrarian receives prior-context block containing all 5 agents.
- Agent failure (None signal): later agents still receive context from successful agents.
- sequential=False → run_ensemble behaves as before (no context injection).
- sequential=True → run_ensemble delegates to run_sequential_ensemble.
- build_sequential_graph() compiles without cycles.
- CANONICAL_ORDER in graph.py matches agent_context.py.

No real LLMs. Uses mock LLMs that capture input messages.
Uses tmp_path for isolated AgentContextStore.
"""

from __future__ import annotations

import json
import shutil

import pytest

from hifi.knowledge.agent_context import AgentContextStore

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

_FUND_STUB = json.dumps({
    "decision": "Buy",
    "confidence": 0.72,
    "rationale": "P/E of 22 is below sector average and earnings growth is strong.",
    "key_concern": "Valuation rich vs. peers.",
})

_TECH_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "RSI at 52 — neutral. MACD histogram slightly positive.",
    "key_concern": "Volume declining on recent rally.",
    "time_horizon": "short-term",
})

_RISK_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.55,
    "rationale": "hist_vol_20d of 0.20 is moderate. No drawdown alert.",
    "key_concern": "VIX elevated at 22.",
    "risk_assessment": "Moderate",
    "recommended_position_size": 0.05,
})

_MACRO_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.50,
    "rationale": "Fed funds at 5.25%. Yield curve flat.",
    "key_concern": "Rate uncertainty.",
    "regime_assessment": "Restrictive",
    "macro_rationale": "High rates compress multiples.",
})

_SENT_STUB = json.dumps({
    "decision": "Buy",
    "confidence": 0.65,
    "rationale": "Management guidance raised. Services revenue strong.",
    "key_concern": "China headwinds mentioned.",
    "notable_signals": ["raised guidance"],
})

_CONT_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "Consensus Buy — fading the crowd given valuation.",
    "key_concern": "Macro headwinds underpriced.",
    "contrarian_thesis": "Hold vs. Buy consensus.",
})


def _capturing_llm(capture_list: list, response: str) -> object:
    """Mock LLM that appends all messages to capture_list and returns response."""
    class _Cap:
        model_name = "mock-llm"
        def invoke(self, messages):
            capture_list.extend(messages)
            class _R:
                content = response
            return _R()
    return _Cap()


def _simple_stub_llm(response: str) -> object:
    class _Stub:
        model_name = "mock-llm"
        def invoke(self, messages):
            class _R:
                content = response
            return _R()
    return _Stub()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixtures_data_dir(tmp_path):
    import os
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
def store(tmp_path) -> AgentContextStore:
    db = str(tmp_path / "knowledge.lance")
    return AgentContextStore(namespace="test-seq", db_path=db)


@pytest.fixture
def aapl_snapshot_json():
    import datetime

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
        fetched_at=datetime.datetime(2023, 4, 1),
        provenance=ProvenanceRecord(
            source="test", fetched_at=datetime.datetime(2023, 4, 1)
        ),
    )
    return snap.model_dump_json()


# ---------------------------------------------------------------------------
# Context injection tests
# ---------------------------------------------------------------------------


def test_fundamental_receives_no_prior_context(fixtures_data_dir, aapl_snapshot_json, store):
    """Fundamental is first — its memory_prefix must not contain a prior-context block."""
    captured = []
    from hifi.agents.ensemble_runner import run_sequential_ensemble
    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_store=store,
        _test_llms={
            "fundamental": _capturing_llm(captured, _FUND_STUB),
            "technical": _simple_stub_llm(_TECH_STUB),
            "risk": _simple_stub_llm(_RISK_STUB),
            "macro": _simple_stub_llm(_MACRO_STUB),
            "sentiment": _simple_stub_llm(_SENT_STUB),
            "contrarian": _simple_stub_llm(_CONT_STUB),
        },
    )
    # Fundamental should NOT have any "[Prior Agent Analyses..." in its messages
    fund_texts = [
        m.content for m in captured if hasattr(m, "content")
    ]
    assert not any("[Prior Agent Analyses" in t for t in fund_texts), (
        "Fundamental received a prior-context block — it should be empty for the first agent."
    )


def test_technical_receives_fundamental_summary(fixtures_data_dir, aapl_snapshot_json, store):
    """Technical (2nd) should receive Fundamental's decision in its memory_prefix."""
    tech_captured = []
    from hifi.agents.ensemble_runner import run_sequential_ensemble
    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_store=store,
        _test_llms={
            "fundamental": _simple_stub_llm(_FUND_STUB),
            "technical": _capturing_llm(tech_captured, _TECH_STUB),
            "risk": _simple_stub_llm(_RISK_STUB),
            "macro": _simple_stub_llm(_MACRO_STUB),
            "sentiment": _simple_stub_llm(_SENT_STUB),
            "contrarian": _simple_stub_llm(_CONT_STUB),
        },
    )
    tech_texts = [m.content for m in tech_captured if hasattr(m, "content")]
    assert any("[Prior Agent Analyses" in t for t in tech_texts), (
        "Technical did not receive a prior-context block from Fundamental."
    )
    assert any("Fundamental Agent" in t for t in tech_texts), (
        "Technical's context block does not mention Fundamental Agent."
    )


def test_contrarian_receives_all_five_prior_contexts(
    fixtures_data_dir, aapl_snapshot_json, store
):
    """Contrarian is last — its context should reference all 5 prior agents."""
    cont_captured = []
    from hifi.agents.ensemble_runner import run_sequential_ensemble
    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_store=store,
        _test_llms={
            "fundamental": _simple_stub_llm(_FUND_STUB),
            "technical": _simple_stub_llm(_TECH_STUB),
            "risk": _simple_stub_llm(_RISK_STUB),
            "macro": _simple_stub_llm(_MACRO_STUB),
            "sentiment": _simple_stub_llm(_SENT_STUB),
            "contrarian": _capturing_llm(cont_captured, _CONT_STUB),
        },
    )
    cont_texts = " ".join(m.content for m in cont_captured if hasattr(m, "content"))
    assert "[Prior Agent Analyses" in cont_texts
    for agent in ["Fundamental", "Technical", "Risk", "Macro", "Sentiment"]:
        assert agent in cont_texts, f"Contrarian context missing {agent} Agent reference."


def test_agent_failure_later_agents_get_successful_context(
    fixtures_data_dir, aapl_snapshot_json, store
):
    """If an agent fails (returns None signal), later agents still get context from others."""
    # Technical stub returns invalid JSON → signal will be None
    risk_captured = []
    from hifi.agents.ensemble_runner import run_sequential_ensemble
    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_store=store,
        _test_llms={
            "fundamental": _simple_stub_llm(_FUND_STUB),
            "technical": _simple_stub_llm("not valid json"),  # will produce None signal
            "risk": _capturing_llm(risk_captured, _RISK_STUB),
            "macro": _simple_stub_llm(_MACRO_STUB),
            "sentiment": _simple_stub_llm(_SENT_STUB),
            "contrarian": _simple_stub_llm(_CONT_STUB),
        },
    )
    risk_texts = [m.content for m in risk_captured if hasattr(m, "content")]
    # Risk should see Fundamental (which succeeded) in its context
    fund_in_risk = any("Fundamental Agent" in t for t in risk_texts)
    # Technical failed so may or may not appear — we only verify fundamental made it
    assert fund_in_risk, (
        "Risk agent did not receive context from Fundamental, even though Fundamental succeeded."
    )


# ---------------------------------------------------------------------------
# AgentContextStore populated after run
# ---------------------------------------------------------------------------


def test_store_populated_after_sequential_run(fixtures_data_dir, aapl_snapshot_json, store):
    """After run_sequential_ensemble, the store should have ≥ 1 record."""
    from hifi.agents.ensemble_runner import run_sequential_ensemble
    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_store=store,
        _test_llms={
            "fundamental": _simple_stub_llm(_FUND_STUB),
            "technical": _simple_stub_llm(_TECH_STUB),
            "risk": _simple_stub_llm(_RISK_STUB),
            "macro": _simple_stub_llm(_MACRO_STUB),
            "sentiment": _simple_stub_llm(_SENT_STUB),
            "contrarian": _simple_stub_llm(_CONT_STUB),
        },
    )
    # At least fundamental should be stored (has valid signal)
    df = store._table.to_pandas()
    assert len(df) >= 1
    assert "fundamental" in df["agent_type"].values


# ---------------------------------------------------------------------------
# sequential=False preserves original behaviour
# ---------------------------------------------------------------------------


def test_sequential_false_does_not_inject_context(fixtures_data_dir, aapl_snapshot_json, store):
    """run_ensemble(sequential=False) must not inject any [Prior Agent Analyses...] block."""
    fund_captured = []
    tech_captured = []
    from hifi.agents.ensemble_runner import run_ensemble
    run_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        sequential=False,
        _test_store=store,
        _test_llms={
            "fundamental": _capturing_llm(fund_captured, _FUND_STUB),
            "technical": _capturing_llm(tech_captured, _TECH_STUB),
            "risk": _simple_stub_llm(_RISK_STUB),
            "macro": _simple_stub_llm(_MACRO_STUB),
            "sentiment": _simple_stub_llm(_SENT_STUB),
            "contrarian": _simple_stub_llm(_CONT_STUB),
        },
    )
    all_texts = [m.content for m in fund_captured + tech_captured if hasattr(m, "content")]
    assert not any("[Prior Agent Analyses" in t for t in all_texts), (
        "sequential=False should not inject any prior-context blocks."
    )


def test_sequential_true_delegates_to_run_sequential_ensemble(
    fixtures_data_dir, aapl_snapshot_json, store
):
    """run_ensemble(sequential=True) must inject context (delegation confirmed)."""
    tech_captured = []
    from hifi.agents.ensemble_runner import run_ensemble
    run_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        sequential=True,
        _test_store=store,
        _test_llms={
            "fundamental": _simple_stub_llm(_FUND_STUB),
            "technical": _capturing_llm(tech_captured, _TECH_STUB),
            "risk": _simple_stub_llm(_RISK_STUB),
            "macro": _simple_stub_llm(_MACRO_STUB),
            "sentiment": _simple_stub_llm(_SENT_STUB),
            "contrarian": _simple_stub_llm(_CONT_STUB),
        },
    )
    tech_texts = [m.content for m in tech_captured if hasattr(m, "content")]
    assert any("[Prior Agent Analyses" in t for t in tech_texts), (
        "sequential=True should delegate to run_sequential_ensemble which injects context."
    )


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


def test_run_sequential_ensemble_returns_ensemble_output(
    fixtures_data_dir, aapl_snapshot_json, store
):
    from hifi.agents.ensemble_runner import run_sequential_ensemble
    from hifi.collective.schemas import EnsembleOutput
    result = run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_store=store,
        _test_llms={
            "fundamental": _simple_stub_llm(_FUND_STUB),
            "technical": _simple_stub_llm(_TECH_STUB),
            "risk": _simple_stub_llm(_RISK_STUB),
            "macro": _simple_stub_llm(_MACRO_STUB),
            "sentiment": _simple_stub_llm(_SENT_STUB),
            "contrarian": _simple_stub_llm(_CONT_STUB),
        },
    )
    assert isinstance(result, EnsembleOutput)
    assert result.ticker == "AAPL"
    assert result.as_of_date == "2023-03-31"


# ---------------------------------------------------------------------------
# Graph topology tests (E3-T3)
# ---------------------------------------------------------------------------


def test_sequential_graph_compiles_without_error():
    """build_sequential_graph() must compile (no cycles detected by LangGraph)."""
    from hifi.agents.graph import build_sequential_graph
    g = build_sequential_graph()
    assert g is not None


def test_sequential_graph_canonical_order():
    """Canonical order in graph.py must match agent_context.py."""
    from hifi.agents.graph import CANONICAL_ORDER
    from hifi.knowledge.agent_context import CANONICAL_ORDER as CTX_CANONICAL_ORDER
    assert CANONICAL_ORDER == CTX_CANONICAL_ORDER


def test_sequential_graph_has_all_agent_nodes():
    """All 6 agents must appear as nodes in the compiled graph."""
    from hifi.agents.graph import CANONICAL_ORDER, build_sequential_graph
    g = build_sequential_graph()
    graph_repr = g.get_graph()
    node_names = set(graph_repr.nodes.keys())
    for agent in CANONICAL_ORDER:
        assert agent in node_names, f"Node '{agent}' missing from sequential graph."


def test_sequential_graph_edges_in_order():
    """Edges must connect agents in CANONICAL_ORDER sequence."""
    from hifi.agents.graph import CANONICAL_ORDER, build_sequential_graph
    g = build_sequential_graph()
    graph_repr = g.get_graph()
    # Collect (source, target) pairs, excluding __start__/__end__ virtual nodes
    edge_pairs = {
        (e.source, e.target)
        for e in graph_repr.edges
        if not e.source.startswith("__") and not e.target.startswith("__")
    }
    for i in range(len(CANONICAL_ORDER) - 1):
        src, dst = CANONICAL_ORDER[i], CANONICAL_ORDER[i + 1]
        assert (src, dst) in edge_pairs, (
            f"Expected edge {src} → {dst} not found in graph."
        )
