"""
Holistic test for the Phase 8 agent population (P8-E7).

Validates the full 6-agent ensemble pipeline end-to-end:
  fundamental, technical, risk, macro, sentiment (fail-open), contrarian

No live LM Studio required. LLMs are monkeypatched with stub objects.
Sentiment agent triggers fail-open (knowledge_server unavailable → empty
retrieved_context → default Hold/0.0 signal without calling the LLM).
MCP tool calls for risk and macro use Phase 1 parquet fixtures.

What this test validates:
1. run_ensemble(agents=None): all 4 Phase 8 fields populated in EnsembleOutput
2. risk_analysis, macro_analysis, sentiment_analysis, contrarian_analysis not None
3. Backward compat: agents=["fundamental","technical"] → 4 new fields are None
4. EnsembleOutput round-trips through JSON without error
5. Contrarian analysis fields are non-empty; confidence in [0, 1]
6. Sentiment fail-open: decision=Hold, confidence=0.0, "Insufficient" summary
"""

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.agents.ensemble_runner import run_ensemble
from hifi.collective.schemas import EnsembleOutput
from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

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


# ---------------------------------------------------------------------------
# Stub LLM responses
# ---------------------------------------------------------------------------

_FUND_RESPONSE = json.dumps({
    "decision": "Buy",
    "confidence": 0.80,
    "rationale": "P/E below historical average; revenue growth solid.",
    "key_concern": "Interest rate sensitivity for growth assets.",
    "time_horizon": "medium-term",
})

_TECH_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.65,
    "rationale": "RSI neutral territory; MACD shows weak signal.",
    "key_concern": "Hist vol elevated at 0.25.",
    "time_horizon": "short-term",
})

_RISK_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "hist_vol_20d of 0.22 is moderate. Sharpe of 0.82 is acceptable.",
    "key_concern": "max_drawdown_252d of -0.28 is a tail risk.",
    "risk_assessment": "Moderate volatility regime; acceptable risk-adjusted return.",
    "recommended_position_size": 0.05,
})

_MACRO_RESPONSE = json.dumps({
    "decision": "Buy",
    "confidence": 0.55,
    "rationale": "Fed near peak rate; CPI trending down.",
    "key_concern": "Unemployment uptick could signal recession ahead.",
    "regime_assessment": "Late tightening cycle with easing bias.",
    "macro_rationale": "CPI deceleration supports risk asset recovery.",
})

_CONTRARIAN_RESPONSE = json.dumps({
    "alternative_thesis": "Rate cuts are priced in; real rates remain restrictive.",
    "risk_scenario": "Credit spread widening triggers equity correction of 15%.",
    "counterargument": "Consensus underestimates duration of high-rate environment.",
    "confidence": 0.70,
})


def _stub_llm(response_content: str, model_name: str = "stub-model"):
    class _Stub:
        def invoke(self, _messages):
            class _R:
                content = response_content
            return _R()
    stub = _Stub()
    stub.model_name = model_name
    return stub


@pytest.fixture
def patched_llms(monkeypatch):
    """
    Patch make_llm in all 5 voting+contrarian agent modules.

    Also patches sentiment_agent.call_tool to return empty passages, forcing
    the fail-open path (Hold/0.0 default signal without calling the LLM).
    This ensures the test is deterministic regardless of whether the live
    knowledge store or LM Studio is available.
    """
    import hifi.agents.contrarian_agent as ca
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.macro_agent as ma
    import hifi.agents.risk_agent as ra
    import hifi.agents.sentiment_agent as sa_mod
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm(_FUND_RESPONSE))
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm(_TECH_RESPONSE))
    monkeypatch.setattr(ra, "make_llm", lambda *a, **kw: _stub_llm(_RISK_RESPONSE))
    monkeypatch.setattr(ma, "make_llm", lambda *a, **kw: _stub_llm(_MACRO_RESPONSE))
    monkeypatch.setattr(ca, "make_llm", lambda *a, **kw: _stub_llm(_CONTRARIAN_RESPONSE))
    # Force sentiment fail-open: return empty passages so no LLM call is made
    monkeypatch.setattr(sa_mod, "call_tool", lambda *a, **kw: {"passages": []})


# ---------------------------------------------------------------------------
# Test 1: Full 6-agent ensemble — all Phase 8 fields populated
# ---------------------------------------------------------------------------


def test_run_ensemble_all_agents_populates_phase8_fields(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """
    agents=None runs all 6 agents. All 4 Phase 8 EnsembleOutput fields
    must be populated (not None).
    """
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir
    )

    assert isinstance(output, EnsembleOutput)
    assert output.ticker == "AAPL"
    assert output.as_of_date == "2023-03-31"
    assert output.fundamental_analysis.signal is not None
    assert output.technical_analysis.signal is not None
    assert output.risk_analysis is not None
    assert output.macro_analysis is not None
    assert output.sentiment_analysis is not None
    assert output.contrarian_analysis is not None


# ---------------------------------------------------------------------------
# Test 2: Contrarian analysis fields are non-empty and well-formed
# ---------------------------------------------------------------------------


def test_contrarian_analysis_fields_are_non_empty(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """ContrarianAnalysis must have non-empty text fields and valid confidence."""
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir
    )

    c = output.contrarian_analysis
    assert c.alternative_thesis.strip()
    assert c.risk_scenario.strip()
    assert c.counterargument.strip()
    assert 0.0 <= c.confidence <= 1.0


# ---------------------------------------------------------------------------
# Test 3: Backward compat — agents=["fundamental","technical"] → Phase 8 None
# ---------------------------------------------------------------------------


def test_phase4_agent_subset_gives_none_phase8_fields(
    monkeypatch, fixtures_data_dir, aapl_snapshot_json
):
    """
    agents=["fundamental","technical"] is the Phase 4/6/7 backward-compat call.
    All 4 Phase 8 EnsembleOutput fields must be None.
    """
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm(_FUND_RESPONSE))
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm(_TECH_RESPONSE))

    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
    )

    assert output.risk_analysis is None
    assert output.macro_analysis is None
    assert output.sentiment_analysis is None
    assert output.contrarian_analysis is None


# ---------------------------------------------------------------------------
# Test 4: EnsembleOutput is JSON-serializable
# ---------------------------------------------------------------------------


def test_ensemble_output_is_json_serializable(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """Full 6-agent EnsembleOutput must round-trip through JSON without error."""
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir
    )

    serialized = output.model_dump_json()
    parsed = json.loads(serialized)
    assert parsed["ticker"] == "AAPL"
    assert parsed["contrarian_analysis"] is not None
    assert parsed["risk_analysis"] is not None


# ---------------------------------------------------------------------------
# Test 5: Sentiment fail-open produces default Hold/0.0 signal
# ---------------------------------------------------------------------------


def test_sentiment_fail_open_produces_insufficient_data_signal(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """
    Without a LanceDB knowledge store, _retrieve_context returns "" and the
    agent short-circuits to the default "Insufficient Data" signal (Hold, 0.0).
    This validates the DJ-038 fail-open design for the Sentiment Agent.
    """
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir
    )

    sa = output.sentiment_analysis
    assert sa is not None
    assert sa.signal is not None
    assert sa.signal.decision == "Hold"
    assert sa.signal.confidence == pytest.approx(0.0)
    assert "Insufficient" in sa.sentiment_summary
