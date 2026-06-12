"""
Integration tests for RAG-augmented agents (Phase 7 / E7).

Covers:
- use_rag=False (regression): graph structure identical to Phase 6, v1 prompt used
- use_rag=True with mock passages: retrieved_context non-empty, v2 prompt selected
- use_rag=True with retrieval failure: fail-open to "", v1 prompt used
- run_ensemble(use_rag=True): use_rag forwarded to both agents
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

import hifi.agents.fundamental_agent as fa
import hifi.agents.technical_agent as ta
from hifi.agents.ensemble_runner import run_ensemble
from hifi.agents.schemas import AgentSignal, FundamentalAnalysis, TechnicalAnalysis
from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TICKER = "AAPL"
_DATE = "2023-03-31"

_STUB_SIGNAL = AgentSignal(
    ticker=_TICKER,
    as_of_date=_DATE,
    decision="Hold",
    confidence=0.6,
    rationale="Test rationale.",
    key_concern="Test concern.",
    data_gaps=[],
    call_ids=[],
    model_id="stub-model",
    agent_type="fundamental",
)

_STUB_TECH_SIGNAL = AgentSignal(
    ticker=_TICKER,
    as_of_date=_DATE,
    decision="Hold",
    confidence=0.5,
    rationale="Test rationale.",
    key_concern="Test concern.",
    data_gaps=[],
    call_ids=[],
    model_id="stub-model",
    agent_type="technical",
)

_MOCK_PASSAGES = [
    {
        "rank": 1,
        "ticker": _TICKER,
        "filing_type": "10-K",
        "section": "MD&A",
        "period": "2022-12-31",
        "text": "Apple reported strong services revenue growth.",
        "score": 0.92,
    }
]


@pytest.fixture()
def snapshot_json() -> str:
    _dt = datetime(2023, 4, 1)
    snap = FundamentalsSnapshot(
        ticker=_TICKER,
        period_end=_DATE,
        source="test",
        fetched_at=_dt,
        provenance=ProvenanceRecord(source="test", fetched_at=_dt),
    )
    return snap.model_dump_json()


# ---------------------------------------------------------------------------
# Fundamental Agent — regression: use_rag=False
# ---------------------------------------------------------------------------


class _FundamentalStub:
    model_name = "stub-model"

    def invoke(self, messages):
        class _R:
            content = json.dumps(
                {
                    "decision": "Hold",
                    "confidence": 0.6,
                    "rationale": "P/E of 28.0 is fair.",
                    "key_concern": "Macro risk.",
                }
            )

        return _R()


def _stub_tool_results(tool_name, params, *, data_dir=None, server_module=None):
    """Stub call_tool: returns minimal dict for financial tools, empty passages for knowledge."""
    if tool_name == "retrieve_context":
        return {"passages": _MOCK_PASSAGES}
    return {
        "call_id": "abc123",
        "pe_ratio": 28.0,
        "roe": 0.24,
        "revenue_growth_yoy": 0.05,
        "market_cap": 2_600_000_000_000,
        "rsi_14": 52.0,
        "macd": 0.5,
        "macd_signal": 0.3,
        "sharpe_252d": 1.2,
        "hist_vol_20d": 0.18,
    }


def _stub_no_rag_tool_results(tool_name, params, *, data_dir=None, server_module=None):
    """Stub call_tool: retrieve_context returns empty passages."""
    if tool_name == "retrieve_context":
        return {"passages": []}
    return _stub_tool_results(tool_name, params, data_dir=data_dir, server_module=server_module)


def _stub_rag_failure(tool_name, params, *, data_dir=None, server_module=None):
    """Stub call_tool: retrieve_context raises an exception."""
    if tool_name == "retrieve_context":
        raise RuntimeError("Knowledge server unavailable")
    return _stub_tool_results(tool_name, params, data_dir=data_dir, server_module=server_module)


def test_fundamental_use_rag_false_uses_v1_prompt(monkeypatch, snapshot_json):
    """use_rag=False: prompt_version is fundamental_v1 and graph has no retrieve_context node."""
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _FundamentalStub())
    monkeypatch.setattr(fa, "call_tool", _stub_no_rag_tool_results)

    result = fa.run_analysis(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snapshot_json,
        use_rag=False,
    )

    assert isinstance(result, FundamentalAnalysis)
    assert result.prompt_version == "fundamental_v1"
    assert result.signal is not None
    assert result.signal.decision == "Hold"


def test_fundamental_use_rag_true_with_passages_uses_v2_prompt(monkeypatch, snapshot_json):
    """use_rag=True with non-empty passages: prompt_version is fundamental_v2."""
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _FundamentalStub())
    monkeypatch.setattr(fa, "call_tool", _stub_tool_results)

    result = fa.run_analysis(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snapshot_json,
        use_rag=True,
    )

    assert isinstance(result, FundamentalAnalysis)
    assert result.prompt_version == "fundamental_v2"
    assert result.signal is not None


def test_fundamental_use_rag_true_retrieval_failure_falls_back_to_v1(monkeypatch, snapshot_json):
    """use_rag=True but retrieval fails: fail-open to v1 prompt, signal still produced."""
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _FundamentalStub())
    monkeypatch.setattr(fa, "call_tool", _stub_rag_failure)

    result = fa.run_analysis(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snapshot_json,
        use_rag=True,
    )

    assert isinstance(result, FundamentalAnalysis)
    assert result.prompt_version == "fundamental_v1"
    assert result.signal is not None


def test_fundamental_use_rag_true_empty_passages_falls_back_to_v1(monkeypatch, snapshot_json):
    """use_rag=True but empty passages list: fall back to v1 prompt."""
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _FundamentalStub())
    monkeypatch.setattr(fa, "call_tool", _stub_no_rag_tool_results)

    result = fa.run_analysis(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snapshot_json,
        use_rag=True,
    )

    assert isinstance(result, FundamentalAnalysis)
    assert result.prompt_version == "fundamental_v1"


# ---------------------------------------------------------------------------
# Technical Agent — regression and RAG tests
# ---------------------------------------------------------------------------


class _TechnicalStub:
    model_name = "stub-model"

    def invoke(self, messages):
        class _R:
            content = json.dumps(
                {
                    "decision": "Hold",
                    "confidence": 0.5,
                    "rationale": "RSI of 52.0 is neutral.",
                    "key_concern": "Volatility.",
                    "time_horizon": "medium-term",
                }
            )

        return _R()


def test_technical_use_rag_false_uses_v1_prompt(monkeypatch):
    """use_rag=False: prompt_version is technical_v1."""
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _TechnicalStub())
    monkeypatch.setattr(ta, "call_tool", _stub_no_rag_tool_results)

    result = ta.run_technical_analysis(
        ticker=_TICKER,
        as_of_date=_DATE,
        use_rag=False,
    )

    assert isinstance(result, TechnicalAnalysis)
    assert result.prompt_version == "technical_v1"
    assert result.signal is not None
    assert result.signal.decision == "Hold"


def test_technical_use_rag_true_with_passages_uses_v2_prompt(monkeypatch):
    """use_rag=True with non-empty passages: prompt_version is technical_v2."""
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _TechnicalStub())
    monkeypatch.setattr(ta, "call_tool", _stub_tool_results)

    result = ta.run_technical_analysis(
        ticker=_TICKER,
        as_of_date=_DATE,
        use_rag=True,
    )

    assert isinstance(result, TechnicalAnalysis)
    assert result.prompt_version == "technical_v2"
    assert result.signal is not None


def test_technical_use_rag_true_retrieval_failure_falls_back_to_v1(monkeypatch):
    """use_rag=True but retrieval fails: fail-open, v1 prompt used."""
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _TechnicalStub())
    monkeypatch.setattr(ta, "call_tool", _stub_rag_failure)

    result = ta.run_technical_analysis(
        ticker=_TICKER,
        as_of_date=_DATE,
        use_rag=True,
    )

    assert isinstance(result, TechnicalAnalysis)
    assert result.prompt_version == "technical_v1"
    assert result.signal is not None


# ---------------------------------------------------------------------------
# run_ensemble — use_rag forwarding
# ---------------------------------------------------------------------------


def test_ensemble_use_rag_false_default(monkeypatch, snapshot_json):
    """run_ensemble default (use_rag=False): both agents produce v1 prompts."""
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _FundamentalStub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _TechnicalStub())
    monkeypatch.setattr(fa, "call_tool", _stub_no_rag_tool_results)
    monkeypatch.setattr(ta, "call_tool", _stub_no_rag_tool_results)

    from hifi.collective.schemas import EnsembleOutput

    output = run_ensemble(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snapshot_json,
        use_rag=False,
    )

    assert isinstance(output, EnsembleOutput)
    assert output.fundamental_analysis.prompt_version == "fundamental_v1"
    assert output.technical_analysis.prompt_version == "technical_v1"


def test_ensemble_use_rag_true_forwards_to_both_agents(monkeypatch, snapshot_json):
    """run_ensemble(use_rag=True): both agents see passages and use v2 prompts."""
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _FundamentalStub())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _TechnicalStub())
    monkeypatch.setattr(fa, "call_tool", _stub_tool_results)
    monkeypatch.setattr(ta, "call_tool", _stub_tool_results)

    from hifi.collective.schemas import EnsembleOutput

    output = run_ensemble(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snapshot_json,
        use_rag=True,
    )

    assert isinstance(output, EnsembleOutput)
    assert output.fundamental_analysis.prompt_version == "fundamental_v2"
    assert output.technical_analysis.prompt_version == "technical_v2"
