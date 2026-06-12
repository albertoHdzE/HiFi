"""
Holistic test for the Phase 7 RAG pipeline (P7-E7-T1 to T5).

Uses DeterministicEmbeddingModel and stub LLMs — no LM Studio or LanceDB
live server required.

What this test validates:
1. Full RAG pipeline: KnowledgeStore → retriever → retrieved_context → v2 prompt → EnsembleOutput
2. Phase 6 regression: run_ensemble(use_rag=False, agents=["fundamental", "technical"])
   identical structure to Phase 6
3. Phase 5 regression: verify_ensemble() works on Phase 7 EnsembleOutput
4. Fail-open: retrieval failure → v1 prompt, valid output still produced
5. RAG output is JSON-safe (serialisable with no errors)
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path

import pytest

from hifi.collective.schemas import EnsembleOutput
from hifi.knowledge.document_ingestion import DocumentIngestionPipeline
from hifi.knowledge.retrieval import KnowledgeRetriever
from hifi.knowledge.schemas import FilingDocument
from hifi.knowledge.vector_store import KnowledgeStore
from tests.conftest import DeterministicEmbeddingModel

_DIM = 32
_TICKER = "AAPL"
_DATE = "2023-03-31"
_PERIOD = date(2023, 3, 31)

# ---------------------------------------------------------------------------
# Shared LLM stubs
# ---------------------------------------------------------------------------

_FUND_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "P/E of 28.0 is fair. Per the 10-K MD&A, services revenue grew.",
    "key_concern": "Macro headwinds.",
})

_TECH_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.55,
    "rationale": "RSI of 52.0 is neutral. ATR consistent with low volatility.",
    "key_concern": "Overbought risk.",
    "time_horizon": "medium-term",
})


def _stub_fundamental_llm():
    class _Stub:
        model_name = "stub-fundamental"

        def invoke(self, _):
            class _R:
                content = _FUND_RESPONSE
            return _R()
    return _Stub()


def _stub_technical_llm():
    class _Stub:
        model_name = "stub-technical"

        def invoke(self, _):
            class _R:
                content = _TECH_RESPONSE
            return _R()
    return _Stub()


# ---------------------------------------------------------------------------
# Shared snapshot fixture
# ---------------------------------------------------------------------------


def _make_snapshot():
    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

    return FundamentalsSnapshot(
        ticker=_TICKER,
        period_end=_DATE,
        eps=6.11,
        market_cap=2_500_000_000_000,
        revenue=394_330_000_000,
        net_income=99_803_000_000,
        total_assets=352_755_000_000,
        total_equity=62_146_000_000,
        total_liabilities=290_437_000_000,
        source="test",
        fetched_at=datetime(2023, 4, 1),
        provenance=ProvenanceRecord(source="test", fetched_at=datetime(2023, 4, 1)),
    )


# ---------------------------------------------------------------------------
# Knowledge store fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def knowledge_store_and_retriever(tmp_path_factory):
    """Build a real KnowledgeStore with deterministic AAPL chunks for testing."""
    store_dir = tmp_path_factory.mktemp("ks_holistic")

    filing = FilingDocument(
        ticker=_TICKER,
        cik="0000320193",
        filing_type="10-K",
        accession_number="0000320193-22-000001",
        period_of_report=_PERIOD,
        filed_date=date(2022, 10, 28),
        sections={
            "MD&A": (
                "Apple's net sales decreased 5 percent to $117.2 billion in Q4 2022. "
                "Services revenue grew 5 percent year-over-year to $20.8 billion. "
                "Gross margin was 42.3 percent for the quarter. "
                "iPhone revenue declined slightly due to foreign exchange headwinds. "
            ) * 10,
            "Risk Factors": (
                "Foreign currency fluctuations adversely affect international revenues. "
                "Competition in the smartphone market is intense and growing. "
                "Supply chain disruptions could increase product costs. "
            ) * 10,
        },
        source_url="https://example.com",
        fetched_at=datetime(2023, 4, 1),
    )

    model = DeterministicEmbeddingModel(dimensions=_DIM)
    store = KnowledgeStore(data_dir=store_dir, chunking_config="A", dimensions=_DIM)
    pipeline = DocumentIngestionPipeline("A")
    chunks = pipeline.chunk_document(filing)
    embeddings = model.embed([c.text for c in chunks])
    store.index_chunks(chunks, embeddings)

    retriever = KnowledgeRetriever(store=store, embedding_model=model)
    return retriever


# ---------------------------------------------------------------------------
# Fixtures data dir (market + macro parquet files for financial MCP tools)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixtures_data_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    fixtures_root = Path(__file__).parent.parent / "fixtures"
    for subdir in ("market", "macro"):
        dst = tmp / subdir
        dst.mkdir()
        src = fixtures_root / subdir
        if src.is_dir():
            for f in src.iterdir():
                if f.suffix == ".parquet":
                    shutil.copy(f, dst / f.name)
    return str(tmp)


# ---------------------------------------------------------------------------
# call_tool factory
# ---------------------------------------------------------------------------

_STUB_FINANCIAL = {
    "call_id": "holistic0000",
    "pe_ratio": 28.0,
    "roe": 0.24,
    "revenue_growth_yoy": 0.05,
    "market_cap": 2_600_000_000_000,
    "rsi_14": 52.0,
    "macd": 0.5,
    "macd_signal": 0.3,
    "sharpe_252d": 1.2,
    "hist_vol_20d": 0.18,
    "net_income": 99_803_000_000,
    "total_equity": 62_146_000_000,
    "revenue": 394_330_000_000,
    "fed_funds_rate": 4.75,
    "cpi_yoy": 0.050,
    "unemployment_rate": 0.034,
}


def _make_call_tool(retriever: KnowledgeRetriever, fail_rag: bool = False):
    """Return a call_tool stub that routes retrieve_context to the real retriever."""

    def _call_tool(tool_name, params, *, data_dir=None, server_module=None):
        if tool_name == "retrieve_context":
            if fail_rag:
                raise RuntimeError("Simulated knowledge server failure")
            ticker = params.get("ticker", _TICKER)
            query = params.get("query", "")
            k = params.get("top_k", 5)
            chunks = retriever.retrieve(query=query, ticker=ticker, top_k=k)
            passages = [
                {
                    "rank": i + 1,
                    "filing_type": c.filing_type,
                    "section": c.section,
                    "period": c.period.isoformat(),
                    "text": c.text,
                }
                for i, c in enumerate(chunks)
            ]
            return {"call_id": "retrieval00a0", "passages": passages, "n_retrieved": len(passages)}
        return dict(_STUB_FINANCIAL)

    return _call_tool


# ---------------------------------------------------------------------------
# T1: Full RAG pipeline — v2 prompts, valid EnsembleOutput
# ---------------------------------------------------------------------------


def test_rag_full_pipeline_produces_ensemble_output(
    monkeypatch, knowledge_store_and_retriever
):
    """
    P7-E7-T1: Full pipeline with real KnowledgeStore and stub LLM.

    Both agents retrieve AAPL passages → use v2 prompts → produce EnsembleOutput.
    """
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble

    retriever = knowledge_store_and_retriever
    ct = _make_call_tool(retriever)

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_fundamental_llm())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_technical_llm())
    monkeypatch.setattr(fa, "call_tool", ct)
    monkeypatch.setattr(ta, "call_tool", ct)

    snap = _make_snapshot()
    output = run_ensemble(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snap.model_dump_json(),
        use_rag=True,
        agents=["fundamental", "technical"],
    )

    assert isinstance(output, EnsembleOutput)
    assert output.ticker == _TICKER
    assert output.as_of_date == _DATE
    assert output.ensemble_decision is not None
    assert output.fundamental_analysis.prompt_version == "fundamental_v2"
    assert output.technical_analysis.prompt_version == "technical_v2"
    assert output.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# T2: Phase 6 regression — use_rag=False identical structure
# ---------------------------------------------------------------------------


def test_phase6_regression_rag_false_identical_structure(
    monkeypatch, knowledge_store_and_retriever
):
    """
    P7-E7-T2: use_rag=False produces same output structure as Phase 6 (v1 prompts).
    """
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble

    retriever = knowledge_store_and_retriever
    ct = _make_call_tool(retriever)

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_fundamental_llm())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_technical_llm())
    monkeypatch.setattr(fa, "call_tool", ct)
    monkeypatch.setattr(ta, "call_tool", ct)

    snap = _make_snapshot()
    output = run_ensemble(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snap.model_dump_json(),
        use_rag=False,
        agents=["fundamental", "technical"],
    )

    assert isinstance(output, EnsembleOutput)
    assert output.fundamental_analysis.prompt_version == "fundamental_v1"
    assert output.technical_analysis.prompt_version == "technical_v1"
    # Must have valid signals
    assert output.fundamental_analysis.signal is not None
    assert output.technical_analysis.signal is not None


# ---------------------------------------------------------------------------
# T3: Phase 5 regression — verify_ensemble works on Phase 7 output
# ---------------------------------------------------------------------------


def test_phase5_regression_verify_ensemble_on_rag_output(
    monkeypatch, knowledge_store_and_retriever
):
    """
    P7-E7-T3: verify_ensemble() produces a valid EnsembleVerificationReport
    on Phase 7 RAG output. Phase 5 verification layer unchanged.
    """
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble
    from hifi.verification.schemas import EnsembleVerificationReport
    from hifi.verification.verifier import verify_ensemble

    retriever = knowledge_store_and_retriever
    ct = _make_call_tool(retriever)

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_fundamental_llm())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_technical_llm())
    monkeypatch.setattr(fa, "call_tool", ct)
    monkeypatch.setattr(ta, "call_tool", ct)

    snap = _make_snapshot()
    output = run_ensemble(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snap.model_dump_json(),
        use_rag=True,
        agents=["fundamental", "technical"],
    )

    report = verify_ensemble(output, always_verify=True)

    assert isinstance(report, EnsembleVerificationReport)
    assert report.ticker == _TICKER
    assert 0.0 <= report.fundamental_report.hallucination_rate <= 1.0
    assert 0.0 <= report.technical_report.hallucination_rate <= 1.0


# ---------------------------------------------------------------------------
# T4: Fail-open — RAG server failure → v1 prompt, valid output
# ---------------------------------------------------------------------------


def test_rag_server_failure_fail_open(monkeypatch, knowledge_store_and_retriever):
    """
    P7-E7-T4: When knowledge server raises an exception, agents fall back to
    v1 prompts transparently. Output is still a valid EnsembleOutput.
    """
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble

    retriever = knowledge_store_and_retriever
    ct_fail = _make_call_tool(retriever, fail_rag=True)

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_fundamental_llm())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_technical_llm())
    monkeypatch.setattr(fa, "call_tool", ct_fail)
    monkeypatch.setattr(ta, "call_tool", ct_fail)

    snap = _make_snapshot()
    output = run_ensemble(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snap.model_dump_json(),
        use_rag=True,
        agents=["fundamental", "technical"],
    )

    assert isinstance(output, EnsembleOutput)
    # Fail-open: fell back to v1 prompts
    assert output.fundamental_analysis.prompt_version == "fundamental_v1"
    assert output.technical_analysis.prompt_version == "technical_v1"
    # Signal still produced despite retrieval failure
    assert output.fundamental_analysis.signal is not None
    assert output.technical_analysis.signal is not None


# ---------------------------------------------------------------------------
# T5: Output JSON-safe
# ---------------------------------------------------------------------------


def test_rag_output_is_json_safe(monkeypatch, knowledge_store_and_retriever):
    """P7-E7-T5: Full RAG EnsembleOutput can be serialised to JSON without error."""
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.technical_agent as ta
    from hifi.agents.ensemble_runner import run_ensemble

    retriever = knowledge_store_and_retriever
    ct = _make_call_tool(retriever)

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_fundamental_llm())
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_technical_llm())
    monkeypatch.setattr(fa, "call_tool", ct)
    monkeypatch.setattr(ta, "call_tool", ct)

    snap = _make_snapshot()
    output = run_ensemble(
        ticker=_TICKER,
        as_of_date=_DATE,
        snapshot_json=snap.model_dump_json(),
        use_rag=True,
        agents=["fundamental", "technical"],
    )

    serialised = json.dumps(output.model_dump())
    assert len(serialised) > 100
