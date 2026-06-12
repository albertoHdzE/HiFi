"""Unit tests for the knowledge MCP server (P7-E6)."""

from __future__ import annotations

import re
from datetime import date
from math import ceil

import pytest

from hifi.knowledge.schemas import DocumentChunk
from hifi.knowledge.vector_store import KnowledgeStore
from tests.conftest import DeterministicEmbeddingModel

_DIM = 32
_PERIOD = date(2023, 3, 31)


def _make_chunk(ticker, section, text, idx=0, ft="10-K"):
    return DocumentChunk(
        chunk_id=DocumentChunk.make_chunk_id(ticker, ft, _PERIOD, section, idx, "A"),
        ticker=ticker, filing_type=ft, period=_PERIOD, section=section,
        chunk_index=idx, text=text, char_count=len(text),
        approx_tokens=ceil(len(text) / 4), chunking_config="A",
    )


@pytest.fixture(autouse=True)
def reset_knowledge_server_singletons():
    """Reset module-level singletons between tests."""
    import hifi.mcp.knowledge_server as ks
    ks._store = None
    ks._retriever = None
    yield
    ks._store = None
    ks._retriever = None


# P7-E6-T4: correct response schema
def test_retrieve_context_returns_correct_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("HIFI_KNOWLEDGE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HIFI_KNOWLEDGE_CHUNKING_CONFIG", "A")

    # Pre-populate store
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    chunks = [_make_chunk("AAPL", "MD&A", f"apple revenue text {i}", i) for i in range(3)]
    store.index_chunks(chunks, model.embed([c.text for c in chunks]))

    # Patch the module to use our test store and model
    import hifi.mcp.knowledge_server as ks
    from hifi.knowledge.retrieval import KnowledgeRetriever
    ks._retriever = KnowledgeRetriever(store=store, embedding_model=model)

    from hifi.mcp.knowledge_server import retrieve_context
    result = retrieve_context(query="Apple revenue", ticker="AAPL", top_k=3)

    assert "call_id" in result
    assert "ticker" in result
    assert "query" in result
    assert "passages" in result
    assert "n_retrieved" in result
    assert result["ticker"] == "AAPL"
    assert result["query"] == "Apple revenue"
    assert isinstance(result["passages"], list)
    assert result["n_retrieved"] == len(result["passages"])


# P7-E6-T5: empty store returns empty passages without crash
def test_retrieve_context_empty_store_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HIFI_KNOWLEDGE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HIFI_KNOWLEDGE_CHUNKING_CONFIG", "A")

    import hifi.mcp.knowledge_server as ks
    from hifi.knowledge.retrieval import KnowledgeRetriever
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    ks._retriever = KnowledgeRetriever(store=store, embedding_model=model)

    from hifi.mcp.knowledge_server import retrieve_context
    result = retrieve_context(query="anything", ticker="AAPL", top_k=5)
    assert result["passages"] == []
    assert result["n_retrieved"] == 0


# P7-E6-T6: call_id is 12-char hex
def test_retrieve_context_call_id_is_12_char_hex(tmp_path, monkeypatch):
    import hifi.mcp.knowledge_server as ks
    from hifi.knowledge.retrieval import KnowledgeRetriever
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    ks._retriever = KnowledgeRetriever(store=store, embedding_model=model)

    from hifi.mcp.knowledge_server import retrieve_context
    result = retrieve_context(query="test", ticker="AAPL")
    cid = result["call_id"]
    assert len(cid) == 12
    assert re.match(r"^[0-9a-f]{12}$", cid), f"call_id not hex: {cid!r}"


# fail-open: unavailable store does not crash
def test_retrieve_context_fails_open_when_no_retriever(tmp_path, monkeypatch):
    monkeypatch.setenv("HIFI_KNOWLEDGE_DATA_DIR", str(tmp_path / "nonexistent"))
    import hifi.mcp.knowledge_server as ks
    ks._retriever = None

    from hifi.mcp.knowledge_server import retrieve_context
    result = retrieve_context(query="test", ticker="AAPL")
    assert "call_id" in result
    assert result["passages"] == []
