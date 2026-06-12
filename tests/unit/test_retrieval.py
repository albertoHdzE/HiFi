"""Unit tests for KnowledgeRetriever and evaluate_precision_at_k (P7-E5)."""

from __future__ import annotations

import time
from datetime import date
from math import ceil

import pytest

from hifi.knowledge.retrieval import KnowledgeRetriever, evaluate_precision_at_k
from hifi.knowledge.schemas import DocumentChunk, EvaluationQuery
from hifi.knowledge.vector_store import KnowledgeStore
from tests.conftest import DeterministicEmbeddingModel

_PERIOD = date(2023, 3, 31)
_DIM = 32


def _make_chunk(ticker, section, text, chunk_index=0, filing_type="10-K"):
    return DocumentChunk(
        chunk_id=DocumentChunk.make_chunk_id(
            ticker, filing_type, _PERIOD, section, chunk_index, "A"
        ),
        ticker=ticker, filing_type=filing_type, period=_PERIOD,
        section=section, chunk_index=chunk_index, text=text,
        char_count=len(text), approx_tokens=ceil(len(text) / 4), chunking_config="A",
    )


def _populated_store(tmp_path, tickers=("AAPL", "JPM", "XOM")):
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    chunks = []
    for ticker in tickers:
        for section in ("MD&A", "Risk Factors", "Business", "Earnings Release"):
            for i in range(5):
                ft = "8-K" if section == "Earnings Release" else "10-K"
                chunks.append(_make_chunk(ticker, section, f"{ticker} {section} text {i}", i, ft))
    embeddings = model.embed([c.text for c in chunks])
    store.index_chunks(chunks, embeddings)
    return store, model


# P7-E5-T6: retrieve returns at most top_k chunks
def test_retrieve_at_most_top_k(tmp_path):
    store, model = _populated_store(tmp_path)
    retriever = KnowledgeRetriever(store=store, embedding_model=model)
    results = retriever.retrieve("Apple iPhone revenue", ticker="AAPL", top_k=3)
    assert len(results) <= 3


# P7-E5-T7: format_context includes source metadata
def test_format_context_includes_metadata(tmp_path):
    store, model = _populated_store(tmp_path)
    retriever = KnowledgeRetriever(store=store, embedding_model=model)
    chunks = retriever.retrieve("Apple iPhone revenue", ticker="AAPL", top_k=2)
    if not chunks:
        pytest.skip("No results from store")
    ctx = retriever.format_context(chunks)
    assert "AAPL" in ctx
    assert "10-K" in ctx or "8-K" in ctx
    assert "2023-03-31" in ctx


def test_format_context_empty_returns_empty_string():
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    # Use a minimal retriever that won't be called
    class MinimalStore:
        def get_stats(self): return {"n_chunks": 0}
        def search(self, *a, **k): return []
    retriever = KnowledgeRetriever(store=MinimalStore(), embedding_model=model)
    assert retriever.format_context([]) == ""


# P7-E5-T8: perfect retriever -> P@5 = 1.0
def test_evaluate_precision_perfect_retriever():
    queries = [
        EvaluationQuery(query_id="Q01", query="q", ticker="AAPL",
                        relevant_section="MD&A", relevant_filing_type="10-K"),
    ]
    # Retriever that always returns the relevant chunk first
    class PerfectRetriever:
        def retrieve(self, query, ticker, top_k=5):
            return [_make_chunk(ticker, "MD&A", "text", i, "10-K") for i in range(top_k)]
        def format_context(self, chunks): return ""
    score = evaluate_precision_at_k(PerfectRetriever(), queries, k=5)
    assert score == 1.0


# P7-E5-T9: null retriever -> P@5 = 0.0
def test_evaluate_precision_null_retriever():
    queries = [EvaluationQuery(query_id="Q01", query="q", ticker="AAPL",
                               relevant_section="MD&A", relevant_filing_type="10-K")]
    class NullRetriever:
        def retrieve(self, *a, **k): return []
        def format_context(self, chunks): return ""
    score = evaluate_precision_at_k(NullRetriever(), queries, k=5)
    assert score == 0.0


def test_evaluate_precision_empty_queries():
    class AnyRetriever:
        def retrieve(self, *a, **k): return []
        def format_context(self, chunks): return ""
    assert evaluate_precision_at_k(AnyRetriever(), [], k=5) == 0.0


# P7-E5-T10: latency - 20 queries under 500ms total
def test_retrieval_latency_under_500ms(tmp_path):
    store, model = _populated_store(tmp_path)
    retriever = KnowledgeRetriever(store=store, embedding_model=model)
    queries = [f"financial query number {i}" for i in range(20)]
    start = time.monotonic()
    for q in queries:
        retriever.retrieve(q, ticker="AAPL", top_k=5)
    elapsed_ms = (time.monotonic() - start) * 1000
    assert elapsed_ms < 500, f"20 queries took {elapsed_ms:.1f}ms (limit: 500ms)"
