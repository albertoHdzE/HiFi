"""
Unit tests for KnowledgeStore (P7-E4).

All tests use tmp_path for isolated LanceDB databases and
DeterministicEmbeddingModel for embeddings. No LM Studio required.
"""

from __future__ import annotations

from datetime import date

import pytest

from hifi.knowledge.schemas import DocumentChunk
from hifi.knowledge.vector_store import KnowledgeStore
from tests.conftest import DeterministicEmbeddingModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERIOD = date(2023, 3, 31)
_DIM = 64  # smaller dimensions for faster tests


def _make_chunk(
    ticker: str,
    section: str,
    text: str,
    chunk_index: int = 0,
    config: str = "A",
    filing_type: str = "10-K",
) -> DocumentChunk:
    from math import ceil

    char_count = len(text)
    return DocumentChunk(
        chunk_id=DocumentChunk.make_chunk_id(
            ticker, filing_type, _PERIOD, section, chunk_index, config
        ),
        ticker=ticker,
        filing_type=filing_type,
        period=_PERIOD,
        section=section,
        chunk_index=chunk_index,
        text=text,
        char_count=char_count,
        approx_tokens=ceil(char_count / 4),
        chunking_config=config,
    )


def _make_chunks(n: int, ticker: str = "AAPL", prefix: str = "text") -> list[DocumentChunk]:
    return [
        _make_chunk(ticker, "MD&A", f"{prefix} chunk number {i}", chunk_index=i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# P7-E4-T8: index_chunks + get_stats
# ---------------------------------------------------------------------------


def test_index_chunks_updates_stats(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    chunks = _make_chunks(5)
    embeddings = model.embed([c.text for c in chunks])
    n = store.index_chunks(chunks, embeddings)
    assert n == 5
    stats = store.get_stats()
    assert stats["n_chunks"] == 5


def test_index_empty_chunks_returns_zero(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    n = store.index_chunks([], [])
    assert n == 0
    assert store.get_stats()["n_chunks"] == 0


def test_index_mismatched_lengths_raises(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    chunks = _make_chunks(3)
    with pytest.raises(ValueError, match="same length"):
        store.index_chunks(chunks, [[0.1] * _DIM, [0.2] * _DIM])


# ---------------------------------------------------------------------------
# P7-E4-T9: search returns at most top_k results
# ---------------------------------------------------------------------------


def test_search_returns_at_most_top_k(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    chunks = _make_chunks(10)
    embeddings = model.embed([c.text for c in chunks])
    store.index_chunks(chunks, embeddings)

    query = model.embed_one("Apple iPhone revenue")
    results = store.search(query, ticker="AAPL", top_k=3)
    assert len(results) <= 3


def test_search_empty_store_returns_empty_list(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    query = model.embed_one("Apple iPhone revenue")
    results = store.search(query, ticker="AAPL", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# P7-E4-T10: search filters by ticker
# ---------------------------------------------------------------------------


def test_search_ticker_filter_only_returns_matching_ticker(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)

    aapl_chunks = _make_chunks(5, ticker="AAPL", prefix="apple")
    jpm_chunks = _make_chunks(5, ticker="JPM", prefix="jpmorgan")
    all_chunks = aapl_chunks + jpm_chunks
    embeddings = model.embed([c.text for c in all_chunks])
    store.index_chunks(all_chunks, embeddings)

    query = model.embed_one("Apple revenue")
    results = store.search(query, ticker="AAPL", top_k=10)
    assert all(c.ticker == "AAPL" for c in results), "Search must filter by ticker"


def test_search_unknown_ticker_returns_empty(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    chunks = _make_chunks(5, ticker="AAPL")
    embeddings = model.embed([c.text for c in chunks])
    store.index_chunks(chunks, embeddings)
    query = model.embed_one("query")
    results = store.search(query, ticker="XOM", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# P7-E4-T11: cosine order — identical vector ranks first
# ---------------------------------------------------------------------------


def test_search_cosine_order_identical_vector_ranks_first(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)

    chunks = [
        _make_chunk("AAPL", "MD&A", "target text for cosine test", chunk_index=0),
        _make_chunk("AAPL", "MD&A", "unrelated other text content here", chunk_index=1),
        _make_chunk("AAPL", "MD&A", "completely different financial data", chunk_index=2),
    ]
    # Embed all chunks
    embeddings = model.embed([c.text for c in chunks])
    store.index_chunks(chunks, embeddings)

    # Query with the exact embedding of chunk 0 — it should rank first
    query_embedding = embeddings[0]
    results = store.search(query_embedding, ticker="AAPL", top_k=3)
    assert len(results) > 0
    assert results[0].chunk_id == chunks[0].chunk_id, (
        "Query identical to chunk 0's embedding should return chunk 0 first"
    )


# ---------------------------------------------------------------------------
# P7-E4-T12: clear() resets stats to zero
# ---------------------------------------------------------------------------


def test_clear_resets_stats(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    chunks = _make_chunks(5)
    embeddings = model.embed([c.text for c in chunks])
    store.index_chunks(chunks, embeddings)
    assert store.get_stats()["n_chunks"] == 5

    store.clear()
    stats = store.get_stats()
    assert stats["n_chunks"] == 0


# ---------------------------------------------------------------------------
# P7-E4-T13: configs A and B use separate Lance tables
# ---------------------------------------------------------------------------


def test_config_a_and_b_use_separate_tables(tmp_path):
    store_a = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    store_b = KnowledgeStore(data_dir=tmp_path, chunking_config="B", dimensions=_DIM)

    model = DeterministicEmbeddingModel(dimensions=_DIM)
    chunks_a = _make_chunks(3, prefix="config a", ticker="AAPL")
    chunks_b = _make_chunks(7, prefix="config b", ticker="JPM")

    store_a.index_chunks(chunks_a, model.embed([c.text for c in chunks_a]))
    store_b.index_chunks(chunks_b, model.embed([c.text for c in chunks_b]))

    # Each store sees only its own data
    assert store_a.get_stats()["n_chunks"] == 3
    assert store_b.get_stats()["n_chunks"] == 7


# ---------------------------------------------------------------------------
# P7-E4-T14: all 3 configs coexist in the same directory
# ---------------------------------------------------------------------------


def test_all_three_configs_coexist(tmp_path):
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    for config, n in [("A", 3), ("B", 5), ("C", 7)]:
        store = KnowledgeStore(data_dir=tmp_path, chunking_config=config, dimensions=_DIM)
        chunks = _make_chunks(n, prefix=f"config {config}")
        store.index_chunks(chunks, model.embed([c.text for c in chunks]))

    # Reopen all three stores and verify data isolation
    store_a = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    store_b = KnowledgeStore(data_dir=tmp_path, chunking_config="B", dimensions=_DIM)
    store_c = KnowledgeStore(data_dir=tmp_path, chunking_config="C", dimensions=_DIM)

    assert store_a.get_stats()["n_chunks"] == 3
    assert store_b.get_stats()["n_chunks"] == 5
    assert store_c.get_stats()["n_chunks"] == 7


# ---------------------------------------------------------------------------
# get_stats n_tickers and n_filing_types
# ---------------------------------------------------------------------------


def test_get_stats_counts_unique_tickers_and_filing_types(tmp_path):
    store = KnowledgeStore(data_dir=tmp_path, chunking_config="A", dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    chunks = [
        _make_chunk("AAPL", "MD&A", "aapl text 1", chunk_index=0, filing_type="10-K"),
        _make_chunk("AAPL", "MD&A", "aapl text 2", chunk_index=1, filing_type="10-Q"),
        _make_chunk("JPM", "MD&A", "jpm text 1", chunk_index=0, filing_type="10-K"),
    ]
    embeddings = model.embed([c.text for c in chunks])
    store.index_chunks(chunks, embeddings)
    stats = store.get_stats()
    assert stats["n_chunks"] == 3
    assert stats["n_tickers"] == 2
    assert stats["n_filing_types"] == 2
