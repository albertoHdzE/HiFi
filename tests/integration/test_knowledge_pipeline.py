"""Integration tests for the full knowledge pipeline (P7-E5 T11-T14)."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pytest

from hifi.knowledge.document_ingestion import DocumentIngestionPipeline
from hifi.knowledge.retrieval import KnowledgeRetriever
from hifi.knowledge.schemas import EvaluationQuery, FilingDocument
from hifi.knowledge.vector_store import KnowledgeStore
from tests.conftest import DeterministicEmbeddingModel

_DIM = 32
_PERIOD = date(2023, 3, 31)

# Fixture filings (one per ticker, three sections each)
_FILINGS = [
    FilingDocument(
        ticker="AAPL", cik="0000320193", filing_type="10-K",
        accession_number="0000320193-22-000001",
        period_of_report=_PERIOD, filed_date=date(2022, 10, 28),
        sections={
            "Business": "Apple Inc. designs and markets consumer electronics and software. "
                        "iPhone is the flagship product line. Services is growing rapidly. " * 20,
            "Risk Factors": (
                "Foreign currency risk may adversely affect international revenue. "
                "Competition in smartphone market is intense. Supply chains are complex. " * 20
            ),
            "MD&A": "Net sales decreased 5% to $117.2 billion. "
                    "Gross margin was 42.3 percent. iPhone revenue declined. " * 20,
        },
        source_url="https://example.com", fetched_at=__import__("datetime").datetime(2023, 4, 1),
    ),
    FilingDocument(
        ticker="JPM", cik="0000019617", filing_type="10-K",
        accession_number="0000019617-23-000001",
        period_of_report=_PERIOD, filed_date=date(2023, 2, 21),
        sections={
            "Business": "JPMorgan Chase is a global financial services firm. "
                        "Consumer and community banking serves retail customers. " * 20,
            "Risk Factors": "Credit risk arises from lending activities. "
                            "Market risk results from fluctuations in interest rates. " * 20,
            "MD&A": "Net interest income increased 48 percent to $67.0 billion. "
                    "Credit loss provisions were $9.3 billion. CET1 ratio was 15.0 percent. " * 20,
        },
        source_url="https://example.com", fetched_at=__import__("datetime").datetime(2023, 4, 1),
    ),
    FilingDocument(
        ticker="XOM", cik="0000034088", filing_type="10-K",
        accession_number="0000034088-23-000001",
        period_of_report=_PERIOD, filed_date=date(2023, 2, 22),
        sections={
            "Business": (
                "ExxonMobil is one of the world's largest publicly traded energy companies. "
                "Low carbon solutions division is investing in carbon capture. " * 20
            ),
            "Risk Factors": "Energy transition poses regulatory and market risks. "
                            "Climate change legislation could increase operating costs. " * 20,
            "MD&A": "Upstream production volume averaged 3.7 million BOE per day. "
                    "Refining margins improved significantly in the Energy Products segment. " * 20,
        },
        source_url="https://example.com", fetched_at=__import__("datetime").datetime(2023, 4, 1),
    ),
]


def _build_store(tmp_path: Path, config: str) -> KnowledgeStore:
    store = KnowledgeStore(data_dir=tmp_path, chunking_config=config, dimensions=_DIM)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    pipeline = DocumentIngestionPipeline(config)
    all_chunks = []
    for doc in _FILINGS:
        all_chunks.extend(pipeline.chunk_document(doc))
    embeddings = model.embed([c.text for c in all_chunks])
    store.index_chunks(all_chunks, embeddings)
    return store


@pytest.fixture(scope="module")
def store_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("knowledge")


# P7-E5-T11,T12,T13: each config returns at least 1 chunk per query
@pytest.mark.parametrize("config", ["A", "B", "C"])
def test_config_store_returns_chunks_for_all_queries(tmp_path, config):
    store = _build_store(tmp_path, config)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    retriever = KnowledgeRetriever(store=store, embedding_model=model)
    # Load evaluation queries
    queries_path = (
        Path(__file__).parent.parent / "fixtures" / "retrieval" / "evaluation_queries.json"
    )
    queries = [EvaluationQuery(**q) for q in json.loads(queries_path.read_text())]
    for query in queries:
        results = retriever.retrieve(query.query, ticker=query.ticker, top_k=5)
        assert len(results) >= 1, (
            f"Config {config}: no results for query {query.query_id}"
        )


# P7-E5-T14: retrieval latency for all 20 queries < 500ms total per config
@pytest.mark.parametrize("config", ["A", "B", "C"])
def test_retrieval_latency_under_500ms(tmp_path, config):
    store = _build_store(tmp_path, config)
    model = DeterministicEmbeddingModel(dimensions=_DIM)
    retriever = KnowledgeRetriever(store=store, embedding_model=model)
    queries_path = (
        Path(__file__).parent.parent / "fixtures" / "retrieval" / "evaluation_queries.json"
    )
    queries = [EvaluationQuery(**q) for q in json.loads(queries_path.read_text())]
    start = time.monotonic()
    for q in queries:
        retriever.retrieve(q.query, ticker=q.ticker, top_k=5)
    elapsed_ms = (time.monotonic() - start) * 1000
    assert elapsed_ms < 500, f"Config {config}: 20 queries took {elapsed_ms:.1f}ms"
