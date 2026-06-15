"""
run_phase12_graphrag_eval.py -- Precision@k: plain RAG vs graph-expanded retrieval (P12-E2-T2).

Evaluates retrieval quality for both KnowledgeRetriever (dense ANN only) and
GraphRetriever (graph-expanded dense ANN) against the Phase 7 20-query
evaluation set.

Terminology note (BUG 3 fix, 2026-06-14):
  The GraphRetriever is NOT true GraphRAG (no entity extraction,
  community detection, or relationship summaries). It is
  **graph-expanded dense retrieval**: 2-hop BFS ticker-filter widening
  via FinancialGraph, then standard cosine ANN in LanceDB.

Two-level precision (BUG 2 fix, 2026-06-14):
  - **Document-level**: chunk matches if (ticker, filing_type) match the query.
    Measures whether the retriever finds the correct document type.
  - **Section-level**: chunk matches if (ticker, section, filing_type) all match.
    Measures section-granularity retrieval. Currently limited by the upstream SEC
    parser which produces only 'Full Text' (10-K/10-Q) and 'Earnings Release'
    (8-K), not named sections like 'MD&A' or 'Risk Factors'.
  OQ-K02 uses document-level precision because the section-level gap is a data
  engineering limitation in the SEC parser, not a retrieval algorithm failure.

OQ-K02 threshold: graph-expanded retrieval justified if improvement >= 5pp
absolute (DJ-016).

Output:
  tests/fixtures/baseline/phase12_graphrag_precision.json

Usage:
    uv run python scripts/run_phase12_graphrag_eval.py \
        [--k 5] [--data-dir DIR] [--knowledge-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_EVAL_QUERIES_PATH = _ROOT / "tests" / "fixtures" / "retrieval" / "evaluation_queries.json"
_FIXTURE_OUT = _ROOT / "tests" / "fixtures" / "baseline" / "phase12_graphrag_precision.json"


def _load_eval_queries() -> list:
    from hifi.knowledge.schemas import EvaluationQuery

    if not _EVAL_QUERIES_PATH.exists():
        logger.error("Evaluation queries not found: %s", _EVAL_QUERIES_PATH)
        sys.exit(1)

    raw = json.loads(_EVAL_QUERIES_PATH.read_text())
    return [EvaluationQuery.model_validate(q) for q in raw]


def _build_rag_retriever(data_dir: str, knowledge_dir: str, dimensions: int):
    from hifi.knowledge.embeddings import EmbeddingModel
    from hifi.knowledge.retrieval import KnowledgeRetriever
    from hifi.knowledge.vector_store import KnowledgeStore

    store = KnowledgeStore(
        data_dir=Path(knowledge_dir), chunking_config="A", dimensions=dimensions,
    )
    stats = store.get_stats()
    logger.info("RAG store stats: %s", stats)
    if stats["n_chunks"] == 0:
        logger.error(
            "Knowledge store at %s/knowledge.lance is EMPTY. "
            "Run first: make baseline-phase7",
            knowledge_dir,
        )
        sys.exit(1)
    model = EmbeddingModel()
    return KnowledgeRetriever(store=store, embedding_model=model)


def _build_graph_retriever(
    data_dir: str, knowledge_dir: str, dimensions: int,
):
    from hifi.knowledge.embeddings import EmbeddingModel
    from hifi.knowledge.graph_retrieval import GraphRetriever
    from hifi.knowledge.graph_store import FinancialGraph
    from hifi.knowledge.vector_store import KnowledgeStore

    graph_path = Path(data_dir) / "knowledge_graph" / "financial_graph.json"
    if not graph_path.exists():
        logger.error(
            "Knowledge graph not found at %s. Run: make build-graph", graph_path,
        )
        sys.exit(1)

    graph = FinancialGraph.load(graph_path)
    store = KnowledgeStore(
        data_dir=Path(knowledge_dir), chunking_config="A", dimensions=dimensions,
    )
    model = EmbeddingModel()
    return GraphRetriever(store=store, embedding_model=model, graph=graph)


def _eval_per_query(retriever, queries: list, k: int) -> list[dict]:
    """Run per-query two-level precision evaluation.

    Returns per-query dicts with both document-level and section-level precision.
    """
    per_query: list[dict] = []
    for q in queries:
        chunks = retriever.retrieve(q.query, ticker=q.ticker, top_k=k)

        # Section-level: strict triple match (ticker + section + filing_type)
        section_relevant = sum(
            1
            for c in chunks
            if (
                c.ticker == q.ticker
                and c.section == q.relevant_section
                and c.filing_type == q.relevant_filing_type
            )
        )
        # Document-level: pair match (ticker + filing_type only)
        doc_relevant = sum(
            1
            for c in chunks
            if c.ticker == q.ticker and c.filing_type == q.relevant_filing_type
        )

        per_query.append({
            "query_id": q.query_id,
            "ticker": q.ticker,
            "relevant_section": q.relevant_section,
            "relevant_filing_type": q.relevant_filing_type,
            "section_precision": round(section_relevant / k, 4) if k > 0 else 0.0,
            "document_precision": round(doc_relevant / k, 4) if k > 0 else 0.0,
            "section_relevant_in_top_k": section_relevant,
            "doc_relevant_in_top_k": doc_relevant,
            "n_retrieved": len(chunks),
            "retrieved_sections": sorted({c.section for c in chunks}) if chunks else [],
        })
    return per_query


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 12: Plain RAG vs graph-expanded retrieval Precision@k."
    )
    parser.add_argument("--k", type=int, default=5, help="Retrieval depth k (default 5)")
    parser.add_argument("--data-dir", default=str(_ROOT / "data"))
    parser.add_argument(
        "--knowledge-dir",
        default=str(_ROOT / "data" / "knowledge"),
        help="Path to the knowledge store directory (default: data/knowledge/).",
    )
    parser.add_argument(
        "--dimensions", type=int, default=768, help="Embedding dimensions (default 768)"
    )
    args = parser.parse_args()

    print("Phase 12: Plain RAG vs Graph-Expanded Retrieval Precision@k")
    print("=" * 60)
    print(f"  k               : {args.k}")
    print(f"  data-dir        : {args.data_dir}")
    print(f"  knowledge-dir   : {args.knowledge_dir}")

    queries = _load_eval_queries()
    print(f"  Evaluation set  : {len(queries)} queries")

    # --- Plain RAG (KnowledgeRetriever) ---
    print("\nEvaluating plain RAG (KnowledgeRetriever)...")
    rag_retriever = _build_rag_retriever(args.data_dir, args.knowledge_dir, args.dimensions)
    rag_per_query = _eval_per_query(rag_retriever, queries, args.k)
    rag_doc_p = sum(r["document_precision"] for r in rag_per_query) / len(rag_per_query)
    rag_sec_p = sum(r["section_precision"] for r in rag_per_query) / len(rag_per_query)
    print(f"  RAG Document Precision@{args.k}: {rag_doc_p:.4f}")
    print(f"  RAG Section  Precision@{args.k}: {rag_sec_p:.4f}")

    # --- Graph-expanded retrieval (GraphRetriever) ---
    print("\nEvaluating graph-expanded retrieval (GraphRetriever)...")
    graph_retriever = _build_graph_retriever(args.data_dir, args.knowledge_dir, args.dimensions)
    graph_per_query = _eval_per_query(graph_retriever, queries, args.k)
    graph_doc_p = sum(r["document_precision"] for r in graph_per_query) / len(graph_per_query)
    graph_sec_p = sum(r["section_precision"] for r in graph_per_query) / len(graph_per_query)
    print(f"  Graph Document Precision@{args.k}: {graph_doc_p:.4f}")
    print(f"  Graph Section  Precision@{args.k}: {graph_sec_p:.4f}")

    # --- Delta and per-query comparison (OQ-K02 uses document-level) ---
    doc_delta = graph_doc_p - rag_doc_p
    sec_delta = graph_sec_p - rag_sec_p
    queries_improved = sum(
        1 for r, g in zip(rag_per_query, graph_per_query, strict=False)
        if g["document_precision"] > r["document_precision"]
    )
    queries_degraded = sum(
        1 for r, g in zip(rag_per_query, graph_per_query, strict=False)
        if g["document_precision"] < r["document_precision"]
    )
    queries_unchanged = len(queries) - queries_improved - queries_degraded

    threshold_met = doc_delta >= 0.05

    print(f"\n  Document Delta (Graph - RAG): {doc_delta:+.4f}")
    print(f"  Section  Delta (Graph - RAG): {sec_delta:+.4f}")
    print(f"  Queries improved  : {queries_improved}")
    print(f"  Queries degraded  : {queries_degraded}")
    print(f"  Queries unchanged : {queries_unchanged}")
    oq_label = "PASSED" if threshold_met else "NOT MET"
    dj_label = "ADOPT graph-expanded" if threshold_met else "KEEP plain RAG"
    print(f"\n  OQ-K02 threshold (>= 5pp, doc-level): {oq_label}")
    print(f"  DJ-016 decision: {dj_label}")

    # Indexed section inventory (diagnostic)
    all_retrieved_sections: set[str] = set()
    for pq in rag_per_query + graph_per_query:
        all_retrieved_sections.update(pq["retrieved_sections"])
    query_sections = sorted({q.relevant_section for q in queries})
    idx_label = sorted(all_retrieved_sections) or "(none retrieved)"
    print(f"\n  Indexed sections : {idx_label}")
    print(f"  Query sections   : {query_sections}")
    non_er = set(query_sections) - {"Earnings Release"}
    if all_retrieved_sections and not all_retrieved_sections & non_er:
        print("  WARNING: No overlap between indexed and query sections (except Earnings Release).")
        print("  Section-level precision will be ~0 for 10-K/10-Q queries.")
        print("  Root cause: upstream SEC parser stores 10-K/10-Q content as 'Full Text'.")

    payload = {
        "metadata": {
            "phase": "12",
            "epic": "E2-T2",
            "k": args.k,
            "n_queries": len(queries),
            "run_date": datetime.now(UTC).isoformat(),
            "oq_k02_threshold_pp": 5,
            "precision_levels": {
                "document_level": (
                    "Match on (ticker, filing_type) "
                    "-- measures document retrieval."
                ),
                "section_level": (
                    "Match on (ticker, section, filing_type) "
                    "-- measures section granularity."
                ),
            },
            "oq_k02_uses": "document_level",
            "section_level_note": (
                "Section-level precision is limited by the upstream "
                "SEC parser which produces 'Full Text' (10-K/10-Q) "
                "and 'Earnings Release' (8-K), not named sections "
                "like 'MD&A' or 'Risk Factors'."
            ),
            "terminology_note": (
                "GraphRetriever is graph-expanded dense retrieval "
                "(2-hop BFS ticker widening + cosine ANN), NOT true "
                "GraphRAG (no entity extraction or community detection)."
            ),
        },
        "rag": {
            "document_precision_at_k": round(rag_doc_p, 4),
            "section_precision_at_k": round(rag_sec_p, 4),
        },
        "graph_expanded": {
            "document_precision_at_k": round(graph_doc_p, 4),
            "section_precision_at_k": round(graph_sec_p, 4),
        },
        "delta": {
            "document_level": round(doc_delta, 4),
            "section_level": round(sec_delta, 4),
        },
        "queries_improved": queries_improved,
        "queries_degraded": queries_degraded,
        "queries_unchanged": queries_unchanged,
        "oq_k02_threshold_met": threshold_met,
        "dj_016_decision": (
            "ADOPT graph-expanded retrieval" if threshold_met else "KEEP plain RAG"
        ),
        "per_query": {
            "rag": rag_per_query,
            "graph_expanded": graph_per_query,
        },
    }

    _FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved: {_FIXTURE_OUT}")


if __name__ == "__main__":
    main()
