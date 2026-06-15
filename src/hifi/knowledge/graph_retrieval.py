"""
GraphRAG retrieval for HiFi Phase 12 (P12-E1-T3, DJ-062, DJ-068).

GraphRetriever expands a query ticker to its graph neighborhood (competitors,
sector peers) via FinancialGraph.expand_query_tickers(), then searches the
LanceDB vector store across the expanded ticker set.

This is the graph-augmented path. Plain RAG (KnowledgeRetriever) is the
use_rag=True path. Only one can be active per ensemble run (DJ-068).

Interface compatibility: GraphRetriever exposes the same public API as
KnowledgeRetriever (retrieve, format_context) so agents accept either type.
"""

from __future__ import annotations

import logging

from hifi.knowledge.graph_store import FinancialGraph
from hifi.knowledge.schemas import DocumentChunk
from hifi.knowledge.vector_store import KnowledgeStore

logger = logging.getLogger(__name__)

_PASSAGE_SEPARATOR = "\n---\n"


class GraphRetriever:
    """
    Graph-expanded retrieval: expand query via FinancialGraph, then dense ANN search.

    Retrieval steps
    ---------------
    1. ``graph.expand_query_tickers(ticker, max_hops=2)`` yields a set of
       related tickers (the target ticker itself + its 1-hop competitors +
       2-hop sector peers).
    2. ``store.search_tickers(query_embedding, expanded, top_k)`` returns the
       top_k chunks across that expanded ticker set, ranked by cosine similarity.

    Falls back gracefully when the ticker is not in the graph: the expanded set
    collapses to ``{ticker}`` so the call behaves like plain ANN search.

    Parameters
    ----------
    store : KnowledgeStore
        LanceDB-backed embedding store.
    embedding_model : object
        Any object implementing ``embed_one(text: str) -> list[float]``.
    graph : FinancialGraph
        Loaded financial entity graph (see graph_store.py).
    """

    def __init__(
        self,
        store: KnowledgeStore,
        embedding_model: object,
        graph: FinancialGraph,
    ) -> None:
        self._store = store
        self._model = embedding_model
        self._graph = graph

    def retrieve(
        self,
        query: str,
        ticker: str,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """
        Expand ticker to graph neighbors, then retrieve from the expanded set.

        Parameters
        ----------
        query : str
            Natural language query (e.g. "financial analysis AAPL").
        ticker : str
            Primary ticker to expand via the financial graph.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[DocumentChunk]
            Chunks from the expanded ticker set, ranked by cosine similarity to
            the query embedding. Returns [] on any failure (fail-open).
        """
        try:
            query_embedding = self._model.embed_one(query)
        except Exception as exc:
            logger.warning("GraphRetriever: embedding failed for %r: %s", query, exc)
            return []

        expanded = self._graph.expand_query_tickers(ticker, max_hops=2)
        logger.debug("GraphRetriever: %s expanded to %s", ticker, expanded)

        return self._store.search_tickers(
            query_embedding=query_embedding,
            tickers=expanded,
            top_k=top_k,
        )

    def format_context(self, chunks: list[DocumentChunk]) -> str:
        """
        Format retrieved chunks as a numbered passage block for prompt injection.

        Produces the same format as ``KnowledgeRetriever.format_context()`` so
        the two retrievers are drop-in substitutes for each other in the v2
        agent prompts.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Retrieved and ranked chunks.

        Returns
        -------
        str
            Multi-passage context string ready for injection into an LLM prompt.
            Returns "" if chunks is empty.
        """
        if not chunks:
            return ""

        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            header = (
                f"[{i}] {chunk.ticker} / {chunk.filing_type} / "
                f"{chunk.section} / {chunk.period.isoformat()}"
            )
            lines.append(header)
            lines.append(chunk.text)
            if i < len(chunks):
                lines.append(_PASSAGE_SEPARATOR)

        return "\n".join(lines)
