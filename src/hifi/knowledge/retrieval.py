"""
Retrieval pipeline for HiFi Phase 7 RAG (P7-E5).

KnowledgeRetriever composes KnowledgeStore (ANN search) and EmbeddingModel
(query embedding) to provide end-to-end retrieval: string query -> ranked chunks.

evaluate_precision_at_k() measures retrieval quality against the 20-question
financial evaluation set defined in tests/fixtures/retrieval/evaluation_queries.json.
A chunk is relevant if its (ticker, section, filing_type) matches the query label.
"""

from __future__ import annotations

import logging

from hifi.knowledge.schemas import DocumentChunk, EvaluationQuery
from hifi.knowledge.vector_store import KnowledgeStore

logger = logging.getLogger(__name__)

# Separator used between retrieved passages in formatted context blocks
_PASSAGE_SEPARATOR = "\n---\n"


class KnowledgeRetriever:
    """
    End-to-end retrieval: embed query -> search store -> format context.

    Composes a KnowledgeStore (vector search) and an embedding model
    (any object with embed_one(str) -> list[float] interface).
    """

    def __init__(self, store: KnowledgeStore, embedding_model: object) -> None:
        """
        Parameters
        ----------
        store : KnowledgeStore
            LanceDB-backed vector store.
        embedding_model : object
            Any object implementing embed_one(text: str) -> list[float].
            Can be EmbeddingModel (live) or DeterministicEmbeddingModel (tests).
        """
        self._store = store
        self._model = embedding_model

    def retrieve(
        self,
        query: str,
        ticker: str,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """
        Return the top_k most relevant chunks for a (query, ticker) pair.

        Parameters
        ----------
        query : str
            Natural language query.
        ticker : str
            Ticker symbol to restrict search (e.g. "AAPL").
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[DocumentChunk]
            Chunks ranked by cosine similarity to the query embedding.
        """
        try:
            query_embedding = self._model.embed_one(query)
        except Exception as exc:
            logger.warning("Embedding query failed: %s", exc)
            return []
        return self._store.search(query_embedding, ticker=ticker, top_k=top_k)

    def format_context(self, chunks: list[DocumentChunk]) -> str:
        """
        Format retrieved chunks as a numbered passage block for prompt injection.

        Each passage includes source metadata (ticker, filing_type, section,
        period) followed by the extracted text.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Retrieved and ranked chunks.

        Returns
        -------
        str
            A multi-passage context string ready for injection into an LLM prompt.
            Returns an empty string if chunks is empty.
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


def evaluate_precision_at_k(
    retriever: KnowledgeRetriever,
    queries: list[EvaluationQuery],
    k: int = 5,
) -> float:
    """
    Compute mean Precision@k over a labelled evaluation query set.

    A retrieved chunk is considered relevant if:
      chunk.ticker == query.ticker
      AND chunk.section == query.relevant_section
      AND chunk.filing_type == query.relevant_filing_type

    Precision@k for a single query = (# relevant in top-k) / k.
    Mean Precision@k = average over all queries.

    Parameters
    ----------
    retriever : KnowledgeRetriever
        The retriever to evaluate.
    queries : list[EvaluationQuery]
        Labelled queries with ground-truth source metadata.
    k : int
        Number of results to retrieve per query.

    Returns
    -------
    float
        Mean Precision@k in [0.0, 1.0]. Returns 0.0 if queries is empty.
    """
    if not queries:
        return 0.0

    precision_scores: list[float] = []
    for query in queries:
        chunks = retriever.retrieve(query.query, ticker=query.ticker, top_k=k)
        if not chunks:
            precision_scores.append(0.0)
            continue
        relevant = sum(
            1
            for c in chunks
            if (
                c.ticker == query.ticker
                and c.section == query.relevant_section
                and c.filing_type == query.relevant_filing_type
            )
        )
        precision_scores.append(relevant / k)

    return sum(precision_scores) / len(precision_scores)
