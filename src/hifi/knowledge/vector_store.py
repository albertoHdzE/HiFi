"""
LanceDB vector store for HiFi Phase 7 RAG (P7-E4).

Stores DocumentChunk embeddings in a LanceDB database. One Lance table per
chunking configuration (chunks_a, chunks_b, chunks_c) within the same database
directory. The Arrow-native format mirrors the Parquet storage philosophy (DJ-026).

After DJ-030 selects the winning chunking config, the production knowledge
server uses only that config's table.
"""

from __future__ import annotations

import logging
from datetime import date
from math import ceil
from pathlib import Path
from typing import Any

import pyarrow as pa

from hifi.knowledge.namespaced_store import NamespacedLanceDB
from hifi.knowledge.schemas import DocumentChunk

logger = logging.getLogger(__name__)

_VALID_CONFIGS = frozenset(["A", "B", "C"])


def _make_schema(dimensions: int) -> pa.Schema:
    """Build the PyArrow schema for a chunk embedding table."""
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("ticker", pa.string()),
            pa.field("filing_type", pa.string()),
            pa.field("period", pa.string()),      # ISO date string "2023-03-31"
            pa.field("section", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("approx_tokens", pa.int32()),
            pa.field("chunking_config", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), dimensions)),
        ]
    )


class KnowledgeStore:
    """
    LanceDB-backed embedding store for document chunks.

    One Lance table per chunking config. All three config tables (A, B, C)
    can coexist within the same LanceDB database directory.

    Parameters
    ----------
    data_dir : Path
        Directory for the LanceDB database (``knowledge.lance/`` is created here).
    chunking_config : str
        Which config table to open ("A", "B", or "C").
    dimensions : int
        Embedding vector dimensionality (must match the model used to produce
        embeddings). Default 768 (nomic-embed-text-v1.5 full Matryoshka).
    """

    def __init__(
        self,
        data_dir: Path,
        chunking_config: str = "A",
        dimensions: int = 768,
        namespace: str = "",
    ) -> None:
        config = chunking_config.upper()
        if config not in _VALID_CONFIGS:
            raise ValueError(
                f"chunking_config must be one of {sorted(_VALID_CONFIGS)}; "
                f"got {chunking_config!r}"
            )
        self._config = config
        self._dimensions = dimensions
        self._schema = _make_schema(dimensions)
        self._table_name = f"chunks_{config.lower()}"

        db_path = Path(data_dir) / "knowledge.lance"
        db_path.mkdir(parents=True, exist_ok=True)

        self._ns = NamespacedLanceDB(str(db_path), namespace)
        self._table = self._open_or_create_table()

    def _open_or_create_table(self) -> Any:
        """Open the table if it exists, otherwise create it."""
        return self._ns.open_or_create_table(self._table_name, self._schema)

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Insert chunks and their embeddings into the Lance table.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Chunk metadata objects.
        embeddings : list[list[float]]
            One embedding vector per chunk (same length as chunks).

        Returns
        -------
        int
            Number of rows inserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must have the same length"
            )
        if not chunks:
            return 0

        rows = {
            "chunk_id": [c.chunk_id for c in chunks],
            "ticker": [c.ticker for c in chunks],
            "filing_type": [c.filing_type for c in chunks],
            "period": [c.period.isoformat() for c in chunks],
            "section": [c.section for c in chunks],
            "chunk_index": [c.chunk_index for c in chunks],
            "text": [c.text for c in chunks],
            "approx_tokens": [c.approx_tokens for c in chunks],
            "chunking_config": [c.chunking_config for c in chunks],
            "embedding": [
                pa.array(emb, type=pa.float32()) for emb in embeddings
            ],
        }
        batch = pa.table(rows, schema=self._schema)
        self._table.add(batch)
        logger.debug("Indexed %d chunks into table %s", len(chunks), self._table_name)
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        ticker: str,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """
        Cosine ANN search filtered by ticker. Returns up to top_k chunks.

        Parameters
        ----------
        query_embedding : list[float]
            Query vector (must have the same dimensionality as stored embeddings).
        ticker : str
            Ticker symbol to filter on (e.g. "AAPL").
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[DocumentChunk]
            Most similar chunks for the queried ticker, ranked by cosine similarity.
        """
        stats = self.get_stats()
        if stats["n_chunks"] == 0:
            return []

        escaped = ticker.replace('"', '\\"')
        where_clause = f'ticker = "{escaped}"'

        try:
            results = (
                self._table.search(query_embedding, vector_column_name="embedding")
                .metric("cosine")
                .where(where_clause, prefilter=True)
                .limit(top_k)
                .to_list()
            )
        except Exception as exc:
            logger.warning("LanceDB search failed: %s", exc)
            return []

        chunks: list[DocumentChunk] = []
        for row in results:
            try:
                text = row["text"]
                char_count = len(text)
                chunk = DocumentChunk(
                    chunk_id=row["chunk_id"],
                    ticker=row["ticker"],
                    filing_type=row["filing_type"],
                    period=date.fromisoformat(row["period"]),
                    section=row["section"],
                    chunk_index=int(row["chunk_index"]),
                    text=text,
                    char_count=char_count,
                    approx_tokens=ceil(char_count / 4),
                    chunking_config=row["chunking_config"],
                )
                chunks.append(chunk)
            except Exception as exc:
                logger.warning("Failed to reconstruct DocumentChunk: %s", exc)
        return chunks

    def search_tickers(
        self,
        query_embedding: list[float],
        tickers: list[str],
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """
        Cosine ANN search across multiple tickers. Returns up to top_k chunks merged.

        Used by GraphRetriever to search the expanded ticker neighborhood from
        FinancialGraph.expand_query_tickers() (P12-E1-T3, DJ-068).

        Parameters
        ----------
        query_embedding : list[float]
            Query vector (same dimensionality as stored embeddings).
        tickers : list[str]
            Ticker symbols to include in search.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[DocumentChunk]
            Most similar chunks across all provided tickers, ranked by similarity.
        """
        if not tickers:
            return []
        stats = self.get_stats()
        if stats["n_chunks"] == 0:
            return []

        escaped = [t.replace('"', '\\"') for t in tickers]
        ticker_list = ", ".join(f'"{t}"' for t in escaped)
        where_clause = f"ticker IN ({ticker_list})"

        try:
            results = (
                self._table.search(query_embedding, vector_column_name="embedding")
                .metric("cosine")
                .where(where_clause, prefilter=True)
                .limit(top_k)
                .to_list()
            )
        except Exception as exc:
            logger.warning("LanceDB search_tickers failed: %s", exc)
            return []

        chunks: list[DocumentChunk] = []
        for row in results:
            try:
                text = row["text"]
                char_count = len(text)
                chunk = DocumentChunk(
                    chunk_id=row["chunk_id"],
                    ticker=row["ticker"],
                    filing_type=row["filing_type"],
                    period=date.fromisoformat(row["period"]),
                    section=row["section"],
                    chunk_index=int(row["chunk_index"]),
                    text=text,
                    char_count=char_count,
                    approx_tokens=ceil(char_count / 4),
                    chunking_config=row["chunking_config"],
                )
                chunks.append(chunk)
            except Exception as exc:
                logger.warning("Failed to reconstruct DocumentChunk: %s", exc)
        return chunks

    def get_stats(self) -> dict[str, int]:
        """
        Return summary statistics for the current table.

        Returns
        -------
        dict with keys: n_chunks, n_tickers, n_filing_types
        """
        try:
            df = self._table.to_pandas()
            if df.empty:
                return {"n_chunks": 0, "n_tickers": 0, "n_filing_types": 0}
            return {
                "n_chunks": len(df),
                "n_tickers": df["ticker"].nunique(),
                "n_filing_types": df["filing_type"].nunique(),
            }
        except Exception:
            return {"n_chunks": 0, "n_tickers": 0, "n_filing_types": 0}

    def clear(self) -> None:
        """Drop and recreate the table. Used in tests and experiment resets."""
        import contextlib

        with contextlib.suppress(Exception):
            self._ns.drop_table(self._table_name)
        self._table = self._open_or_create_table()
        logger.debug("Cleared table %s", self._table_name)
