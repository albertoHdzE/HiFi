"""
HiFi Knowledge Retrieval MCP Server (P7-E6).

Exposes a single MCP tool: retrieve_context. Agents call this tool via the
same subprocess MCP client (mcp_client.py) as the financial server. The server
loads the LanceDB knowledge store and embedding model at startup, then serves
retrieval requests until the subprocess exits.

Configuration (environment variables):
  HIFI_KNOWLEDGE_DATA_DIR      Path to the knowledge database directory.
                                Default: "data/knowledge" relative to CWD.
  HIFI_KNOWLEDGE_CHUNKING_CONFIG  Winning chunking config from DJ-030.
                                Default: "A"

Fail-open contract: if the store is not populated or LM Studio is unavailable,
retrieve_context returns empty passages rather than crashing the agent pipeline.

Tool response schema:
  {
    "call_id":    "<12-char SHA-256>",
    "ticker":     "<ticker>",
    "query":      "<query>",
    "passages":   [{"rank": 1, "filing_type": ..., "section": ...,
                    "period": ..., "text": ...}, ...],
    "n_retrieved": <int>
  }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("hifi-knowledge-retrieval")


# ---------------------------------------------------------------------------
# Module-level lazy singletons (initialised on first retrieve_context call)
# ---------------------------------------------------------------------------

_store: Any = None       # KnowledgeStore | None
_retriever: Any = None   # KnowledgeRetriever | None


def _data_dir() -> Path:
    return Path(os.environ.get("HIFI_KNOWLEDGE_DATA_DIR", "data/knowledge"))


def _chunking_config() -> str:
    return os.environ.get("HIFI_KNOWLEDGE_CHUNKING_CONFIG", "A").upper()


def _call_id(**kwargs: Any) -> str:
    """12-char SHA-256 of sorted JSON params (Phase 2 audit trail pattern)."""
    payload = json.dumps(kwargs, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _get_retriever() -> Any:
    """Return the module-level KnowledgeRetriever, initialising it if needed."""
    global _store, _retriever

    if _retriever is not None:
        return _retriever

    try:
        from hifi.knowledge.embeddings import EmbeddingModel
        from hifi.knowledge.retrieval import KnowledgeRetriever
        from hifi.knowledge.vector_store import KnowledgeStore

        store = KnowledgeStore(
            data_dir=_data_dir(),
            chunking_config=_chunking_config(),
        )
        model = EmbeddingModel()
        _store = store
        _retriever = KnowledgeRetriever(store=store, embedding_model=model)
        logger.info(
            "Knowledge server initialised: data_dir=%s config=%s stats=%s",
            _data_dir(),
            _chunking_config(),
            store.get_stats(),
        )
    except Exception as exc:
        logger.warning(
            "Failed to initialise knowledge store; retrieve_context will "
            "return empty results. Error: %s",
            exc,
        )
        _retriever = None

    return _retriever


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------


@mcp.tool()
def retrieve_context(
    query: str,
    ticker: str,
    top_k: int = 5,
) -> dict:
    """
    Retrieve relevant passages from SEC filings for a given query and ticker.

    Parameters
    ----------
    query : str
        Natural language question or context description.
    ticker : str
        Company ticker symbol (e.g. "AAPL").
    top_k : int
        Number of passages to retrieve (default 5).

    Returns
    -------
    dict
        call_id : str    -- 12-char audit trail identifier
        ticker : str     -- echo of input ticker
        query : str      -- echo of input query
        passages : list  -- ranked passages with source metadata
        n_retrieved : int
    """
    cid = _call_id(tool="retrieve_context", ticker=ticker, query=query[:64])

    retriever = _get_retriever()
    if retriever is None:
        return {
            "call_id": cid,
            "ticker": ticker,
            "query": query,
            "passages": [],
            "n_retrieved": 0,
        }

    try:
        chunks = retriever.retrieve(query=query, ticker=ticker, top_k=top_k)
    except Exception as exc:
        logger.warning("retrieve_context failed for %s: %s", ticker, exc)
        return {
            "call_id": cid,
            "ticker": ticker,
            "query": query,
            "passages": [],
            "n_retrieved": 0,
        }

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

    return {
        "call_id": cid,
        "ticker": ticker,
        "query": query,
        "passages": passages,
        "n_retrieved": len(passages),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
