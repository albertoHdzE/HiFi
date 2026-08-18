"""
EDGAR MD&A temporal retriever for Phase 15 walk-forward evaluation (Wave 2).

Queries the hifi-dev-sec-sec-mda (or hifi-eval-sec-mda) LanceDB table with
strict temporal discipline: only filings with period <= as_of_date are returned,
preventing look-ahead leakage in the walk-forward evaluation.

The EDGAR table schema (see ingest_edgar_mda.py):
  ticker      : str  — e.g. "AAPL"
  filing_type : str  — "10-K" or "10-Q"
  period      : str  — ISO 8601 fiscal period end (e.g. "2022-09-24")
  section     : str  — always "mda"
  chunk_index : int  — chunk position within the filing
  text        : str  — MD&A text chunk (~512 tokens)
  approx_tokens : int

No vector search is used — retrieval is by temporal filter + recency.
The most recent available filing (period <= as_of_date) is returned.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHUNKS = 4   # ~2048 tokens of MD&A context
_DEFAULT_TABLE = "sec-mda"


def _rank_chunks_by_query(texts: list[str], query: str, max_chunks: int) -> list[str]:
    """Rank MD&A chunks by term overlap with ``query``, most relevant first.

    Deliberately lexical (no embedding model): the knowledge server already owns
    the vector-search path, and this selector has to run inside the agent loop
    for 98 tickers a night without loading a second model.

    Its purpose is diversity, not precision. The fundamental agent takes the
    head of the MD&A; giving the sentiment agent the same bytes would make two
    nominally independent ensemble members correlated by construction, which
    would inflate agreement in an experiment whose dependent variable *is*
    agreement. Query-scored selection keeps their evidence bases distinct.
    Ties fall back to document order, so the result is deterministic.
    """
    terms = {t for t in query.lower().split() if len(t) > 3}
    if not terms:
        return texts[:max_chunks]
    scored = []
    for i, text in enumerate(texts):
        low = text.lower()
        score = sum(low.count(t) for t in terms)
        scored.append((-score, i, text))
    scored.sort()
    return [t for _, _, t in scored[:max_chunks]]


def retrieve_mda_context(
    ticker: str,
    as_of_date: str,
    namespace: str = "hifi-dev-sec",
    db_path: str | None = None,
    max_chunks: int = _DEFAULT_MAX_CHUNKS,
    query: str | None = None,
) -> str:
    """
    Retrieve MD&A text for ticker from the most recent filing on or before as_of_date.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. "AAPL").
    as_of_date : str
        ISO 8601 evaluation date.  Only filings with period <= as_of_date
        are considered (temporal discipline for walk-forward evaluation).
    namespace : str
        LanceDB table prefix. Dev: "hifi-dev-sec", Eval: "hifi-eval".
        Table name = "{namespace}-mda" → e.g. "hifi-dev-sec-sec-mda".
        Note: when namespace="hifi-eval", table = "hifi-eval-sec-mda".
    db_path : str | None
        LanceDB directory. Defaults to $HIFI_DATA_DIR/knowledge.lance.
    max_chunks : int
        Maximum number of ~512-token chunks to include (default 4 ≈ 2048 tokens).
    query : str | None
        When given, select the chunks with the highest lexical overlap with the
        query instead of the head of the MD&A. Callers sharing a filing should
        pass distinct queries so they do not receive identical context.

    Returns
    -------
    str
        Formatted MD&A context block, or "" when no filing is available.
    """
    _db_path = db_path or str(
        Path(os.environ.get("HIFI_DATA_DIR", "data")) / "knowledge.lance"
    )
    table_name = f"{namespace}-sec-mda"

    try:
        import lancedb

        db = lancedb.connect(_db_path)
        all_tables = db.list_tables()
        all_names = list(all_tables.tables) if hasattr(all_tables, "tables") else list(all_tables)
        if table_name not in all_names:
            logger.debug("EDGAR table %s not found in %s", table_name, _db_path)
            return ""

        table = db.open_table(table_name)
        df = table.to_pandas()

        # Temporal filter: only filings available on the evaluation date
        mask = (df["ticker"] == ticker) & (df["period"] <= as_of_date)
        filtered = df[mask]
        if filtered.empty:
            logger.debug("No EDGAR MD&A for %s as_of %s in %s", ticker, as_of_date, table_name)
            return ""

        # Select most recent filing period
        most_recent_period = filtered["period"].max()
        filing_df = filtered[filtered["period"] == most_recent_period].sort_values("chunk_index")
        filing_type = filing_df["filing_type"].iloc[0] if len(filing_df) > 0 else "10-K"

        # Head of the MD&A by default; query-scored selection when a caller
        # needs evidence distinct from another agent's (see _rank_chunks_by_query).
        if query:
            chunks = _rank_chunks_by_query(filing_df["text"].tolist(), query, max_chunks)
        else:
            chunks = filing_df["text"].head(max_chunks).tolist()
        if not chunks:
            return ""

        header = (
            f"[EDGAR MD&A — {ticker} {filing_type} period={most_recent_period}"
            f" as_of={as_of_date}]"
        )
        body = "\n\n".join(chunks)
        return f"{header}\n{body}"

    except Exception as exc:
        logger.warning(
            "EDGAR retrieval failed for %s as_of %s namespace=%s: %s",
            ticker, as_of_date, namespace, exc,
        )
        return ""
