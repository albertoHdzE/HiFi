"""
Minimal FundamentalsSnapshot builder for Phase 15 walk-forward evaluation.

In the walk-forward, fundamental insight comes primarily from EDGAR MD&A context
retrieved from the hifi-eval namespace (populated by make eval-ingest-through DATE=D).
The FundamentalsSnapshot passed to run_sequential_ensemble() is therefore minimal:
all financial fields are None.  The Fundamental Agent's LLM fills in parametric
knowledge; the RAG context provides the grounded MD&A excerpt.

This approach avoids the need to pre-fetch point-in-time financial statements
for 98 tickers × 264 monthly dates (25,872 snapshots), which would require a
separate financial data API beyond what Phase 1-2 acquired.
"""

from __future__ import annotations

from datetime import UTC, datetime


def build_minimal_snapshot(ticker: str, as_of_date: str) -> str:
    """
    Return a JSON-serialized FundamentalsSnapshot with None financial fields.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. "AAPL").
    as_of_date : str
        ISO 8601 evaluation date used as period_end (e.g. "2022-01-31").

    Returns
    -------
    str
        JSON string ready to pass as snapshot_json to run_sequential_ensemble().
    """
    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

    fetched_at = datetime.now(UTC)
    snap = FundamentalsSnapshot(
        ticker=ticker,
        period_end=as_of_date,
        revenue=None,
        net_income=None,
        total_assets=None,
        total_liabilities=None,
        total_equity=None,
        eps=None,
        pe_ratio=None,
        market_cap=None,
        source="walk_forward_eval",
        fetched_at=fetched_at,
        provenance=ProvenanceRecord(
            source="walk_forward_eval",
            fetched_at=fetched_at,
            parameters={"ticker": ticker, "as_of_date": as_of_date},
        ),
    )
    return snap.model_dump_json()
