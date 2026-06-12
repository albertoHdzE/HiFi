"""
Knowledge system schemas for HiFi Phase 7 RAG (P7-E1).

Defines Pydantic models for:
- FilingDocument: a parsed SEC filing with extracted section text
- DocumentChunk: a text chunk ready for embedding and retrieval
- EvaluationQuery: a labelled query for Precision@k evaluation
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

from pydantic import BaseModel


class FilingDocument(BaseModel):
    """A parsed SEC filing with extracted section text."""

    ticker: str
    cik: str
    filing_type: str           # "10-K", "10-Q", "8-K"
    accession_number: str      # e.g., "0000320193-23-000006"
    period_of_report: date     # Q1 2023 = 2023-03-31
    filed_date: date
    sections: dict[str, str]   # section_name -> extracted_text
    source_url: str
    fetched_at: datetime


class DocumentChunk(BaseModel):
    """A text chunk from a FilingDocument, ready for embedding and retrieval."""

    chunk_id: str           # SHA-256[:16] of content fields
    ticker: str
    filing_type: str
    period: date
    section: str            # "MD&A", "Risk Factors", "Business", "Earnings Release"
    chunk_index: int
    text: str
    char_count: int         # actual character count
    approx_tokens: int      # ceil(char_count / 4)
    chunking_config: str    # "A", "B", or "C"

    @classmethod
    def make_chunk_id(
        cls,
        ticker: str,
        filing_type: str,
        period: date,
        section: str,
        chunk_index: int,
        config: str,
    ) -> str:
        """Compute the deterministic SHA-256[:16] chunk ID."""
        key = (
            f"{ticker}|{filing_type}|{period.isoformat()}"
            f"|{section}|{chunk_index}|{config}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]


class EvaluationQuery(BaseModel):
    """A labelled query for Precision@k evaluation of the retrieval system."""

    query_id: str
    query: str
    ticker: str
    relevant_section: str      # expected source section
    relevant_filing_type: str  # expected source filing type
    notes: str = ""
