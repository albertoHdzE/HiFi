"""
EDGAR MD&A targeted section extraction (E2-T3, DJ-090).

Extracts Item 7 (MD&A, 10-K) and Item 2 (MD&A, 10-Q) from SEC EDGAR filings.
Strips HTML, splits into ~512-token chunks, and ingests into LanceDB namespace
``hifi-dev-sec`` (or the namespace supplied by the caller).

Key design:
  - Uses SEC EDGAR full-text search API (efts.sec.gov) for CIK lookup.
  - Downloads filing index from EDGAR; identifies target sections by heading.
  - Strips HTML with a lightweight regex approach (no BeautifulSoup dependency).
  - Does NOT use full-filing downloads — only the target section text.

Why this matters (from DJ-087):
  Phase 13 retrieved legal boilerplate from 8-K cover pages because the corpus
  lacked section discrimination.  Targeted MD&A extraction gives the Sentiment
  agent quotable earnings commentary rather than XBRL certifications.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Approximate characters per token (used for chunk sizing)
_CHARS_PER_TOKEN = 4
_TARGET_TOKENS = 512
_CHUNK_SIZE = _TARGET_TOKENS * _CHARS_PER_TOKEN   # ~2048 chars


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


def strip_html(text: str) -> str:
    """Remove HTML tags and normalise whitespace."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Section boundary detection
# ---------------------------------------------------------------------------

_ITEM7_PATTERNS = [
    re.compile(r"item\s+7[^a-z0-9].*?management", re.IGNORECASE),
    re.compile(r"management.{0,30}discussion.{0,30}analysis", re.IGNORECASE),
]

_ITEM2_PATTERNS = [
    re.compile(r"item\s+2[^a-z0-9].*?management", re.IGNORECASE),
]

_NEXT_ITEM_PATTERN = re.compile(r"item\s+[89]\b", re.IGNORECASE)


def extract_mda_section(html_text: str, filing_type: str) -> str:
    """
    Extract the MD&A section text from an HTML filing.

    Parameters
    ----------
    html_text : str
        Raw HTML content of the filing.
    filing_type : str
        "10-K" → look for Item 7; "10-Q" → look for Item 2.

    Returns
    -------
    str
        Extracted plain-text MD&A content, or "" if not found.
    """
    plain = strip_html(html_text)

    patterns = _ITEM7_PATTERNS if "10-K" in filing_type.upper() else _ITEM2_PATTERNS

    start_pos = -1
    for pat in patterns:
        m = pat.search(plain)
        if m:
            start_pos = m.start()
            break

    if start_pos == -1:
        logger.warning("MD&A section not found in %s filing", filing_type)
        return ""

    # Find end of section (next major item heading)
    end_match = _NEXT_ITEM_PATTERN.search(plain, start_pos + 100)
    end_pos = end_match.start() if end_match else len(plain)

    section = plain[start_pos:end_pos].strip()
    return section


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    ticker: str,
    filing_type: str,
    period: str,
    chunk_size: int = _CHUNK_SIZE,
) -> list[dict]:
    """
    Split section text into ~512-token chunks.

    Parameters
    ----------
    text : str
        Plain-text section content.
    ticker : str
        Ticker symbol.
    filing_type : str
        "10-K" or "10-Q".
    period : str
        Period-of-report ISO date (e.g. "2022-12-31").
    chunk_size : int
        Target chunk size in characters.

    Returns
    -------
    list[dict]
        Each dict: {ticker, filing_type, period, section, chunk_index,
                    text, approx_tokens}.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    current: list[str] = []
    current_len = 0
    chunk_idx = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunk_text_str = " ".join(current)
            chunks.append({
                "ticker": ticker,
                "filing_type": filing_type,
                "period": period,
                "section": "mda",
                "chunk_index": chunk_idx,
                "text": chunk_text_str,
                "approx_tokens": len(chunk_text_str) // _CHARS_PER_TOKEN,
            })
            current = []
            current_len = 0
            chunk_idx += 1

    if current:
        chunk_text_str = " ".join(current)
        chunks.append({
            "ticker": ticker,
            "filing_type": filing_type,
            "period": period,
            "section": "mda",
            "chunk_index": chunk_idx,
            "text": chunk_text_str,
            "approx_tokens": len(chunk_text_str) // _CHARS_PER_TOKEN,
        })

    return chunks
