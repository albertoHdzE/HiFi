"""
Document ingestion and chunking for HiFi Phase 7 RAG (P7-E2).

Converts FilingDocument objects into DocumentChunk lists using one of three
chunking configurations:

  Config A — Fixed-size 512 tokens (~2000 chars), 10% overlap (200 chars)
  Config B — Fixed-size 1024 tokens (~4000 chars), 20% overlap (800 chars)
  Config C — Semantic paragraph-based (split on blank lines + section headers)

All configurations produce DocumentChunk objects with deterministic chunk_ids,
correct char_count and approx_tokens, and no empty chunks.
"""

from __future__ import annotations

import re
from datetime import date
from math import ceil

from hifi.knowledge.schemas import DocumentChunk, FilingDocument

# Sentence boundary characters used for window snapping
_SENTENCE_ENDS = frozenset(".!?\n")

# Config A: window ~2000 chars, overlap ~200 chars
_CONFIG_A_WINDOW = 2000
_CONFIG_A_OVERLAP = 200

# Config B: window ~4000 chars, overlap ~800 chars
_CONFIG_B_WINDOW = 4000
_CONFIG_B_OVERLAP = 800

# Config C: max chars per semantic chunk before sub-splitting
_CONFIG_C_MAX_CHARS = 6000

# Sentence boundary search radius for window snapping
_BOUNDARY_RADIUS = 50

# Regex for semantic chunk dividers (all-caps lines, or lines starting with Item/Part/Note)
_SEMANTIC_DIVIDER_RE = re.compile(
    r"(?:^|\n)(?:[A-Z][A-Z\s\-/]{4,}|(?:Item|Part|Note|Section)\s+\S.*?)(?:\n|$)"
)


# ---------------------------------------------------------------------------
# Private chunking functions
# ---------------------------------------------------------------------------


def _find_sentence_boundary(text: str, pos: int, radius: int) -> int:
    """
    Snap position to the nearest sentence boundary within ±radius characters.

    Searches backward from pos for a sentence-ending character. If none found
    within radius, tries forward. Falls back to pos if nothing found.
    """
    # Search backward
    for i in range(pos, max(0, pos - radius) - 1, -1):
        if text[i] in _SENTENCE_ENDS:
            return i + 1  # position after the boundary
    # Search forward
    for i in range(pos, min(len(text), pos + radius)):
        if text[i] in _SENTENCE_ENDS:
            return i + 1
    return pos


def _fixed_size_chunker(
    text: str,
    window: int,
    overlap: int,
) -> list[str]:
    """
    Split text into fixed-size windows with overlap.

    Window and overlap are in characters (~4 chars per token).
    Each window snaps to the nearest sentence boundary within ±50 chars.
    Returns a list of non-empty text chunks.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + window, text_len)

        # Snap to sentence boundary (unless we've reached the end)
        if end < text_len:
            end = _find_sentence_boundary(text, end, _BOUNDARY_RADIUS)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Next window starts at (end - overlap), snapped to sentence boundary
        next_start = end - overlap
        if next_start <= start:
            next_start = end  # prevent infinite loop on very short text
        start = next_start

    return chunks


def _semantic_chunker(text: str) -> list[str]:
    """
    Split text on paragraph boundaries and section dividers.

    Paragraphs are delineated by double newlines or section-header lines.
    Paragraphs longer than _CONFIG_C_MAX_CHARS are sub-split at sentence
    boundaries to keep each chunk manageable.

    Returns a list of non-empty text chunks.
    """
    # Split on double newlines (paragraph boundaries)
    raw_paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= _CONFIG_C_MAX_CHARS:
            chunks.append(para)
        else:
            # Sub-split long paragraphs at sentence boundaries
            sub_chunks = _split_at_sentences(para, _CONFIG_C_MAX_CHARS)
            chunks.extend(sub_chunks)

    return chunks


def _split_at_sentences(text: str, max_chars: int) -> list[str]:
    """
    Split text at sentence boundaries so each piece <= max_chars.

    Sentences are separated by '. ', '! ', '? ', or '\n'.
    """
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if current_len + len(s) + 1 > max_chars and current:
            pieces.append(" ".join(current))
            current = [s]
            current_len = len(s)
        else:
            current.append(s)
            current_len += len(s) + 1

    if current:
        pieces.append(" ".join(current))

    return pieces


# ---------------------------------------------------------------------------
# DocumentIngestionPipeline
# ---------------------------------------------------------------------------


class DocumentIngestionPipeline:
    """
    Convert FilingDocuments into DocumentChunk lists.

    Supports three chunking configurations (A, B, C). The configuration
    determines chunk size and overlap strategy. All configurations produce
    DocumentChunk objects with deterministic chunk_ids.
    """

    def __init__(self, chunking_config: str) -> None:
        """
        Parameters
        ----------
        chunking_config : str
            Must be "A", "B", or "C" (case-insensitive).
        """
        config = chunking_config.upper()
        if config not in ("A", "B", "C"):
            raise ValueError(
                f"chunking_config must be 'A', 'B', or 'C'; got {chunking_config!r}"
            )
        self._config = config

    def chunk_section(
        self,
        text: str,
        ticker: str,
        filing_type: str,
        period: date,
        section: str,
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """
        Chunk a single section of text.

        Parameters
        ----------
        text : str
            Section text to chunk.
        ticker : str
            Ticker symbol (e.g. "AAPL").
        filing_type : str
            Filing type (e.g. "10-K").
        period : date
            Filing period.
        section : str
            Section name (e.g. "MD&A").
        start_index : int
            Chunk index offset (for when multiple sections are chunked together).

        Returns
        -------
        list[DocumentChunk]
            Non-empty chunks with deterministic IDs.
        """
        if self._config == "A":
            raw_chunks = _fixed_size_chunker(text, _CONFIG_A_WINDOW, _CONFIG_A_OVERLAP)
        elif self._config == "B":
            raw_chunks = _fixed_size_chunker(text, _CONFIG_B_WINDOW, _CONFIG_B_OVERLAP)
        else:  # C
            raw_chunks = _semantic_chunker(text)

        result: list[DocumentChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            global_idx = start_index + idx
            chunk_id = DocumentChunk.make_chunk_id(
                ticker, filing_type, period, section, global_idx, self._config
            )
            char_count = len(chunk_text)
            result.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    ticker=ticker,
                    filing_type=filing_type,
                    period=period,
                    section=section,
                    chunk_index=global_idx,
                    text=chunk_text,
                    char_count=char_count,
                    approx_tokens=ceil(char_count / 4),
                    chunking_config=self._config,
                )
            )
        return result

    def chunk_document(self, doc: FilingDocument) -> list[DocumentChunk]:
        """
        Chunk all sections of a FilingDocument.

        Each section is chunked independently. Chunk indices within a section
        start from 0. Returns the concatenated chunk list for the full document.
        """
        all_chunks: list[DocumentChunk] = []
        for section_name, section_text in doc.sections.items():
            chunks = self.chunk_section(
                text=section_text,
                ticker=doc.ticker,
                filing_type=doc.filing_type,
                period=doc.period_of_report,
                section=section_name,
            )
            all_chunks.extend(chunks)
        return all_chunks
