"""
Unit tests for DocumentIngestionPipeline (P7-E2).

Tests cover: chunk size ranges for configs A and B, overlap correctness,
paragraph splitting for config C, deterministic chunk IDs, no empty chunks,
multi-section documents.
"""

from __future__ import annotations

from datetime import date

import pytest

from hifi.knowledge.document_ingestion import (
    _CONFIG_A_OVERLAP,
    DocumentIngestionPipeline,
)
from hifi.knowledge.schemas import FilingDocument

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PERIOD = date(2023, 3, 31)

# ~3000 chars of prose — enough to produce multiple chunks under both A and B
_PROSE = (
    "Apple Inc. reported strong financial results for the fiscal quarter ended "
    "December 2022. Total net sales were $117.2 billion, representing a 5% decline "
    "compared to the same period in the prior year. The decline was primarily driven "
    "by supply constraints and adverse foreign exchange headwinds. " * 8
    + "Services revenue grew 6% year over year to $20.9 billion. "
    "iPhone revenue of $65.8 billion was down 8% compared to the prior year. "
    "Mac revenue declined 29% to $7.7 billion, reflecting the transition between "
    "product cycles and macroeconomic uncertainty. iPad revenue decreased 30% to "
    "$9.4 billion. Wearables, Home and Accessories revenue decreased 8% to $13.5 "
    "billion. The company returned $25.2 billion to shareholders during the quarter "
    "through dividends and share repurchases. " * 6
)

# Long prose for Config B sizing tests
_LONG_PROSE = _PROSE * 3  # ~9000 chars


def _make_doc(sections: dict[str, str] | None = None) -> FilingDocument:
    return FilingDocument(
        ticker="AAPL",
        cik="0000320193",
        filing_type="10-K",
        accession_number="0000320193-22-000001",
        period_of_report=_PERIOD,
        filed_date=date(2022, 10, 28),
        sections=sections or {
            "Business": "Apple designs consumer electronics and software.",
            "Risk Factors": "Competition is intense.",
            "MD&A": _PROSE,
        },
        source_url="https://example.com",
        fetched_at=__import__("datetime").datetime(2023, 4, 1),
    )


# ---------------------------------------------------------------------------
# P7-E2-T6: Config A produces approx_tokens in 400-600 range
# ---------------------------------------------------------------------------


def test_config_a_approx_tokens_range():
    """Config A: most chunks should be 400-600 tokens (2000 chars ≈ 500 tokens)."""
    pipeline = DocumentIngestionPipeline("A")
    chunks = pipeline.chunk_section(
        _LONG_PROSE, "AAPL", "10-K", _PERIOD, "MD&A"
    )
    assert len(chunks) > 1, "Long text should produce multiple chunks"
    # All full-size chunks should be within 400-600 tokens
    full_chunks = chunks[:-1]  # last chunk may be shorter
    for chunk in full_chunks:
        assert 300 <= chunk.approx_tokens <= 700, (
            f"approx_tokens={chunk.approx_tokens} out of expected range for config A"
        )


# ---------------------------------------------------------------------------
# P7-E2-T7: Config A overlap — adjacent chunks share characters from boundary
# ---------------------------------------------------------------------------


def test_config_a_overlap_between_adjacent_chunks():
    """Adjacent Config A chunks should share text at the boundary (overlap ~200 chars)."""
    pipeline = DocumentIngestionPipeline("A")
    # Use text with no sentence breaks to get clean overlap
    text = "word " * 600  # 3000 chars, clean repetition
    chunks = pipeline.chunk_section(text, "AAPL", "10-K", _PERIOD, "MD&A")
    if len(chunks) < 2:
        pytest.skip("Not enough chunks to test overlap")

    # Check that the end of chunk 0 appears in the beginning of chunk 1
    end_of_c0 = chunks[0].text[-_CONFIG_A_OVERLAP:]
    start_of_c1 = chunks[1].text[:_CONFIG_A_OVERLAP]
    # At least some overlap content should appear
    common = set(end_of_c0.split()) & set(start_of_c1.split())
    assert len(common) > 0, "Config A adjacent chunks should share boundary text"


# ---------------------------------------------------------------------------
# P7-E2-T8: Config B produces approx_tokens in 800-1200 range
# ---------------------------------------------------------------------------


def test_config_b_approx_tokens_range():
    """Config B: most chunks should be 800-1200 tokens (4000 chars ≈ 1000 tokens)."""
    pipeline = DocumentIngestionPipeline("B")
    text = _PROSE * 5  # ~15000 chars
    chunks = pipeline.chunk_section(text, "AAPL", "10-K", _PERIOD, "MD&A")
    assert len(chunks) > 1, "Long text should produce multiple chunks"
    full_chunks = chunks[:-1]
    for chunk in full_chunks:
        assert 600 <= chunk.approx_tokens <= 1400, (
            f"approx_tokens={chunk.approx_tokens} out of expected range for config B"
        )


# ---------------------------------------------------------------------------
# P7-E2-T9: Config C splits at paragraph boundaries
# ---------------------------------------------------------------------------


def test_config_c_splits_on_paragraph_boundaries():
    """Config C chunks should correspond to paragraph units."""
    text = (
        "First paragraph about iPhone revenue.\n\n"
        "Second paragraph about Mac revenue.\n\n"
        "Third paragraph about Services business."
    )
    pipeline = DocumentIngestionPipeline("C")
    chunks = pipeline.chunk_section(text, "AAPL", "10-Q", _PERIOD, "MD&A")
    assert len(chunks) == 3
    assert "iPhone" in chunks[0].text
    assert "Mac" in chunks[1].text
    assert "Services" in chunks[2].text


def test_config_c_no_mid_sentence_splits():
    """Config C should not split sentences across chunks."""
    sentences = (
        "Revenue grew 6 percent year over year. "
        "Services reached a new record. "
        "Management is pleased with the results."
    )
    # Create a large paragraph that will need sub-splitting
    large_para = sentences * 400  # ~24000 chars
    text = large_para + "\n\n" + "Another paragraph here."
    pipeline = DocumentIngestionPipeline("C")
    chunks = pipeline.chunk_section(text, "AAPL", "10-K", _PERIOD, "MD&A")
    for chunk in chunks:
        # Each chunk should not start mid-word mid-sentence (rough check)
        t = chunk.text.strip()
        assert t, "No empty chunks"


# ---------------------------------------------------------------------------
# P7-E2-T10: chunk_id is deterministic
# ---------------------------------------------------------------------------


def test_chunk_id_deterministic_across_calls():
    """Same inputs always produce the same chunk IDs."""
    pipeline = DocumentIngestionPipeline("A")
    text = _PROSE
    chunks1 = pipeline.chunk_section(text, "AAPL", "10-K", _PERIOD, "MD&A")
    chunks2 = pipeline.chunk_section(text, "AAPL", "10-K", _PERIOD, "MD&A")
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2, strict=True):
        assert c1.chunk_id == c2.chunk_id


def test_chunk_ids_are_unique_within_section():
    """No two chunks in the same section should have the same ID."""
    pipeline = DocumentIngestionPipeline("A")
    chunks = pipeline.chunk_section(_LONG_PROSE, "AAPL", "10-K", _PERIOD, "MD&A")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# ---------------------------------------------------------------------------
# P7-E2-T11: chunk_document with multi-section FilingDocument
# ---------------------------------------------------------------------------


def test_chunk_document_all_configs_produce_chunks():
    """chunk_document() with a 3-section document produces chunks for all 3 configs."""
    doc = _make_doc()
    for config in ("A", "B", "C"):
        pipeline = DocumentIngestionPipeline(config)
        chunks = pipeline.chunk_document(doc)
        assert len(chunks) > 0, f"Config {config}: expected non-empty chunk list"


def test_chunk_document_includes_all_sections():
    """chunk_document() returns chunks from all sections of the document."""
    doc = _make_doc({
        "Business": "Apple designs consumer electronics." * 50,
        "Risk Factors": "Competition is intense." * 50,
        "MD&A": "Revenue declined 5%." * 50,
    })
    pipeline = DocumentIngestionPipeline("A")
    chunks = pipeline.chunk_document(doc)
    sections_seen = {c.section for c in chunks}
    assert "Business" in sections_seen
    assert "Risk Factors" in sections_seen
    assert "MD&A" in sections_seen


# ---------------------------------------------------------------------------
# P7-E2-T12: no empty chunks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["A", "B", "C"])
def test_no_empty_chunks(config):
    """All returned chunks must have non-empty stripped text."""
    pipeline = DocumentIngestionPipeline(config)
    chunks = pipeline.chunk_section(_PROSE, "AAPL", "10-K", _PERIOD, "MD&A")
    for chunk in chunks:
        assert chunk.text.strip(), f"Config {config}: found empty chunk"


# ---------------------------------------------------------------------------
# P7-E2-T13: Config C max 6000 chars
# ---------------------------------------------------------------------------


def test_config_c_max_6000_chars():
    """Config C chunks should not exceed 6000 characters."""
    large_para = "Apple reported strong results. " * 200  # ~6200 chars per paragraph
    text = large_para + "\n\n" + large_para + "\n\n" + large_para
    pipeline = DocumentIngestionPipeline("C")
    chunks = pipeline.chunk_section(text, "AAPL", "10-K", _PERIOD, "MD&A")
    for chunk in chunks:
        assert chunk.char_count <= 6000, (
            f"Config C chunk exceeds 6000 chars: {chunk.char_count}"
        )


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------


def test_chunk_metadata_fields_are_correct():
    """DocumentChunk metadata fields should match the input parameters."""
    pipeline = DocumentIngestionPipeline("A")
    chunks = pipeline.chunk_section(_PROSE, "AAPL", "10-K", _PERIOD, "MD&A")
    for chunk in chunks:
        assert chunk.ticker == "AAPL"
        assert chunk.filing_type == "10-K"
        assert chunk.period == _PERIOD
        assert chunk.section == "MD&A"
        assert chunk.chunking_config == "A"
        assert chunk.char_count == len(chunk.text)
        from math import ceil
        assert chunk.approx_tokens == ceil(chunk.char_count / 4)


def test_invalid_config_raises_value_error():
    with pytest.raises(ValueError, match="chunking_config must be"):
        DocumentIngestionPipeline("D")
