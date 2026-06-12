"""
Unit tests for EdgarFetcher (P7-E1).

Tests cover: URL construction, CIK zero-padding, User-Agent header, section
extraction for all three filing types, FilingDocument and DocumentChunk schemas.

HTTP calls are mocked using the responses library to avoid internet access.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

import responses as responses_lib

from hifi.data.edgar import _ARCHIVES_BASE, _SUBMISSIONS_BASE, TICKER_CIKS, EdgarFetcher
from hifi.knowledge.schemas import DocumentChunk, EvaluationQuery, FilingDocument

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AAPL_CIK = "0000320193"
_AAPL_ACC = "0000320193-23-000006"
_AAPL_ACC_NODASH = "000032019323000006"


def _sample_submissions(ticker: str = "AAPL", cik: str = "320193") -> dict:
    """Minimal EDGAR submissions JSON for testing."""
    return {
        "cik": cik,
        "name": f"{ticker} Inc.",
        "filings": {
            "recent": {
                "accessionNumber": [_AAPL_ACC, "0000320193-22-000001"],
                "form": ["10-Q", "10-K"],
                "filingDate": ["2023-02-02", "2022-10-28"],
                "reportDate": ["2022-12-31", "2022-09-24"],
                "primaryDocument": ["aapl20221231.htm", "aapl20220924.htm"],
            }
        },
    }


def _sample_10k_html() -> str:
    return """
    <html><body>
    <p>ITEM 1. BUSINESS</p>
    <p>Apple Inc. designs and markets consumer electronics and software.</p>
    <p>The company sells iPhone, Mac, iPad and other products.</p>
    <p>ITEM 1A. RISK FACTORS</p>
    <p>The company faces intense competition in all product categories.</p>
    <p>Currency fluctuations may adversely affect international revenue.</p>
    <p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
    <p>Net sales decreased 2.5% to $394.3 billion compared to the prior year.</p>
    <p>Gross margin was 44.1%, up from 43.3% in the prior year.</p>
    </body></html>
    """


def _sample_10q_html() -> str:
    return """
    <html><body>
    <h2>PART I — FINANCIAL INFORMATION</h2>
    <p>ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
    <p>iPhone net sales decreased 2% year over year.</p>
    <p>Services net sales grew 6% to $20.9 billion.</p>
    </body></html>
    """


def _sample_8k_html() -> str:
    return """
    <html><body>
    <p>Apple Inc. today announced financial results for its fiscal 2023 first quarter.</p>
    <p>Revenue of $117.2 billion, down 5% year over year.</p>
    <p>EPS diluted $1.88, down 10% year over year.</p>
    </body></html>
    """


def _sample_filing_index(primary_doc: str = "aapl20221231.htm") -> dict:
    return {
        "directory": {
            "item": [
                {"name": primary_doc, "type": "10-Q"},
                {"name": "exhibit31.htm", "type": "EX-31.1"},
            ]
        }
    }


# ---------------------------------------------------------------------------
# P7-E1-T8: get_submissions() URL construction + CIK zero-padding
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_get_submissions_constructs_correct_url():
    url = f"{_SUBMISSIONS_BASE}/CIK0000320193.json"
    responses_lib.add(responses_lib.GET, url, json=_sample_submissions())
    fetcher = EdgarFetcher()
    result = fetcher.get_submissions("0000320193")
    assert result["name"] == "AAPL Inc."
    # Verify User-Agent header was sent
    assert responses_lib.calls[0].request.headers["User-Agent"] == EdgarFetcher.DEFAULT_USER_AGENT


@responses_lib.activate
def test_get_submissions_zero_pads_short_cik():
    """CIK "320193" should be padded to "0000320193" in the URL."""
    url = f"{_SUBMISSIONS_BASE}/CIK0000320193.json"
    responses_lib.add(responses_lib.GET, url, json=_sample_submissions())
    fetcher = EdgarFetcher()
    result = fetcher.get_submissions("320193")  # unpadded
    assert result["name"] == "AAPL Inc."


@responses_lib.activate
def test_get_filing_index_constructs_correct_url():
    """Accession number dashes are removed for the URL."""
    url = f"{_ARCHIVES_BASE}/320193/{_AAPL_ACC_NODASH}/index.json"
    responses_lib.add(responses_lib.GET, url, json=_sample_filing_index())
    fetcher = EdgarFetcher()
    result = fetcher.get_filing_index("0000320193", _AAPL_ACC)
    assert "directory" in result


# ---------------------------------------------------------------------------
# P7-E1-T9: extract_text_sections() for 10-K
# ---------------------------------------------------------------------------


def test_extract_text_sections_10k_returns_all_sections():
    fetcher = EdgarFetcher()
    sections = fetcher.extract_text_sections(_sample_10k_html(), "10-K")
    assert "Business" in sections
    assert "Risk Factors" in sections
    assert "MD&A" in sections


def test_extract_text_sections_10k_strips_html_tags():
    html = "<p>ITEM 7. <b>MD&amp;A</b></p><p>Revenue was <em>$100B</em>.</p>"
    fetcher = EdgarFetcher()
    sections = fetcher.extract_text_sections(
        html + "<p>ITEM 8. FINANCIAL STATEMENTS</p><p>See below.</p>", "10-K"
    )
    assert "MD&A" in sections
    text = sections["MD&A"]
    assert "<p>" not in text
    assert "<b>" not in text
    assert "$100B" in text


def test_extract_text_sections_10k_non_empty():
    fetcher = EdgarFetcher()
    sections = fetcher.extract_text_sections(_sample_10k_html(), "10-K")
    for section_name, text in sections.items():
        assert len(text) > 0, f"Section {section_name!r} is empty"


# ---------------------------------------------------------------------------
# P7-E1-T10: extract_text_sections() for 10-Q returns MD&A
# ---------------------------------------------------------------------------


def test_extract_text_sections_10q_returns_mda():
    fetcher = EdgarFetcher()
    sections = fetcher.extract_text_sections(_sample_10q_html(), "10-Q")
    assert "MD&A" in sections
    assert "iPhone" in sections["MD&A"] or len(sections["MD&A"]) > 0


# ---------------------------------------------------------------------------
# P7-E1-T11: extract_text_sections() for 8-K returns full body
# ---------------------------------------------------------------------------


def test_extract_text_sections_8k_returns_earnings_release():
    fetcher = EdgarFetcher()
    sections = fetcher.extract_text_sections(_sample_8k_html(), "8-K")
    assert "Earnings Release" in sections
    assert len(sections["Earnings Release"]) > 0


def test_extract_text_sections_8k_contains_body_text():
    fetcher = EdgarFetcher()
    sections = fetcher.extract_text_sections(_sample_8k_html(), "8-K")
    text = sections["Earnings Release"]
    assert "Apple" in text or "Revenue" in text or "117" in text


# ---------------------------------------------------------------------------
# P7-E1-T12: FilingDocument schema validation
# ---------------------------------------------------------------------------


def test_filing_document_validates_period_of_report_as_date():
    doc = FilingDocument(
        ticker="AAPL",
        cik="0000320193",
        filing_type="10-Q",
        accession_number="0000320193-23-000006",
        period_of_report=date(2022, 12, 31),
        filed_date=date(2023, 2, 2),
        sections={"MD&A": "iPhone revenue decreased 2%."},
        source_url="https://www.sec.gov/Archives/edgar/data/320193/...",
        fetched_at=datetime(2023, 4, 1, 0, 0, 0),
    )
    assert doc.period_of_report == date(2022, 12, 31)
    assert isinstance(doc.period_of_report, date)


def test_filing_document_sections_is_non_empty():
    doc = FilingDocument(
        ticker="AAPL",
        cik="0000320193",
        filing_type="10-K",
        accession_number="0000320193-22-000001",
        period_of_report=date(2022, 9, 24),
        filed_date=date(2022, 10, 28),
        sections={"Business": "Apple sells iPhones.", "MD&A": "Revenue grew."},
        source_url="https://example.com",
        fetched_at=datetime(2023, 4, 1),
    )
    assert len(doc.sections) > 0
    assert "Business" in doc.sections


# ---------------------------------------------------------------------------
# P7-E1-T13: DocumentChunk.chunk_id deterministic SHA-256 prefix
# ---------------------------------------------------------------------------


def test_document_chunk_id_is_deterministic():
    period = date(2023, 3, 31)
    cid1 = DocumentChunk.make_chunk_id("AAPL", "10-K", period, "MD&A", 0, "A")
    cid2 = DocumentChunk.make_chunk_id("AAPL", "10-K", period, "MD&A", 0, "A")
    assert cid1 == cid2


def test_document_chunk_id_is_16_chars():
    period = date(2023, 3, 31)
    cid = DocumentChunk.make_chunk_id("AAPL", "10-K", period, "MD&A", 0, "A")
    assert len(cid) == 16


def test_document_chunk_id_differs_for_different_inputs():
    period = date(2023, 3, 31)
    cid1 = DocumentChunk.make_chunk_id("AAPL", "10-K", period, "MD&A", 0, "A")
    cid2 = DocumentChunk.make_chunk_id("AAPL", "10-K", period, "MD&A", 1, "A")
    cid3 = DocumentChunk.make_chunk_id("JPM", "10-K", period, "MD&A", 0, "A")
    assert cid1 != cid2
    assert cid1 != cid3


def test_document_chunk_id_matches_sha256():
    period = date(2023, 3, 31)
    key = "AAPL|10-K|2023-03-31|MD&A|0|A"
    expected = hashlib.sha256(key.encode()).hexdigest()[:16]
    result = DocumentChunk.make_chunk_id("AAPL", "10-K", period, "MD&A", 0, "A")
    assert result == expected


# ---------------------------------------------------------------------------
# EvaluationQuery schema
# ---------------------------------------------------------------------------


def test_evaluation_query_schema():
    q = EvaluationQuery(
        query_id="Q01",
        query="What did Apple say about iPhone demand?",
        ticker="AAPL",
        relevant_section="MD&A",
        relevant_filing_type="10-Q",
    )
    assert q.query_id == "Q01"
    assert q.notes == ""  # default


# ---------------------------------------------------------------------------
# TICKER_CIKS constant
# ---------------------------------------------------------------------------


def test_ticker_ciks_contains_baseline_tickers():
    assert "AAPL" in TICKER_CIKS
    assert "JPM" in TICKER_CIKS
    assert "XOM" in TICKER_CIKS
    for cik in TICKER_CIKS.values():
        assert len(cik) == 10  # zero-padded
        assert cik.isdigit()
