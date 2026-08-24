"""
Unit tests for src/hifi/data/edgar_mda.py helpers (E2-T3, DJ-090).

Tests strip_html, extract_mda_section, and chunk_text without any
network calls.  A lightweight inline HTML fixture stands in for a
real 10-K/10-Q filing.
"""

from __future__ import annotations

from hifi.data.edgar_mda import chunk_text, extract_mda_section, strip_html

# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------


def test_strip_html_removes_tags():
    assert strip_html("<p>Hello <b>World</b></p>") == "Hello World"


def test_strip_html_normalizes_whitespace():
    result = strip_html("<p>  foo   bar  </p>")
    assert result == "foo bar"


def test_strip_html_decodes_amp():
    assert "&" in strip_html("Revenue &amp; Expenses")


def test_strip_html_decodes_lt_gt():
    result = strip_html("&lt;b&gt;bold&lt;/b&gt;")
    assert "<" in result and ">" in result


def test_strip_html_decodes_nbsp():
    result = strip_html("Line&nbsp;one&#160;two")
    assert "one" in result and "two" in result
    assert "&nbsp;" not in result


def test_strip_html_removes_script_block():
    html = "<script>alert('xss')</script>Real content"
    result = strip_html(html)
    assert "alert" not in result
    assert "Real content" in result


def test_strip_html_removes_style_block():
    html = "<style>.red { color: red; }</style>Visible text"
    result = strip_html(html)
    assert ".red" not in result
    assert "Visible text" in result


def test_strip_html_plain_text_unchanged():
    assert strip_html("no tags here") == "no tags here"


def test_strip_html_empty_string():
    assert strip_html("") == ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_10K_HTML = """
<html><body>
<p>PART II</p>
<p>Item 7. Management Discussion and Analysis of Financial Condition</p>
<p>Revenues increased 15 percent year-over-year driven by cloud services.
Operating expenses were well controlled.  Margins expanded 200 basis points.</p>
<p>We expect growth to continue into fiscal 2024.</p>
<p>Item 8. Financial Statements and Supplementary Data</p>
<p>Balance sheet data appears here and should NOT be extracted.</p>
</body></html>
"""

_10Q_HTML = """
<html><body>
<p>PART I — FINANCIAL INFORMATION</p>
<p>Item 2. Management Discussion and Analysis</p>
<p>Quarterly revenues declined 5 percent due to macroeconomic headwinds.
Gross margins contracted slightly.</p>
<p>Item 3. Quantitative Disclosures About Market Risk</p>
<p>Interest rate risk discussion here.</p>
</body></html>
"""

_NO_MDA_HTML = """
<html><body>
<p>Annual Report Cover Page</p>
<p>No management section in this document.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# extract_mda_section — 10-K
# ---------------------------------------------------------------------------


def test_extract_mda_10k_finds_section():
    result = extract_mda_section(_10K_HTML, "10-K")
    assert "revenues increased" in result.lower()
    assert "Operating expenses" in result


def test_extract_mda_10k_stops_before_item8():
    result = extract_mda_section(_10K_HTML, "10-K")
    assert "Balance sheet" not in result


def test_extract_mda_10k_returns_nonempty_string():
    result = extract_mda_section(_10K_HTML, "10-K")
    assert isinstance(result, str)
    assert len(result) > 50


# ---------------------------------------------------------------------------
# extract_mda_section — 10-Q
# ---------------------------------------------------------------------------


def test_extract_mda_10q_finds_section():
    result = extract_mda_section(_10Q_HTML, "10-Q")
    assert "revenues declined" in result.lower()


def test_extract_mda_10q_nonempty_string():
    result = extract_mda_section(_10Q_HTML, "10-Q")
    assert isinstance(result, str) and len(result) > 20


# ---------------------------------------------------------------------------
# extract_mda_section — not found
# ---------------------------------------------------------------------------


def test_extract_mda_not_found_returns_empty():
    result = extract_mda_section(_NO_MDA_HTML, "10-K")
    assert result == ""


def test_extract_mda_empty_html_returns_empty():
    assert extract_mda_section("", "10-K") == ""


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("", "AAPL", "10-K", "2022-12-31") == []


def test_chunk_text_short_text_single_chunk():
    text = "Short earnings commentary that fits in one chunk."
    chunks = chunk_text(text, "AAPL", "10-K", "2022-12-31")
    assert len(chunks) == 1


def test_chunk_text_metadata_fields():
    chunks = chunk_text("Some text here.", "MSFT", "10-Q", "2023-03-31")
    c = chunks[0]
    assert c["ticker"] == "MSFT"
    assert c["filing_type"] == "10-Q"
    assert c["period"] == "2023-03-31"
    assert c["section"] == "mda"
    assert c["chunk_index"] == 0
    assert "text" in c
    assert c["approx_tokens"] > 0


def test_chunk_text_long_text_multiple_chunks():
    text = ("word " * 1200).strip()
    chunks = chunk_text(text, "MSFT", "10-K", "2023-12-31", chunk_size=2048)
    assert len(chunks) >= 2


def test_chunk_text_chunk_indices_sequential():
    text = ("word " * 1200).strip()
    chunks = chunk_text(text, "MSFT", "10-K", "2023-12-31", chunk_size=2048)
    for i, c in enumerate(chunks):
        assert c["chunk_index"] == i


def test_chunk_text_all_text_preserved():
    """All words in the original text must appear across chunks."""
    text = "alpha beta gamma delta epsilon zeta"
    chunks = chunk_text(text, "AAPL", "10-K", "2022-12-31", chunk_size=10)
    combined = " ".join(c["text"] for c in chunks)
    for word in text.split():
        assert word in combined


def test_chunk_text_approx_tokens_reasonable():
    text = "word " * 200
    chunks = chunk_text(text, "AAPL", "10-K", "2022-12-31")
    for c in chunks:
        # approx_tokens = len(text) // 4, so ~250 for ~1000 chars
        assert 0 < c["approx_tokens"] <= 600
