"""
Unit tests for hifi.knowledge.edgar_retriever (Phase 15 Wave 2, DJ-097).

Tests are split into two groups:
  1. Pure (no-data): mock lancedb to test error-handling and format paths.
  2. Data-required: skip if data/knowledge.lance is absent; otherwise test
     temporal discipline and coverage against the real 209,722-chunk table.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hifi.knowledge.edgar_retriever import retrieve_mda_context

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DB_PATH = str(Path(__file__).resolve().parents[3] / "data" / "knowledge.lance")
_TABLE_EXISTS = Path(_DB_PATH).exists()
_SKIP_REASON = "data/knowledge.lance not present — skipping live EDGAR tests"

pytestmark_data = pytest.mark.skipif(not _TABLE_EXISTS, reason=_SKIP_REASON)

# ---------------------------------------------------------------------------
# 1. Pure unit tests (mocked lancedb)
# ---------------------------------------------------------------------------


def _make_mock_db(rows):
    """Return a mock lancedb.connect() that serves the given rows as a DataFrame."""
    import pandas as pd

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker", "filing_type", "period", "section", "chunk_index", "text"]
    )

    mock_table = MagicMock()
    mock_table.to_pandas.return_value = df

    mock_db = MagicMock()
    mock_db.list_tables.return_value = ["hifi-dev-sec-sec-mda"]
    mock_db.open_table.return_value = mock_table

    return mock_db


def _patch_lancedb(mock_db):
    """Patch lancedb.connect to return mock_db."""
    lancedb_mod = MagicMock()
    lancedb_mod.connect.return_value = mock_db
    return patch.dict("sys.modules", {"lancedb": lancedb_mod})


def test_returns_empty_when_table_missing():
    """Returns '' when the requested table is not in the database."""
    mock_db = MagicMock()
    mock_db.list_tables.return_value = []

    lancedb_mod = MagicMock()
    lancedb_mod.connect.return_value = mock_db

    with patch.dict("sys.modules", {"lancedb": lancedb_mod}):
        result = retrieve_mda_context("AAPL", "2022-01-31", db_path="/fake/path")
    assert result == ""


def test_returns_empty_when_no_rows_match():
    """Returns '' when ticker is not in table."""
    mock_db = _make_mock_db([
        {"ticker": "MSFT", "filing_type": "10-K", "period": "2021-09-25",
         "section": "mda", "chunk_index": 0, "text": "Microsoft results."},
    ])
    with _patch_lancedb(mock_db):
        result = retrieve_mda_context("AAPL", "2022-01-31", db_path="/fake/path")
    assert result == ""


def test_returns_empty_when_all_periods_after_as_of():
    """Returns '' when all filings are future-dated (temporal discipline)."""
    mock_db = _make_mock_db([
        {"ticker": "AAPL", "filing_type": "10-K", "period": "2023-09-30",
         "section": "mda", "chunk_index": 0, "text": "Future content."},
    ])
    with _patch_lancedb(mock_db):
        result = retrieve_mda_context("AAPL", "2022-01-31", db_path="/fake/path")
    assert result == ""


def test_returns_most_recent_period():
    """Selects the latest filing period ≤ as_of_date."""
    mock_db = _make_mock_db([
        {"ticker": "AAPL", "filing_type": "10-Q", "period": "2021-06-26",
         "section": "mda", "chunk_index": 0, "text": "Q3 2021 results."},
        {"ticker": "AAPL", "filing_type": "10-K", "period": "2021-09-25",
         "section": "mda", "chunk_index": 0, "text": "FY2021 results."},
        {"ticker": "AAPL", "filing_type": "10-Q", "period": "2022-06-25",
         "section": "mda", "chunk_index": 0, "text": "Q3 2022 results — future."},
    ])
    with _patch_lancedb(mock_db):
        result = retrieve_mda_context("AAPL", "2022-01-31", db_path="/fake/path")
    assert "FY2021" in result
    assert "Q3 2022" not in result


def test_output_format_includes_header():
    """Result starts with '[EDGAR MD&A — TICKER TYPE period=... as_of=...]'."""
    mock_db = _make_mock_db([
        {"ticker": "AAPL", "filing_type": "10-K", "period": "2021-09-25",
         "section": "mda", "chunk_index": 0, "text": "Apple annual discussion."},
    ])
    with _patch_lancedb(mock_db):
        result = retrieve_mda_context("AAPL", "2022-01-31", db_path="/fake/path")
    assert result.startswith("[EDGAR MD&A — AAPL 10-K period=2021-09-25 as_of=2022-01-31]")


def test_max_chunks_limits_output():
    """Only max_chunks consecutive chunks are returned."""
    rows = [
        {"ticker": "AAPL", "filing_type": "10-K", "period": "2021-09-25",
         "section": "mda", "chunk_index": i, "text": f"Chunk {i}."}
        for i in range(10)
    ]
    mock_db = _make_mock_db(rows)
    with _patch_lancedb(mock_db):
        result = retrieve_mda_context("AAPL", "2022-01-31", db_path="/fake/path", max_chunks=3)
    # Chunks 0–2 included; chunk 3 onwards excluded
    assert "Chunk 0." in result
    assert "Chunk 2." in result
    assert "Chunk 3." not in result


def test_exception_returns_empty_string():
    """Any unexpected exception returns '' (fail-open contract)."""
    lancedb_mod = MagicMock()
    lancedb_mod.connect.side_effect = RuntimeError("DB unavailable")

    with patch.dict("sys.modules", {"lancedb": lancedb_mod}):
        result = retrieve_mda_context("AAPL", "2022-01-31", db_path="/fake/path")
    assert result == ""


def test_unknown_ticker_returns_empty():
    """Unknown ticker with real-looking rows for others returns ''."""
    mock_db = _make_mock_db([
        {"ticker": "AAPL", "filing_type": "10-K", "period": "2021-09-25",
         "section": "mda", "chunk_index": 0, "text": "AAPL content."},
    ])
    with _patch_lancedb(mock_db):
        result = retrieve_mda_context("FAKE", "2022-01-31", db_path="/fake/path")
    assert result == ""


# ---------------------------------------------------------------------------
# 2. Data-required tests (real LanceDB table)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TABLE_EXISTS, reason=_SKIP_REASON)
def test_aapl_2022_non_empty():
    """AAPL has MD&A filing on or before 2022-01-31 (10-K FY2021 filed Oct 2021)."""
    result = retrieve_mda_context("AAPL", "2022-01-31", db_path=_DB_PATH)
    assert result != "", "Expected non-empty MD&A for AAPL as_of 2022-01-31"
    assert "AAPL" in result


@pytest.mark.skipif(not _TABLE_EXISTS, reason=_SKIP_REASON)
def test_aapl_2004_temporal_filter():
    """Before any EDGAR filing → empty string (temporal discipline)."""
    result = retrieve_mda_context("AAPL", "2004-01-31", db_path=_DB_PATH)
    assert result == "", (
        "Expected '' for AAPL as_of 2004-01-31 — no 10-K filed before that date"
    )


@pytest.mark.skipif(not _TABLE_EXISTS, reason=_SKIP_REASON)
def test_fake_ticker_returns_empty():
    """Ticker not in EDGAR table → empty string."""
    result = retrieve_mda_context("FAKE", "2022-01-31", db_path=_DB_PATH)
    assert result == ""


@pytest.mark.skipif(not _TABLE_EXISTS, reason=_SKIP_REASON)
@pytest.mark.parametrize("ticker,date", [
    ("AAPL", "2022-01-31"),
    ("JPM",  "2022-06-30"),
    ("XOM",  "2023-03-31"),
    ("NVDA", "2022-06-30"),
    ("BAC",  "2023-03-31"),
    ("JNJ",  "2022-01-31"),
    ("PG",   "2022-06-30"),
    ("UNH",  "2023-03-31"),
])
def test_temporal_discipline_period_le_as_of_date(ticker, date):
    """
    Temporal discipline: the period in the EDGAR header is ≤ as_of_date.

    This guards against any look-ahead leakage in the walk-forward evaluation.
    """
    result = retrieve_mda_context(ticker, date, db_path=_DB_PATH)
    if not result:
        pytest.skip(f"No EDGAR data for {ticker} as_of {date}")
    # Parse period from header: "[EDGAR MD&A — AAPL 10-K period=2021-09-25 as_of=...]"
    import re
    m = re.search(r"period=(\d{4}-\d{2}-\d{2})", result)
    assert m, f"Could not parse period from EDGAR header: {result[:120]}"
    assert m.group(1) <= date, (
        f"Look-ahead violation: period={m.group(1)} > as_of_date={date} for {ticker}"
    )


@pytest.mark.skipif(not _TABLE_EXISTS, reason=_SKIP_REASON)
def test_all_universe_tickers_have_edgar_data():
    """
    All 98 PHASE14_UNIVERSE tickers have ≥1 chunk in the EDGAR table at some date.

    Uses a permissive as_of_date (2025-12-31) to surface coverage gaps.
    """
    import lancedb

    from hifi.data.universe import PHASE14_UNIVERSE

    db = lancedb.connect(_DB_PATH)
    table = db.open_table("hifi-dev-sec-sec-mda")
    tickers_in_db = set(table.to_pandas()["ticker"].unique())

    universe_tickers = {entry["ticker"] for entry in PHASE14_UNIVERSE}
    missing = universe_tickers - tickers_in_db

    assert not missing, (
        f"{len(missing)} universe tickers missing from EDGAR table: {sorted(missing)}"
    )
