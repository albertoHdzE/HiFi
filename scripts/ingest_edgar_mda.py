"""
EDGAR MD&A targeted ingestion into LanceDB (E2-T3, DJ-090).

Downloads Item 7 (10-K) and Item 2 (10-Q) sections from SEC EDGAR for
the Phase 14 universe, strips HTML, chunks into ~512-token pieces, and
writes to a LanceDB table in the specified namespace.

Usage:
    uv run python scripts/ingest_edgar_mda.py [options]

Options:
    --namespace NS      LanceDB table prefix (default: hifi-dev-sec)
    --through-date DATE Only ingest filings with period-of-report <= DATE
    --data-dir DIR      Root data directory (default: data)
    --tickers T [T...]  Override ticker list (default: full Phase 14 universe)
    --dry-run           Print filings that would be ingested, do not write

Make target: ingest-edgar-mda  (internet required, ~4-8h for 100 stocks)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_EDGAR_BASE = "https://www.sec.gov"
_FILING_TYPES = [("10-K", "Item 7"), ("10-Q", "Item 2")]
_START_DATE = "2004-01-01"
_REQUEST_DELAY = 0.5  # seconds between SEC requests (polite crawl)


# ---------------------------------------------------------------------------
# CIK lookup via EDGAR EFTS
# ---------------------------------------------------------------------------


def lookup_cik(ticker: str, session: Any) -> int | None:
    """Return integer CIK for ``ticker``, or None if not found."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K&dateRange=custom&startdt=2020-01-01&enddt=2025-12-31"
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            eid = src.get("entity_id", "")
            if eid:
                try:
                    return int(eid)
                except ValueError:
                    pass
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


def lookup_cik_by_ticker_api(ticker: str, session: Any) -> int | None:
    """Fallback: use company_tickers.json from SEC."""
    try:
        resp = session.get(
            "https://www.sec.gov/files/company_tickers.json", timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        for _idx, entry in data.items():
            if entry.get("ticker", "").upper() == ticker.upper():
                return int(entry["cik_str"])
    except Exception as exc:
        logger.warning("Ticker→CIK lookup failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# Filing index fetch
# ---------------------------------------------------------------------------


def get_filings(
    cik: int,
    form_type: str,
    through_date: str | None,
    session: Any,
) -> list[dict]:
    """
    Return filing metadata for ``cik`` filtered to ``form_type``.

    Each entry: {accession_number, period_of_report, filing_date, primary_document}.
    Filtered to period_of_report >= '2004-01-01' and <= through_date (if given).
    """
    url = _EDGAR_SUBMISSIONS.format(cik=cik)
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Submissions fetch failed for CIK %d: %s", cik, exc)
        return []

    filings_data = data.get("filings", {}).get("recent", {})
    forms = filings_data.get("form", [])
    accessions = filings_data.get("accessionNumber", [])
    periods = filings_data.get("periodOfReport", [])
    primary_docs = filings_data.get("primaryDocument", [])

    results = []
    for form, accession, period, primary in zip(
        forms, accessions, periods, primary_docs, strict=False
    ):
        if form != form_type:
            continue
        if period < _START_DATE:
            continue
        if through_date and period > through_date:
            continue
        results.append({
            "accession_number": accession,
            "period_of_report": period,
            "primary_document": primary,
            "cik": cik,
        })
    return results


# ---------------------------------------------------------------------------
# Filing HTML fetch
# ---------------------------------------------------------------------------


def fetch_filing_html(filing: dict, session: Any) -> str | None:
    """Download the primary document HTML for a filing, return raw text."""
    cik = filing["cik"]
    accession = filing["accession_number"].replace("-", "")
    doc = filing["primary_document"]
    url = f"{_EDGAR_BASE}/Archives/edgar/data/{cik}/{accession}/{doc}"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("Failed to fetch filing HTML from %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# LanceDB writer
# ---------------------------------------------------------------------------


def _get_or_create_table(db: Any, table_name: str) -> Any:
    """Return existing LanceDB table or create it with the SEC schema."""
    import pyarrow as pa

    schema = pa.schema([
        pa.field("ticker", pa.string()),
        pa.field("filing_type", pa.string()),
        pa.field("period", pa.string()),
        pa.field("section", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("approx_tokens", pa.int32()),
    ])
    try:
        return db.open_table(table_name)
    except Exception:
        return db.create_table(table_name, schema=schema)


def write_chunks(table: Any, chunks: list[dict]) -> None:
    """Write chunk dicts to LanceDB table."""
    import pyarrow as pa

    if not chunks:
        return
    arrays = {
        "ticker": pa.array([c["ticker"] for c in chunks], pa.string()),
        "filing_type": pa.array([c["filing_type"] for c in chunks], pa.string()),
        "period": pa.array([c["period"] for c in chunks], pa.string()),
        "section": pa.array([c["section"] for c in chunks], pa.string()),
        "chunk_index": pa.array([c["chunk_index"] for c in chunks], pa.int32()),
        "text": pa.array([c["text"] for c in chunks], pa.string()),
        "approx_tokens": pa.array([c["approx_tokens"] for c in chunks], pa.int32()),
    }
    batch = pa.table(arrays)
    table.add(batch)


# ---------------------------------------------------------------------------
# Per-ticker ingestion
# ---------------------------------------------------------------------------


def ingest_ticker(
    ticker: str,
    cik: int,
    form_type: str,
    through_date: str | None,
    table: Any,
    session: Any,
    dry_run: bool = False,
) -> int:
    """Ingest all MD&A chunks for one ticker/form_type combination."""
    from hifi.data.edgar_mda import chunk_text, extract_mda_section

    filings = get_filings(cik, form_type, through_date, session)
    if not filings:
        logger.debug("No %s filings for %s (CIK %d)", form_type, ticker, cik)
        return 0

    total = 0
    for filing in filings:
        period = filing["period_of_report"]
        if dry_run:
            logger.info("[DRY-RUN] %s %s %s", ticker, form_type, period)
            continue

        time.sleep(_REQUEST_DELAY)
        html = fetch_filing_html(filing, session)
        if not html:
            continue

        mda_text = extract_mda_section(html, form_type)
        if not mda_text:
            logger.debug("No MD&A found in %s %s %s", ticker, form_type, period)
            continue

        chunks = chunk_text(mda_text, ticker, form_type, period)
        write_chunks(table, chunks)
        total += len(chunks)
        logger.info("Ingested %s %s %s → %d chunks", ticker, form_type, period, len(chunks))

    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", default="hifi-dev-sec")
    p.add_argument("--through-date", default=None, dest="through_date")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import lancedb
    import requests

    from hifi.data.universe import PHASE14_UNIVERSE

    data_dir = Path(args.data_dir or "data")
    db_path = str(data_dir / "knowledge.lance")
    table_name = f"{args.namespace}-sec-mda"

    tickers = args.tickers or [e["ticker"] for e in PHASE14_UNIVERSE]
    logger.info(
        "Ingesting EDGAR MD&A for %d tickers → %s (through=%s)",
        len(tickers),
        table_name,
        args.through_date or "all",
    )

    session = requests.Session()
    session.headers["User-Agent"] = "HiFi Research <research@hifi.local>"

    if not args.dry_run:
        db = lancedb.connect(db_path)
        table = _get_or_create_table(db, table_name)
    else:
        table = None

    total_chunks = 0
    for ticker in tickers:
        time.sleep(_REQUEST_DELAY)
        cik = lookup_cik_by_ticker_api(ticker, session)
        if cik is None:
            logger.warning("Could not resolve CIK for %s — skipping", ticker)
            continue

        for form_type, _label in _FILING_TYPES:
            n = ingest_ticker(
                ticker=ticker,
                cik=cik,
                form_type=form_type,
                through_date=args.through_date,
                table=table,
                session=session,
                dry_run=args.dry_run,
            )
            total_chunks += n

    logger.info("Done. Total chunks ingested: %d", total_chunks)


if __name__ == "__main__":
    main()
