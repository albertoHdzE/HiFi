"""
Record SEC EDGAR fixture filings for Phase 7 testing (P7-E8).

Fetches 9 filings (3 tickers x 3 filing types) from the SEC EDGAR public API
and saves the extracted text sections to tests/fixtures/sec/.

IMPORTANT: Requires internet access. Run once and commit the output.

Usage
-----
    uv run python scripts/record_sec_fixtures.py

Output
------
tests/fixtures/sec/{TICKER}_{FILING_TYPE}_sections.json  (9 files)

Each file contains:
{
  "ticker": "AAPL",
  "cik": "0000320193",
  "filing_type": "10-K",
  "accession_number": "...",
  "period_of_report": "2023-03-31",
  "filed_date": "2022-10-28",
  "sections": {"Business": "...", "Risk Factors": "...", "MD&A": "..."},
  "fetched_at": "..."
}
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hifi.data.edgar import TICKER_CIKS, EdgarFetcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FIXTURES_DIR = _ROOT / "tests" / "fixtures" / "sec"
_AS_OF_DATE = date(2023, 3, 31)
_FILING_TYPES = ["10-K", "10-Q", "8-K"]


def main() -> None:
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = EdgarFetcher()
    errors: list[str] = []

    for ticker, cik in TICKER_CIKS.items():
        for filing_type in _FILING_TYPES:
            out_path = _FIXTURES_DIR / f"{ticker}_{filing_type.replace('-', '_')}_sections.json"

            if out_path.exists():
                logger.info("Skipping %s %s (already recorded)", ticker, filing_type)
                continue

            logger.info("Fetching %s %s ...", ticker, filing_type)
            try:
                doc = fetcher.fetch_filing(
                    ticker=ticker,
                    cik=cik,
                    filing_type=filing_type,
                    as_of_date=_AS_OF_DATE,
                )
                payload = {
                    "ticker": doc.ticker,
                    "cik": doc.cik,
                    "filing_type": doc.filing_type,
                    "accession_number": doc.accession_number,
                    "period_of_report": doc.period_of_report.isoformat(),
                    "filed_date": doc.filed_date.isoformat(),
                    "sections": doc.sections,
                    "fetched_at": doc.fetched_at.isoformat(),
                    "source_url": str(doc.source_url),
                }
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                logger.info(
                    "Saved %s %s: %d sections, %d chars total",
                    ticker, filing_type,
                    len(doc.sections),
                    sum(len(v) for v in doc.sections.values()),
                )
            except Exception as exc:
                msg = f"FAILED {ticker} {filing_type}: {exc}"
                logger.warning(msg)
                errors.append(msg)

    if errors:
        print(f"\n{len(errors)} filing(s) failed:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"\nAll fixtures saved to {_FIXTURES_DIR}")


if __name__ == "__main__":
    main()
