"""
SEC EDGAR API client for HiFi Phase 7 (P7-E1).

Downloads 10-K, 10-Q, and 8-K filings from the SEC EDGAR public API.
No API key required. User-Agent header required per EDGAR terms of service.

API endpoints:
  submissions:   https://data.sec.gov/submissions/CIK{cik_padded}.json
  filing index:  https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/index.json
  filing doc:    https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/{filename}

Rate limit: 10 requests/second. This client enforces <= 8 req/s via a minimum
inter-request sleep of 0.125 seconds.

HTML parsing uses stdlib html.parser (no lxml dependency).
"""

from __future__ import annotations

import html as _html_module
import logging
import os
import re
import time
from datetime import UTC, date, datetime
from html.parser import HTMLParser

import requests

from hifi.knowledge.schemas import FilingDocument

logger = logging.getLogger(__name__)

_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
_MIN_INTERVAL = 0.125  # 8 req/s, below 10/s EDGAR limit

# CIK numbers for Phase 7 baseline tickers
TICKER_CIKS: dict[str, str] = {
    "AAPL": "0000320193",
    "JPM": "0000019617",
    "XOM": "0000034088",
}

# Target sections by filing type: item_number -> section_name
_TEN_K_SECTIONS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "7": "MD&A",
}

_TEN_Q_SECTIONS: dict[str, str] = {
    "2": "MD&A",
}


class _TextExtractor(HTMLParser):
    """Collect plain text from HTML, skipping script/style/iXBRL-header content."""

    _SKIP_TAGS = frozenset(["script", "style"])
    _BLOCK_TAGS = frozenset(
        ["p", "div", "tr", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6"]
    )

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS or t == "ix:header":
            self._skip_depth += 1
        elif t in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS or t == "ix:header":
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        unescaped = _html_module.unescape(raw)
        # Normalise: strip each line, collapse runs of blank lines
        lines = [line.strip() for line in unescaped.splitlines()]
        cleaned: list[str] = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank:
                    cleaned.append("")
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False
        return "\n".join(cleaned).strip()


# Match "ITEM N" or "ITEM 1A" section headers at the start of a line
_ITEM_RE = re.compile(
    r"(?:^|\n)\s*ITEM\s+([0-9]+[A-Z]?)[\s.\,:]",
    re.IGNORECASE,
)


def _extract_items(text: str) -> dict[str, str]:
    """
    Find all ITEM sections in cleaned text. Returns {item_number: text_content}.

    item_number is normalised to uppercase (e.g. "1A", "7").
    When the same item number appears multiple times (TOC and body), the
    occurrence with the longest content wins — that is always the body section.
    """
    matches = list(_ITEM_RE.finditer(text))
    # Collect (content_start, content_end) pairs per item number
    by_item: dict[str, list[tuple[int, int]]] = {}
    for i, m in enumerate(matches):
        item_no = m.group(1).upper()
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if item_no not in by_item:
            by_item[item_no] = []
        by_item[item_no].append((content_start, content_end))

    items: dict[str, str] = {}
    for item_no, positions in by_item.items():
        # Take the occurrence with the most characters (body > TOC entry)
        best_start, best_end = max(positions, key=lambda p: p[1] - p[0])
        content = text[best_start:best_end].strip()
        if content:
            items[item_no] = content
    return items


class EdgarFetcher:
    """
    SEC EDGAR API client.

    Downloads 10-K, 10-Q, and 8-K filings for a given ticker/CIK.
    Rate-limited to <= 8 requests/second per EDGAR policy (0.125 s minimum gap).
    """

    # Override with HIFI_EDGAR_USER_AGENT env var. EDGAR requires a valid
    # "Name email@domain.com" — localhost domains are blocked by www.sec.gov.
    DEFAULT_USER_AGENT = "HiFi Research admin@hifi-research.org"

    def __init__(self) -> None:
        self._session = requests.Session()
        user_agent = os.environ.get("HIFI_EDGAR_USER_AGENT", self.DEFAULT_USER_AGENT)
        self._session.headers.update({"User-Agent": user_agent})
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if needed to keep inter-request gap >= _MIN_INTERVAL."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _get_json(self, url: str) -> dict:
        """GET a URL, parse and return JSON. Enforces rate limit."""
        self._rate_limit()
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get_text(self, url: str) -> str:
        """GET a URL, return response text. Enforces rate limit."""
        self._rate_limit()
        resp = self._session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.text

    # ------------------------------------------------------------------
    # EDGAR API methods
    # ------------------------------------------------------------------

    def get_submissions(self, cik: str) -> dict:
        """
        Fetch company filing history from EDGAR submissions API.

        CIK is zero-padded to 10 digits per EDGAR requirements.

        Returns the full submissions JSON including the ``filings.recent``
        arrays (accessionNumber, form, filingDate, reportDate, etc.).
        """
        padded = cik.lstrip("0").zfill(10)
        url = f"{_SUBMISSIONS_BASE}/CIK{padded}.json"
        logger.debug("Fetching submissions: %s", url)
        return self._get_json(url)

    def get_filing_index(self, cik: str, accession_number: str) -> dict:
        """
        Fetch the index JSON for a specific accession number.

        Returns a dict whose ``directory.item`` list identifies document files.
        The accession number is normalised (dashes removed) for URL construction.
        """
        acc_no_dashes = accession_number.replace("-", "")
        cik_plain = cik.lstrip("0")
        url = f"{_ARCHIVES_BASE}/{cik_plain}/{acc_no_dashes}/index.json"
        logger.debug("Fetching filing index: %s", url)
        return self._get_json(url)

    def get_filing_document(
        self,
        cik: str,
        accession_number: str,
        filename: str,
    ) -> str:
        """
        Download the primary filing document. Returns raw HTML text.
        """
        acc_no_dashes = accession_number.replace("-", "")
        cik_plain = cik.lstrip("0")
        url = f"{_ARCHIVES_BASE}/{cik_plain}/{acc_no_dashes}/{filename}"
        logger.debug("Fetching filing document: %s", url)
        return self._get_text(url)

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def extract_text_sections(
        self,
        raw_html: str,
        filing_type: str,
    ) -> dict[str, str]:
        """
        Extract named sections from SEC filing HTML.

        Strips all HTML tags with stdlib html.parser. Then locates EDGAR
        standard ITEM headers and extracts the text for each target section.

        Filing-type mapping:
          10-K: Business (Item 1), Risk Factors (Item 1A), MD&A (Item 7)
          10-Q: MD&A (Item 2)
          8-K:  Earnings Release (full body — 8-Ks have no standard ITEM labels)

        If no named sections are found, returns full body text under "Full Text".
        """
        extractor = _TextExtractor()
        extractor.feed(raw_html)
        clean_text = extractor.get_text()

        ft = filing_type.upper().replace("/A", "")  # treat 10-K/A same as 10-K

        if ft == "8-K":
            return {"Earnings Release": clean_text}

        if ft == "10-K":
            target_map = _TEN_K_SECTIONS
        elif ft == "10-Q":
            target_map = _TEN_Q_SECTIONS
        else:
            logger.warning("Unknown filing type %s; returning full text", filing_type)
            return {"Full Text": clean_text}

        items = _extract_items(clean_text)
        sections: dict[str, str] = {}
        for item_no, section_name in target_map.items():
            content = items.get(item_no.upper(), "")
            if content:
                sections[section_name] = content
            else:
                logger.debug(
                    "Section %s (Item %s) not found in %s filing",
                    section_name,
                    item_no,
                    filing_type,
                )

        if not sections:
            logger.warning(
                "No named sections extracted from %s; falling back to full text",
                filing_type,
            )
            sections["Full Text"] = clean_text

        return sections

    # ------------------------------------------------------------------
    # Private orchestration helpers
    # ------------------------------------------------------------------

    def _scan_filing_batch(
        self,
        batch: dict,
        filing_type: str,
        as_of_date: date,
    ) -> list[tuple[date, date, str]]:
        """Scan one batch of filings (recent or paginated) for matching entries."""
        forms = batch.get("form", [])
        acc_numbers = batch.get("accessionNumber", [])
        filing_dates_raw = batch.get("filingDate", [])
        report_dates_raw = batch.get("reportDate", [])

        ft_norm = filing_type.upper()
        candidates: list[tuple[date, date, str]] = []

        for i, form in enumerate(forms):
            if form.upper() != ft_norm:
                continue
            try:
                filed = date.fromisoformat(filing_dates_raw[i])
            except (ValueError, IndexError):
                continue
            if filed > as_of_date:
                continue
            try:
                period_str = (
                    report_dates_raw[i] if i < len(report_dates_raw) else ""
                )
                period = date.fromisoformat(period_str) if period_str else filed
            except ValueError:
                period = filed
            candidates.append((filed, period, acc_numbers[i]))

        return candidates

    def _find_accession_number(
        self,
        submissions: dict,
        filing_type: str,
        as_of_date: date,
    ) -> tuple[str, date, date]:
        """
        Find the most recent accession number for a filing type at or before as_of_date.

        Searches ``filings.recent`` first. High-frequency filers (e.g. JPMorgan)
        may have their annual/quarterly filings pushed beyond the ~40-entry
        ``recent`` window; for those, paginated history files listed in
        ``filings.files`` are fetched until a match is found.

        Returns (accession_number, period_of_report, filed_date).
        Raises ValueError if no matching filing is found.
        """
        filings = submissions.get("filings", {})
        recent = filings.get("recent", {})

        candidates = self._scan_filing_batch(recent, filing_type, as_of_date)

        # If no match in recent, walk paginated history files
        if not candidates:
            extra_files = filings.get("files", [])
            for file_entry in extra_files:
                fname = file_entry.get("name", "")
                if not fname:
                    continue
                page_url = f"{_SUBMISSIONS_BASE}/{fname}"
                logger.debug("Fetching paginated submissions: %s", page_url)
                try:
                    page_data = self._get_json(page_url)
                except Exception as exc:
                    logger.warning("Failed to fetch %s: %s", page_url, exc)
                    continue
                page_candidates = self._scan_filing_batch(
                    page_data, filing_type, as_of_date
                )
                candidates.extend(page_candidates)
                if candidates:
                    # Found enough; stop paginating (files are newest-first)
                    break

        if not candidates:
            raise ValueError(
                f"No {filing_type} filing found on or before {as_of_date}"
            )

        # Most recently filed first
        candidates.sort(key=lambda x: x[0], reverse=True)
        filed_date, period_of_report, acc_number = candidates[0]
        return acc_number, period_of_report, filed_date

    def _find_primary_document(self, index_data: dict) -> str:
        """
        Find the primary HTML document filename from a filing index.

        Raises ValueError if no HTML document is found.
        """
        items = index_data.get("directory", {}).get("item", [])

        # Prefer items explicitly typed with the form type
        for item in items:
            name = item.get("name", "")
            item_type = item.get("type", "")
            if item_type and item_type.upper() in (
                "10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"
            ):
                return name

        # Fall back to first .htm/.html file that is not an index
        for item in items:
            name = item.get("name", "")
            if name.endswith((".htm", ".html")) and "index" not in name.lower():
                return name

        # Last resort: any .htm file
        for item in items:
            name = item.get("name", "")
            if name.endswith((".htm", ".html")):
                return name

        raise ValueError(
            f"No primary HTML document found in filing index. "
            f"First items: {items[:3]}"
        )

    # ------------------------------------------------------------------
    # High-level entrypoint
    # ------------------------------------------------------------------

    def fetch_filing(
        self,
        ticker: str,
        cik: str,
        filing_type: str,
        as_of_date: date,
    ) -> FilingDocument:
        """
        Find, download, and extract the most recent filing of the given type.

        Orchestrates: get_submissions → _find_accession_number
            → get_filing_index → _find_primary_document
            → get_filing_document → extract_text_sections
            → FilingDocument

        Parameters
        ----------
        ticker : str
            Company ticker (e.g. "AAPL").
        cik : str
            SEC CIK number (zero-padded 10-digit string or raw integer string).
        filing_type : str
            "10-K", "10-Q", or "8-K".
        as_of_date : date
            Find the most recent filing filed on or before this date.

        Returns
        -------
        FilingDocument
            Parsed filing with extracted text sections.
        """
        fetched_at = datetime.now(UTC)

        submissions = self.get_submissions(cik)
        acc_number, period_of_report, filed_date = self._find_accession_number(
            submissions, filing_type, as_of_date
        )
        logger.info(
            "%s %s: found accession %s (filed %s, period %s)",
            ticker, filing_type, acc_number, filed_date, period_of_report,
        )

        index_data = self.get_filing_index(cik, acc_number)
        primary_doc = self._find_primary_document(index_data)
        logger.info("%s %s: primary document: %s", ticker, filing_type, primary_doc)

        raw_html = self.get_filing_document(cik, acc_number, primary_doc)
        sections = self.extract_text_sections(raw_html, filing_type)
        logger.info(
            "%s %s: extracted %d sections: %s",
            ticker, filing_type, len(sections), list(sections.keys()),
        )

        acc_no_dashes = acc_number.replace("-", "")
        cik_plain = cik.lstrip("0")
        source_url = f"{_ARCHIVES_BASE}/{cik_plain}/{acc_no_dashes}/{primary_doc}"

        return FilingDocument(
            ticker=ticker,
            cik=cik,
            filing_type=filing_type,
            accession_number=acc_number,
            period_of_report=period_of_report,
            filed_date=filed_date,
            sections=sections,
            source_url=source_url,
            fetched_at=fetched_at,
        )
