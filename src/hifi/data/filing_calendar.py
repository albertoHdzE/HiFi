"""Point-in-time filing calendar: when each fiscal period actually became public.

Why this exists (DJ-133a)
-------------------------
``data/fundamentals/<TICKER>/quarterly.parquet`` is indexed by *fiscal period
end*, which is not the date the numbers were knowable. AEP's quarter ending
2026-03-31 was not filed with the SEC until weeks later; treating the period
end as the availability date would let an agent read results before they were
published. That is lookahead bias, and for a study whose whole claim rests on
out-of-sample discipline it is a worse defect than the blindness it replaces.

The honest gate is the actual EDGAR ``filingDate``, which the SEC publishes in
the submissions API alongside ``reportDate``. A fixed lag (45 days, 90 days)
would be an arbitrary constant standing in for a fact we can simply look up --
exactly the class of unexamined knob that produced DJ-122 and DJ-131. So we
look it up, cache it, and gate on it.

Cadence
-------
The calendar is a slowly-changing artefact: one new row per ticker per quarter.
It is fetched once and cached to Parquet; refresh it when a new quarter's
filings land. The fetch is rate-limited to EDGAR's published limit by
``EdgarFetcher``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CALENDAR_PATH",
    "build_filing_calendar",
    "latest_filed_period",
    "load_filing_calendar",
    "ticker_to_cik",
]

DEFAULT_CALENDAR_PATH = "fundamentals/filing_calendar.parquet"

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_PERIODIC_FORMS = ("10-Q", "10-K")

#: Periodic filings to collect per ticker before we stop paginating. Eight
#: covers two years, which is more than the four quarters a TTM needs plus
#: slack. This bound matters: ``filings.recent`` holds only the most recent
#: filings of ALL types, so a heavy 8-K filer (every large bank) can have its
#: 10-Qs pushed out of that window entirely. Reading only ``recent`` gave JPM,
#: BAC, GS, MS, BLK, PEP and COST fewer than four filed quarters, which silently
#: dropped every TTM-based ratio (P/E, P/S, ROE) for exactly those names while
#: the balance-sheet ratios kept working -- a partial blindness far harder to
#: notice than the total blindness of DJ-133a.
_MIN_PERIODIC_FILINGS = 8


def _user_agent() -> str:
    """EDGAR rejects requests without a contact string; reuse the client's."""
    from hifi.data.edgar import EdgarFetcher

    return os.environ.get("HIFI_EDGAR_USER_AGENT", EdgarFetcher.DEFAULT_USER_AGENT)


def ticker_to_cik(tickers: list[str] | None = None) -> dict[str, str]:
    """Map ticker -> zero-padded CIK from the SEC's published index.

    Returns only the tickers that SEC actually lists. A ticker missing here has
    no EDGAR filings under that symbol, which the caller must treat as "no
    fundamentals available" rather than silently substituting another company.
    """
    req = urllib.request.Request(_COMPANY_TICKERS_URL, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed SEC URL
        raw = json.loads(resp.read())

    wanted = set(tickers) if tickers else None
    out: dict[str, str] = {}
    for entry in raw.values():
        sym = str(entry["ticker"]).upper()
        if wanted is None or sym in wanted:
            out[sym] = str(entry["cik_str"]).zfill(10)
    return out


def _scan_periodic(ticker: str, batch: dict) -> list[dict]:
    """Extract 10-Q/10-K rows from one EDGAR submissions batch."""
    import pandas as pd

    forms = batch.get("form", []) or []
    filed = batch.get("filingDate", []) or []
    reported = batch.get("reportDate", []) or []
    accessions = batch.get("accessionNumber", []) or []

    out: list[dict] = []
    for i, form in enumerate(forms):
        if form not in _PERIODIC_FORMS:
            continue
        # A filing with no reportDate cannot be tied to a fiscal period and
        # therefore cannot gate anything; skipping is the safe choice.
        if i >= len(reported) or not reported[i] or i >= len(filed) or not filed[i]:
            continue
        out.append({
            "ticker": ticker,
            "form": form,
            "period_end": pd.Timestamp(reported[i]),
            "filing_date": pd.Timestamp(filed[i]),
            "accession": accessions[i] if i < len(accessions) else "",
        })
    return out


def build_filing_calendar(
    tickers: list[str],
    data_dir: str | Path = "data",
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch and cache (ticker, form, period_end, filing_date) for periodic filings.

    Only 10-Q and 10-K are collected: those are the forms that carry the
    financial statements behind the ratios. 8-K and proxy statements move
    prices but do not restate the quarterly numbers this calendar gates.
    """
    from hifi.data.edgar import EdgarFetcher

    client = EdgarFetcher()
    ciks = ticker_to_cik(tickers)
    missing = sorted(set(tickers) - set(ciks))
    if missing:
        logger.warning("No EDGAR CIK for %d ticker(s): %s", len(missing), ", ".join(missing))

    rows: list[dict] = []
    for ticker in sorted(ciks):
        try:
            subs = client.get_submissions(ciks[ticker])
        except Exception as exc:
            logger.error("Submissions fetch failed for %s: %s", ticker, exc)
            continue

        filings = subs.get("filings") or {}
        found = _scan_periodic(ticker, filings.get("recent") or {})

        # ``recent`` is capped and mixes all form types, so walk the paginated
        # history shards until we have enough periodic filings (see
        # _MIN_PERIODIC_FILINGS). Shards are newest-first.
        if len(found) < _MIN_PERIODIC_FILINGS:
            for entry in filings.get("files", []):
                name = entry.get("name", "")
                if not name:
                    continue
                try:
                    page = client._get_json(f"{_SUBMISSIONS_BASE}/{name}")
                except Exception as exc:
                    logger.warning("Failed to fetch shard %s for %s: %s", name, ticker, exc)
                    continue
                found.extend(_scan_periodic(ticker, page))
                if len(found) >= _MIN_PERIODIC_FILINGS:
                    break

        if len(found) < 4:
            logger.warning(
                "%s: only %d periodic filing(s) found; TTM ratios will be unavailable",
                ticker, len(found),
            )
        rows.extend(found)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("Filing calendar is empty -- no filings collected")
        return df

    df = df.sort_values(["ticker", "period_end", "filing_date"]).reset_index(drop=True)

    target = Path(out_path) if out_path else Path(data_dir) / DEFAULT_CALENDAR_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    logger.info(
        "Filing calendar: %d filings across %d tickers -> %s",
        len(df), df["ticker"].nunique(), target,
    )
    return df


def load_filing_calendar(
    data_dir: str | Path = "data",
    path: str | Path | None = None,
) -> pd.DataFrame | None:
    """Load the cached calendar, or None when it has not been built yet.

    None is returned rather than raising so a caller can degrade explicitly and
    loudly, instead of a missing cache silently becoming a missing agent.
    """
    target = Path(path) if path else Path(data_dir) / DEFAULT_CALENDAR_PATH
    if not target.exists():
        return None
    try:
        return pd.read_parquet(target)
    except Exception as exc:
        logger.error("Filing calendar at %s is unreadable: %s", target, exc)
        return None


def latest_filed_period(
    ticker: str,
    as_of_date: str | date | pd.Timestamp,
    calendar: pd.DataFrame,
) -> pd.Timestamp | None:
    """Latest fiscal period whose filing was public on ``as_of_date``.

    The gate is ``filing_date <= as_of_date``, never ``period_end <=
    as_of_date``: the second is the lookahead this module exists to prevent.
    Returns None when nothing had been filed yet, which is a real state for a
    recently listed company and must not be papered over with a stale period.
    """
    if calendar is None or calendar.empty:
        return None
    as_of = pd.Timestamp(as_of_date)
    rows = calendar[
        (calendar["ticker"] == ticker.upper()) & (calendar["filing_date"] <= as_of)
    ]
    if rows.empty:
        return None
    return pd.Timestamp(rows["period_end"].max())
