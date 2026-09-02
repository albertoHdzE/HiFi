"""Bringing the local stores up to date without losing what they already hold.

Three sources, one merge rule: union the periods, let fresh values win on
overlap, never drop existing history. That rule exists because the acquisition
scripts do the opposite — ``acquire_fundamentals`` writes
``combined.to_parquet(...)``, and yfinance serves only five to seven quarters,
so a plain re-run buys the newest quarter at the cost of the oldest ones. Silent
history loss in exactly the data the TTM ratios and the walk-forward depend on.

Two failures shaped what is here.

DJ-133a: 91 of 97 tickers were missing a quarter that had been public since late
July while everyone sat at 2026-03-31. The point-in-time gate meant this was
safe rather than lookahead, but every agent was reading a quarter-old book.

DJ-133c: the first version of the macro refresh wrote its files with a plain
``df.to_parquet()``. ``write_macro`` embeds series_id, name, frequency, unit and
provenance in the Parquet *schema metadata*, and ``read_macro`` raises without
it — so all seven series became unreadable, ``_load_all_macro`` swallowed the
per-file exception and returned ``{}``, and the macro agent reported
NO_MACRO_DATA and voted Hold on 193 of 194 passes. The DJ-120 pattern
reproduced by the very script written to prevent staleness: an agent blinded by
a data-path change, rendering the blindness as a confident decision. Every macro
write is now round-tripped through ``read_macro`` before it is committed.

Every write is hashed into the DatasetRegistry and, for OHLCV, scored by the
DataQualityChecker. Both existed since Phase 1 and neither was called by
anything, which is why the David evaluation scored §4.5 (reproducibility) as
half-met: a registry nothing writes to records nothing.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from hifi.data.quality import DataQualityChecker
from hifi.data.versioning import DatasetRegistry

logger = logging.getLogger(__name__)

_STATEMENTS = ("quarterly_income_stmt", "quarterly_balance_sheet", "quarterly_cashflow")

#: FRED series the agents actually read, with their native publication cadence.
SERIES = {
    "VIXCLS": "daily",
    "GS10": "monthly",
    "GS2": "monthly",
    "FEDFUNDS": "monthly",
    "CPIAUCSL": "monthly",
    "UNRATE": "monthly",
    "A191RL1Q225SBEA": "quarterly",
}

#: Below this fraction of expected trading days an OHLCV history is a defect
#: rather than thin data.
#:
#: NOT the 0.98 that DataQualityChecker's own docstring calls acceptable. That
#: figure is unreachable here and would fire on every ticker, every refresh.
#: Completeness counts weekdays and does not subtract market holidays, so a
#: perfect history scores below 100% by exactly the holiday rate. Measured on
#: AAPL, 2004-01-02 to 2026-09-01: 5,913 weekdays, 5,702 bars, 211 missing —
#: 9.31 per year against the 9 US market holidays per year (10 since Juneteenth
#: in 2022), with zero gaps and zero anomalies detected. The same 96.4% appears
#: for MSFT, JPM, XOM and SPY.
#:
#: 0.95 leaves roughly four unexplained sessions a year of headroom over the
#: holiday floor. A threshold that fires on healthy data is worse than none: it
#: teaches the reader to ignore the warning, which is how DJ-120 stayed
#: invisible for a month.
#:
#: ``gap_count`` is the sharper instrument and is reported alongside: it counts
#: runs of more than five consecutive missing weekdays, which holidays never
#: produce and an outage always does.
MIN_COMPLETENESS = 0.95


def _fetch(ticker: str):
    """Fresh quarterly statements from yfinance, period-indexed, or None."""
    import pandas as pd
    import yfinance as yf

    t = yf.Ticker(ticker)
    frames = []
    for attr in _STATEMENTS:
        df = getattr(t, attr, None)
        if df is not None and not df.empty:
            frames.append(df.T)
    if not frames:
        return None
    # sort=False is explicit: the frames share a period index and we sort once
    # below, so letting pandas sort per-concat only costs work and a warning.
    combined = pd.concat(frames, axis=1, sort=False)
    combined.index = pd.to_datetime(combined.index)
    # Duplicate column labels appear when two statements share a line item;
    # keeping the first occurrence avoids a reindex explosion on merge.
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.sort_index()



def refresh_ticker(ticker: str, data_dir: Path, quiet: bool = False) -> dict:
    """Merge fresh statements into the cached parquet. Returns a change report."""
    import pandas as pd

    out_dir = data_dir / "fundamentals" / ticker
    path = out_dir / "quarterly.parquet"

    existing = None
    if path.exists():
        try:
            existing = pd.read_parquet(path)
        except Exception as exc:
            logger.error("%s: existing parquet unreadable (%s); refusing to clobber", ticker, exc)
            return {"ticker": ticker, "status": "unreadable"}

    try:
        fresh = _fetch(ticker)
    except Exception as exc:
        logger.error("%s: fetch failed: %s", ticker, exc)
        return {"ticker": ticker, "status": "fetch_failed"}

    if fresh is None or fresh.empty:
        logger.warning("%s: yfinance returned no statements", ticker)
        return {"ticker": ticker, "status": "empty_response"}

    if existing is None or existing.empty:
        merged, added, kept = fresh, list(fresh.index), []
    else:
        added = [p for p in fresh.index if p not in set(existing.index)]
        kept = [p for p in existing.index if p not in set(fresh.index)]
        # Fresh wins on overlap so restatements propagate; union of columns so a
        # newly reported line item is not dropped.
        merged = fresh.combine_first(existing)
        merged = merged.reindex(sorted(set(existing.index) | set(fresh.index)))

    out_dir.mkdir(parents=True, exist_ok=True)
    merged = merged.sort_index()
    merged.to_parquet(path)
    _register(f"fundamentals/{ticker}", "yfinance", path,
              str(merged.index.min().date()), str(merged.index.max().date()))

    report = {
        "ticker": ticker,
        "status": "ok",
        "periods_before": 0 if existing is None else len(existing),
        "periods_after": len(merged),
        "added": [str(p.date()) for p in added],
        "preserved_only_locally": [str(p.date()) for p in kept],
        "latest": str(merged.index.max().date()),
    }
    if not quiet:
        logger.info(
            "%-6s %d -> %d quarters, latest %s, added %s%s",
            ticker, report["periods_before"], report["periods_after"],
            report["latest"], report["added"] or "none",
            f", kept {len(kept)} local-only" if kept else "",
        )
    return report



def _load_env(root: Path) -> None:
    """Read FRED_API_KEY from .env when it is not already in the environment."""
    if os.environ.get("FRED_API_KEY"):
        return
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.startswith("FRED_API_KEY="):
            os.environ["FRED_API_KEY"] = line.split("=", 1)[1].strip()
            return



def refresh_series(series_id: str, data_dir: Path, fred, quiet: bool = False) -> dict:
    import pandas as pd

    from hifi.data.macro import SERIES_METADATA
    from hifi.data.schemas import MacroDataset, MacroIndicator, ProvenanceRecord
    from hifi.data.storage import read_macro, write_macro

    path = data_dir / "macro" / f"{series_id}.parquet"
    existing = None
    if path.exists():
        try:
            existing = pd.read_parquet(path)[["date", "value"]]
        except Exception as exc:
            logger.error("%s: existing parquet unreadable (%s); refusing to clobber",
                         series_id, exc)
            return {"series": series_id, "status": "unreadable"}

    try:
        raw = fred.get_series(series_id)
    except Exception as exc:
        logger.error("%s: FRED fetch failed: %s", series_id, exc)
        return {"series": series_id, "status": "fetch_failed"}

    fresh = (
        raw.rename("value").rename_axis("date").reset_index().dropna(subset=["value"])
    )
    fresh["date"] = pd.to_datetime(fresh["date"])

    if existing is None or existing.empty:
        merged = fresh
    else:
        existing["date"] = pd.to_datetime(existing["date"])
        # Fresh last so it wins on duplicate dates: FRED revises published values.
        merged = (
            pd.concat([existing, fresh], ignore_index=True)
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

    before = 0 if existing is None else len(existing)
    prev_latest = None if existing is None or existing.empty else existing["date"].max()

    meta = SERIES_METADATA.get(series_id, {})
    fetched_at = datetime.now(UTC)
    merged["date"] = pd.to_datetime(merged["date"])
    dataset = MacroDataset(
        series_id=series_id,
        name=meta.get("name", series_id),
        frequency=meta.get("frequency", "unknown"),
        unit=meta.get("unit", "unknown"),
        observations=[
            MacroIndicator(series_id=series_id, date=r.date.date(), value=float(r.value))
            for r in merged.itertuples()
        ],
        source="FRED",
        fetched_at=fetched_at,
        date_from=merged["date"].min().date(),
        date_to=merged["date"].max().date(),
        provenance=ProvenanceRecord(
            source="FRED",
            fetched_at=fetched_at,
            parameters={"series_id": series_id},
        ),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    write_macro(dataset, path)
    _register(f"macro/{series_id}", "FRED", path,
              str(merged['date'].min().date()), str(merged['date'].max().date()))

    # Round-trip before declaring success. The defect this guards against was
    # invisible at write time and only surfaced as an agent voting Hold on
    # everything three days later.
    try:
        back = read_macro(path)
        if back.series_id != series_id or len(back.observations) != len(merged):
            raise ValueError(
                f"round trip mismatch: {back.series_id} "
                f"{len(back.observations)} vs {len(merged)}"
            )
    except Exception as exc:
        logger.error("%s: WROTE AN UNREADABLE FILE (%s)", series_id, exc)
        return {"series": series_id, "status": "unreadable_after_write"}

    report = {
        "series": series_id,
        "status": "ok",
        "rows_before": before,
        "rows_after": len(merged),
        "was": None if prev_latest is None else str(prev_latest.date()),
        "now": str(merged["date"].max().date()),
    }
    if not quiet:
        logger.info(
            "%-18s %5d -> %5d rows, latest %s -> %s",
            series_id, before, len(merged), report["was"], report["now"],
        )
    return report



# ---------------------------------------------------------------------------
# Provenance and quality
# ---------------------------------------------------------------------------


def _register(dataset_id: str, source: str, path: Path,
              date_from: str, date_to: str) -> None:
    """Record the written file's SHA-256 in the dataset registry (fail-open).

    A hash says what is in a file; a timestamp only says when it was touched.
    FRED revises published history in place, so two refreshes that "changed
    nothing" but produce different hashes are the only way to notice that the
    provider rewrote the past underneath a completed experiment.

    Fail-open on purpose: the registry is an audit trail, not a precondition.
    Losing an entry is a gap in the record; refusing to refresh because the
    audit trail is unwritable would be a gap in the data itself, which is worse.
    """
    try:
        DatasetRegistry().register(
            dataset_id=dataset_id, source=source,
            date_from=date_from, date_to=date_to, file_path=path,
        )
    except Exception as exc:
        logger.warning("Could not register %s in the dataset registry: %s",
                       dataset_id, exc)


def check_ohlcv_quality(tickers: list[str], data_dir: Path,
                        quiet: bool = False) -> list[dict]:
    """Score each ticker's bar completeness after a refresh.

    Answers a different question from the runner's coverage gate. That gate asks
    how many tickers resolve to a file at all — the DJ-120 failure, where 83 of
    98 resolved to nothing. This asks how complete each resolved file is, which
    is the quieter half of the same defect: the 15 tickers that *did* resolve
    were being analysed on 2023 prices, and a per-file completeness score is
    what would have said so.

    Completeness counts weekdays and does not subtract market holidays, so it
    reads 2-3 percentage points below reality by construction.
    """
    # The loader lives in the MCP server because that is the only other place
    # a validated OHLCVDataset is built from the store; importing it here keeps
    # one parser rather than a second that could disagree about the layout.
    from hifi.mcp.financial_server import _load_ohlcv

    checker = DataQualityChecker()
    poor = []
    for ticker in tickers:
        try:
            dataset = _load_ohlcv(ticker)
        except Exception as exc:
            poor.append({"ticker": ticker, "completeness": None, "error": str(exc)[:120]})
            continue
        report = checker.check(dataset)
        if report.completeness < MIN_COMPLETENESS or report.gap_count:
            poor.append({"ticker": ticker, "completeness": report.completeness,
                         "gaps": report.gap_count, "anomalies": report.anomaly_count})
            if not quiet:
                logger.warning("%s: completeness %.1f%% below %.0f%% (%d gap(s))",
                               ticker, report.completeness * 100,
                               MIN_COMPLETENESS * 100, report.gap_count)
    return poor
