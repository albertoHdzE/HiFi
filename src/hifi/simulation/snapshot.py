"""
FundamentalsSnapshot builders for the agent ensemble.

``build_pointintime_snapshot`` is the production path. ``build_minimal_snapshot``
is the Phase 15 walk-forward scaffold, retained for that harness alone.

DJ-133a -- the scaffold was the production path for the entire live record
--------------------------------------------------------------------------
``build_minimal_snapshot`` sets every financial field to None by design, and
says so in its docstring. It was written for the walk-forward, where the
reasoning below applies. But ``run_phase15_orchestrator`` calls
``run_agent_pass`` without ``snapshot_json``, and ``agent_executor`` fell back
to the minimal builder -- so the *live* fundamental agent received an empty
snapshot on every ticker, every night.

The measured consequence, 2026-08-24 to 08-27: ``pe``, ``pb``, ``ps``,
``ev_ebitda``, ``roe`` and ``roa`` were absent on 97/97 tickers on all four
days, and the agent whose sole remit is valuation voted Hold on 97/97 for three
consecutive days. Real quarterly financials sat unread in
``data/fundamentals/<TICKER>/quarterly.parquet`` the whole time.

This is DJ-124's pattern one layer down: an artefact scoped to one purpose left
wired into another, with nothing asserting the difference. The guard added here
is that a snapshot now carries its own provenance in ``source`` -- a blind
fundamental agent is visible in the stored record rather than being
indistinguishable from a working one.

Point-in-time discipline
------------------------
Fundamentals are gated on the actual EDGAR ``filingDate``, never on the fiscal
``period_end``: a quarter ending 2026-03-31 is not knowable on 2026-04-01. See
``hifi.data.filing_calendar``.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Parquet column -> FundamentalsSnapshot field. The parquet carries 183 raw
# statement lines under yfinance's names; these are the six the ratio engine
# consumes. Left explicit rather than fuzzy-matched so a renamed upstream
# column fails loudly instead of silently reintroducing DJ-133a.
_FIELD_MAP = {
    "revenue": "Total Revenue",
    "net_income": "Net Income",
    "total_assets": "Total Assets",
    "total_liabilities": "Total Debt",
    "total_equity": "Stockholders Equity",
    "eps": "Diluted EPS",
    # DJ-134: present in the statements all along, never carried through.
    "ebitda": "EBITDA",
    "cash_and_equivalents": "Cash And Cash Equivalents",
    "current_assets": "Current Assets",
    "current_liabilities": "Current Liabilities",
    "cost_of_revenue": "Cost Of Revenue",
    "operating_income": "Operating Income",
}

#: Fields that are flows (summed over four quarters) rather than stocks.
#: Getting this wrong is how P/S comes out four times too high, and how an
#: EV/EBITDA multiple silently quadruples.
_FLOW_FIELDS = frozenset({
    "revenue", "net_income", "eps", "ebitda", "cost_of_revenue", "operating_income",
})


def _cell_float(value: Any) -> float | None:
    """A numeric cell as a float, or None when it is absent (DJ-142).

    ``DataFrame.at[period, col]`` is typed as a union spanning str, bytes, date,
    timedelta and more, because a frame can hold anything. These frames hold
    reported financials, so the values are numeric or missing — but "or missing"
    is the whole point: an unreported quarter must stay None and propagate as an
    absence, never become 0.0. A zero here is a company that reported nothing
    looking like a company that reported nothing *of value*, which is a
    different claim and one the fundamental agent would act on.
    """
    import pandas as pd  # noqa: PLC0415

    if value is None or pd.isna(value):
        return None
    return float(value)


def _close_on_or_before(ticker: str, as_of_date: str, data_dir: str | Path) -> float | None:
    """Unadjusted close on ``as_of_date``, or the last session before it."""
    import pandas as pd

    path = Path(data_dir) / "market" / ticker / "ohlcv.parquet"
    if not path.exists():
        return None
    try:
        bars = pd.read_parquet(path)
        upto = bars[bars.index <= pd.Timestamp(as_of_date)]
        if upto.empty:
            return None
        return float(upto["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("Price lookup failed for %s @ %s: %s", ticker, as_of_date, exc)
        return None


#: Days a local statement period may differ from EDGAR's reportDate and still
#: be the same fiscal quarter. The parquet normalises to the calendar quarter
#: end; EDGAR carries the true fiscal close. Apple's 13-week quarter ended
#: 2026-03-28 (3 days out) and PepsiCo's 4-4-5 retail quarter ended 2026-06-13
#: (17 days out), so a tight tolerance silently drops exactly the companies
#: with non-calendar fiscal years. Quarters are ~91 days apart, so anything
#: below ~45 cannot confuse two adjacent periods; 25 leaves that margin
#: intact while covering every 4-4-5 calendar in the universe.
_PERIOD_MATCH_TOLERANCE_DAYS = 25


def _match_periods(local_periods: list, published) -> dict:
    """Map each local statement period to the filing date that made it public.

    Returns only periods that were actually filed by the caller's cut-off, so
    an unmatched period is dropped rather than assumed available.
    """
    import pandas as pd

    out: dict = {}
    if published.empty:
        return out
    for period in local_periods:
        p = pd.Timestamp(period)
        gap = (published["period_end"] - p).abs()
        within = published[gap <= pd.Timedelta(days=_PERIOD_MATCH_TOLERANCE_DAYS)]
        if within.empty:
            continue
        # Nearest period, not latest filing: with a wide tolerance an amended
        # or adjacent filing could otherwise win and attach the wrong date.
        nearest = within.loc[gap[within.index].idxmin()]
        out[p] = pd.Timestamp(nearest["filing_date"])
    return out


def build_pointintime_snapshot(
    ticker: str,
    as_of_date: str,
    data_dir: str | Path = "data",
) -> str | None:
    """Serialise a FundamentalsSnapshot from data public on ``as_of_date``.

    Returns None when no filing had been published yet, or when the local
    fundamentals or filing calendar are absent. None is deliberate: the caller
    must decide what to do about a blind agent, loudly. Returning an
    all-None snapshot instead is precisely the DJ-133a failure.

    Accounting treatment
    --------------------
    Flow quantities (revenue, net income, EPS) are summed over the trailing
    four filed quarters; stock quantities (assets, equity, debt) are taken from
    the most recent filed quarter alone. Mixing the two conventions is the
    usual way P/S comes out four times too high: one quarter of revenue against
    a full market capitalisation.

    ``market_cap`` uses the close on ``as_of_date`` and the share count from
    the latest filed quarter, so it is a price the market actually printed.
    """
    import pandas as pd

    from hifi.data.filing_calendar import load_filing_calendar
    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

    root = Path(data_dir)
    fundamentals = root / "fundamentals" / ticker / "quarterly.parquet"
    if not fundamentals.exists():
        logger.warning("No fundamentals parquet for %s", ticker)
        return None

    calendar = load_filing_calendar(data_dir=root)
    if calendar is None or calendar.empty:
        logger.warning(
            "No filing calendar at %s; cannot establish what was public on %s. "
            "Run hifi.data.filing_calendar.build_filing_calendar.",
            root / "fundamentals" / "filing_calendar.parquet", as_of_date,
        )
        return None

    try:
        quarters = pd.read_parquet(fundamentals)
    except Exception as exc:
        logger.error("Unreadable fundamentals for %s: %s", ticker, exc)
        return None

    as_of = pd.Timestamp(as_of_date)
    published = calendar[
        (calendar["ticker"] == ticker.upper()) & (calendar["filing_date"] <= as_of)
    ]
    if published.empty:
        logger.warning("No filing published for %s on or before %s", ticker, as_of_date)
        return None

    # Join the two period conventions with a tolerance. yfinance normalises a
    # fiscal period to the calendar quarter end; EDGAR reports the true one.
    # Apple's Q2 2026 is 2026-03-28 at the SEC and 2026-03-31 in the parquet,
    # so an exact-match join silently blinds every company whose fiscal
    # calendar is not month-aligned -- reintroducing DJ-133a for a subset.
    filed_dates = _match_periods(sorted(quarters.index), published)
    available = sorted(filed_dates)
    if not available:
        logger.warning(
            "%s: no local statement period had been filed by %s "
            "(local periods end %s, latest filed period %s)",
            ticker, as_of_date,
            max(quarters.index).date() if len(quarters.index) else "none",
            published["period_end"].max().date(),
        )
        return None

    latest = available[-1]
    newest_first = list(reversed(available))

    # Select per field, not per row. The source emits a placeholder row for the
    # most recent quarter with most columns NaN (Oracle's 2026-05-31 carries an
    # EPS and nothing else), so taking "the latest row" wholesale discarded the
    # balance sheet for ORCL and MDT and the TTM for PEP and COST. Walking back
    # per field uses the most recent *reported* value, which is what a reader of
    # the filings would have.
    def _stock(field: str) -> float | None:
        col = _FIELD_MAP[field]
        if col not in quarters.columns:
            return None
        for period in newest_first:
            got = _cell_float(quarters.at[period, col])
            if got is not None:
                return got
        return None

    def _flow(field: str) -> float | None:
        """Sum the four most recent quarters that actually report the field."""
        col = _FIELD_MAP[field]
        if col not in quarters.columns:
            return None
        vals = []
        for period in newest_first:
            got = _cell_float(quarters.at[period, col])
            if got is not None:
                vals.append(got)
            if len(vals) == 4:
                return float(sum(vals))
        return None  # fewer than four reported quarters: no honest TTM

    def _nth_reported(field: str, n: int) -> float | None:
        """The n-th most recent quarter that actually reports ``field`` (0-based).

        Counting reported quarters rather than indexing by date keeps the
        year-ago comparison aligned when the source omits a quarter, which it
        routinely does for the oldest rows it serves.
        """
        col = _FIELD_MAP[field]
        if col not in quarters.columns:
            return None
        seen = 0
        for period in newest_first:
            got = _cell_float(quarters.at[period, col])
            if got is None:
                continue
            if seen == n:
                return got
            seen += 1
        return None

    shares = None
    if "Ordinary Shares Number" in quarters.columns:
        for period in newest_first:
            got = _cell_float(quarters.at[period, "Ordinary Shares Number"])
            if got is not None:
                shares = got
                break

    price = _close_on_or_before(ticker, as_of_date, root)
    market_cap = price * shares if (price is not None and shares) else None

    eps = _flow("eps")
    if eps is None:
        # EPS is sparsely reported in the source; derive it rather than lose
        # P/E entirely, and only when both inputs are genuinely present.
        ni_ttm = _flow("net_income")
        if ni_ttm is not None and shares:
            eps = ni_ttm / shares

    fetched_at = datetime.now(UTC)
    snap = FundamentalsSnapshot(
        ticker=ticker,
        period_end=latest.date(),
        revenue=_flow("revenue"),
        net_income=_flow("net_income"),
        total_assets=_stock("total_assets"),
        total_liabilities=_stock("total_liabilities"),
        total_equity=_stock("total_equity"),
        eps=eps,
        pe_ratio=None,  # computed by the ratio engine from price and eps
        market_cap=market_cap,
        # DJ-134
        ebitda=_flow("ebitda"),
        cash_and_equivalents=_stock("cash_and_equivalents"),
        current_assets=_stock("current_assets"),
        current_liabilities=_stock("current_liabilities"),
        cost_of_revenue=_flow("cost_of_revenue"),
        operating_income=_flow("operating_income"),
        revenue_latest_q=_nth_reported("revenue", 0),
        revenue_year_ago_q=_nth_reported("revenue", 4),
        net_income_latest_q=_nth_reported("net_income", 0),
        net_income_year_ago_q=_nth_reported("net_income", 4),
        source="edgar_pointintime",
        fetched_at=fetched_at,
        provenance=ProvenanceRecord(
            source="edgar_pointintime",
            fetched_at=fetched_at,
            parameters={
                "ticker": ticker,
                "as_of_date": as_of_date,
                "period_end": str(latest.date()),
                "filing_date": str(filed_dates[latest].date()),
                "periods_considered": [str(p.date()) for p in available[-4:]],
                "price_used": str(price),
            },
        ),
    )
    return snap.model_dump_json()


def build_minimal_snapshot(ticker: str, as_of_date: str) -> str:
    """
    Return a JSON-serialized FundamentalsSnapshot with None financial fields.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. "AAPL").
    as_of_date : str
        ISO 8601 evaluation date used as period_end (e.g. "2022-01-31").

    Returns
    -------
    str
        JSON string ready to pass as snapshot_json to run_sequential_ensemble().
    """
    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

    fetched_at = datetime.now(UTC)
    snap = FundamentalsSnapshot(
        ticker=ticker,
        # Explicit, not left to Pydantic's ISO coercion: this is the field that
        # decides which fundamentals an agent may see, so a malformed date must
        # fail here and name itself rather than inside a validator (DJ-142).
        period_end=date.fromisoformat(as_of_date),
        revenue=None,
        net_income=None,
        total_assets=None,
        total_liabilities=None,
        total_equity=None,
        eps=None,
        pe_ratio=None,
        market_cap=None,
        source="walk_forward_eval",
        fetched_at=fetched_at,
        provenance=ProvenanceRecord(
            source="walk_forward_eval",
            fetched_at=fetched_at,
            parameters={"ticker": ticker, "as_of_date": as_of_date},
        ),
    )
    return snap.model_dump_json()
