"""
Market data acquisition from Yahoo Finance via yfinance.

Provides two fetchers:
- MarketDataFetcher: downloads OHLCV price history and normalises to OHLCVDataset
- FundamentalsFetcher: downloads quarterly financial statement data and normalises
  to FundamentalsSnapshot

Design decisions:
- Uses auto_adjust=False to get both Close and Adj Close separately. This gives
  explicit control over which price is used downstream (unadjusted for volume
  analysis, adjusted for return calculations).
- Bars with any NaN OHLCV field are silently dropped and logged. A NaN bar is a
  data gap, not an error; the DataQualityChecker (P1-E4) will measure how many
  gaps exist across the full universe.
- The fetcher attaches a ProvenanceRecord to every dataset so downstream code
  can trace exactly which yfinance call produced each file.
- All timestamps are stored in UTC regardless of the America/New_York timezone
  that yfinance returns in the index.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any, cast

import pandas as pd
import yfinance as yf

from hifi.data.schemas import (
    FundamentalsSnapshot,
    OHLCVBar,
    OHLCVDataset,
    ProvenanceRecord,
)

logger = logging.getLogger(__name__)

# yfinance column names returned by Ticker.history(auto_adjust=False)
_YFINANCE_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Volume": "volume",
}


def _now_utc() -> datetime:
    return datetime.now(UTC)


class MarketDataFetcher:
    """
    Downloads OHLCV price history from Yahoo Finance and normalises to OHLCVDataset.

    The fetcher is stateless: each call to fetch_ohlcv creates a fresh yfinance
    Ticker and issues one history request. It does not cache or retry.
    """

    def __init__(self, source: str = "yfinance") -> None:
        self._source = source

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        _test_download: pd.DataFrame | None = None,
    ) -> OHLCVDataset:
        """
        Download OHLCV data for a single ticker over [start, end).

        The end date is exclusive in yfinance convention (consistent with Python
        range semantics). If start == end, an empty dataset is returned.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol (e.g. "AAPL").
        start : date
            First date to include (inclusive).
        end : date
            Last date to include (exclusive).

        Returns
        -------
        OHLCVDataset
            Dataset containing all trading days with complete OHLCV data.
            Days with any NaN field are excluded; the count of dropped rows is
            logged at WARNING level.
        """
        fetched_at = _now_utc()
        start_str = start.isoformat()
        end_str = end.isoformat()

        raw_df = (
            _test_download if _test_download is not None
            else self._download(ticker, start_str, end_str)
        )
        bars = self._normalise(ticker, raw_df)

        provenance = ProvenanceRecord(
            source=self._source,
            fetched_at=fetched_at,
            parameters={"ticker": ticker, "start": start_str, "end": end_str},
        )

        return OHLCVDataset(
            ticker=ticker,
            bars=bars,
            source=self._source,
            fetched_at=fetched_at,
            date_from=start,
            date_to=end,
            provenance=provenance,
        )

    def _download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Issue the yfinance download. Isolated for patching in tests."""
        t = yf.Ticker(ticker)
        # yfinance ships no stubs, so this is Any at the boundary. cast rather
        # than a blanket ignore: the declared return type is the contract the
        # rest of this class relies on.
        return cast("pd.DataFrame", t.history(start=start, end=end, auto_adjust=False))

    def _normalise(self, ticker: str, df: pd.DataFrame) -> list[OHLCVBar]:
        """
        Normalise a raw yfinance DataFrame to a list of OHLCVBar objects.

        Rows where any of Open, High, Low, Close have NaN are dropped and
        logged. Volume NaN is treated as zero (some data sources omit volume
        for certain instruments).
        """
        if df.empty:
            return []

        # Rename to lowercase canonical names
        present = {k: v for k, v in _YFINANCE_RENAME.items() if k in df.columns}
        df = df.rename(columns=present)

        # Strip timezone from index and convert to date. isinstance, not
        # hasattr: it narrows for the reader and the type checker alike, and a
        # non-datetime index here would otherwise reach .date and raise.
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            dates = df.index.date
        else:
            dates = pd.to_datetime(df.index).date

        bars: list[OHLCVBar] = []
        dropped = 0

        for i, bar_date in enumerate(dates):
            row = df.iloc[i]
            try:
                o = float(row["open"])
                h = float(row["high"])
                lv = float(row["low"])
                c = float(row["close"])
                vol = float(row.get("volume", 0) or 0)
                adj = row.get("adjusted_close")
                adj_val: float | None = None if pd.isna(adj) else float(adj)
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue

            # Skip bars with missing prices
            import math

            if any(math.isnan(x) for x in [o, h, lv, c]):
                dropped += 1
                continue

            try:
                bar = OHLCVBar(
                    ticker=ticker,
                    date=bar_date,
                    open=o,
                    high=h,
                    low=lv,
                    close=c,
                    volume=vol,
                    adjusted_close=adj_val,
                    source=self._source,
                )
                bars.append(bar)
            except Exception as exc:
                dropped += 1
                logger.warning("Dropped bar %s %s: %s", ticker, bar_date, exc)

        if dropped:
            logger.warning("%s: dropped %d bars with missing/invalid data", ticker, dropped)

        return bars


class FundamentalsFetcher:
    """
    Downloads quarterly financial statement data from Yahoo Finance.

    Retrieves the most recently available annual financials and balance sheet
    data. Returns a single FundamentalsSnapshot for the most recent period.

    Limitation: yfinance provides limited historical depth for fundamentals.
    This fetcher is adequate for Phase 1 (universe selection and basic agent
    context). More detailed historical fundamentals are a Phase 7+ concern.
    """

    def __init__(self, source: str = "yfinance") -> None:
        self._source = source

    def fetch_snapshot(self, ticker: str) -> FundamentalsSnapshot:
        """
        Download the most recent available fundamental data for a single ticker.

        Returns a FundamentalsSnapshot where unavailable fields are None.
        The period_end is set to the most recent annual reporting period date.
        """
        fetched_at = _now_utc()
        info = self._get_info(ticker)
        period_end = self._extract_period_end(info)

        provenance = ProvenanceRecord(
            source=self._source,
            fetched_at=fetched_at,
            parameters={"ticker": ticker},
        )

        return FundamentalsSnapshot(
            ticker=ticker,
            period_end=period_end,
            revenue=_to_float(info.get("totalRevenue")),
            net_income=_to_float(info.get("netIncomeToCommon")),
            total_assets=_to_float(info.get("totalAssets")),
            total_liabilities=_to_float(info.get("totalDebt")),
            total_equity=_to_float(info.get("bookValue")),
            eps=_to_float(info.get("trailingEps")),
            pe_ratio=_to_float(info.get("trailingPE")),
            market_cap=_to_float(info.get("marketCap")),
            source=self._source,
            fetched_at=fetched_at,
            provenance=provenance,
        )

    def _get_info(self, ticker: str) -> dict[str, Any]:
        """Issue the yfinance info request. Isolated for patching in tests."""
        return cast("dict[str, Any]", yf.Ticker(ticker).info)

    def _extract_period_end(self, info: dict[str, Any]) -> date:
        """
        Extract the most recent fiscal year end date from yfinance info.

        Falls back to today's date if not available, which is a documented
        approximation (period_end may be inaccurate for recently listed tickers).
        """
        raw = info.get("mostRecentQuarter") or info.get("lastFiscalYearEnd")
        if raw is not None:
            try:
                # yfinance returns Unix timestamp integers for date fields
                return date.fromtimestamp(int(raw))
            except (TypeError, ValueError, OSError):
                pass
        return date.today()


def _to_float(v: Any) -> float | None:
    """Convert a yfinance value to float, returning None if unavailable."""
    if v is None:
        return None
    try:
        result = float(v)
        import math

        return None if math.isnan(result) or math.isinf(result) else result
    except (TypeError, ValueError):
        return None
