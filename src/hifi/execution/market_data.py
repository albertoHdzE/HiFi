"""Live market data fetcher via Alpaca (Phase 16).

Pulls current OHLCV bars to extend the local data/market/ parquet files
through today, so the ensemble agents and MCP pipeline have fresh prices.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from hifi.execution.retry import with_retry

logger = logging.getLogger(__name__)


def get_data_client(
    api_key: str | None = None,
    secret_key: str | None = None,
) -> StockHistoricalDataClient:
    # Market data is account-independent: any valid key pair works.
    if not api_key or not secret_key:
        for suffix in ("", "_FIRST", "_SECOND", "_THIRD", "_A", "_B", "_C"):
            api_key = os.environ.get(f"ALPACA_API_KEY{suffix}")
            secret_key = os.environ.get(f"ALPACA_SECRET{suffix}")
            if api_key and secret_key:
                break
    if not api_key or not secret_key:
        raise KeyError("No ALPACA_API_KEY[_suffix] / ALPACA_SECRET[_suffix] found in env")
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


@with_retry()
def fetch_bars(
    tickers: list[str],
    start: datetime,
    end: datetime | None = None,
    client: StockHistoricalDataClient | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV bars from Alpaca for the given tickers and date range.

    Returns a DataFrame with columns: date, ticker, open, high, low, close, volume.
    """
    if client is None:
        client = get_data_client()

    if end is None:
        # Free tier forbids SIP data from the last 15 minutes; stay behind it.
        end = datetime.now() - timedelta(minutes=16)

    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(req)
    rows = []
    for bar in bars.data.values():
        for b in bar:
            rows.append({
                "date": b.timestamp.strftime("%Y-%m-%d"),
                "ticker": b.symbol,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume),
            })

    if not rows:
        logger.warning("No bars returned for %s (%s → %s)", tickers, start, end)
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def update_local_ohlcv(
    tickers: list[str],
    market_dir: str = "data/market",
    client: StockHistoricalDataClient | None = None,
) -> dict[str, int]:
    """Extend local parquet files with fresh bars from Alpaca.

    Returns dict of {ticker: new_rows_added}.
    """
    from pathlib import Path
    result: dict[str, int] = {}
    fetch_end = datetime.now() - timedelta(minutes=16)

    for ticker in tickers:
        parquet_path = Path(market_dir) / ticker / "ohlcv.parquet"
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            # Normalize: existing parquets have Date as DatetimeIndex + capitalized columns
            if existing.index.name and existing.index.name.lower() == "date":
                existing = existing.reset_index()
            existing.columns = existing.columns.str.lower()
            last_date = pd.to_datetime(existing["date"]).max()
            # Re-fetch the last stored day: if it was fetched intraday it is a
            # partial bar (wrong close/volume) and must be replaced by the
            # settled bar once the session ends.
            start = last_date
            n_existing = len(existing)
        else:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            existing = pd.DataFrame()
            start = datetime(2020, 1, 1)
            n_existing = 0

        if start >= fetch_end:
            result[ticker] = 0
            continue

        try:
            new_bars = fetch_bars([ticker], start=start, end=fetch_end, client=client)
        except Exception as exc:
            logger.error("%s: fetch failed: %s", ticker, exc)
            result[ticker] = 0
            continue
        if new_bars.empty:
            result[ticker] = 0
            continue

        new_bars = new_bars.drop(columns=["ticker"], errors="ignore")
        combined = pd.concat([existing, new_bars], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        # keep="last": freshly fetched bars replace stale intraday partials
        combined = (
            combined.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        new_count = len(combined) - n_existing

        # Write back in original format: Date as index, capitalized columns
        out = combined.copy()
        out.columns = [c.capitalize() for c in out.columns]
        out = out.rename(columns={"Date": "Date"}).set_index("Date")
        out.to_parquet(parquet_path)

        result[ticker] = new_count
        if new_count > 0:
            logger.info("%s: +%d bars (through %s)", ticker, new_count,
                        combined["date"].max().strftime("%Y-%m-%d"))

    return result
