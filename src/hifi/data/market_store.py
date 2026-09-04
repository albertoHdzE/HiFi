"""Canonical resolver for the on-disk OHLCV market store (DJ-120).

Background
----------
HiFi has two OHLCV layouts on disk:

1. **Nested (canonical, live)** — ``data/market/<TICKER>/ohlcv.parquet``.
   A DatetimeIndex named ``Date`` plus capitalised ``Open/High/Low/Close/Volume``
   columns. Written and extended nightly by
   :func:`hifi.execution.market_data.update_local_ohlcv`. Covers the full
   98-ticker universe from 2004 to the present.

2. **Flat (legacy fixtures)** — ``data/market/<TICKER>_<from>_<to>.parquet``.
   Raw yfinance frames with a ``Date`` column, left over from early phases.
   Only 16 tickers exist and every file stops at 2023-06-30.

Five call sites independently globbed the *flat* pattern, so they silently saw
83 of 98 tickers as missing and the remaining 15 as three years stale. Because
the MCP tools return ``TICKER_NOT_FOUND`` rather than raising, the agents
treated the gap as information ("no data available -> Sell") and the defect
surfaced as a plausible-looking bearish signal instead of an error. See
``doc/bitacora/DJ_120_DATA_STARVATION.md``.

This module is the single place that knows how to find a ticker's bars.
Resolution is nested-first, flat-fallback, so test fixtures that only ship the
flat layout keep working while production always gets the canonical store.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas is imported lazily; this is for the annotations only
    import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "coverage_report",
    "load_ohlcv_frame",
    "market_dir",
    "resolve_ohlcv_path",
]


def market_dir(data_dir: str | Path | None = None) -> Path:
    """Return the market store root, honouring ``HIFI_DATA_DIR``."""
    base = data_dir if data_dir is not None else os.environ.get("HIFI_DATA_DIR", "data")
    return Path(base) / "market"


def resolve_ohlcv_path(ticker: str, data_dir: str | Path | None = None) -> Path:
    """Return the parquet path holding ``ticker``'s bars.

    Prefers the canonical nested layout; falls back to the most recent flat
    legacy fixture so existing test data still resolves.

    Raises
    ------
    FileNotFoundError
        When neither layout has data for the ticker. Callers that translate
        this into a tool-level error code must make the resulting payload
        clearly an *error*, never an absence of signal.
    """
    root = market_dir(data_dir)

    nested = root / ticker / "ohlcv.parquet"
    if nested.exists():
        return nested

    flat = sorted(glob.glob(str(root / f"{ticker}_*.parquet")))
    if flat:
        path = Path(flat[-1])
        logger.warning(
            "%s resolved to legacy flat fixture %s; the canonical nested store "
            "%s is absent. Bars may be stale.",
            ticker, path.name, nested,
        )
        return path

    raise FileNotFoundError(
        f"No OHLCV data for ticker '{ticker}': looked for {nested} and "
        f"{root / f'{ticker}_*.parquet'}"
    )


#: Column spellings that have meant "the trading date" in this store.
_DATE_ALIASES = ("date", "Date", "DATE", "index", "level_0")


def load_ohlcv_frame(ticker: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return ``ticker``'s bars as one normalised frame (DJ-141).

    Lowercase columns, the trading date as a ``date`` column of dtype
    ``datetime64``, sorted ascending, with a fresh RangeIndex.

    Seven call sites used to normalise this file themselves and did not agree
    about where the date lived. Four reset a date index into a column, two
    treated the index *as* the date, and one did neither and took ``.iloc[-1]``
    unsorted. All seven worked against the file as written, which means six
    worked by accident, and the accidents were not equal in cost:

    * ``live.market._last_completed_session`` took ``read_parquet(p).index.max()``
      and stamped every arm's decision with it. Against a store whose date sits
      in a column that index is a RangeIndex, ``max()`` is an integer, and the
      cycle dates itself 1970-01-01 — silently, for every arm.
    * ``live.market._latest_prices`` never sorted, and its number sizes orders.
    * ``live.walkforward`` filtered ``df.index <= as_of_date`` on that same
      RangeIndex and returned everything or nothing, without raising.

    That is DJ-120's shape one layer up: the fix then centralised where a
    ticker's bars are *found*; how they are *read* stayed scattered.

    ``data_dir`` is passed through to :func:`resolve_ohlcv_path`, so the legacy
    flat fixtures resolve too. **Live callers must pass it explicitly** — when it
    is None the path resolver honours ``HIFI_DATA_DIR``, and
    ``hifi.live.paths`` deliberately does not (a stray export would redirect a
    night's decisions into a temp directory and look like a night that never
    ran).

    Raises
    ------
    FileNotFoundError
        Propagated from :func:`resolve_ohlcv_path` when no layout has the
        ticker. Callers that translate this into a tool-level error must make
        the payload clearly an *error*, never an absence of signal (DJ-120).
    """
    import pandas as pd  # noqa: PLC0415

    df = pd.read_parquet(resolve_ohlcv_path(ticker, data_dir))

    # The date is in the index in the canonical store and in a column in the
    # flat fixtures. Normalise to a column before touching anything else.
    if "date" not in {str(c).lower() for c in df.columns}:
        df = df.reset_index()

    df = df.rename(columns={c: "date" for c in df.columns
                            if str(c) in _DATE_ALIASES})
    df.columns = [str(c).lower() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError(
            f"OHLCV store for {ticker!r} has no recognisable date: columns "
            f"{list(df.columns)}. Reading it would date the bars by position."
        )

    # Refuse a positional index promoted to "date". reset_index() on a frame
    # with a plain RangeIndex yields an integer column named "index", and
    # pd.to_datetime turns 0, 1, 2 into 1970-01-01 without complaint — which is
    # precisely the failure this function exists to make impossible.
    if pd.api.types.is_integer_dtype(df["date"]):
        raise ValueError(
            f"OHLCV store for {ticker!r} has an integer date column; the bars "
            "carry no trading date and would all be read as 1970-01-01"
        )

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def coverage_report(
    tickers: list[str], data_dir: str | Path | None = None
) -> dict[str, dict]:
    """Per-ticker store diagnostics: layout, row count and last bar date.

    Used by the live pre-flight and the report notebook's provenance panel so
    a starved universe is visible before agents run on it, not inferred weeks
    later from suspiciously bearish signals.
    """
    import pandas as pd  # noqa: PLC0415

    out: dict[str, dict] = {}
    for ticker in tickers:
        try:
            path = resolve_ohlcv_path(ticker, data_dir)
        except FileNotFoundError:
            out[ticker] = {"found": False, "layout": None, "rows": 0, "last_date": None}
            continue
        layout = "nested" if path.name == "ohlcv.parquet" else "flat-legacy"
        try:
            df = pd.read_parquet(path)
            idx: pd.DatetimeIndex | None = (
                df.index if isinstance(df.index, pd.DatetimeIndex) else None)
            if idx is None:
                for col in ("Date", "date"):
                    if col in df.columns:
                        idx = pd.DatetimeIndex(pd.to_datetime(df[col]))
                        break
            last = str(idx.max().date()) if idx is not None and len(idx) else None
            out[ticker] = {
                "found": True, "layout": layout, "rows": int(len(df)), "last_date": last,
            }
        except Exception as exc:  # pragma: no cover - corrupt parquet
            logger.warning("Failed reading %s for %s: %s", path, ticker, exc)
            out[ticker] = {"found": False, "layout": layout, "rows": 0, "last_date": None}
    return out
