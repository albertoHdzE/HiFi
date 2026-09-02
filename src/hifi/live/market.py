"""Price data for the cycle: refresh, session date, and last closes.

These three belong together because they are ordered: the refresh must run
before the session date is read (or it returns yesterday), and the session date
determines which close the prices below belong to.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from hifi.live import paths

logger = logging.getLogger(__name__)


def update_data(tickers: list[str]) -> dict[str, int]:
    from hifi.execution.market_data import update_local_ohlcv

    # DJ-130 companion: SPY is not a universe member but is the regime
    # classifier's benchmark input. Without nightly refresh the classifier
    # would silently run on stale bars (the exact failure that kept it pinned
    # at "neutral" — its old paths died in the DJ-120 store migration).
    fetch = list(tickers) + ["SPY"]
    result = update_local_ohlcv(fetch, market_dir=os.path.join(paths._DATA_DIR, "market"))
    total_new = sum(result.values())
    logger.info("OHLCV update: %d new bars across %d tickers (+SPY benchmark)",
                total_new, len(fetch))
    return result


def _last_completed_session(tickers: list[str], sample: int = 5) -> str | None:
    """Date of the newest bar in the OHLCV store, or None (DJ-121).

    The store is the authoritative trading calendar: the broker only returns
    bars for sessions that actually happened, so the newest bar IS the last
    completed session. That is correct across weekends AND market holidays
    without hard-coding a calendar we would have to maintain — the gap called
    out when the DJ-118 clock guard went in.

    Why the decision date must be the session and not the wall-clock date: the
    agents see that session's close and nothing later, and the orders fill at
    the NEXT open. A Sunday-evening run is Friday's cycle executed late, not a
    new observation. Dating it "Sunday" would invent a decision on a day with
    no price, and would let a Friday-night run and a Sunday run both record
    against the same information — double-counting one observation in the IC.
    Resolving both to Friday makes already_decided() collapse them correctly.

    Must be called AFTER update_data(), or it returns a stale session. Takes the
    max over several tickers so one halted or delisted name cannot drag it back.
    """
    import pandas as pd

    newest: str | None = None
    for ticker in tickers[:sample]:
        pq = Path(paths._DATA_DIR) / "market" / ticker / "ohlcv.parquet"
        if not pq.exists():
            continue
        try:
            idx = pd.read_parquet(pq).index
            if len(idx) == 0:
                continue
            last = pd.Timestamp(idx.max()).strftime("%Y-%m-%d")
        except Exception as exc:
            logger.warning("Could not read %s for session date: %s", ticker, exc)
            continue
        if newest is None or last > newest:
            newest = last
    return newest


def _latest_prices(tickers: list[str]) -> dict[str, float]:
    import pandas as pd

    prices: dict[str, float] = {}
    for ticker in tickers:
        pq = Path(paths._DATA_DIR) / "market" / ticker / "ohlcv.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            df.columns = df.columns.str.lower()
            prices[ticker] = float(df["close"].iloc[-1])
    return prices
