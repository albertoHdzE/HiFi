"""riskbudget external strategy adapter (Phase 16, DJ-113).

Wraps the external ``riskbudget`` MCP provider (v1.1.0, strategy
``calm_exposure``) as a separately-attributed HiFi paper-trading strategy.

riskbudget is a finished, versioned dependency — its internals are never
modified. It is deterministic, point-in-time causal, no-LLM, no-network:
same closes + same as_of_date -> same signals.

Data path: riskbudget's own parquet loader expects a flat
``data/market/<TICKER>_*.parquet`` layout with a ``date`` column, which does
NOT match HiFi's nested ``data/market/<TICKER>/ohlcv.parquet`` (Date index,
capitalised columns). Rather than mirror the store, we use riskbudget's
first-class ``closes=`` data-independent mode: HiFi reads its own store,
slices each series point-in-time to <= as_of_date, and passes the closes in.
HiFi thereby owns the causality guarantee (README section "Data-independent
mode").
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SERVER_MODULE = "riskbudget.mcp_server"
_STRATEGY = "calm_exposure"


def _venv_python() -> str:
    """Absolute path to HiFi's venv interpreter (has riskbudget installed)."""
    return str(Path(__file__).resolve().parents[3] / ".venv" / "bin" / "python")


def point_in_time_closes(ticker: str, as_of_date: str, data_dir: str) -> list[float]:
    """Ordered (oldest->newest) close series for ``ticker`` with date <= as_of.

    Reads HiFi's own parquet store; returns [] when the ticker has no file.
    """
    from hifi.data.market_store import load_ohlcv_frame  # noqa: PLC0415

    try:
        df = load_ohlcv_frame(ticker, data_dir)
    except FileNotFoundError:
        return []
    except ValueError as exc:
        # Arm D sizes from these closes. An empty list is a real answer here,
        # but it must never be a quiet one (DJ-120).
        logger.warning("No usable bars for %s; arm D sees no history: %s",
                       ticker, exc)
        return []
    if "close" not in df.columns:
        return []
    df = df[df["date"].dt.strftime("%Y-%m-%d") <= as_of_date]
    return [float(x) for x in df["close"].tolist() if x == x]


def get_riskbudget_signals(
    tickers: list[str],
    as_of_date: str,
    data_dir: str,
    sectors: dict[str, str] | None = None,
) -> dict:
    """Return riskbudget signals for the universe via the closes= bypass.

    Returns the raw provider payload:
      {signals: [...], skipped: [...], strategy, strategy_version, as_of_date, call_id}
    Each signals entry {ticker, decision, confidence, sector, target_exposure, reason}
    is directly consumable by compose_portfolio.
    """
    from hifi.agents.mcp_client import call_tool  # noqa: PLC0415

    sectors = sectors or {}
    closes = {t: point_in_time_closes(t, as_of_date, data_dir) for t in tickers}
    closes = {t: c for t, c in closes.items() if c}

    result = call_tool(
        tool_name="get_signals",
        params={
            "tickers": tickers,
            "as_of_date": as_of_date,
            "strategy": _STRATEGY,
            "sectors": {t: sectors.get(t, "Unknown") for t in tickers},
            "closes": closes,
        },
        server_module=_SERVER_MODULE,
        data_dir=data_dir,
        python_executable=_venv_python(),
    )

    if "error" in result:
        logger.error("riskbudget get_signals error: %s", result)
        return {"signals": [], "skipped": [], "error": result.get("error")}

    n = len(result.get("signals", []))
    logger.info(
        "riskbudget %s v%s: %d signals, %d skipped (call_id=%s)",
        _STRATEGY, result.get("strategy_version"), n,
        len(result.get("skipped", [])), result.get("call_id"),
    )
    return result
