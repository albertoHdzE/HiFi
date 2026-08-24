"""
HiFi Portfolio Composer MCP Server (E4-T1, DJ-091).

Exposes one MCP tool: ``compose_portfolio``.  Pure deterministic math,
no LLMs, no external data fetches.  The ensemble provides signals; this
server decides position sizing subject to hard constraints.

Algorithm (applied in order):
  1. Filter to Buy signals only (long-only mode).
  2. Confidence-proportional initial weights (normalised to sum 1.0).
  3. Per-stock cap: iteratively cap weights at max_single_stock and
     redistribute excess to uncapped positions.  If all positions hit
     the cap simultaneously, excess flows to cash (weights sum < 1.0).
  4. Sector cap: scale down positions in sectors whose total exceeds
     max_sector pro-rata.  Excess flows to cash.
  5. Minimum position: remove positions below min_position, redistribute
     their weight proportionally to remaining positions.

Output: dict[str, float] mapping ticker -> portfolio weight.
Weights sum to 1.0 when no caps bind.  When caps reduce investable
capital, the remainder is implicitly held as cash.

Transport: stdio MCP (DJ-009).  No SSE/HTTP until Phase 15.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("hifi-portfolio-composer")


# ---------------------------------------------------------------------------
# Internal algorithm helpers (importable for unit testing)
# ---------------------------------------------------------------------------


def _apply_stock_cap(
    weights: dict[str, float],
    max_weight: float,
) -> dict[str, float]:
    """Iteratively cap per-stock weights and redistribute excess.

    Excess from capped positions is redistributed proportionally to
    uncapped positions.  When all positions are capped simultaneously
    (no room to redistribute), the excess flows to cash and the total
    portfolio weight falls below 1.0.

    Parameters
    ----------
    weights : dict[str, float]
        Current weights (may not sum to 1.0 if prior steps capped them).
    max_weight : float
        Maximum allowed weight per individual stock.

    Returns
    -------
    dict[str, float]
        Weights with no single position exceeding max_weight.
    """
    w = dict(weights)
    for _ in range(200):  # bounded iterations for safety
        over = {t: v for t, v in w.items() if v > max_weight + 1e-9}
        if not over:
            break
        excess = sum(v - max_weight for v in over.values())
        for t in over:
            w[t] = max_weight
        under = {t: v for t, v in w.items() if v < max_weight - 1e-9}
        if not under:
            # No room to redistribute; excess goes to cash
            break
        total_under = sum(under.values())
        for t in under:
            w[t] += excess * (w[t] / total_under)
    return w


def _apply_sector_cap(
    weights: dict[str, float],
    sectors: dict[str, str],
    max_sector: float,
    existing_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Scale down positions in overweight sectors pro-rata.

    Each sector is handled independently.  Excess within an overweight
    sector flows to cash (not redistributed to other sectors).

    ``existing_weights`` are positions already held that are NOT part of this
    allocation — typically names the ensemble marked Hold. They consume sector
    budget even though they are not being traded, so the cap must be applied
    to the *combined* book. Omitting them let arm A's held NVDA push
    Information Technology to 21.86% against a 20% cap while every individual
    check passed (DJ-122).

    Parameters
    ----------
    weights : dict[str, float]
        Weights being allocated now.
    sectors : dict[str, str]
        Mapping of ticker -> GICS sector, covering both dicts.
    max_sector : float
        Maximum allowed aggregate weight per GICS sector.
    existing_weights : dict[str, float] | None
        Already-held positions not in ``weights``, as portfolio weights.

    Returns
    -------
    dict[str, float]
        Weights whose sector totals, combined with existing holdings, do not
        exceed max_sector. A sector already over budget from holdings alone
        receives zero new allocation rather than a negative one.
    """
    w = dict(weights)
    held = {t: v for t, v in (existing_weights or {}).items() if t not in w}

    # Budget consumed by positions we are not reallocating.
    held_by_sector: dict[str, float] = {}
    for ticker, weight in held.items():
        s = sectors.get(ticker, "Unknown")
        held_by_sector[s] = held_by_sector.get(s, 0.0) + weight

    new_by_sector: dict[str, float] = {}
    for ticker, weight in w.items():
        s = sectors.get(ticker, "Unknown")
        new_by_sector[s] = new_by_sector.get(s, 0.0) + weight

    for sector, new_total in new_by_sector.items():
        budget = max_sector - held_by_sector.get(sector, 0.0)
        if budget <= 0:
            # Holdings alone already fill or exceed the sector: allocate nothing new.
            for ticker in list(w):
                if sectors.get(ticker, "Unknown") == sector:
                    w[ticker] = 0.0
            continue
        if new_total > budget + 1e-9:
            scale = budget / new_total
            for ticker in list(w):
                if sectors.get(ticker, "Unknown") == sector:
                    w[ticker] *= scale
    return w


def _apply_min_position(
    weights: dict[str, float],
    min_position: float,
) -> dict[str, float]:
    """Remove positions below min_position and redistribute their weight.

    Positions removed by this step have their weight redistributed
    proportionally to the remaining positions, preserving the total
    invested weight.  If all positions fall below min_position the
    function returns an empty dict.

    Parameters
    ----------
    weights : dict[str, float]
        Current weights.
    min_position : float
        Minimum allowed weight per position.

    Returns
    -------
    dict[str, float]
        Weights with all positions >= min_position, or empty dict.
    """
    w = dict(weights)
    while True:
        to_remove = [t for t, v in w.items() if v < min_position - 1e-9]
        if not to_remove:
            break
        removed_weight = sum(w.pop(t) for t in to_remove)
        if not w:
            return {}
        total_remaining = sum(w.values())
        if total_remaining <= 0:
            return {}
        for t in w:
            w[t] += removed_weight * (w[t] / total_remaining)
    return w


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------


@mcp.tool()
def compose_portfolio(
    signals_json: str,
    max_single_stock: float = 0.05,
    max_sector: float = 0.20,
    min_position: float = 0.01,
    long_only: bool = True,
    existing_weights: dict[str, float] | None = None,
    existing_sectors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Compose a portfolio from agent signals with constraint enforcement.

    The limit defaults below are legacy absolutes kept for backwards
    compatibility. Production callers should pass values from
    ``hifi.portfolio.PortfolioPolicy``, which derives them from the number of
    candidates so one knob governs every book width (DJ-122).

    Parameters
    ----------
    signals_json : str
        JSON array of signal objects.  Each object must have:
        - ``ticker``     : str  (e.g. "AAPL")
        - ``decision``   : str  ("Buy" | "Hold" | "Sell")
        - ``confidence`` : float  [0.0, 1.0]
        - ``sector``     : str  (GICS sector)
    max_single_stock : float, default 0.05
        Maximum weight for any single stock (5% default).
    max_sector : float, default 0.20
        Maximum aggregate weight for any GICS sector (20% default).
    min_position : float, default 0.01
        Minimum weight to include a position (1% default).
        Positions below this threshold after capping are removed and
        their weight redistributed to remaining positions.
    long_only : bool, default True
        When True, only Buy signals are included; Hold and Sell are ignored.
    existing_weights : dict[str, float] | None
        Portfolio weights of positions already held that are not part of this
        allocation (typically Holds). They consume sector budget and must be
        counted, or the combined book can breach the sector cap while every
        individual check passes.
    existing_sectors : dict[str, str] | None
        Ticker -> sector for ``existing_weights``, since those tickers are not
        in the signal list being allocated.

    Returns
    -------
    dict
        ``{"weights": {ticker: weight, ...}}`` on success.
        ``{}`` (empty weights key) when no actionable signals exist.
        ``{"error": ..., "detail": ...}`` on parse failure.
    """
    try:
        raw = json.loads(signals_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return {"error": "INVALID_SIGNALS_JSON", "detail": str(exc)}

    if not isinstance(raw, list):
        return {"error": "INVALID_SIGNALS_JSON", "detail": "signals_json must be a JSON array"}

    # Parse signals
    buy_signals: list[dict[str, Any]] = []
    for item in raw:
        try:
            decision = str(item["decision"])
            ticker = str(item["ticker"])
            confidence = float(item["confidence"])
            sector = str(item["sector"])
        except (KeyError, TypeError, ValueError) as exc:
            return {"error": "INVALID_SIGNAL", "detail": f"Malformed signal {item!r}: {exc}"}

        if long_only and decision != "Buy":
            continue
        if not long_only and decision not in ("Buy", "Sell"):
            continue
        if confidence <= 0:
            continue
        buy_signals.append(
            {"ticker": ticker, "decision": decision, "confidence": confidence, "sector": sector}
        )

    if not buy_signals:
        return {}

    # Step 2: confidence-proportional initial weights
    total_conf = sum(s["confidence"] for s in buy_signals)
    weights: dict[str, float] = {
        s["ticker"]: s["confidence"] / total_conf for s in buy_signals
    }
    sectors: dict[str, str] = {s["ticker"]: s["sector"] for s in buy_signals}

    # Step 3: per-stock cap
    weights = _apply_stock_cap(weights, max_single_stock)

    # Step 4: sector cap, against the combined book (new + already held)
    if existing_weights:
        for ticker, sector in (existing_sectors or {}).items():
            sectors.setdefault(ticker, sector)
    weights = _apply_sector_cap(weights, sectors, max_sector, existing_weights)
    weights = {t: v for t, v in weights.items() if v > 1e-9}

    # Step 5: minimum position
    weights = _apply_min_position(weights, min_position)

    return weights


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
