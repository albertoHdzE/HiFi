"""
HiFi Portfolio Composer MCP Server (E4-T1, DJ-091).

Exposes one MCP tool: ``compose_portfolio``.  Pure deterministic math,
no LLMs, no external data fetches.  The ensemble provides signals; this
server decides position sizing subject to hard constraints.

Algorithm:
  1. Filter to Buy signals only (long-only mode).
  2. Confidence-proportional initial weights, scaled to ``target_deployment``.
  3. Solve to a fixed point (``_solve_to_fixed_point``): per-stock cap,
     per-sector cap, fill the residual toward the target using whatever
     headroom remains, drop dust below ``min_position``, and repeat until
     the weights stop moving.
  4. Verify the post-conditions and raise ``PortfolioConstraintError`` if
     any of them fails.

Output: dict[str, float] mapping ticker -> portfolio weight, summing to
``target_deployment`` whenever the constraints admit it.  A shortfall is
always attributable to a named binding constraint and is logged as such.

DJ-132 -- why this is a fixed point and not a pipeline
------------------------------------------------------
Until 2026-08-30 the three constraints were applied exactly once each, in
order, with no recheck.  Each step is individually correct; composing them
once is not, because every step invalidates the ones before it.  Three
distinct failures were reproduced on the real 2026-08-24 arm-A basket:

* **The sector cap sent freed weight to cash instead of to names with
  headroom.**  Seven names in one sector deployed 25.0% of the book and
  discarded five of the seven picks entirely.  Worse, it inverted the
  ensemble's ordering: scaling Information Technology down left the
  highest-conviction name (0.6800) holding 12.66% while a 0.4459-conviction
  name in an unconstrained sector held 12.68%.  The portfolio ranked the
  book backwards relative to the signal that produced it.
* **``_apply_min_position`` ran last and could breach the caps it had just
  satisfied.**  Redistributing dust pushed one sector to 25.92% against a
  25.00% cap -- a risk limit silently violated in the returned book.
* **Nothing consumed ``target_deployment``.**  ``PortfolioPolicy`` emitted
  it (DJ-131) and the composer never read it, so no layer was responsible
  for the book actually being invested.

The repair is the same principle as DJ-131 one layer up: a constraint may
express *do not concentrate*; it may not also silently express *do not
invest*.  Cash is now only ever the result of a constraint that genuinely
binds, and the composer says which one.

Transport: stdio MCP (DJ-009).  No SSE/HTTP until Phase 15.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("hifi-portfolio-composer")

# Tolerance for all constraint comparisons. Loose enough to absorb the float
# error of repeated proportional redistribution, tight enough that a real
# breach (the 0.92pp sector overshoot of DJ-132) can never hide inside it.
TOL = 1e-9


class PortfolioConstraintError(RuntimeError):
    """The composed book violates a constraint, or the solver did not converge.

    Raised rather than returned because there is no safe way to continue: the
    weights on the table breach a risk limit, and executing them is strictly
    worse than trading nothing. Callers are expected to catch this, log it,
    and stand the arm down for the cycle -- not to fall back to a best effort.

    Under the pre-DJ-132 one-pass composer this condition existed and was
    silent: ``_apply_min_position`` could return a book 0.92pp over its sector
    cap and nothing anywhere looked again.
    """


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


def _sector_budgets(
    sectors: dict[str, str],
    max_sector: float,
    existing_weights: dict[str, float] | None,
    allocating: set[str],
) -> dict[str, float]:
    """Weight each sector may still receive, after already-held positions.

    Held names that are not being reallocated consume sector budget (DJ-122),
    so the budget is the cap less what the Holds already occupy, floored at
    zero so a sector that is already full can never yield a negative target.
    """
    held_by_sector: dict[str, float] = {}
    for ticker, weight in (existing_weights or {}).items():
        if ticker in allocating:
            continue  # already represented in the weights being solved
        s = sectors.get(ticker, "Unknown")
        held_by_sector[s] = held_by_sector.get(s, 0.0) + weight
    return {
        s: max(0.0, max_sector - held_by_sector.get(s, 0.0))
        for s in set(sectors.values()) | set(held_by_sector)
    }


def _fill_toward_target(
    weights: dict[str, float],
    confidence: dict[str, float],
    sectors: dict[str, str],
    max_weight: float,
    budgets: dict[str, float],
    target: float,
) -> dict[str, float]:
    """Distribute the residual to names that still have headroom.

    This step is what DJ-132 was missing. Capping is subtractive: every cap
    frees weight, and if nothing puts that weight back the book drifts below
    its target for reasons no one decided. Here the residual flows to names
    with room under *both* their own cap and their sector's remaining budget,
    in proportion to confidence -- so the ensemble's ordering survives the
    constraints instead of being inverted by them.

    Water-filling: each pass either closes the residual or saturates at least
    one name, so it terminates in at most ``len(weights)`` effective passes.
    """
    w = dict(weights)
    for _ in range(len(w) + 2):
        residual = target - sum(w.values())
        if residual <= TOL:
            break

        used: dict[str, float] = {}
        for t, v in w.items():
            s = sectors.get(t, "Unknown")
            used[s] = used.get(s, 0.0) + v

        room = {}
        for t in w:
            s = sectors.get(t, "Unknown")
            headroom = min(
                max_weight - w[t],
                budgets.get(s, max_weight) - used.get(s, 0.0),
            )
            if headroom > TOL:
                room[t] = headroom
        if not room:
            break  # genuinely constrained: the shortfall is real and reportable

        total_conf = sum(confidence.get(t, 0.0) for t in room)
        if total_conf <= 0:
            break
        added = 0.0
        for t, headroom in room.items():
            share = residual * confidence.get(t, 0.0) / total_conf
            delta = min(headroom, share)
            w[t] += delta
            added += delta
        if added <= TOL:
            break
    return w


def _binding_constraint(
    weights: dict[str, float],
    sectors: dict[str, str],
    max_weight: float,
    budgets: dict[str, float],
) -> str:
    """Name the constraint responsible for a shortfall, for the run log.

    Idle cash must always be attributable (DJ-131). This turns "the book is
    87% invested" into "the book is 87% invested because Information
    Technology is at its cap", which is the difference between a reported
    constraint and a silent defect.
    """
    used: dict[str, float] = {}
    for t, v in weights.items():
        s = sectors.get(t, "Unknown")
        used[s] = used.get(s, 0.0) + v
    full = [s for s, v in used.items() if v >= budgets.get(s, max_weight) - TOL]
    if full:
        return f"sector cap reached: {', '.join(sorted(full))}"
    capped = [t for t, v in weights.items() if v >= max_weight - TOL]
    if capped:
        return f"max_single_stock reached on all of: {', '.join(sorted(capped))}"
    return "no candidate retained headroom"


def _verify(
    weights: dict[str, float],
    sectors: dict[str, str],
    max_weight: float,
    budgets: dict[str, float],
    min_position: float,
    target: float,
) -> None:
    """Assert the post-conditions the one-pass composer could not guarantee.

    Raises rather than logging: each of these is a risk limit, and a book that
    breaches one must not reach the broker. The sector check is the one that
    was live-violated (25.92% against a 25.00% cap, DJ-132).
    """
    over = {t: v for t, v in weights.items() if v > max_weight + TOL}
    if over:
        raise PortfolioConstraintError(
            f"max_single_stock={max_weight:.6f} violated by {over}"
        )

    used: dict[str, float] = {}
    for t, v in weights.items():
        s = sectors.get(t, "Unknown")
        used[s] = used.get(s, 0.0) + v
    breached = {
        s: v for s, v in used.items() if v > budgets.get(s, max_weight) + TOL
    }
    if breached:
        detail = {
            s: (round(v, 6), round(budgets.get(s, max_weight), 6))
            for s, v in breached.items()
        }
        raise PortfolioConstraintError(f"sector budgets violated: {detail}")

    dust = {t: v for t, v in weights.items() if v < min_position - TOL}
    if dust:
        raise PortfolioConstraintError(
            f"min_position={min_position:.6f} violated by {dust}"
        )

    total = sum(weights.values())
    if total > target + TOL:
        raise PortfolioConstraintError(
            f"deployed {total:.6f} exceeds target_deployment {target:.6f}"
        )


def _solve_to_fixed_point(
    confidence: dict[str, float],
    sectors: dict[str, str],
    max_weight: float,
    budgets: dict[str, float],
    min_position: float,
    target: float,
) -> dict[str, float]:
    """Apply every constraint repeatedly until the weights stop moving.

    The four steps are each individually correct and each invalidates the
    others: capping frees weight, filling can re-breach a cap, and dropping
    dust redistributes into both. Iterating to a fixed point is what makes
    "all constraints hold simultaneously" true rather than "each constraint
    held at the moment it was applied" (DJ-132).

    Termination: ``_apply_min_position`` only ever removes names, so the
    candidate set is monotonically shrinking and the outer loop can run at
    most ``n + 2`` times. Exhausting it means the constraint set has no fixed
    point under this solver, which is a defect, not a market condition --
    hence the raise rather than a best-effort return.
    """
    # Structural infeasibility, checked before any arithmetic: if the dust
    # threshold sits above the per-name cap, no weight can satisfy both and
    # every name is dropped. That returns an empty book -- safe, but silent,
    # and a silent degenerate outcome is the DJ-131 failure mode. Say it
    # instead. PortfolioPolicy already clamps this; direct callers do not.
    if min_position > max_weight + TOL:
        raise PortfolioConstraintError(
            f"min_position={min_position:.6f} exceeds max_single_stock="
            f"{max_weight:.6f}: no position can satisfy both"
        )

    n = len(confidence)
    total_conf = sum(confidence.values())
    if total_conf <= 0:
        return {}
    w = {t: target * c / total_conf for t, c in confidence.items()}

    for _ in range(n + 2):
        previous = dict(w)

        w = _apply_stock_cap(w, max_weight)
        w = _apply_sector_cap_to_budgets(w, sectors, budgets)
        w = _fill_toward_target(w, confidence, sectors, max_weight, budgets, target)
        w = _apply_min_position(w, min_position)
        w = {t: v for t, v in w.items() if v > TOL}

        if set(w) == set(previous) and all(
            abs(w[t] - previous[t]) <= TOL for t in w
        ):
            _verify(w, sectors, max_weight, budgets, min_position, target)
            return w

    raise PortfolioConstraintError(
        f"constraint solver did not converge in {n + 2} passes "
        f"(n={n}, max_single={max_weight:.6f}, min_position={min_position:.6f}, "
        f"target={target:.6f})"
    )


def _apply_sector_cap_to_budgets(
    weights: dict[str, float],
    sectors: dict[str, str],
    budgets: dict[str, float],
) -> dict[str, float]:
    """Scale each sector down to its pre-computed remaining budget.

    Separate from ``_apply_sector_cap`` so the budget is derived once, outside
    the loop, instead of being recomputed from held positions on every pass.
    """
    w = dict(weights)
    totals: dict[str, float] = {}
    for t, v in w.items():
        s = sectors.get(t, "Unknown")
        totals[s] = totals.get(s, 0.0) + v

    for sector, total in totals.items():
        budget = budgets.get(sector, 0.0)
        if total <= budget + TOL:
            continue
        scale = (budget / total) if total > 0 else 0.0
        for t in w:
            if sectors.get(t, "Unknown") == sector:
                w[t] *= scale
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
    target_deployment: float = 1.0,
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
    target_deployment : float, default 1.0
        Fraction of capital to invest. The composer solves *toward* this
        number and any shortfall is logged with the constraint that caused
        it. Before DJ-132 nothing here read this field, so no layer owned the
        question of whether the book was actually invested.

    Returns
    -------
    dict
        ``{ticker: weight, ...}`` on success, summing to ``target_deployment``
        unless a constraint genuinely binds.
        ``{}`` when no actionable signals exist.
        ``{"error": ..., "detail": ...}`` on parse failure or on a constraint
        violation the solver could not resolve.

    Raises
    ------
    Never propagates ``PortfolioConstraintError``: it is converted to an
    ``error`` dict so a single arm's infeasible book stands that arm down for
    the cycle instead of aborting every other arm's run (DJ-123).
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
            # `conf`, not `confidence`: below, `confidence` is the ticker -> value
            # map the solver takes. One name for a scalar and then for a dict of
            # those scalars, in one function, is how the wrong one gets passed.
            conf = float(item["confidence"])
            item_sector = str(item["sector"])
        except (KeyError, TypeError, ValueError) as exc:
            return {"error": "INVALID_SIGNAL", "detail": f"Malformed signal {item!r}: {exc}"}

        if long_only and decision != "Buy":
            continue
        if not long_only and decision not in ("Buy", "Sell"):
            continue
        if conf <= 0:
            continue
        buy_signals.append(
            {"ticker": ticker, "decision": decision, "confidence": conf,
             "sector": item_sector}
        )

    if not buy_signals:
        return {}

    confidence: dict[str, float] = {s["ticker"]: s["confidence"] for s in buy_signals}
    sectors: dict[str, str] = {s["ticker"]: s["sector"] for s in buy_signals}
    if existing_weights:
        for ticker, sector in (existing_sectors or {}).items():
            sectors.setdefault(ticker, sector)

    budgets = _sector_budgets(
        sectors, max_sector, existing_weights, allocating=set(confidence)
    )

    try:
        weights = _solve_to_fixed_point(
            confidence=confidence,
            sectors=sectors,
            max_weight=max_single_stock,
            budgets=budgets,
            min_position=min_position,
            target=target_deployment,
        )
    except PortfolioConstraintError as exc:
        logger.error("compose_portfolio failed closed: %s", exc)
        return {"error": "CONSTRAINTS_UNSATISFIABLE", "detail": str(exc)}

    # Idle cash is only ever a reported constraint, never an accident (DJ-131).
    deployed = sum(weights.values())
    if deployed < target_deployment - 1e-4:
        logger.warning(
            "Deployed %.2f%% of a %.2f%% target across %d name(s): %s",
            deployed * 100, target_deployment * 100, len(weights),
            _binding_constraint(weights, sectors, max_single_stock, budgets),
        )

    return weights


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
