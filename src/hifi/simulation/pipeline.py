"""
End-to-end MCP pipeline for one evaluation date (E1-T1, DJ-108).

Chains the three MCP tools into a single function:

    compose_portfolio -> check_risk_limits -> allocate_capital

Usage
-----
    signals = [
        {"ticker": "AAPL", "decision": "Buy", "confidence": 0.8,
         "sector": "Information Technology"},
        ...
    ]
    portfolio_state = {
        "portfolio": {},          # {ticker: {"weight": float, "sector": str}}
        "portfolio_value": 100_000.0,
        "hwm_value": 100_000.0,
        "holdings": {},           # {ticker: int}
        "prices": {},             # {ticker: float}
    }
    constraints = {
        "max_single_stock": 0.05,
        "max_sector": 0.20,
        "min_position": 0.01,
        "capital": 100_000.0,
        "current_capital": 0.0,
    }
    ohlcv = {"AAPL": [{"date": "2022-01-31", "close": 170.0}, ...], ...}

    snapshot = run_pipeline(signals, ohlcv, portfolio_state, constraints)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class PortfolioSnapshot:
    """
    Full output of run_pipeline() for one evaluation date.

    Captures the compose -> risk -> allocate chain for observability,
    checkpoint storage, and replication notebook analysis.
    """

    signals: list[dict]
    """Input signals (ticker, decision, confidence, sector)."""

    weights: dict[str, float]
    """Risk-filtered portfolio weights from compose_portfolio() (ticker -> weight)."""

    risk_report: dict[str, Any]
    """Full output of check_risk_limits(): approved_signals, blocked_signals, var_*, ..."""

    orders: list[dict[str, Any]]
    """MARKET orders from allocate_capital() (ticker, side, quantity, ...)."""

    n_buy: int
    """Count of Buy signals in input."""

    n_hold: int
    """Count of Hold signals in input."""

    n_sell: int
    """Count of Sell signals in input."""

    sector_exposure: dict[str, float]
    """Aggregate portfolio weight per GICS sector (approved positions only)."""

    total_estimated_value: float
    """Sum of estimated_value across all orders ($)."""

    constraints: dict[str, Any] = field(default_factory=dict)
    """Constraints used in this pipeline run (for traceability)."""

    def to_json(self) -> str:
        """JSON-serialize the snapshot. All fields are JSON-compatible by construction."""
        return json.dumps(
            {
                "signals": self.signals,
                "weights": self.weights,
                "risk_report": self.risk_report,
                "orders": self.orders,
                "n_buy": self.n_buy,
                "n_hold": self.n_hold,
                "n_sell": self.n_sell,
                "sector_exposure": self.sector_exposure,
                "total_estimated_value": self.total_estimated_value,
                "constraints": self.constraints,
            },
            indent=2,
            default=str,
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    signals: list[dict],
    ohlcv: dict[str, list[dict]],
    portfolio_state: dict[str, Any],
    constraints: dict[str, Any],
) -> PortfolioSnapshot:
    """
    Run the full compose -> risk -> allocate pipeline for one evaluation date.

    Parameters
    ----------
    signals : list[dict]
        Per-ticker ensemble decisions. Each dict must have:
        - ticker: str
        - decision: "Buy" | "Hold" | "Sell"
        - confidence: float [0, 1]
        - sector: str (GICS sector)
    ohlcv : dict[str, list[dict]]
        OHLCV time series per ticker:
        ``{ticker: [{"date": str, "close": float, ...}, ...]}``.
        Used by risk manager for VaR and correlation checks.
    portfolio_state : dict
        Current portfolio state with keys:
        - portfolio: dict — ``{ticker: {"weight": float, "sector": str, ...}}``
        - portfolio_value: float — current marked-to-market value
        - hwm_value: float — high-water mark
        - holdings: dict — ``{ticker: int}`` current shares
        - prices: dict — ``{ticker: float}`` current prices
    constraints : dict
        Pipeline constraints:
        - max_single_stock: float (default 0.05)
        - max_sector: float (default 0.20)
        - min_position: float (default 0.01)
        - capital: float — total capital to deploy
        - current_capital: float — current portfolio value (0 = all cash)
        - available_cash: float | None — buying power; when present, BUY
          orders are scaled to fit it (optional)

    Returns
    -------
    PortfolioSnapshot
        Full pipeline output: weights, risk_report, orders, and summary stats.
    """
    from hifi.mcp.capital_allocator import generate_orders
    from hifi.mcp.portfolio_composer import compose_portfolio
    from hifi.mcp.risk_manager import compute_risk_report

    max_single_stock = float(constraints.get("max_single_stock", 0.05))
    max_sector = float(constraints.get("max_sector", 0.20))
    min_position = float(constraints.get("min_position", 0.01))
    capital = float(constraints.get("capital", 100_000.0))
    current_capital = float(constraints.get("current_capital", 0.0))

    portfolio = dict(portfolio_state.get("portfolio", {}))
    portfolio_value = float(portfolio_state.get("portfolio_value", capital))
    hwm_value = float(portfolio_state.get("hwm_value", capital))
    # float, not int: fractional shares are supported end to end (DJ-121).
    holdings = {str(k): float(v) for k, v in portfolio_state.get("holdings", {}).items()}
    prices = {str(k): float(v) for k, v in portfolio_state.get("prices", {}).items()}

    # Positions already held consume sector budget even when they are not
    # being reallocated (the ensemble marked them Hold). The sector cap has to
    # see them or the combined book can breach it (DJ-122).
    existing_weights = {
        str(sym): float(meta.get("weight", 0.0))
        for sym, meta in portfolio.items()
        if isinstance(meta, dict)
    }
    existing_sectors = {
        str(sym): str(meta.get("sector", "Unknown"))
        for sym, meta in portfolio.items()
        if isinstance(meta, dict)
    }

    # --- Step 1: compose_portfolio ---
    weights: dict[str, float] = compose_portfolio(
        signals_json=json.dumps(signals),
        max_single_stock=max_single_stock,
        max_sector=max_sector,
        min_position=min_position,
        long_only=True,
        existing_weights=existing_weights,
        existing_sectors=existing_sectors,
    )
    if "error" in weights:
        logger.error("compose_portfolio error: %s", weights)
        weights = {}

    # --- Step 2: check_risk_limits ---
    risk_report = compute_risk_report(
        portfolio=portfolio,
        signals=signals,
        ohlcv=ohlcv,
        portfolio_value=portfolio_value,
        hwm_value=hwm_value,
        max_sector=max_sector,
    )

    # Keep only risk-approved tickers in portfolio weights
    approved_set = set(risk_report.get("approved_signals", []))
    approved_weights = {t: w for t, w in weights.items() if t in approved_set}

    # Exits (DJ-127). compose_portfolio is long-only, so it assigns weights to
    # Buys and says nothing about anything else; generate_orders then iterates
    # over those weights alone. A held name the ensemble marked Sell was
    # therefore never touched: across the whole live record, 4,206 Sell signals
    # produced 0 sell orders on a Sell-signalled ticker. Half of each arm's
    # conviction never reached the portfolio, while IC scored it as if it had.
    #
    # A Sell on a name we hold becomes an explicit target weight of 0, which
    # generate_orders turns into a full exit. Hold means "keep what you have",
    # so held-and-Hold is deliberately left alone rather than forced to a
    # target. A Sell on a name we do not hold is a no-op: the book is long-only.
    #
    # Risk approval is not required to LEAVE a position. The risk manager gates
    # taking on exposure; blocking an exit would trap the arm in a name it has
    # already decided against.
    decisions = {
        str(s.get("ticker")): str(s.get("decision"))
        for s in signals if s.get("ticker")
    }
    exits = {
        t: 0.0 for t, qty in holdings.items()
        if qty > 0 and decisions.get(t) == "Sell" and t not in approved_weights
    }
    if exits:
        logger.info("Exiting %d position(s) on Sell signals: %s",
                    len(exits), ",".join(sorted(exits)))
        approved_weights = {**approved_weights, **exits}

    # --- Step 3: allocate_capital ---
    # Fall back to latest OHLCV close when prices not in portfolio_state
    if not prices and ohlcv:
        for ticker, rows in ohlcv.items():
            if rows:
                prices[ticker] = float(rows[-1].get("close", 0.0))

    orders = generate_orders(
        weights=approved_weights,
        prices=prices,
        holdings=holdings,
        capital=capital,
        current_capital=current_capital,
        available_cash=constraints.get("available_cash"),
    )

    # --- Summary statistics ---
    decision_counts = {"Buy": 0, "Hold": 0, "Sell": 0}
    for sig in signals:
        d = sig.get("decision", "Hold")
        if d in decision_counts:
            decision_counts[d] += 1

    sector_exposure: dict[str, float] = {}
    sig_sector_map = {sig.get("ticker", ""): sig.get("sector", "Unknown") for sig in signals}
    for ticker, w in approved_weights.items():
        sector = sig_sector_map.get(ticker, "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + w

    total_estimated_value = sum(float(o.get("estimated_value", 0.0)) for o in orders)

    return PortfolioSnapshot(
        signals=signals,
        weights=approved_weights,
        risk_report=risk_report,
        orders=orders,
        n_buy=decision_counts["Buy"],
        n_hold=decision_counts["Hold"],
        n_sell=decision_counts["Sell"],
        sector_exposure=sector_exposure,
        total_estimated_value=total_estimated_value,
        constraints=dict(constraints),
    )
