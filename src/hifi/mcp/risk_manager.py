"""
HiFi Risk Manager MCP Server (E4-T2, DJ-091).

Exposes one MCP tool: ``check_risk_limits``.  Pure deterministic math,
no LLMs, no external data fetches.

Risk checks applied in order:
  1. Max drawdown: if portfolio is down > 15% from HWM, block all Buy signals.
  2. VaR (95%, 99%): historical simulation on 252-day portfolio returns.
     Block Buy signals if VaR_95 > 5% or VaR_99 > 8%.
  3. Sector concentration: block tickers whose sector already exceeds max_sector.
  4. Correlation-aware sizing: for pairs with r > 0.85 (60-day window),
     flag the lower-confidence ticker in block_reasons (does not block, reduces).

Transport: stdio MCP (DJ-009).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("hifi-risk-manager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VAR_95_LIMIT = 0.05    # block if VaR_95 > 5%
_VAR_99_LIMIT = 0.08    # block if VaR_99 > 8%
_MAX_DD_LIMIT = 0.15    # block all buys if drawdown > 15% from HWM
_CORR_THRESHOLD = 0.85  # reduce lower-confidence position if r > threshold
_VAR_WINDOW = 252       # trading days for VaR computation
_CORR_WINDOW = 60       # trading days for correlation computation


# ---------------------------------------------------------------------------
# Internal helpers (importable for unit testing)
# ---------------------------------------------------------------------------


def _build_returns(
    tickers: list[str],
    ohlcv: dict[str, list[dict]],
) -> dict[str, list[float]]:
    """Convert OHLCV close sequences to daily percentage-return lists."""
    result: dict[str, list[float]] = {}
    for ticker in tickers:
        rows = ohlcv.get(ticker, [])
        closes = [float(r["close"]) for r in rows if "close" in r]
        if len(closes) < 2:
            result[ticker] = []
            continue
        rets: list[float] = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            rets.append((closes[i] - prev) / prev if prev != 0 else 0.0)
        result[ticker] = rets
    return result


def compute_portfolio_var(
    portfolio: dict[str, dict],
    returns: dict[str, list[float]],
    window: int = _VAR_WINDOW,
) -> tuple[float, float]:
    """
    Historical simulation VaR at 95% and 99% confidence.

    Parameters
    ----------
    portfolio : dict
        {ticker: {"weight": float, ...}}
    returns : dict
        Pre-computed daily return lists per ticker.
    window : int
        Look-back window in trading days.

    Returns
    -------
    (var_95, var_99) as positive loss magnitudes (e.g. 0.03 = 3% daily loss).
    Returns (0.0, 0.0) when insufficient data.
    """
    if not portfolio:
        return 0.0, 0.0

    weighted: list[tuple[float, list[float]]] = []
    min_len: int | None = None

    for ticker, info in portfolio.items():
        r = returns.get(ticker, [])[-window:]
        if not r:
            continue
        n = len(r)
        min_len = n if min_len is None else min(min_len, n)
        weighted.append((float(info.get("weight", 0.0)), r))

    if min_len is None or min_len < 2:
        return 0.0, 0.0

    port_returns = np.zeros(min_len)
    for w, r in weighted:
        port_returns += w * np.array(r[-min_len:])

    var_95 = float(-np.percentile(port_returns, 5))
    var_99 = float(-np.percentile(port_returns, 1))
    return max(var_95, 0.0), max(var_99, 0.0)


def compute_correlation_matrix(
    tickers: list[str],
    returns: dict[str, list[float]],
    window: int = _CORR_WINDOW,
) -> dict[tuple[str, str], float]:
    """Compute pairwise Pearson correlation over the last ``window`` days."""
    corr: dict[tuple[str, str], float] = {}
    for i, ta in enumerate(tickers):
        ra = returns.get(ta, [])[-window:]
        for j, tb in enumerate(tickers):
            if j <= i:
                continue
            rb = returns.get(tb, [])[-window:]
            n = min(len(ra), len(rb))
            if n < 2:
                continue
            a = np.array(ra[-n:], dtype=float)
            b = np.array(rb[-n:], dtype=float)
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            corr[(ta, tb)] = float(np.corrcoef(a, b)[0, 1])
    return corr


def max_drawdown_breached(
    portfolio_value: float,
    hwm_value: float,
    limit: float = _MAX_DD_LIMIT,
) -> bool:
    """Return True if the portfolio has fallen more than ``limit`` from HWM."""
    if hwm_value <= 0:
        return False
    return (hwm_value - portfolio_value) / hwm_value > limit


# ---------------------------------------------------------------------------
# Core logic (separated for unit testing)
# ---------------------------------------------------------------------------


def compute_risk_report(
    portfolio: dict[str, dict],
    signals: list[dict],
    ohlcv: dict[str, list[dict]],
    portfolio_value: float = 100_000.0,
    hwm_value: float = 100_000.0,
    max_sector: float = 0.20,
) -> dict[str, Any]:
    """
    Evaluate risk limits for proposed Buy signals.

    Parameters
    ----------
    portfolio : dict
        Current positions: {ticker: {"weight": float, "confidence": float, "sector": str}}.
    signals : list[dict]
        Proposed signals: [{"ticker": str, "decision": str, "confidence": float, "sector": str}].
    ohlcv : dict
        OHLCV time series: {ticker: [{"date": str, "close": float}]}.
    portfolio_value : float
        Current portfolio value.
    hwm_value : float
        Portfolio high-water mark.
    max_sector : float
        Maximum sector concentration allowed.

    Returns
    -------
    dict matching RiskReport schema.
    """
    buy_signals = [s for s in signals if s.get("decision") == "Buy"]
    approved: list[str] = []
    blocked: list[str] = []
    reasons: dict[str, str] = {}

    # --- Check 1: max drawdown ---
    dd_breached = max_drawdown_breached(portfolio_value, hwm_value)

    # --- Build return series for all relevant tickers ---
    all_tickers = list(portfolio) + [s["ticker"] for s in buy_signals]
    returns = _build_returns(list(dict.fromkeys(all_tickers)), ohlcv)

    # --- Check 2: VaR on current portfolio ---
    var_95, var_99 = compute_portfolio_var(portfolio, returns)
    var_blocked = (var_95 > _VAR_95_LIMIT) or (var_99 > _VAR_99_LIMIT)

    # --- Sector totals in current portfolio ---
    sector_totals: dict[str, float] = {}
    for _t, info in portfolio.items():
        s = str(info.get("sector", ""))
        sector_totals[s] = sector_totals.get(s, 0.0) + float(info.get("weight", 0.0))

    # --- Check 4: correlation matrix for buy-signal tickers ---
    signal_tickers = [s["ticker"] for s in buy_signals]
    corr_matrix = compute_correlation_matrix(signal_tickers, returns)
    signal_conf: dict[str, float] = {
        s["ticker"]: float(s.get("confidence", 0.5)) for s in buy_signals
    }
    corr_reduced: dict[str, str] = {}
    for (ta, tb), r in corr_matrix.items():
        if r > _CORR_THRESHOLD:
            ca = signal_conf.get(ta, 0.5)
            cb = signal_conf.get(tb, 0.5)
            weaker = ta if ca < cb else tb
            peer = tb if weaker == ta else ta
            corr_reduced[weaker] = (
                f"CORR_REDUCED: r={r:.3f}>{_CORR_THRESHOLD} with {peer} — size at 50%"
            )

    # --- Evaluate each buy signal ---
    for sig in buy_signals:
        ticker = sig["ticker"]
        sector = str(sig.get("sector", ""))

        if dd_breached:
            blocked.append(ticker)
            reasons[ticker] = "MAX_DRAWDOWN: portfolio down >15% from HWM"
            continue

        if var_blocked:
            parts = []
            if var_95 > _VAR_95_LIMIT:
                parts.append(f"VaR_95={var_95:.3f}>{_VAR_95_LIMIT}")
            if var_99 > _VAR_99_LIMIT:
                parts.append(f"VaR_99={var_99:.3f}>{_VAR_99_LIMIT}")
            blocked.append(ticker)
            reasons[ticker] = "VAR_LIMIT: " + ", ".join(parts)
            continue

        current_sector_total = sector_totals.get(sector, 0.0)
        if current_sector_total >= max_sector:
            blocked.append(ticker)
            reasons[ticker] = (
                f"SECTOR_CAP: {sector} at {current_sector_total:.1%} >= {max_sector:.0%}"
            )
            continue

        approved.append(ticker)
        if ticker in corr_reduced:
            reasons[ticker] = corr_reduced[ticker]

    parts = [
        f"VaR_95={var_95:.4f}",
        f"VaR_99={var_99:.4f}",
        f"dd_breached={dd_breached}",
        f"approved={len(approved)}",
        f"blocked={len(blocked)}",
    ]
    if corr_reduced:
        parts.append(f"corr_reduced={sorted(corr_reduced)}")

    return {
        "approved_signals": approved,
        "blocked_signals": blocked,
        "block_reasons": reasons,
        "var_95": round(var_95, 6),
        "var_99": round(var_99, 6),
        "portfolio_risk_summary": " | ".join(parts),
    }


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------


@mcp.tool()
def check_risk_limits(
    portfolio_json: str,
    signals_json: str,
    ohlcv_json: str,
    portfolio_value: float = 100_000.0,
    hwm_value: float = 100_000.0,
    max_sector: float = 0.20,
) -> dict[str, Any]:
    """
    Evaluate risk limits for proposed buy signals against the current portfolio.

    Parameters
    ----------
    portfolio_json : str
        JSON object: {ticker: {"weight": float, "confidence": float, "sector": str}}.
    signals_json : str
        JSON array: [{"ticker": str, "decision": str, "confidence": float, "sector": str}].
        Only Buy signals are evaluated; Hold/Sell pass through as approved.
    ohlcv_json : str
        JSON object: {ticker: [{"date": str, "close": float}]}.
        At least 252 rows recommended for VaR; 60 for correlation.
    portfolio_value : float
        Current marked-to-market portfolio value.
    hwm_value : float
        Portfolio high-water mark.
    max_sector : float
        Maximum sector weight allowed (default 0.20 = 20%).

    Returns
    -------
    dict
        approved_signals, blocked_signals, block_reasons, var_95, var_99,
        portfolio_risk_summary.  On parse error: {"error": ..., "detail": ...}.
    """
    try:
        portfolio = json.loads(portfolio_json)
        if not isinstance(portfolio, dict):
            raise ValueError("portfolio_json must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return {"error": "INVALID_PORTFOLIO_JSON", "detail": str(exc)}

    try:
        signals = json.loads(signals_json)
        if not isinstance(signals, list):
            raise ValueError("signals_json must be a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        return {"error": "INVALID_SIGNALS_JSON", "detail": str(exc)}

    try:
        ohlcv = json.loads(ohlcv_json)
        if not isinstance(ohlcv, dict):
            raise ValueError("ohlcv_json must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return {"error": "INVALID_OHLCV_JSON", "detail": str(exc)}

    return compute_risk_report(
        portfolio=portfolio,
        signals=signals,
        ohlcv=ohlcv,
        portfolio_value=portfolio_value,
        hwm_value=hwm_value,
        max_sector=max_sector,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
