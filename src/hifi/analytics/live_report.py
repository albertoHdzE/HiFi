"""Live 4-arm paper-trading report builders (Phase 16, DJ-115).

Pure data → DataFrame helpers for the live experiment report. Presentation
(plots, notebook, future published page) imports these; the logic lives here
so it is testable and reused by every renderer.

Three analysis layers, each degrading gracefully on thin data (the experiment
is ongoing; early days have too few points for stable statistics):

1. Financial  — equity curves + a QuantStats-backed metrics table per arm.
2. Decisions  — Buy/Hold/Sell distribution over time per arm.
3. AI science — herding (unanimity + disagreement entropy) for the LLM arms.

Forward-return Information Coefficient is intentionally NOT here yet: it needs
60-day forward labels (Phase 16 E3), which do not exist until enough time has
elapsed. The report marks it "pending" rather than computing a noisy proxy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hifi.execution.portfolio_recorder import load_equity_curve, load_returns

# Arm registry mirrors run_phase16_live._ACCOUNTS (kept local to avoid importing
# the script). label + LLM flag + walk-forward condition for sidecar lookup.
ARMS: dict[str, dict] = {
    "A": {"label": "parallel ensemble (champion)", "llm": True, "condition": "parallel"},
    "B": {"label": "full sequential (herding contrast)", "llm": True, "condition": "full"},
    "C": {"label": "equal-weight control (null model)", "llm": False, "condition": "control"},
    "D": {"label": "riskbudget calm_exposure (quant)", "llm": False, "condition": "riskbudget"},
}


# ---------------------------------------------------------------------------
# Financial layer
# ---------------------------------------------------------------------------


def _safe_stat(fn, *args, **kwargs):
    """QuantStats stats raise / return NaN on too-short series; guard them."""
    try:
        v = fn(*args, **kwargs)
        v = float(v)
        return v if v == v else None  # drop NaN
    except Exception:
        return None


def equity_curves(data_dir: str = "data", rebased: bool = True) -> pd.DataFrame:
    """Aligned daily equity per arm (columns=arms). Rebased to 100 by default."""
    cols = {}
    for arm in ARMS:
        s = load_equity_curve(arm, data_dir)
        if s.empty:
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        cols[arm] = (100.0 * s / s.iloc[0]) if rebased and s.iloc[0] else s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def metrics_table(data_dir: str = "data", min_points: int = 20) -> pd.DataFrame:
    """Per-arm performance metrics. Metrics needing history are None until
    ``min_points`` daily returns exist (marked so the report says 'accumulating').
    """
    import quantstats as qs  # noqa: PLC0415

    rows = []
    for arm, meta in ARMS.items():
        eq = load_equity_curve(arm, data_dir)
        rets = load_returns(arm, data_dir)
        n = len(rets)
        enough = n >= min_points
        row = {
            "arm": arm,
            "strategy": meta["label"],
            "days": int(len(eq)),
            "return_days": int(n),
            "current_equity": float(eq.iloc[-1]) if len(eq) else None,
            "total_return_pct": (float(eq.iloc[-1] / eq.iloc[0] - 1) * 100
                                 if len(eq) > 1 and eq.iloc[0] else None),
            # history-dependent metrics: only once statistically meaningful
            "cagr_pct": (_safe_stat(qs.stats.cagr, rets) or 0) * 100 if enough else None,
            "sharpe": _safe_stat(qs.stats.sharpe, rets, periods=252) if enough else None,
            "sortino": _safe_stat(qs.stats.sortino, rets, periods=252) if enough else None,
            "max_drawdown_pct": ((_safe_stat(qs.stats.max_drawdown, rets) or 0) * 100
                                 if enough else None),
            "volatility_pct": ((_safe_stat(qs.stats.volatility, rets, periods=252) or 0) * 100
                               if enough else None),
            "enough_data": enough,
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def signal_distribution(account: str, data_dir: str = "data") -> pd.DataFrame:
    """Per decision_date: counts of Buy/Hold/Sell and orders placed."""
    eps = _read_jsonl(Path(data_dir) / "live" / account / "decisions.jsonl")
    rows = []
    for ep in eps:
        counts = {"Buy": 0, "Hold": 0, "Sell": 0}
        for s in ep.get("signals", []):
            d = s.get("decision")
            if d in counts:
                counts[d] += 1
        rows.append({
            "decision_date": ep.get("decision_date"),
            **counts,
            "n_orders": ep.get("n_orders", 0),
        })
    if not rows:
        return pd.DataFrame(columns=["decision_date", "Buy", "Hold", "Sell", "n_orders"])
    return pd.DataFrame(rows).drop_duplicates("decision_date", keep="last")


def exposure_series(account: str, data_dir: str = "data") -> pd.DataFrame:
    """Per decision_date capital deployment: invested / equity (DJ-119).

    The arms are not equally invested — as of 2026-07-28 A held one position on
    ~5% of equity while C held 98 on ~99%. Raw return, Sharpe and drawdown at
    that spread measure exposure, not signal quality, so any cross-arm reading
    of the financial layer has to carry this column beside it.
    """
    rows = []
    for snap in _read_jsonl(Path(data_dir) / "live" / account / "equity.jsonl"):
        equity = snap.get("equity") or 0.0
        invested = sum(p.get("market_value", 0.0) for p in snap.get("positions", []))
        rows.append({
            "decision_date": snap.get("decision_date"),
            "equity": equity,
            "cash": snap.get("cash"),
            "invested": invested,
            "n_positions": snap.get("n_positions", 0),
            "exposure": (invested / equity) if equity else None,
        })
    if not rows:
        return pd.DataFrame(columns=["decision_date", "equity", "cash", "invested",
                                     "n_positions", "exposure"])
    return pd.DataFrame(rows).drop_duplicates("decision_date", keep="last")


def exposure_adjusted_returns(account: str, data_dir: str = "data") -> pd.Series:
    """Daily returns divided by that day's exposure — the return the arm would
    have earned fully invested (DJ-119).

    This is the minimum correction that makes arms with different deployment
    comparable at all. It is deliberately crude: it assumes the uninvested
    remainder earns zero and that exposure is constant within the day. It is a
    sanity overlay on the financial layer, NOT a substitute for IC — the
    headline diversity claim rests on IC and herding, which never touch
    position sizing. Days with exposure below 1% are dropped rather than
    divided by a near-zero denominator.
    """
    returns = load_returns(account, data_dir)
    if returns is None or len(returns) == 0:
        return pd.Series(dtype=float)
    exp = exposure_series(account, data_dir)
    if exp.empty:
        return pd.Series(dtype=float)
    exp = exp.dropna(subset=["exposure"])
    exp["decision_date"] = pd.to_datetime(exp["decision_date"])
    lookup = exp.set_index("decision_date")["exposure"]
    lookup = lookup[lookup >= 0.01]
    aligned = lookup.reindex(pd.to_datetime(returns.index)).ffill()
    out = returns / aligned.values
    return out.dropna()


def halted_days(account: str, data_dir: str = "data") -> pd.DataFrame:
    """Days the circuit breaker stopped this arm from trading (DJ-119).

    Rows written before DJ-119 have no "action" key and were all halts. Flags
    (a position breach too small to matter) are excluded — they are observations,
    not interventions, and must not be read as no-trade days.
    """
    rows = []
    for e in _read_jsonl(Path(data_dir) / "live" / account / "circuit_breakers.jsonl"):
        if e.get("action", "halt") != "halt":
            continue
        rows.append({
            "date": (e.get("timestamp") or "")[:10],
            "trigger": e.get("trigger"),
            "ticker": e.get("ticker") or None,
            "value": e.get("value"),
        })
    if not rows:
        return pd.DataFrame(columns=["date", "trigger", "ticker", "value"])
    return pd.DataFrame(rows).drop_duplicates("date", keep="last")


# ---------------------------------------------------------------------------
# AI-science layer (LLM arms only)
# ---------------------------------------------------------------------------


def herding_series(account: str, data_dir: str = "data") -> pd.DataFrame:
    """Per decision_date herding metrics from ensemble sidecars (LLM arms).

    - unanimity: fraction of tickers where all agents agreed (Page-theorem herding)
    - mean_entropy: mean cross-agent disagreement entropy (higher = more diverse)
    Empty for non-LLM arms (C, D) which have no ensemble.
    """
    base = Path(data_dir) / "live" / account / "walkforward"
    if not base.exists():
        return pd.DataFrame()
    rows: dict[str, dict] = {}
    # rglob is layout-robust: works for the month-keyed legacy path and the
    # date-partitioned live path (issue #2). Group by the file's own as_of_date.
    for f in base.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        ed = d.get("ensemble_decision") or {}
        date = d.get("as_of_date")
        if not date or not ed:
            continue
        agent_decs = ed.get("agent_decisions") or []
        unanimous = 1 if agent_decs and len(set(agent_decs)) == 1 else 0
        entropy = ed.get("disagreement_entropy")
        r = rows.setdefault(date, {"n": 0, "unanimous": 0, "entropy_sum": 0.0, "entropy_n": 0})
        r["n"] += 1
        r["unanimous"] += unanimous
        if entropy is not None:
            r["entropy_sum"] += float(entropy)
            r["entropy_n"] += 1
    out = []
    for date, r in sorted(rows.items()):
        out.append({
            "decision_date": date,
            "n_tickers": r["n"],
            "unanimity": r["unanimous"] / r["n"] if r["n"] else None,
            "mean_entropy": r["entropy_sum"] / r["entropy_n"] if r["entropy_n"] else None,
        })
    return pd.DataFrame(out)


def experiment_status(data_dir: str = "data") -> dict:
    """Headline metadata for the report header (days running, arms live, etc.)."""
    curves = equity_curves(data_dir, rebased=False)
    live_arms = [a for a in ARMS if not load_equity_curve(a, data_dir).empty]
    return {
        "arms_total": len(ARMS),
        "arms_live": live_arms,
        "days_of_data": int(curves.shape[0]) if not curves.empty else 0,
        "first_date": str(curves.index[0].date()) if not curves.empty else None,
        "last_date": str(curves.index[-1].date()) if not curves.empty else None,
        "ic_status": "pending — needs 60-day forward labels (Phase 16 E3)",
    }
