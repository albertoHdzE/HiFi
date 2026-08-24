"""Per-stock decision traceability for the live experiment (DJ-120).

Answers three questions for any (arm, ticker, date):

- **Why**  — each agent's decision, confidence, rationale, key concern and the
  data gaps it declared, then how the ensemble aggregated them.
- **How**  — which model produced each signal, which MCP tool calls backed it
  (by call_id), whether those calls succeeded, and how long they took.
- **When** — the decision date, the filing period behind the fundamental view,
  and whether the decision reached the broker as an order.

Provenance is a first-class column, not a footnote. The defect that motivated
this module (DJ-120) was invisible precisely because a failed tool call and a
considered bearish opinion looked identical downstream: agents told
``TICKER_NOT_FOUND`` replied "no data available -- Sell" at confidence 1.0, and
83 of 98 tickers were in that state for the entire first month of live trading.
Any view of these decisions that cannot distinguish evidence from absence will
hide the same class of failure again, so every function here carries tool-call
health alongside the decision it produced.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

# Agents that emit a signal, in canonical ensemble order. The contrarian pass
# reviews the others and contributes no standalone decision.
AGENTS = ["fundamental", "technical", "risk", "macro", "sentiment"]

__all__ = [
    "AGENTS",
    "agent_behaviour",
    "degenerate_agents",
    "format_trace",
    "iter_sidecars",
    "order_index",
    "provenance_matrix",
    "ticker_history",
    "trace",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _walkforward_dir(account: str, data_dir: str = "data") -> Path:
    return Path(data_dir) / "live" / account / "walkforward"


def iter_sidecars(
    account: str,
    data_dir: str = "data",
    ticker: str | None = None,
    date: str | None = None,
) -> Iterator[dict]:
    """Yield ensemble sidecars for an arm, optionally filtered.

    ``rglob`` keeps this robust to both walk-forward layouts: the month-keyed
    legacy path and the date-partitioned live path introduced for issue #2.
    """
    base = _walkforward_dir(account, data_dir)
    if not base.exists():
        return
    pattern = f"{ticker}.json" if ticker else "*.json"
    for f in sorted(base.rglob(pattern)):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if date and d.get("as_of_date") != date:
            continue
        yield d


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def order_index(account: str, data_dir: str = "data") -> dict[tuple[str, str], dict]:
    """Map (decision_date, ticker) -> the order actually placed, if any.

    This is what closes the loop between an opinion and a trade. An arm can
    hold a strong view and still place nothing: the pipeline's rebalance band,
    position caps or the circuit breaker can all intervene between signal and
    order, and the report must show that gap rather than imply the LLM declined.
    """
    idx: dict[tuple[str, str], dict] = {}
    for ep in _read_jsonl(Path(data_dir) / "live" / account / "decisions.jsonl"):
        d = ep.get("decision_date")
        for o in ep.get("orders") or []:
            if o.get("ticker"):
                idx[(d, o["ticker"])] = o
    return idx


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _tool_calls(block: dict) -> list[tuple[str, str | None]]:
    """Return (tool_name, error_code_or_None) for each MCP call in an agent block."""
    out = []
    for k, v in block.items():
        if k == "signal" or not isinstance(v, dict):
            continue
        if "call_id" in v or "error" in v:
            out.append((k, v.get("error")))
    return out


def provenance_matrix(account: str, data_dir: str = "data") -> pd.DataFrame:
    """Per (date, agent): tool-call counts, failures and the failure rate.

    A healthy arm sits near zero. Sustained high rates mean the agents are
    reasoning about missing data, and every decision metric downstream --
    signal mix, herding, IC -- is measuring the data layer, not the models.
    """
    rows: dict[tuple[str, str], dict] = {}
    for d in iter_sidecars(account, data_dir):
        date = d.get("as_of_date")
        if not date:
            continue
        for key, block in d.items():
            if not key.endswith("_analysis") or not isinstance(block, dict):
                continue
            agent = (block.get("signal") or {}).get("agent_type") or key[: -len("_analysis")]
            calls = _tool_calls(block)
            r = rows.setdefault((date, agent), {"calls": 0, "failed": 0, "tickers": 0})
            r["calls"] += len(calls)
            r["failed"] += sum(1 for _, e in calls if e)
            r["tickers"] += 1
    if not rows:
        return pd.DataFrame(
            columns=["decision_date", "agent", "tickers", "calls", "failed", "failure_rate"]
        )
    out = [
        {
            "decision_date": date,
            "agent": agent,
            "tickers": r["tickers"],
            "calls": r["calls"],
            "failed": r["failed"],
            "failure_rate": (r["failed"] / r["calls"]) if r["calls"] else 0.0,
        }
        for (date, agent), r in sorted(rows.items())
    ]
    return pd.DataFrame(out)


def agent_behaviour(
    account: str, data_dir: str = "data", date: str | None = None
) -> pd.DataFrame:
    """Per agent: decision mix, mean confidence and tool-failure rate.

    Splits each agent's decisions by whether its tool calls succeeded. That
    split is the diagnostic: an agent whose behaviour changes completely
    between the two columns is reporting on data availability, not on the
    security.

    ``date`` scopes the analysis to one decision date. Without it every
    sidecar ever written is pooled, which after a defect makes the *history*
    the finding rather than the present state: on 2026-08-18 the pooled view
    still flagged fundamental and sentiment as constant purely on the strength
    of the DJ-120 starved period, while that day alone had no agent above a
    0.948 modal share. Pass a date when asking "is the ensemble healthy now".
    """
    acc: dict[str, dict] = {}
    for d in iter_sidecars(account, data_dir, date=date):
        for key, block in d.items():
            if not key.endswith("_analysis") or not isinstance(block, dict):
                continue
            sig = block.get("signal") or {}
            dec = sig.get("decision")
            if not dec:
                continue
            agent = sig.get("agent_type") or key[: -len("_analysis")]
            failed = any(e for _, e in _tool_calls(block))
            a = acc.setdefault(agent, {
                "n": 0, "conf_sum": 0.0, "failed": 0,
                "clean": {}, "starved": {}, "models": set(),
            })
            a["n"] += 1
            a["conf_sum"] += float(sig.get("confidence") or 0.0)
            bucket = a["starved" if failed else "clean"]
            bucket[dec] = bucket.get(dec, 0) + 1
            if failed:
                a["failed"] += 1
            if sig.get("model_id"):
                a["models"].add(str(sig["model_id"]).split("/")[-1])

    rows = []
    for agent, a in sorted(acc.items()):
        merged: dict[str, int] = {}
        for b in (a["clean"], a["starved"]):
            for k, v in b.items():
                merged[k] = merged.get(k, 0) + v
        top = max(merged.values()) if merged else 0
        rows.append({
            "agent": agent,
            "n": a["n"],
            "mean_confidence": a["conf_sum"] / a["n"] if a["n"] else None,
            "tool_failure_rate": a["failed"] / a["n"] if a["n"] else 0.0,
            "modal_share": top / a["n"] if a["n"] else None,
            "decisions": merged,
            "when_tools_ok": a["clean"],
            "when_tools_failed": a["starved"],
            "models": sorted(a["models"]),
        })
    return pd.DataFrame(rows)


def degenerate_agents(
    account: str, data_dir: str = "data", threshold: float = 0.98,
    date: str | None = None,
) -> pd.DataFrame:
    """Agents emitting the same decision on at least ``threshold`` of passes.

    A near-constant member adds no information to an ensemble and deflates
    every disagreement statistic. Because unanimity requires all members to
    agree, a constant agent also *suppresses* measured herding -- so this
    check has to run before any Page-theorem claim is read off the data.

    Pass ``date`` to ask about a single decision date. Pooled across all dates
    this keeps reporting agents that were constant only during a past defect,
    which is history, not current state (see ``agent_behaviour``).
    """
    beh = agent_behaviour(account, data_dir, date=date)
    if beh.empty:
        return beh
    flagged = beh[beh["modal_share"] >= threshold].copy()
    flagged["verdict"] = "constant — contributes no information"
    return flagged[["agent", "n", "modal_share", "mean_confidence", "decisions", "verdict"]]


# ---------------------------------------------------------------------------
# Per-stock traceability
# ---------------------------------------------------------------------------


def trace(
    account: str, ticker: str, date: str, data_dir: str = "data"
) -> dict[str, Any] | None:
    """Full why/how/when record for one (arm, ticker, date).

    Returns None when the arm produced no ensemble sidecar for that pair --
    which is itself meaningful for the non-LLM arms C and D, neither of which
    runs agents at all.
    """
    found = next(iter_sidecars(account, data_dir, ticker=ticker, date=date), None)
    if found is None:
        return None

    agents = []
    for key, block in found.items():
        if not key.endswith("_analysis") or not isinstance(block, dict):
            continue
        sig = block.get("signal") or {}
        calls = _tool_calls(block)
        agents.append({
            "agent": sig.get("agent_type") or key[: -len("_analysis")],
            "decision": sig.get("decision"),
            "confidence": sig.get("confidence"),
            "rationale": sig.get("rationale"),
            "key_concern": sig.get("key_concern"),
            "data_gaps": sig.get("data_gaps") or [],
            "model_id": (str(sig.get("model_id")).split("/")[-1]
                         if sig.get("model_id") else None),
            "latency_ms": block.get("latency_ms"),
            "prompt_version": block.get("prompt_version"),
            "call_ids": sig.get("call_ids") or [],
            "tool_calls": calls,
            "tools_failed": [t for t, e in calls if e],
        })
    order = AGENTS + [a["agent"] for a in agents if a["agent"] not in AGENTS]
    agents.sort(key=lambda a: order.index(a["agent"]) if a["agent"] in order else 99)

    ens = found.get("ensemble_decision") or {}
    placed = order_index(account, data_dir).get((date, ticker))

    return {
        "account": account,
        "ticker": ticker,
        "date": date,
        "agents": agents,
        "ensemble": ens,
        "order": placed,
        "n_tool_failures": sum(len(a["tools_failed"]) for a in agents),
        "evidence_complete": all(not a["tools_failed"] for a in agents),
    }


def ticker_history(
    account: str, ticker: str, data_dir: str = "data"
) -> pd.DataFrame:
    """One row per decision date for a ticker: the ensemble view over time.

    Carries ``n_tool_failures`` beside every decision so a run of Sells that
    merely tracks a broken data feed cannot be mistaken for a sustained view.
    """
    orders = order_index(account, data_dir)
    rows = []
    for d in iter_sidecars(account, data_dir, ticker=ticker):
        date = d.get("as_of_date")
        ens = d.get("ensemble_decision") or {}
        fails = 0
        per_agent = {}
        for key, block in d.items():
            if not key.endswith("_analysis") or not isinstance(block, dict):
                continue
            sig = block.get("signal") or {}
            agent = sig.get("agent_type") or key[: -len("_analysis")]
            if sig.get("decision"):
                per_agent[agent] = sig["decision"]
            fails += sum(1 for _, e in _tool_calls(block) if e)
        o = orders.get((date, ticker))
        rows.append({
            "decision_date": date,
            "ensemble": ens.get("collective_decision"),
            "confidence": ens.get("collective_confidence"),
            "entropy": ens.get("disagreement_entropy"),
            "agreement": ens.get("agreement"),
            "n_tool_failures": fails,
            "ordered": bool(o),
            "order_side": (o or {}).get("side"),
            "order_qty": (o or {}).get("qty"),
            **{f"a_{k}": v for k, v in per_agent.items()},
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_trace(t: dict[str, Any] | None, width: int = 100) -> str:
    """Render a trace as readable text for the notebook."""
    if t is None:
        return "No ensemble sidecar (arm runs no agents, or no decision that date)."

    def wrap(text: str | None, indent: int = 9) -> str:
        if not text:
            return "-"
        import textwrap  # noqa: PLC0415
        return textwrap.fill(
            text, width=width, subsequent_indent=" " * indent
        )

    lines = [
        f"{t['account']} / {t['ticker']} / {t['date']}",
        "=" * width,
    ]
    for a in t["agents"]:
        if a["decision"] is None and not a["rationale"]:
            continue
        bad = f"  [TOOL FAILURES: {', '.join(a['tools_failed'])}]" if a["tools_failed"] else ""
        lines.append("")
        lines.append(f"{a['agent'].upper():<12} {a['decision']} @ {a['confidence']}{bad}")
        lines.append(f"  model:   {a['model_id']}  ({a['latency_ms']} ms)")
        lines.append(f"  why:     {wrap(a['rationale'])}")
        if a["key_concern"]:
            lines.append(f"  concern: {wrap(a['key_concern'])}")
        if a["data_gaps"]:
            lines.append(f"  gaps:    {', '.join(a['data_gaps'])}")

    e = t["ensemble"]
    lines += ["", "-" * width]
    lines.append(
        f"ENSEMBLE     {e.get('collective_decision')} @ "
        f"{e.get('collective_confidence')}   "
        f"votes={e.get('agent_decisions')}"
    )
    lines.append(
        f"  entropy={e.get('disagreement_entropy')}  "
        f"dispersion={e.get('opinion_dispersion')}  "
        f"unanimous={e.get('agreement')}"
    )
    o = t["order"]
    lines.append(
        f"  ORDER:  {o['side']} {o['qty']} ({o.get('status')})" if o
        else "  ORDER:  none placed — signal did not clear the allocator "
             "(rebalance band, cap, or breaker)"
    )
    lines.append(
        "  EVIDENCE: complete" if t["evidence_complete"]
        else f"  EVIDENCE: INCOMPLETE — {t['n_tool_failures']} failed tool call(s); "
             "treat this decision as uninformed"
    )
    return "\n".join(lines)
