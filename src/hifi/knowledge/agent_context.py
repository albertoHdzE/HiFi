"""
AgentContextStore: per-run LanceDB context store for sequential ensemble (E3-T1, DJ-089b).

Stores each agent's decision summary in a LanceDB table keyed by run_id.
``run_sequential_ensemble`` reads prior agents' summaries and injects them as
structured context before each agent's analytical prompt.

Canonical execution order
--------------------------
fundamental → technical → risk → macro → sentiment → contrarian

``read_prior(run_id, "macro")`` returns records for fundamental, technical, risk.
``read_prior(run_id, "fundamental")`` returns an empty list (no predecessors).

Helper
------
``format_prior_context(records, ticker, date)`` produces the human-readable
text block injected into each agent's ``memory_prefix``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pyarrow as pa
from pydantic import BaseModel

from hifi.knowledge.namespaced_store import NamespacedLanceDB

logger = logging.getLogger(__name__)

CANONICAL_ORDER: list[str] = [
    "fundamental",
    "technical",
    "risk",
    "macro",
    "sentiment",
    "contrarian",
]

_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("date", pa.string()),
        pa.field("agent_type", pa.string()),
        pa.field("analysis_summary", pa.string()),
        pa.field("decision", pa.string()),
        pa.field("confidence", pa.float64()),
        pa.field("created_at", pa.string()),
    ]
)


class AgentContextRecord(BaseModel):
    """Single agent analysis record stored per sequential ensemble run."""

    run_id: str
    ticker: str
    date: str
    agent_type: str
    analysis_summary: str  # ≤300 char excerpt from agent rationale
    decision: str          # "Buy" | "Hold" | "Sell"
    confidence: float
    created_at: str        # ISO 8601


class AgentContextStore:
    """
    LanceDB-backed store for inter-agent context records.

    Parameters
    ----------
    namespace : str
        LanceDB namespace prefix (default "hifi-dev-context").
        Use different namespaces for dev / eval / live isolation.
    db_path : str | None
        Path to LanceDB directory. Defaults to
        ``{HIFI_DATA_DIR}/knowledge.lance`` (env var, else ``data/knowledge.lance``).
    """

    TABLE = "agent_context"

    def __init__(
        self,
        namespace: str = "hifi-dev-context",
        db_path: str | None = None,
    ) -> None:
        _path = db_path or str(
            Path(os.environ.get("HIFI_DATA_DIR", "data")) / "knowledge.lance"
        )
        Path(_path).mkdir(parents=True, exist_ok=True)
        self._ns = NamespacedLanceDB(_path, namespace=namespace)
        self._table = self._ns.open_or_create_table(self.TABLE, _SCHEMA)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, record: AgentContextRecord) -> None:
        """Append a single context record to the table."""
        row = pa.table(
            {
                "run_id": [record.run_id],
                "ticker": [record.ticker],
                "date": [record.date],
                "agent_type": [record.agent_type],
                "analysis_summary": [record.analysis_summary],
                "decision": [record.decision],
                "confidence": [float(record.confidence)],
                "created_at": [record.created_at],
            },
            schema=_SCHEMA,
        )
        self._table.add(row)
        logger.debug(
            "Stored context for run_id=%s agent=%s decision=%s",
            record.run_id,
            record.agent_type,
            record.decision,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_prior(
        self,
        run_id: str,
        before_agent: str,
    ) -> list[AgentContextRecord]:
        """
        Return context records for all agents that precede ``before_agent``.

        Records are returned in canonical execution order.

        Parameters
        ----------
        run_id : str
            Identifier for this ensemble run.
        before_agent : str
            Agent type whose predecessors are requested (e.g. "macro" returns
            records for "fundamental", "technical", "risk").
        """
        try:
            idx = CANONICAL_ORDER.index(before_agent)
        except ValueError:
            idx = len(CANONICAL_ORDER)
        prior_agents = set(CANONICAL_ORDER[:idx])

        if not prior_agents:
            return []

        df = self._table.to_pandas()
        if df.empty:
            return []

        mask = (df["run_id"] == run_id) & (df["agent_type"].isin(prior_agents))
        rows = df[mask].copy()
        if rows.empty:
            return []

        order_map = {a: i for i, a in enumerate(CANONICAL_ORDER)}
        rows["_order"] = rows["agent_type"].map(order_map).fillna(len(CANONICAL_ORDER))
        rows = rows.sort_values("_order")

        return [
            AgentContextRecord(
                run_id=str(r["run_id"]),
                ticker=str(r["ticker"]),
                date=str(r["date"]),
                agent_type=str(r["agent_type"]),
                analysis_summary=str(r["analysis_summary"]),
                decision=str(r["decision"]),
                confidence=float(r["confidence"]),
                created_at=str(r["created_at"]),
            )
            for _, r in rows.iterrows()
        ]

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear_run(self, run_id: str) -> None:
        """Delete all context records for a given run_id."""
        self._table.delete(f"run_id = '{run_id}'")
        logger.debug("Cleared context records for run_id=%s", run_id)


# ---------------------------------------------------------------------------
# Shared formatting helper (used by ensemble_runner and graph)
# ---------------------------------------------------------------------------


def format_prior_context(
    records: list[AgentContextRecord],
    ticker: str,
    date: str,
) -> str:
    """
    Format prior agent context records as a structured text block.

    The returned string is prepended to each agent's ``memory_prefix`` by
    ``run_sequential_ensemble``.  Returns ``""`` when there are no records.

    Example output::

        [Prior Agent Analyses for AAPL on 2023-03-31]
        Fundamental Agent: Buy (conf=0.72) — P/E of 28.3 and ROE of 0.24 are solid…
        Technical Agent: Hold (conf=0.60) — RSI at 52 suggests neutral momentum…
    """
    if not records:
        return ""
    header = f"[Prior Agent Analyses for {ticker} on {date}]"
    lines = [header]
    for rec in records:
        display = rec.agent_type.replace("_", " ").title() + " Agent"
        summary = rec.analysis_summary.replace("\n", " ")
        lines.append(
            f"{display}: {rec.decision} (conf={rec.confidence:.2f}) — {summary}"
        )
    return "\n".join(lines)
