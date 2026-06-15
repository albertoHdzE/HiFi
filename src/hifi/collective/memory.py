"""
Agent memory store for Phase 13 (P13-E4, DJ-076).

Implements in-context decision history for all 5 voting agents.
Persistence: JSON file per agent per ticker at {store_path}/{agent_type}/{ticker}.json.
Recall returns the last N records, most recent first.
format_for_prompt() builds the structured prefix injected before the analytical prompt.

Design (DJ-076):
- In-context prefix (last 3 decisions): low complexity, zero retrieval latency.
- 3 decisions x ~200 tokens = 600 tokens total (within qwen2.5-coder-32b 32K context).
- actual_60d_return populated by label-outcomes pipeline (Phase 10 / DJ-060).
- Memory decay calibration deferred to Phase 14 (OQ-AG02).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


class AgentMemoryRecord(BaseModel):
    """
    One agent decision record for a (ticker, as_of_date) pair (P13-E4-T1).

    actual_60d_return and outcome_correct are None until the 60-day forward
    window closes and the label-outcomes pipeline writes the result (DJ-060).
    """

    ticker: str
    as_of_date: str          # ISO 8601 date string
    agent_type: str          # "fundamental" | "technical" | "risk" | "macro" | "sentiment"
    decision: Literal["Buy", "Hold", "Sell"]
    confidence: float        # [0.0, 1.0]
    actual_60d_return: float | None = None
    outcome_correct: bool | None = None

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v


class AgentMemoryStore:
    """
    Persistent per-agent per-ticker decision store (P13-E4-T2).

    Storage layout:
        {store_path}/{agent_type}/{ticker}.json  — JSON array of AgentMemoryRecord dicts

    Files are appended-only (new records go to the end, oldest-first).
    recall() returns the last n records, most recent first.
    """

    def __init__(self, store_path: str | Path) -> None:
        self._root = Path(store_path)

    def _file(self, agent_type: str, ticker: str) -> Path:
        return self._root / agent_type / f"{ticker}.json"

    def record(self, rec: AgentMemoryRecord) -> None:
        """Append rec to the per-agent per-ticker JSON file (creates if missing)."""
        p = self._file(rec.agent_type, rec.ticker)
        p.parent.mkdir(parents=True, exist_ok=True)

        existing: list[dict] = []
        if p.exists():
            with p.open() as f:
                existing = json.load(f)

        existing.append(rec.model_dump())
        with p.open("w") as f:
            json.dump(existing, f)

    def recall(
        self,
        ticker: str,
        agent_type: str,
        n: int = 3,
    ) -> list[AgentMemoryRecord]:
        """
        Return last n records for (ticker, agent_type), most recent first.

        Returns [] when no history exists or the file is missing.
        If fewer than n records exist, returns all available records.
        """
        p = self._file(agent_type, ticker)
        if not p.exists():
            return []

        with p.open() as f:
            raw: list[dict] = json.load(f)

        records = [AgentMemoryRecord(**r) for r in raw]
        return list(reversed(records[-n:]))

    def format_for_prompt(self, records: list[AgentMemoryRecord]) -> str:
        """
        Build the structured memory prefix for injection into an agent prompt (DJ-076).

        Returns the no-history sentinel string when records is empty.
        Format (per DJ-076):
            [Agent Memory — last N decisions for {ticker}]
            {as_of_date}: {decision} (confidence={confidence:.2f}) → actual_60d_return={r:.1%}
            ...
        The outcome line is omitted when actual_60d_return is None.
        """
        if not records:
            return "No prior decisions recorded for this ticker."

        ticker = records[0].ticker
        lines = [f"[Agent Memory — last {len(records)} decisions for {ticker}]"]
        for r in records:
            outcome = (
                f" \u2192 actual_60d_return={r.actual_60d_return:.1%}"
                if r.actual_60d_return is not None
                else ""
            )
            lines.append(
                f"{r.as_of_date}: {r.decision} (confidence={r.confidence:.2f}){outcome}"
            )
        return "\n".join(lines)
