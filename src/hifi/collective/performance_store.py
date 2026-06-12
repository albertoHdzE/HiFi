"""
Agent performance history store (P9-E4).

Manages persistence and retrieval of AgentPerformanceHistory, which tracks
per-agent decision accuracy to power performance_weighted_vote() (D-07).

Design:
- Storage: data/agent_performance_history.json (flat JSON at Phase 9 scale)
- Atomic writes: write to .tmp, then rename to prevent corruption on crash
- Fallback: when the file is absent, return uniform weights (1/N per agent type)
  so run_ensemble() works before the bootstrap has been run
- compute_weights() is a pure function: given a list of labeled DecisionRecords,
  returns accuracy per agent_type (number correct / number labeled per type)

Phase 10 will migrate to Parquet when the bootstrap expands to 20+ tickers.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from hifi.collective.schemas import AgentPerformanceHistory, DecisionRecord

_DEFAULT_FILENAME = "agent_performance_history.json"
_INITIAL_AGENT_TYPES = ("fundamental", "technical", "risk", "macro")
_INITIAL_WEIGHT = 0.25  # 1/4 uniform — four voting agent types in Phase 9


def _history_path(data_dir: str | None) -> Path:
    root = data_dir or os.environ.get("HIFI_DATA_DIR", ".")
    return Path(root) / _DEFAULT_FILENAME


def load_history(data_dir: str | None = None) -> AgentPerformanceHistory:
    """
    Load AgentPerformanceHistory from {data_dir}/agent_performance_history.json.

    If the file does not exist, return a fresh history with uniform weights and
    no records — safe for first-run before the bootstrap has been executed.
    """
    path = _history_path(data_dir)
    if not path.exists():
        return AgentPerformanceHistory(
            records=[],
            weights={t: _INITIAL_WEIGHT for t in _INITIAL_AGENT_TYPES},
            last_updated="",
            n_labeled=0,
        )
    return AgentPerformanceHistory.model_validate_json(path.read_text())


def save_history(
    history: AgentPerformanceHistory, data_dir: str | None = None
) -> None:
    """
    Persist AgentPerformanceHistory atomically to {data_dir}/agent_performance_history.json.

    Uses write-to-tmp + rename to prevent partial writes on crash.
    """
    path = _history_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(history.model_dump_json(indent=2))
    tmp.rename(path)


def compute_weights(records: list[DecisionRecord]) -> dict[str, float]:
    """
    Compute accuracy weights per agent_type from labeled DecisionRecords.

    accuracy(agent_type) = n_correct / n_labeled  for records with outcome_correct
    not None. Only labeled records contribute; unlabeled records are ignored.

    Returns uniform weights (_INITIAL_WEIGHT per type) when no labeled records
    exist, or fills missing agent types with _INITIAL_WEIGHT so callers always
    receive a complete dict for all four canonical agent types.
    """
    labeled = [r for r in records if r.outcome_correct is not None]
    if not labeled:
        return {t: _INITIAL_WEIGHT for t in _INITIAL_AGENT_TYPES}

    # Accumulate correct/total per agent_type
    correct_by_type: dict[str, int] = {}
    total_by_type: dict[str, int] = {}
    for r in labeled:
        correct_by_type[r.agent_type] = correct_by_type.get(r.agent_type, 0) + (
            1 if r.outcome_correct else 0
        )
        total_by_type[r.agent_type] = total_by_type.get(r.agent_type, 0) + 1

    weights: dict[str, float] = {
        t: correct_by_type[t] / total_by_type[t] for t in total_by_type
    }
    # Fill any canonical agent type absent from records with the initial weight
    for t in _INITIAL_AGENT_TYPES:
        weights.setdefault(t, _INITIAL_WEIGHT)
    return weights


def get_weights(data_dir: str | None = None) -> dict[str, float]:
    """
    Convenience: load history and return current weights dict.

    Called by ensemble_runner at the start of every run_ensemble() invocation.
    Fast path: if the history file is absent, returns uniform weights immediately.
    """
    return load_history(data_dir).weights


def update_and_save(
    records: list[DecisionRecord],
    data_dir: str | None = None,
) -> AgentPerformanceHistory:
    """
    Load existing history, append new records, recompute weights, and save.

    Returns the updated AgentPerformanceHistory. Used by the bootstrap script
    and future live-labeling routines (Phase 10).
    """
    history = load_history(data_dir)
    all_records = history.records + records
    updated_weights = compute_weights(all_records)
    updated = AgentPerformanceHistory(
        records=all_records,
        weights=updated_weights,
        last_updated=datetime.now(tz=UTC).isoformat(),
        n_labeled=0,  # auto-computed by model_validator
    )
    save_history(updated, data_dir)
    return updated
