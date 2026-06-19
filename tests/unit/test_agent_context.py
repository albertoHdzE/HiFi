"""
Unit tests for AgentContextStore (E3-T1, DJ-089b).

All tests use tmp_path for isolated LanceDB databases. No LLMs, no network.
Tests verify: write/read round-trip, read_prior canonical ordering,
clear_run removal, empty-store behaviour, format_prior_context output.
"""

from __future__ import annotations

from hifi.knowledge.agent_context import (
    CANONICAL_ORDER,
    AgentContextRecord,
    AgentContextStore,
    format_prior_context,
)

_TS = "2025-01-01T00:00:00+00:00"


def _record(
    run_id: str,
    agent_type: str,
    decision: str = "Buy",
    confidence: float = 0.70,
    summary: str = "Fundamentals look solid.",
    ticker: str = "AAPL",
    date: str = "2023-03-31",
) -> AgentContextRecord:
    return AgentContextRecord(
        run_id=run_id,
        ticker=ticker,
        date=date,
        agent_type=agent_type,
        analysis_summary=summary,
        decision=decision,
        confidence=confidence,
        created_at=_TS,
    )


def _store(tmp_path) -> AgentContextStore:
    db = str(tmp_path / "knowledge.lance")
    return AgentContextStore(namespace="test", db_path=db)


# ---------------------------------------------------------------------------
# Write / read round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_back(tmp_path):
    store = _store(tmp_path)
    rec = _record("run1", "fundamental")
    store.write(rec)

    results = store.read_prior("run1", "technical")
    assert len(results) == 1
    assert results[0].agent_type == "fundamental"
    assert results[0].decision == "Buy"
    assert results[0].run_id == "run1"


def test_write_multiple_and_read_all_prior(tmp_path):
    store = _store(tmp_path)
    store.write(_record("run1", "fundamental", decision="Buy"))
    store.write(_record("run1", "technical", decision="Hold"))
    store.write(_record("run1", "risk", decision="Sell"))

    results = store.read_prior("run1", "macro")
    agent_types = [r.agent_type for r in results]
    assert "fundamental" in agent_types
    assert "technical" in agent_types
    assert "risk" in agent_types
    assert "macro" not in agent_types
    assert "sentiment" not in agent_types


def test_read_prior_returns_canonical_order(tmp_path):
    store = _store(tmp_path)
    # Write in reverse order to verify sort
    store.write(_record("run1", "risk"))
    store.write(_record("run1", "fundamental"))
    store.write(_record("run1", "technical"))

    results = store.read_prior("run1", "macro")
    types = [r.agent_type for r in results]
    assert types == ["fundamental", "technical", "risk"]


# ---------------------------------------------------------------------------
# read_prior boundary conditions
# ---------------------------------------------------------------------------


def test_read_prior_fundamental_returns_empty(tmp_path):
    """fundamental is first in canonical order — no predecessors."""
    store = _store(tmp_path)
    store.write(_record("run1", "fundamental"))
    assert store.read_prior("run1", "fundamental") == []


def test_read_prior_empty_store_returns_empty(tmp_path):
    store = _store(tmp_path)
    assert store.read_prior("run_missing", "technical") == []


def test_read_prior_different_run_id_not_returned(tmp_path):
    store = _store(tmp_path)
    store.write(_record("run1", "fundamental"))
    store.write(_record("run2", "fundamental"))

    results = store.read_prior("run1", "technical")
    assert all(r.run_id == "run1" for r in results)
    assert len(results) == 1


def test_read_prior_contrarian_sees_all_five(tmp_path):
    """Contrarian reads prior context from all 5 voting agents."""
    store = _store(tmp_path)
    for agent in CANONICAL_ORDER[:-1]:  # all except contrarian
        store.write(_record("run1", agent))

    results = store.read_prior("run1", "contrarian")
    assert len(results) == 5
    assert [r.agent_type for r in results] == CANONICAL_ORDER[:-1]


def test_read_prior_unknown_agent_returns_all(tmp_path):
    """Unknown before_agent defaults to returning all records for the run."""
    store = _store(tmp_path)
    for agent in CANONICAL_ORDER:
        store.write(_record("run1", agent))

    results = store.read_prior("run1", "unknown_agent")
    assert len(results) == len(CANONICAL_ORDER)


# ---------------------------------------------------------------------------
# clear_run
# ---------------------------------------------------------------------------


def test_clear_run_removes_all_records(tmp_path):
    store = _store(tmp_path)
    store.write(_record("run1", "fundamental"))
    store.write(_record("run1", "technical"))
    store.write(_record("run2", "fundamental"))

    store.clear_run("run1")

    assert store.read_prior("run1", "contrarian") == []


def test_clear_run_does_not_affect_other_runs(tmp_path):
    store = _store(tmp_path)
    store.write(_record("run1", "fundamental"))
    store.write(_record("run2", "fundamental"))

    store.clear_run("run1")

    results = store.read_prior("run2", "technical")
    assert len(results) == 1
    assert results[0].run_id == "run2"


# ---------------------------------------------------------------------------
# format_prior_context
# ---------------------------------------------------------------------------


def test_format_prior_context_empty_returns_empty():
    assert format_prior_context([], "AAPL", "2023-03-31") == ""


def test_format_prior_context_header_present():
    recs = [_record("r1", "fundamental", decision="Buy", confidence=0.72)]
    text = format_prior_context(recs, "AAPL", "2023-03-31")
    assert "[Prior Agent Analyses for AAPL on 2023-03-31]" in text


def test_format_prior_context_contains_decision_and_confidence():
    recs = [_record("r1", "fundamental", decision="Buy", confidence=0.72)]
    text = format_prior_context(recs, "AAPL", "2023-03-31")
    assert "Buy" in text
    assert "0.72" in text


def test_format_prior_context_contains_all_agents():
    recs = [
        _record("r1", "fundamental", decision="Buy"),
        _record("r1", "technical", decision="Hold"),
    ]
    text = format_prior_context(recs, "AAPL", "2023-03-31")
    assert "Fundamental Agent" in text
    assert "Technical Agent" in text


def test_format_prior_context_newlines_stripped_from_summary():
    recs = [_record("r1", "fundamental", summary="Line one.\nLine two.")]
    text = format_prior_context(recs, "AAPL", "2023-03-31")
    # The summary line should not contain a raw newline
    lines = text.splitlines()
    agent_line = next(line for line in lines if "Fundamental" in line)
    assert "\n" not in agent_line


# ---------------------------------------------------------------------------
# CANONICAL_ORDER sanity
# ---------------------------------------------------------------------------


def test_canonical_order_length():
    assert len(CANONICAL_ORDER) == 6


def test_canonical_order_first_and_last():
    assert CANONICAL_ORDER[0] == "fundamental"
    assert CANONICAL_ORDER[-1] == "contrarian"


def test_canonical_order_contains_all_agents():
    expected = {"fundamental", "technical", "risk", "macro", "sentiment", "contrarian"}
    assert set(CANONICAL_ORDER) == expected
