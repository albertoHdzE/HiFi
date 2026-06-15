"""Tests for Agent Memory (P13-E4-T1/T2, DJ-076)."""

import json
import pytest
from pathlib import Path

from hifi.collective.memory import AgentMemoryRecord, AgentMemoryStore


# ---------------------------------------------------------------------------
# AgentMemoryRecord
# ---------------------------------------------------------------------------

def test_record_valid():
    r = AgentMemoryRecord(
        ticker="AAPL", as_of_date="2023-03-31", agent_type="fundamental",
        decision="Buy", confidence=0.8,
    )
    assert r.decision == "Buy"
    assert r.actual_60d_return is None
    assert r.outcome_correct is None


def test_record_confidence_bounds():
    with pytest.raises(Exception):
        AgentMemoryRecord(
            ticker="AAPL", as_of_date="2023-03-31", agent_type="technical",
            decision="Hold", confidence=1.5,
        )


def test_record_invalid_decision():
    with pytest.raises(Exception):
        AgentMemoryRecord(
            ticker="AAPL", as_of_date="2023-03-31", agent_type="risk",
            decision="Strong Buy", confidence=0.9,
        )


def test_record_with_outcome():
    r = AgentMemoryRecord(
        ticker="JPM", as_of_date="2022-12-31", agent_type="macro",
        decision="Sell", confidence=0.7,
        actual_60d_return=-0.12, outcome_correct=True,
    )
    assert r.actual_60d_return == pytest.approx(-0.12)
    assert r.outcome_correct is True


# ---------------------------------------------------------------------------
# AgentMemoryStore
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return AgentMemoryStore(tmp_path)


def _make_record(ticker="AAPL", date="2023-03-31", agent="fundamental",
                 decision="Buy", confidence=0.75):
    return AgentMemoryRecord(
        ticker=ticker, as_of_date=date, agent_type=agent,
        decision=decision, confidence=confidence,
    )


def test_store_record_creates_file(store, tmp_path):
    r = _make_record()
    store.record(r)
    assert (tmp_path / "fundamental" / "AAPL.json").exists()


def test_store_recall_empty(store):
    assert store.recall("AAPL", "fundamental") == []


def test_store_recall_single(store):
    r = _make_record()
    store.record(r)
    result = store.recall("AAPL", "fundamental")
    assert len(result) == 1
    assert result[0].decision == "Buy"


def test_store_recall_order_most_recent_first(store):
    for date in ["2022-03-31", "2022-06-30", "2022-09-30"]:
        store.record(_make_record(date=date, decision="Hold"))
    result = store.recall("AAPL", "fundamental", n=3)
    assert result[0].as_of_date == "2022-09-30"
    assert result[-1].as_of_date == "2022-03-31"


def test_store_recall_n_limit(store):
    for date in ["2021-12-31", "2022-03-31", "2022-06-30", "2022-09-30"]:
        store.record(_make_record(date=date))
    result = store.recall("AAPL", "fundamental", n=3)
    assert len(result) == 3
    assert result[0].as_of_date == "2022-09-30"


def test_store_recall_partial_history(store):
    store.record(_make_record(date="2023-01-31"))
    result = store.recall("AAPL", "fundamental", n=3)
    assert len(result) == 1


def test_store_different_agents_isolated(store):
    store.record(_make_record(agent="fundamental", decision="Buy"))
    store.record(_make_record(agent="technical", decision="Sell"))
    fund = store.recall("AAPL", "fundamental")
    tech = store.recall("AAPL", "technical")
    assert fund[0].decision == "Buy"
    assert tech[0].decision == "Sell"


def test_store_different_tickers_isolated(store):
    store.record(_make_record(ticker="AAPL", decision="Buy"))
    store.record(_make_record(ticker="JPM", decision="Sell"))
    assert store.recall("AAPL", "fundamental")[0].decision == "Buy"
    assert store.recall("JPM", "fundamental")[0].decision == "Sell"


def test_store_roundtrip(store):
    r = AgentMemoryRecord(
        ticker="XOM", as_of_date="2022-09-30", agent_type="risk",
        decision="Sell", confidence=0.65,
        actual_60d_return=-0.08, outcome_correct=True,
    )
    store.record(r)
    recalled = store.recall("XOM", "risk")[0]
    assert recalled.actual_60d_return == pytest.approx(-0.08)
    assert recalled.outcome_correct is True


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

def test_format_empty_returns_sentinel(store):
    result = store.format_for_prompt([])
    assert "No prior decisions recorded" in result


def test_format_single_record(store):
    r = _make_record(date="2023-03-31", decision="Buy", confidence=0.80)
    result = store.format_for_prompt([r])
    assert "Agent Memory" in result
    assert "2023-03-31" in result
    assert "Buy" in result
    assert "0.80" in result


def test_format_with_outcome(store):
    r = AgentMemoryRecord(
        ticker="AAPL", as_of_date="2022-12-31", agent_type="fundamental",
        decision="Hold", confidence=0.60, actual_60d_return=0.05,
    )
    result = store.format_for_prompt([r])
    assert "5.0%" in result or "5%" in result


def test_format_without_outcome_no_return_line(store):
    r = _make_record()
    result = store.format_for_prompt([r])
    assert "actual_60d_return" not in result


def test_format_multiple_records_count_in_header(store):
    records = [_make_record(date=f"2022-0{i+1}-31") for i in range(3)]
    result = store.format_for_prompt(records)
    assert "last 3 decisions" in result
