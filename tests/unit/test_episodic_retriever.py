"""
Unit tests for EpisodicRetriever (E5-T3, DJ-092).

Tests:
- Retrieval from a populated store returns formatted prefix.
- Empty store returns empty string.
- Format correctness: header, date, ticker, regime, decision, outcome.
- n capping: at most n episodes returned.
- Temporal discipline: no future-date episodes returned.
"""

from __future__ import annotations

import uuid

import pytest

from hifi.knowledge.episodic_retriever import EpisodicRetriever, _format_episodes
from hifi.knowledge.episodic_store import EpisodeRecord, EpisodicStore


class _MockEmbedding:
    @property
    def dimensions(self):
        return 32

    def embed(self, texts):
        return [[0.1] * 32] * len(texts)

    def embed_one(self, text):
        return [0.1] * 32


@pytest.fixture
def store(tmp_path):
    return EpisodicStore(
        embedding_model=_MockEmbedding(),
        namespace="test-retriever",
        db_path=str(tmp_path / "knowledge.lance"),
    )


@pytest.fixture
def retriever(store):
    return EpisodicRetriever(store)


def _ep(
    decision_date: str = "2022-01-01",
    outcome_correct: bool | None = True,
    forward_return: float | None = 0.08,
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=str(uuid.uuid4()),
        ticker="AAPL",
        decision_date=decision_date,
        regime_label="bull_low_vol",
        sector="Information Technology",
        agent_type="fundamental",
        decision="Buy",
        confidence=0.75,
        collective_decision="Buy",
        forward_return=forward_return,
        outcome_correct=outcome_correct,
        reasoning_summary="Strong earnings and solid cash flow.",
        labeled_at="2022-03-15" if outcome_correct is not None else None,
    )


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------


def test_empty_store_returns_empty_string(retriever):
    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
    )
    assert result == ""


# ---------------------------------------------------------------------------
# Populated store
# ---------------------------------------------------------------------------


def test_retrieve_returns_formatted_string(store, retriever):
    store.add(_ep())
    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
    )
    assert result != ""
    assert "[Episodic Memory" in result


def test_format_contains_required_fields(store, retriever):
    store.add(_ep(forward_return=0.08))
    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
    )
    assert "Date:" in result
    assert "AAPL" in result
    assert "bull_low_vol" in result
    assert "Buy" in result
    assert "CORRECT" in result
    assert "+8.0%" in result


def test_format_no_forward_return(store, retriever):
    ep = _ep(forward_return=None, outcome_correct=True)
    ep2 = ep.model_copy(update={"labeled_at": "2022-03-01"})
    store.add(ep2)
    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
    )
    assert "CORRECT" in result


# ---------------------------------------------------------------------------
# n capping
# ---------------------------------------------------------------------------


def test_n_capping(store, retriever):
    for _ in range(10):
        store.add(_ep())
    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
        n=3,
    )
    # Count "Date:" occurrences — one per episode
    assert result.count("Date:") <= 3


def test_n_one(store, retriever):
    for _ in range(5):
        store.add(_ep())
    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
        n=1,
    )
    assert result.count("Date:") == 1
    assert "1 successful past decision" in result


# ---------------------------------------------------------------------------
# Temporal discipline
# ---------------------------------------------------------------------------


def test_temporal_discipline_no_future_episodes(store, retriever):
    """Episodes with decision_date >= as_of_date must not be returned."""
    future_ep = _ep(decision_date="2023-06-01")  # after as_of_date
    store.add(future_ep)

    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",   # as_of_date is before the stored episode
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
    )
    assert result == ""


def test_temporal_discipline_same_date_excluded(store, retriever):
    """Episodes with decision_date == as_of_date must not be returned."""
    ep = _ep(decision_date="2023-01-01")
    store.add(ep)

    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
    )
    assert result == ""


def test_temporal_discipline_past_included(store, retriever):
    """Episodes with decision_date < as_of_date are included."""
    past_ep = _ep(decision_date="2022-06-01")
    store.add(past_ep)

    result = retriever.retrieve(
        ticker="AAPL",
        date="2023-01-01",
        agent_type="fundamental",
        regime="bull_low_vol",
        sector="Information Technology",
    )
    assert result != ""


# ---------------------------------------------------------------------------
# _format_episodes helper
# ---------------------------------------------------------------------------


def test_format_episodes_singular():
    ep = EpisodeRecord(
        episode_id=str(uuid.uuid4()),
        ticker="JPM",
        decision_date="2022-06-30",
        regime_label="rate_shock",
        sector="Financials",
        agent_type="risk",
        decision="Sell",
        confidence=0.80,
        collective_decision="Sell",
        forward_return=-0.12,
        outcome_correct=True,
        reasoning_summary="Rate shock pressuring bank margins.",
        labeled_at="2022-09-01",
    )
    text = _format_episodes([ep], n=1)
    assert "1 successful past decision in similar conditions" in text
    assert "rate_shock" in text
    assert "Sell" in text
    assert "-12.0%" in text


def test_format_episodes_plural():
    eps = [
        EpisodeRecord(
            episode_id=str(uuid.uuid4()),
            ticker="AAPL",
            decision_date=f"2022-0{i+1}-01",
            regime_label="bull_low_vol",
            sector="IT",
            agent_type="fundamental",
            decision="Buy",
            confidence=0.7,
            reasoning_summary="Solid growth.",
            forward_return=0.05,
            outcome_correct=True,
        )
        for i in range(3)
    ]
    text = _format_episodes(eps, n=3)
    assert "3 successful past decisions in similar conditions" in text
