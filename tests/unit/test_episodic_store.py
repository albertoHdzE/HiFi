"""
Unit tests for EpisodeRecord schema and EpisodicStore (E5-T2, DJ-092).

Tests:
- add/search round-trip.
- outcome_correct filter (True/None/False).
- get_unlabeled_past_horizon returns only episodes past the horizon with no label.
- Sentinel encoding: "" for str None, NaN for float None, -1 for int8 None.
- Update (delete + re-add) is idempotent.

Mock EmbeddingModel injected at dim=32 (no LM Studio required).
"""

from __future__ import annotations

import math
import uuid
from datetime import date, timedelta

import pytest

from hifi.knowledge.episodic_store import EpisodeRecord, EpisodicStore

# ---------------------------------------------------------------------------
# Mock embedding model (dim=32)
# ---------------------------------------------------------------------------


class _MockEmbedding:
    """Deterministic dim=32 mock embedding model."""

    @property
    def dimensions(self) -> int:
        return 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(hash(t) % 100) / 100.0] * 32 for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model() -> _MockEmbedding:
    return _MockEmbedding()


@pytest.fixture
def store(tmp_path, mock_model) -> EpisodicStore:
    db = str(tmp_path / "knowledge.lance")
    return EpisodicStore(
        embedding_model=mock_model,
        namespace="test-episodes",
        db_path=db,
    )


def _make_episode(
    ticker: str = "AAPL",
    regime: str = "bull_low_vol",
    sector: str = "Information Technology",
    agent_type: str = "fundamental",
    decision: str = "Buy",
    confidence: float = 0.75,
    outcome_correct: bool | None = None,
    forward_return: float | None = None,
    labeled_at: str | None = None,
    decision_date: str = "2023-03-31",
    reasoning: str = "Strong earnings growth and solid balance sheet.",
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=str(uuid.uuid4()),
        ticker=ticker,
        decision_date=decision_date,
        regime_label=regime,
        sector=sector,
        agent_type=agent_type,
        decision=decision,
        confidence=confidence,
        collective_decision="Buy",
        forward_return=forward_return,
        outcome_correct=outcome_correct,
        reasoning_summary=reasoning,
        labeled_at=labeled_at,
    )


# ---------------------------------------------------------------------------
# Add / retrieve round-trip
# ---------------------------------------------------------------------------


def test_add_and_count(store):
    ep = _make_episode()
    store.add(ep)
    assert store.count() == 1


def test_add_multiple(store):
    for _ in range(5):
        store.add(_make_episode())
    assert store.count() == 5


def test_add_retrieves_correct_fields(store):
    ep = _make_episode(
        ticker="JPM",
        regime="rate_shock",
        sector="Financials",
        decision="Hold",
        confidence=0.60,
        reasoning="Yield curve inversion pressuring bank margins.",
    )
    store.add(ep)

    df = store._table.to_pandas()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "JPM"
    assert row["regime_label"] == "rate_shock"
    assert row["sector"] == "Financials"
    assert row["decision"] == "Hold"
    assert abs(row["confidence"] - 0.60) < 1e-6
    assert row["episode_id"] == ep.episode_id


# ---------------------------------------------------------------------------
# Sentinel encoding
# ---------------------------------------------------------------------------


def test_str_none_sentinel(store):
    ep = _make_episode(labeled_at=None)
    store.add(ep)
    df = store._table.to_pandas()
    assert df.iloc[0]["labeled_at"] == ""


def test_float_none_sentinel(store):
    ep = _make_episode(forward_return=None)
    store.add(ep)
    df = store._table.to_pandas()
    val = df.iloc[0]["forward_return"]
    assert math.isnan(val)


def test_outcome_correct_none_sentinel(store):
    ep = _make_episode(outcome_correct=None)
    store.add(ep)
    df = store._table.to_pandas()
    assert df.iloc[0]["outcome_correct"] == -1


def test_outcome_correct_true_encoding(store):
    ep = _make_episode(outcome_correct=True)
    store.add(ep)
    df = store._table.to_pandas()
    assert df.iloc[0]["outcome_correct"] == 1


def test_outcome_correct_false_encoding(store):
    ep = _make_episode(outcome_correct=False)
    store.add(ep)
    df = store._table.to_pandas()
    assert df.iloc[0]["outcome_correct"] == 0


# ---------------------------------------------------------------------------
# Sentinel decoding (round-trip via _from_row)
# ---------------------------------------------------------------------------


def test_roundtrip_outcome_correct_none(store):
    ep = _make_episode(outcome_correct=None)
    store.add(ep)
    df = store._table.to_pandas()
    recovered = store._from_row(df.iloc[0].to_dict())
    assert recovered.outcome_correct is None


def test_roundtrip_outcome_correct_true(store):
    ep = _make_episode(outcome_correct=True)
    store.add(ep)
    df = store._table.to_pandas()
    recovered = store._from_row(df.iloc[0].to_dict())
    assert recovered.outcome_correct is True


def test_roundtrip_forward_return_none(store):
    ep = _make_episode(forward_return=None)
    store.add(ep)
    df = store._table.to_pandas()
    recovered = store._from_row(df.iloc[0].to_dict())
    assert recovered.forward_return is None


def test_roundtrip_forward_return_value(store):
    ep = _make_episode(forward_return=0.123)
    store.add(ep)
    df = store._table.to_pandas()
    recovered = store._from_row(df.iloc[0].to_dict())
    assert recovered.forward_return == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# search() — outcome_correct filter
# ---------------------------------------------------------------------------


def test_search_outcome_correct_true_filter(store):
    """search(outcome_correct=True) returns only correctly-labeled episodes."""
    store.add(_make_episode(
        ticker="AAPL", outcome_correct=True, reasoning="Strong buy signal."
    ))
    store.add(_make_episode(
        ticker="AAPL", outcome_correct=False, reasoning="Wrong call."
    ))
    store.add(_make_episode(
        ticker="AAPL", outcome_correct=None, reasoning="Unlabeled episode."
    ))
    results = store.search(
        ticker="AAPL",
        regime="bull_low_vol",
        sector="Information Technology",
        outcome_correct=True,
        n=10,
    )
    assert all(r.outcome_correct is True for r in results)


def test_search_outcome_correct_none_returns_all(store):
    """search(outcome_correct=None) returns all episodes regardless of label."""
    for _ in range(3):
        store.add(_make_episode(
            ticker="AAPL", outcome_correct=None, reasoning="Unlabeled."
        ))
    results = store.search(
        ticker="AAPL",
        regime="bull_low_vol",
        sector="Information Technology",
        outcome_correct=None,
        n=10,
    )
    assert len(results) == 3


def test_search_empty_store_returns_empty(store):
    results = store.search(
        ticker="AAPL",
        regime="bull_low_vol",
        sector="Information Technology",
    )
    assert results == []


def test_search_wrong_regime_returns_empty(store):
    store.add(_make_episode(regime="bear_high_vol", outcome_correct=True))
    results = store.search(
        ticker="AAPL",
        regime="bull_low_vol",
        sector="Information Technology",
        outcome_correct=True,
    )
    assert results == []


def test_search_wrong_sector_returns_empty(store):
    store.add(_make_episode(sector="Financials", outcome_correct=True))
    results = store.search(
        ticker="AAPL",
        regime="bull_low_vol",
        sector="Information Technology",
        outcome_correct=True,
    )
    assert results == []


def test_search_respects_n_limit(store):
    for _ in range(10):
        store.add(_make_episode(outcome_correct=True))
    results = store.search(
        ticker="AAPL",
        regime="bull_low_vol",
        sector="Information Technology",
        outcome_correct=True,
        n=3,
    )
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# get_unlabeled_past_horizon
# ---------------------------------------------------------------------------


def test_get_unlabeled_past_horizon_returns_past_episodes(store):
    """Episodes 90 days old with no label should be returned."""
    old_date = (date.today() - timedelta(days=90)).isoformat()
    ep = _make_episode(decision_date=old_date, labeled_at=None, outcome_correct=None)
    store.add(ep)

    results = store.get_unlabeled_past_horizon(horizon_days=60)
    assert len(results) == 1
    assert results[0].episode_id == ep.episode_id


def test_get_unlabeled_past_horizon_excludes_recent(store):
    """Episodes within the 60-day horizon should NOT be returned."""
    recent_date = (date.today() - timedelta(days=30)).isoformat()
    ep = _make_episode(decision_date=recent_date, labeled_at=None)
    store.add(ep)

    results = store.get_unlabeled_past_horizon(horizon_days=60)
    assert results == []


def test_get_unlabeled_past_horizon_excludes_already_labeled(store):
    """Already-labeled episodes should NOT be returned."""
    old_date = (date.today() - timedelta(days=90)).isoformat()
    ep = _make_episode(
        decision_date=old_date,
        labeled_at="2024-01-01",
        outcome_correct=True,
    )
    store.add(ep)

    results = store.get_unlabeled_past_horizon(horizon_days=60)
    assert results == []


def test_get_unlabeled_past_horizon_custom_today(store):
    """Using a custom today date shifts the horizon cutoff."""
    ep = _make_episode(decision_date="2023-01-01", labeled_at=None)
    store.add(ep)

    # With today = 2023-03-15, horizon=60 → cutoff = 2023-01-14
    # 2023-01-01 <= 2023-01-14 → included
    results = store.get_unlabeled_past_horizon(
        horizon_days=60, today=date(2023, 3, 15)
    )
    assert len(results) == 1

    # With today = 2023-02-01, horizon=60 → cutoff = 2022-12-03
    # 2023-01-01 > 2022-12-03 → NOT included
    results2 = store.get_unlabeled_past_horizon(
        horizon_days=60, today=date(2023, 2, 1)
    )
    assert results2 == []


# ---------------------------------------------------------------------------
# update (delete + re-add)
# ---------------------------------------------------------------------------


def test_update_replaces_record(store):
    ep = _make_episode(forward_return=None, outcome_correct=None)
    store.add(ep)

    # Label the episode
    ep_labeled = ep.model_copy(update={
        "forward_return": 0.08,
        "outcome_correct": True,
        "labeled_at": "2024-01-01",
    })
    store.update(ep_labeled)

    assert store.count() == 1
    df = store._table.to_pandas()
    recovered = store._from_row(df.iloc[0].to_dict())
    assert recovered.forward_return == pytest.approx(0.08)
    assert recovered.outcome_correct is True
    assert recovered.labeled_at == "2024-01-01"


def test_update_idempotent(store):
    """Calling update twice doesn't create duplicates."""
    ep = _make_episode(outcome_correct=True)
    store.add(ep)
    store.update(ep)
    store.update(ep)
    assert store.count() == 1
