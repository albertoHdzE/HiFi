"""
Unit tests for scripts/label_outcomes.py (E5-T4, DJ-092).

Tests:
- Buy + positive return → outcome_correct=True.
- Buy + negative return → outcome_correct=False.
- Sell + negative return → outcome_correct=True.
- Hold + small |return| → outcome_correct=True.
- Hold + large |return| → outcome_correct=False.
- Idempotency: already-labeled records not re-labeled.
- Horizon enforcement: records < 60 days old are not labeled.
- Missing yfinance data: episode skipped, count=0.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from scripts.label_outcomes import (
    compute_outcome_correct,
    fetch_forward_return,
    label_unlabeled_episodes,
)

from hifi.knowledge.episodic_store import EpisodeRecord, EpisodicStore

# ---------------------------------------------------------------------------
# compute_outcome_correct
# ---------------------------------------------------------------------------


class TestComputeOutcomeCorrect:
    def test_buy_positive_return(self):
        assert compute_outcome_correct("Buy", 0.08) is True

    def test_buy_negative_return(self):
        assert compute_outcome_correct("Buy", -0.03) is False

    def test_buy_zero_return(self):
        assert compute_outcome_correct("Buy", 0.0) is False

    def test_sell_negative_return(self):
        assert compute_outcome_correct("Sell", -0.05) is True

    def test_sell_positive_return(self):
        assert compute_outcome_correct("Sell", 0.05) is False

    def test_hold_small_return_correct(self):
        assert compute_outcome_correct("Hold", 0.03) is True

    def test_hold_large_return_incorrect(self):
        assert compute_outcome_correct("Hold", 0.10) is False

    def test_hold_large_negative_return_incorrect(self):
        assert compute_outcome_correct("Hold", -0.10) is False

    def test_hold_exactly_at_threshold(self):
        # abs(0.05) < 0.05 is False → not correct
        assert compute_outcome_correct("Hold", 0.05) is False

    def test_hold_below_threshold(self):
        assert compute_outcome_correct("Hold", 0.049) is True


# ---------------------------------------------------------------------------
# Mock yfinance for fetch_forward_return
# ---------------------------------------------------------------------------


def _make_yf_mock(prices: dict[str, float]) -> MagicMock:
    """
    Build a mock yfinance module where download() returns a DataFrame
    with a 'Close' column.
    """
    yf_mock = MagicMock()

    def _download(ticker, start, end, progress=False, auto_adjust=True):
        dates = sorted(prices.keys())
        close_series = pd.Series(prices, name="Close")
        close_series.index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        return pd.DataFrame({"Close": close_series})

    yf_mock.download = _download
    return yf_mock


class TestFetchForwardReturn:
    def test_basic_positive_return(self):
        # t0=2023-01-01, horizon=60 → t1=2023-03-02; use 2023-03-01 (within window)
        prices = {
            "2023-01-01": 100.0,
            "2023-03-01": 108.0,
        }
        yf = _make_yf_mock(prices)
        result = fetch_forward_return("AAPL", "2023-01-01", horizon_days=60, yf=yf)
        assert result == pytest.approx(0.08)

    def test_negative_return(self):
        prices = {
            "2023-01-01": 100.0,
            "2023-03-01": 95.0,
        }
        yf = _make_yf_mock(prices)
        result = fetch_forward_return("AAPL", "2023-01-01", horizon_days=60, yf=yf)
        assert result == pytest.approx(-0.05)

    def test_empty_dataframe_returns_none(self):
        yf = MagicMock()
        yf.download.return_value = pd.DataFrame()
        result = fetch_forward_return("AAPL", "2023-01-01", yf=yf)
        assert result is None

    def test_download_exception_returns_none(self):
        yf = MagicMock()
        yf.download.side_effect = ConnectionError("network failure")
        result = fetch_forward_return("AAPL", "2023-01-01", yf=yf)
        assert result is None


# ---------------------------------------------------------------------------
# Mock EpisodeRecord + EpisodicStore helpers
# ---------------------------------------------------------------------------


class _MockEmbedding:
    @property
    def dimensions(self):
        return 32

    def embed(self, texts):
        return [[0.1] * 32] * len(texts)

    def embed_one(self, text):
        return [0.1] * 32


def _make_episode(
    decision: str = "Buy",
    decision_date: str | None = None,
    labeled_at: str | None = None,
    outcome_correct: bool | None = None,
) -> EpisodeRecord:
    if decision_date is None:
        decision_date = (date.today() - timedelta(days=90)).isoformat()
    return EpisodeRecord(
        episode_id=str(uuid.uuid4()),
        ticker="AAPL",
        decision_date=decision_date,
        regime_label="bull_low_vol",
        sector="Information Technology",
        agent_type="fundamental",
        decision=decision,
        confidence=0.75,
        collective_decision="Buy",
        forward_return=None,
        outcome_correct=outcome_correct,
        reasoning_summary="Strong earnings growth.",
        labeled_at=labeled_at,
    )


@pytest.fixture
def episodic_store(tmp_path):
    return EpisodicStore(
        embedding_model=_MockEmbedding(),
        namespace="test-label",
        db_path=str(tmp_path / "knowledge.lance"),
    )


# ---------------------------------------------------------------------------
# label_unlabeled_episodes
# ---------------------------------------------------------------------------


class TestLabelUnlabeledEpisodes:
    def _make_yf(self, forward_return: float = 0.08) -> MagicMock:
        # t0=2023-01-01, horizon=60 → t1=2023-03-02; use 2023-03-01 (within window)
        prices = {
            "2023-01-01": 100.0,
            "2023-03-01": 100.0 * (1 + forward_return),
        }
        return _make_yf_mock(prices)

    def test_buy_positive_return_labeled_correct(self, episodic_store):
        ep = _make_episode("Buy", decision_date="2023-01-01")
        episodic_store.add(ep)

        n = label_unlabeled_episodes(
            store=episodic_store,
            today=date(2023, 4, 1),
            yf=self._make_yf(forward_return=0.08),
        )
        assert n == 1

        df = episodic_store._table.to_pandas()
        recovered = episodic_store._from_row(df.iloc[0].to_dict())
        assert recovered.outcome_correct is True
        assert recovered.labeled_at is not None
        assert recovered.forward_return == pytest.approx(0.08)

    def test_sell_negative_return_labeled_correct(self, episodic_store):
        ep = _make_episode("Sell", decision_date="2023-01-01")
        episodic_store.add(ep)

        n = label_unlabeled_episodes(
            store=episodic_store,
            today=date(2023, 4, 1),
            yf=self._make_yf(forward_return=-0.05),
        )
        assert n == 1
        df = episodic_store._table.to_pandas()
        recovered = episodic_store._from_row(df.iloc[0].to_dict())
        assert recovered.outcome_correct is True

    def test_hold_small_return_labeled_correct(self, episodic_store):
        ep = _make_episode("Hold", decision_date="2023-01-01")
        episodic_store.add(ep)
        n = label_unlabeled_episodes(
            store=episodic_store,
            today=date(2023, 4, 1),
            yf=self._make_yf(forward_return=0.02),
        )
        assert n == 1
        df = episodic_store._table.to_pandas()
        recovered = episodic_store._from_row(df.iloc[0].to_dict())
        assert recovered.outcome_correct is True

    def test_idempotency_already_labeled_not_relabeled(self, episodic_store):
        """Already-labeled episodes should not be relabeled."""
        ep = _make_episode(
            "Buy",
            decision_date="2023-01-01",
            labeled_at="2023-04-01",
            outcome_correct=True,
        )
        # Manually set forward_return for storage
        ep2 = ep.model_copy(update={"forward_return": 0.05})
        episodic_store.add(ep2)

        n = label_unlabeled_episodes(
            store=episodic_store,
            today=date(2023, 6, 1),
            yf=self._make_yf(0.08),
        )
        assert n == 0

    def test_horizon_enforcement_recent_episode_skipped(self, episodic_store):
        """Episodes within 60 days of today are not labeled."""
        recent_date = (date.today() - timedelta(days=30)).isoformat()
        ep = _make_episode("Buy", decision_date=recent_date)
        episodic_store.add(ep)

        n = label_unlabeled_episodes(
            store=episodic_store,
            yf=self._make_yf(0.08),
        )
        assert n == 0

    def test_missing_yfinance_data_skipped(self, episodic_store):
        """When yfinance returns no data, episode is skipped (not labeled)."""
        ep = _make_episode("Buy", decision_date="2023-01-01")
        episodic_store.add(ep)

        yf = MagicMock()
        yf.download.return_value = pd.DataFrame()  # empty → no data

        n = label_unlabeled_episodes(
            store=episodic_store,
            today=date(2023, 4, 1),
            yf=yf,
        )
        assert n == 0
        df = episodic_store._table.to_pandas()
        assert df.iloc[0]["labeled_at"] == ""  # still unlabeled

    def test_labels_multiple_episodes(self, episodic_store):
        """Multiple unlabeled past-horizon episodes are all labeled."""
        for _i in range(3):
            ep = _make_episode("Buy", decision_date="2023-01-01")
            episodic_store.add(ep)

        n = label_unlabeled_episodes(
            store=episodic_store,
            today=date(2023, 4, 1),
            yf=self._make_yf(0.05),
        )
        assert n == 3
