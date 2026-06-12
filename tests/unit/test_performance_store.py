"""
Unit tests for AgentPerformanceHistory persistence and weight computation (P9-E4).

Covers load_history, save_history, compute_weights, get_weights, and update_and_save.
All file I/O uses pytest's tmp_path to avoid touching data/.
"""

import json

import pytest

from hifi.collective.performance_store import (
    _INITIAL_AGENT_TYPES,
    _INITIAL_WEIGHT,
    compute_weights,
    get_weights,
    load_history,
    save_history,
    update_and_save,
)
from hifi.collective.schemas import AgentPerformanceHistory, DecisionRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    agent_type: str,
    decision: str = "Buy",
    confidence: float = 0.75,
    outcome_correct: bool | None = None,
    ticker: str = "AAPL",
    date: str = "2022-03-31",
) -> DecisionRecord:
    return DecisionRecord(
        ticker=ticker,
        analysis_date=date,
        agent_type=agent_type,
        decision=decision,
        confidence=confidence,
        outcome_correct=outcome_correct,
    )


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------


def test_load_history_missing_file_returns_uniform_defaults(tmp_path):
    """No file → fresh history with uniform weights and no records."""
    history = load_history(data_dir=str(tmp_path))

    assert history.records == []
    assert history.n_labeled == 0
    assert set(history.weights.keys()) == set(_INITIAL_AGENT_TYPES)
    for w in history.weights.values():
        assert w == pytest.approx(_INITIAL_WEIGHT)


def test_load_history_parses_existing_file(tmp_path):
    """Saved file is correctly loaded and validated."""
    original = AgentPerformanceHistory(
        records=[_record("fundamental", outcome_correct=True)],
        weights={"fundamental": 0.80, "technical": 0.70, "risk": 0.60, "macro": 0.65},
        last_updated="2024-01-01T00:00:00+00:00",
        n_labeled=0,  # auto-computed by model_validator
    )
    save_history(original, data_dir=str(tmp_path))
    loaded = load_history(data_dir=str(tmp_path))

    assert loaded.n_labeled == 1  # auto-computed
    assert loaded.weights["fundamental"] == pytest.approx(0.80)
    assert len(loaded.records) == 1
    assert loaded.records[0].agent_type == "fundamental"


# ---------------------------------------------------------------------------
# save_history
# ---------------------------------------------------------------------------


def test_save_history_creates_file(tmp_path):
    history = AgentPerformanceHistory(
        records=[],
        weights={t: _INITIAL_WEIGHT for t in _INITIAL_AGENT_TYPES},
        last_updated="2024-06-12T00:00:00+00:00",
        n_labeled=0,
    )
    save_history(history, data_dir=str(tmp_path))

    path = tmp_path / "agent_performance_history.json"
    assert path.exists()


def test_save_history_content_is_valid_json(tmp_path):
    history = AgentPerformanceHistory(
        records=[_record("technical", outcome_correct=False)],
        weights={"technical": 0.0},
        last_updated="2024-06-12T00:00:00+00:00",
        n_labeled=0,
    )
    save_history(history, data_dir=str(tmp_path))

    path = tmp_path / "agent_performance_history.json"
    parsed = json.loads(path.read_text())
    assert "records" in parsed
    assert "weights" in parsed


def test_save_history_atomic_no_tmp_file_after(tmp_path):
    """After save, the .tmp file must not exist (atomic rename completed)."""
    history = AgentPerformanceHistory(
        records=[], weights={}, last_updated="", n_labeled=0
    )
    save_history(history, data_dir=str(tmp_path))

    tmp = tmp_path / "agent_performance_history.json.tmp"
    assert not tmp.exists()


def test_save_and_load_round_trips(tmp_path):
    """save_history + load_history is lossless."""
    records = [
        _record("fundamental", outcome_correct=True),
        _record("technical", outcome_correct=False),
        _record("risk"),  # unlabeled
    ]
    history = AgentPerformanceHistory(
        records=records,
        weights={"fundamental": 0.75},
        last_updated="2024-06-12T00:00:00+00:00",
        n_labeled=0,
    )
    save_history(history, data_dir=str(tmp_path))
    loaded = load_history(data_dir=str(tmp_path))

    assert loaded.n_labeled == 2        # 2 labeled records
    assert len(loaded.records) == 3
    assert loaded.weights["fundamental"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# compute_weights
# ---------------------------------------------------------------------------


def test_compute_weights_no_labeled_returns_uniform():
    """Zero labeled records → uniform weights for all canonical types."""
    records = [_record("fundamental"), _record("technical")]  # both unlabeled
    weights = compute_weights(records)

    assert set(weights.keys()) == set(_INITIAL_AGENT_TYPES)
    for w in weights.values():
        assert w == pytest.approx(_INITIAL_WEIGHT)


def test_compute_weights_perfect_accuracy():
    """Agent correct every time → weight = 1.0."""
    records = [_record("fundamental", outcome_correct=True)] * 4
    weights = compute_weights(records)

    assert weights["fundamental"] == pytest.approx(1.0)


def test_compute_weights_zero_accuracy():
    """Agent wrong every time → weight = 0.0."""
    records = [_record("technical", outcome_correct=False)] * 3
    weights = compute_weights(records)

    assert weights["technical"] == pytest.approx(0.0)


def test_compute_weights_mixed_accuracy():
    """3 correct, 1 wrong → weight = 0.75."""
    records = [
        _record("risk", outcome_correct=True),
        _record("risk", outcome_correct=True),
        _record("risk", outcome_correct=True),
        _record("risk", outcome_correct=False),
    ]
    weights = compute_weights(records)

    assert weights["risk"] == pytest.approx(0.75)


def test_compute_weights_fills_missing_types_with_initial():
    """Agent types absent from labeled records → filled with _INITIAL_WEIGHT."""
    records = [_record("fundamental", outcome_correct=True)]
    weights = compute_weights(records)

    assert set(weights.keys()) == set(_INITIAL_AGENT_TYPES)
    # fundamental: 1.0 (all correct); others: initial weight
    assert weights["fundamental"] == pytest.approx(1.0)
    assert weights["technical"] == pytest.approx(_INITIAL_WEIGHT)
    assert weights["risk"] == pytest.approx(_INITIAL_WEIGHT)
    assert weights["macro"] == pytest.approx(_INITIAL_WEIGHT)


def test_compute_weights_unlabeled_records_excluded():
    """Unlabeled records (outcome_correct=None) do not affect weights."""
    records = [
        _record("macro", outcome_correct=True),     # labeled
        _record("macro", outcome_correct=None),     # unlabeled — excluded
        _record("macro", outcome_correct=None),     # unlabeled — excluded
    ]
    weights = compute_weights(records)

    # Only 1 labeled: 1 correct / 1 total = 1.0
    assert weights["macro"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_weights
# ---------------------------------------------------------------------------


def test_get_weights_missing_file_returns_uniform(tmp_path):
    """get_weights on non-existent file → uniform dict with all 4 types."""
    weights = get_weights(data_dir=str(tmp_path))

    assert set(weights.keys()) == set(_INITIAL_AGENT_TYPES)
    for w in weights.values():
        assert w == pytest.approx(_INITIAL_WEIGHT)


def test_get_weights_existing_file_returns_stored(tmp_path):
    stored = {"fundamental": 0.88, "technical": 0.72, "risk": 0.65, "macro": 0.60}
    history = AgentPerformanceHistory(
        records=[],
        weights=stored,
        last_updated="2024-06-12T00:00:00+00:00",
        n_labeled=0,
    )
    save_history(history, data_dir=str(tmp_path))
    weights = get_weights(data_dir=str(tmp_path))

    assert weights["fundamental"] == pytest.approx(0.88)
    assert weights["technical"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# update_and_save
# ---------------------------------------------------------------------------


def test_update_and_save_appends_records(tmp_path):
    """update_and_save adds new records on top of existing ones."""
    initial = [_record("fundamental", outcome_correct=True)]
    update_and_save(initial, data_dir=str(tmp_path))

    new_records = [_record("technical", outcome_correct=False)]
    update_and_save(new_records, data_dir=str(tmp_path))

    loaded = load_history(data_dir=str(tmp_path))
    assert len(loaded.records) == 2
    assert loaded.n_labeled == 2


def test_update_and_save_recomputes_weights(tmp_path):
    """Weights are recomputed from all records after update."""
    records = [_record("macro", outcome_correct=True)] * 3 + [
        _record("macro", outcome_correct=False)
    ]
    result = update_and_save(records, data_dir=str(tmp_path))

    assert result.weights["macro"] == pytest.approx(0.75)
