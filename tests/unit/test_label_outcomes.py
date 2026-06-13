"""
Unit tests for the label-outcomes functionality (P11-E5-T2, DJ-060).

Tests the underlying library functions that run_label_outcomes.py orchestrates:
- compute_forward_return() from hifi.collective.labeler
- DJ-042 labeling rules applied to DecisionRecord fields
- Idempotency: already-labeled records are never re-labeled

Uses synthetic OHLCV Parquet files written to pytest's tmp_path fixture.
No LLM, no live services required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hifi.collective.labeler import compute_forward_return
from hifi.collective.schemas import DecisionRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABEL_THRESHOLD = 0.02  # DJ-042


def _apply_label(decision: str, forward_return: float) -> bool:
    """Inline DJ-042 rules (mirrors run_label_outcomes._apply_label)."""
    if decision == "Buy":
        return forward_return > _LABEL_THRESHOLD
    if decision == "Sell":
        return forward_return < -_LABEL_THRESHOLD
    return abs(forward_return) <= _LABEL_THRESHOLD


def _make_ohlcv_parquet(tmp_path: Path, ticker: str, n_days: int = 200) -> None:
    """
    Write a synthetic OHLCV Parquet to tmp_path/market/{ticker}_2020-01-01.parquet.

    Uses business-day calendar starting 2020-01-01. Prices follow a random walk
    seeded at 42 for reproducibility. _load_prices() in labeler.py handles the
    DatetimeIndex-with-Close-column format this produces.
    """
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)
    rng = np.random.default_rng(42)
    prices = 100.0 * (1 + rng.normal(0.001, 0.01, n_days)).cumprod()

    df = pd.DataFrame({"Close": prices}, index=dates)
    df.index.name = "Date"

    market_dir = tmp_path / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(market_dir / f"{ticker}_2020-01-01.parquet")


def _simulate_labeling(  # noqa: E501
    records: list[DecisionRecord], data_dir: str
) -> tuple[list[DecisionRecord], int]:
    """
    Mirror the core loop in run_label_outcomes.label_unlabeled_records().

    Returns (new_records, n_newly_labeled). Uses a fixed timestamp for determinism.
    """
    fixed_ts = "2023-01-01T00:00:00+00:00"
    n_newly_labeled = 0
    new_records = []

    for record in records:
        if record.outcome_correct is not None:
            new_records.append(record)
            continue

        fwd = compute_forward_return(
            record.ticker, record.analysis_date, data_dir, record.horizon_days
        )
        if fwd is None:
            new_records.append(record)
            continue

        labeled = record.model_copy(update={
            "forward_return": fwd,
            "outcome_correct": _apply_label(record.decision, fwd),
            "outcome_labeled_at": fixed_ts,
        })
        new_records.append(labeled)
        n_newly_labeled += 1

    return new_records, n_newly_labeled


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_unlabeled_records_get_labeled(tmp_path: Path) -> None:
    """
    A record with analysis_date well within the OHLCV range (enough forward data)
    is labeled with a non-None outcome_correct.
    """
    ticker = "TEST"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)

    record = DecisionRecord(
        ticker=ticker,
        analysis_date="2020-01-02",  # 2nd trading day; 60 forward days well within 200
        agent_type="technical",
        decision="Buy",
        confidence=0.7,
        outcome_correct=None,
        horizon_days=60,
    )

    new_records, n_newly_labeled = _simulate_labeling([record], str(tmp_path))

    assert n_newly_labeled == 1
    assert new_records[0].outcome_correct is not None
    assert new_records[0].forward_return is not None
    assert new_records[0].outcome_labeled_at is not None


def test_future_records_stay_unlabeled(tmp_path: Path) -> None:
    """
    A record where only 30 trading days exist after analysis_date (horizon=60)
    remains unlabeled — compute_forward_return returns None.
    """
    ticker = "FUTURE"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=30)

    fwd = compute_forward_return("FUTURE", "2020-01-02", str(tmp_path), horizon_days=60)
    assert fwd is None, "Should return None when fewer than 60 forward trading days exist"

    record = DecisionRecord(
        ticker="FUTURE",
        analysis_date="2020-01-02",
        agent_type="fundamental",
        decision="Hold",
        confidence=0.5,
        outcome_correct=None,
        horizon_days=60,
    )
    new_records, n_newly_labeled = _simulate_labeling([record], str(tmp_path))

    assert n_newly_labeled == 0
    assert new_records[0].outcome_correct is None


def test_idempotent(tmp_path: Path) -> None:
    """
    Running the labeling logic twice on the same records produces the same result.
    Records already labeled in the first pass are skipped in the second pass.
    """
    ticker = "IDEM"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)

    record = DecisionRecord(
        ticker=ticker,
        analysis_date="2020-01-02",
        agent_type="risk",
        decision="Sell",
        confidence=0.6,
        outcome_correct=None,
        horizon_days=60,
    )

    first_pass, n_first = _simulate_labeling([record], str(tmp_path))
    assert n_first == 1

    second_pass, n_second = _simulate_labeling(first_pass, str(tmp_path))
    assert n_second == 0  # already labeled, skipped

    assert second_pass[0].outcome_correct == first_pass[0].outcome_correct
    assert second_pass[0].forward_return == first_pass[0].forward_return
    assert second_pass[0].outcome_labeled_at == first_pass[0].outcome_labeled_at


def test_already_labeled_records_not_overwritten(tmp_path: Path) -> None:
    """
    Records with existing outcome_correct (from a previous run) are never touched.
    """
    ticker = "KEEP"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)

    record = DecisionRecord(
        ticker=ticker,
        analysis_date="2020-01-02",
        agent_type="macro",
        decision="Buy",
        confidence=0.8,
        outcome_correct=True,  # already labeled
        forward_return=0.05,
        outcome_labeled_at="2020-01-01T00:00:00+00:00",
        horizon_days=60,
    )

    new_records, n_newly_labeled = _simulate_labeling([record], str(tmp_path))

    assert n_newly_labeled == 0
    assert new_records[0].outcome_correct is True
    assert new_records[0].forward_return == 0.05
    assert new_records[0].outcome_labeled_at == "2020-01-01T00:00:00+00:00"


def test_dj042_labeling_rules_buy() -> None:
    """BUY is correct when forward_return > 0.02."""
    assert _apply_label("Buy", 0.03) is True
    assert _apply_label("Buy", 0.02) is False  # threshold is exclusive (> not >=)
    assert _apply_label("Buy", -0.01) is False


def test_dj042_labeling_rules_sell() -> None:
    """SELL is correct when forward_return < -0.02."""
    assert _apply_label("Sell", -0.03) is True
    assert _apply_label("Sell", -0.02) is False  # exclusive
    assert _apply_label("Sell", 0.01) is False


def test_dj042_labeling_rules_hold() -> None:
    """HOLD is correct when abs(forward_return) <= 0.02."""
    assert _apply_label("Hold", 0.01) is True
    assert _apply_label("Hold", -0.01) is True
    assert _apply_label("Hold", 0.02) is True   # inclusive
    assert _apply_label("Hold", 0.03) is False
