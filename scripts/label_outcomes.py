"""
Phase 14 episodic outcome labeling (E5-T4, DJ-092).

Fetches 60-day forward close prices for unlabeled EpisodeRecords and updates
outcome_correct, forward_return, and labeled_at fields.

Differs from scripts/run_label_outcomes.py (Phase 11, DJ-060), which labels
agent performance records in agent_performance_history.json.  This script
labels EpisodeRecords in the LanceDB episodic store.

Usage:
    uv run python scripts/label_outcomes.py [--namespace NS] [--data-dir DIR]

Make target: label-outcomes (called after the Phase 11 run_label_outcomes.py)
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Labeling logic (importable for unit tests)
# ---------------------------------------------------------------------------


def compute_outcome_correct(
    decision: str,
    forward_return: float,
    hold_threshold: float = 0.05,
) -> bool:
    """
    Determine whether an agent decision was correct given the realized return.

    Parameters
    ----------
    decision : str
        "Buy", "Hold", or "Sell".
    forward_return : float
        60-day realized price return (e.g. 0.08 = +8%).
    hold_threshold : float
        Absolute return threshold for a "correct" Hold call.

    Returns
    -------
    bool
    """
    if decision == "Buy":
        return forward_return > 0.0
    if decision == "Sell":
        return forward_return < 0.0
    # Hold is correct when |return| < threshold
    return abs(forward_return) < hold_threshold


def fetch_forward_return(
    ticker: str,
    decision_date: str,
    horizon_days: int = 60,
    yf: object | None = None,
) -> float | None:
    """
    Fetch the realized return from decision_date to decision_date + horizon_days
    trading days using yfinance.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    decision_date : str
        ISO 8601 date of the original decision.
    horizon_days : int
        Number of *calendar* days forward.
    yf : module | None
        yfinance module injection (for testing).

    Returns
    -------
    float | None
        Realized return, or None if data unavailable.
    """
    if yf is None:
        import yfinance as yf  # type: ignore[no-redef]

    t0 = date.fromisoformat(decision_date)
    t1 = t0 + timedelta(days=horizon_days)

    # Add a small buffer past t1 to ensure we have data at t1
    end = t1 + timedelta(days=10)

    try:
        df = yf.download(
            ticker,
            start=t0.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.warning("yfinance download failed for %s: %s", ticker, exc)
        return None

    if df.empty:
        return None

    # Handle MultiIndex columns (yfinance ≥ 0.2)
    if hasattr(df.columns, "levels"):
        close_col = ("Close", ticker) if ("Close", ticker) in df.columns else None
        if close_col is None:
            try:
                df = df["Close"]
            except KeyError:
                return None
        else:
            df = df[close_col]
    elif "Close" in df.columns:
        df = df["Close"]
    else:
        return None

    df = df.dropna().sort_index()

    # Price at decision_date (or first available day on/after t0)
    t0_ts = str(t0)
    avail_start = df[df.index >= t0_ts]
    if avail_start.empty:
        return None
    price_start = float(avail_start.iloc[0])

    # Price at t1 (or last available day on/before t1)
    avail_end = df[df.index <= str(t1)]
    if avail_end.empty:
        return None
    price_end = float(avail_end.iloc[-1])

    if price_start == 0:
        return None

    return (price_end - price_start) / price_start


def label_unlabeled_episodes(
    store: object,
    horizon_days: int = 60,
    today: date | None = None,
    yf: object | None = None,
) -> int:
    """
    Label all unlabeled episodes past the horizon.

    Parameters
    ----------
    store : EpisodicStore
        Connected episodic store instance.
    horizon_days : int
        Labeling horizon (calendar days after decision_date).
    today : date | None
        Override for current date (for testing).
    yf : module | None
        yfinance module injection (for testing).

    Returns
    -------
    int
        Number of episodes labeled in this run.
    """
    unlabeled = store.get_unlabeled_past_horizon(
        horizon_days=horizon_days, today=today
    )
    labeled_count = 0

    for episode in unlabeled:
        forward_return = fetch_forward_return(
            ticker=episode.ticker,
            decision_date=episode.decision_date,
            horizon_days=horizon_days,
            yf=yf,
        )
        if forward_return is None:
            logger.warning(
                "Could not fetch forward return for %s %s; skipping",
                episode.ticker, episode.decision_date,
            )
            continue

        outcome = compute_outcome_correct(episode.decision, forward_return)
        labeled_episode = episode.model_copy(update={
            "forward_return": forward_return,
            "outcome_correct": outcome,
            "labeled_at": datetime.now(UTC).date().isoformat(),
        })
        store.update(labeled_episode)
        labeled_count += 1
        logger.info(
            "Labeled %s %s %s: return=%.4f correct=%s",
            episode.ticker, episode.decision_date, episode.decision,
            forward_return, outcome,
        )

    return labeled_count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label Phase 14 episodic outcomes")
    p.add_argument(
        "--namespace",
        default="hifi-episodes",
        help="LanceDB namespace for the episodic store (default: hifi-episodes)",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Path to data directory (default: HIFI_DATA_DIR env var or 'data')",
    )
    p.add_argument(
        "--horizon-days",
        type=int,
        default=60,
        help="Labeling horizon in calendar days (default: 60)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()

    data_dir = args.data_dir or os.environ.get("HIFI_DATA_DIR", "data")
    db_path = str(Path(data_dir) / "knowledge.lance")

    # Import here to avoid circular imports at module level
    from hifi.knowledge.embeddings import EmbeddingModel
    from hifi.knowledge.episodic_store import EpisodicStore

    embedding_model = EmbeddingModel()
    store = EpisodicStore(
        embedding_model=embedding_model,
        namespace=args.namespace,
        db_path=db_path,
    )

    n = label_unlabeled_episodes(store, horizon_days=args.horizon_days)
    print(f"label_outcomes: labeled {n} episodes in namespace={args.namespace}")


if __name__ == "__main__":
    main()
