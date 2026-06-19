"""
Phase 15 Walk-Forward Simulation (DJ-097).

Runs the sequential ensemble over historical evaluation dates under four ablation
conditions (DJ-096), storing one EnsembleOutput JSON per (condition, date, ticker).

Walk-forward periods (DJ-095):
  training       2004-2019  (not evaluated by this script — calibration only)
  validation     2020-2021  COVID regime
  held_out_test  2022-2023  Primary result — rate-shock regime
  walk_forward   2024-2025  Sequential monthly

Four conditions (DJ-096):
  full          Sequential 5-org ensemble + episodic RAG from hifi-eval namespace
  parallel      Parallel 5-org ensemble (no inter-agent context sharing)
  homogeneous   Sequential with Phase 13 qwen-dominant model config
  no-memory     Sequential 5-org, no episodic prefix (ablates memory contribution)

Checkpoint-resume
-----------------
Output files are written to:
  {output_dir}/{condition}/{year}/{month}/{ticker}.json

If a file already exists, that (condition, date, ticker) is skipped without
calling the LLM.  Re-run after a failure picks up exactly where it left off.

Temporal discipline
-------------------
- Dates are processed in strictly ascending chronological order.
- Within each date, tickers are evaluated sequentially (no cross-ticker dependency).
- The eval-ingest-through Makefile target must be run for DATE before this script
  evaluates that date.  This script does NOT call eval-ingest-through automatically.

Usage
-----
    # Dry-run: show schedule without calling LLMs
    uv run python scripts/run_phase15_walkforward.py --dry-run \\
        --condition full --period held-out-test

    # Full held-out test (requires LM Studio + all 6 models)
    uv run python scripts/run_phase15_walkforward.py \\
        --condition full --period held-out-test

    # Single ticker smoke test
    uv run python scripts/run_phase15_walkforward.py \\
        --condition full --period held-out-test \\
        --tickers AAPL --start-date 2022-01-31 --end-date 2022-03-31

    # Status: count completed JSONs
    uv run python scripts/run_phase15_walkforward.py --status \\
        --condition full --period held-out-test

Outputs
-------
  data/walkforward/{condition}/{YYYY}/{MM}/{ticker}.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase 13 homogeneous model config — used for the Homogeneous ablation condition.
# These are the model IDs that produced entropy=0.000 herding=1.000 in Phase 12.1.
_HOMOGENEOUS_ENV = {
    "HIFI_FUNDAMENTAL_MODEL": "qwen2.5-coder-32b-instruct-mlx",
    "HIFI_TECHNICAL_MODEL":   "qwen2.5-coder-32b-instruct-mlx",
    "HIFI_RISK_MODEL":        "gemma-3-4b-it",
    "HIFI_MACRO_MODEL":       "mlx-community-qwen3-235b-a22b",  # Phase 13 default
    "HIFI_SENTIMENT_MODEL":   "qwen2.5-coder-32b-instruct-mlx",
    "HIFI_CONTRARIAN_MODEL":  "mlx-community-qwen3-235b-a22b",
}

_DEFAULT_OUTPUT_DIR = "data/walkforward"
_EVAL_CONTEXT_NAMESPACE = "hifi-eval-context"
_EVAL_EPISODE_NAMESPACE = "hifi-eval"
_AGENT_TYPES = ["fundamental", "technical", "risk", "macro", "sentiment"]


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def output_path(output_dir: str, condition: str, date: str, ticker: str) -> Path:
    """Return the output JSON path for one (condition, date, ticker) triple."""
    year, month, _ = date.split("-")
    return Path(output_dir) / condition / year / month / f"{ticker}.json"


def count_completed(output_dir: str, condition: str, tickers: list[str], dates: list[str]) -> int:
    """Count how many (date, ticker) pairs have a completed output JSON."""
    return sum(
        1 for d in dates for t in tickers
        if output_path(output_dir, condition, d, t).exists()
    )


# ---------------------------------------------------------------------------
# Episodic memory retrieval (for "full" condition only)
# ---------------------------------------------------------------------------


def _build_episodic_prefixes(
    ticker: str,
    date: str,
    regime: str,
    sector: str,
    db_path: str,
) -> dict[str, str]:
    """
    Build per-agent episodic memory prefixes from the hifi-eval episodic store.

    Returns an empty dict on any failure (fail-open).
    """
    try:
        from hifi.knowledge.embeddings import EmbeddingModel
        from hifi.knowledge.episodic_retriever import EpisodicRetriever
        from hifi.knowledge.episodic_store import EpisodicStore

        embedding_model = EmbeddingModel()
        store = EpisodicStore(
            embedding_model=embedding_model,
            namespace=_EVAL_EPISODE_NAMESPACE,
            db_path=db_path,
        )
        retriever = EpisodicRetriever(store=store)
        prefixes: dict[str, str] = {}
        for agent_type in _AGENT_TYPES:
            prefix = retriever.retrieve(
                ticker=ticker,
                date=date,
                agent_type=agent_type,
                regime=regime,
                sector=sector,
                n=3,
            )
            if prefix:
                prefixes[agent_type] = prefix
        return prefixes
    except Exception as exc:
        logger.debug("Episodic prefix build failed for %s %s: %s", ticker, date, exc)
        return {}


# ---------------------------------------------------------------------------
# Regime + sector helpers
# ---------------------------------------------------------------------------


def _get_regime(ticker: str, date: str, data_dir: str) -> str:
    try:
        import pandas as pd

        from hifi.data.regime import classify_regime

        spy_path = Path(data_dir) / "market" / "SPY" / "ohlcv.parquet"
        macro_path = Path(data_dir) / "macro" / "macro.parquet"
        if not spy_path.exists() or not macro_path.exists():
            return "neutral"
        spy = pd.read_parquet(spy_path)
        macro = pd.read_parquet(macro_path)
        if not hasattr(spy.index, "freq"):
            spy.index = pd.to_datetime(spy.index)
        if not hasattr(macro.index, "freq"):
            macro.index = pd.to_datetime(macro.index)
        return classify_regime(date, spy, macro)
    except Exception:
        return "neutral"


def _get_sector(ticker: str) -> str:
    try:
        from hifi.data.universe import get_sector
        return get_sector(ticker) or "Unknown"
    except Exception:
        return "Unknown"


# ---------------------------------------------------------------------------
# Per-run dispatcher (one ticker × one date × one condition)
# ---------------------------------------------------------------------------


def run_one(
    ticker: str,
    date: str,
    condition: str,
    data_dir: str,
    output_dir: str,
    *,
    _test_llms: dict | None = None,
) -> Path | None:
    """
    Run the ensemble for one (ticker, date) under the given condition.

    Returns the output Path on success, None on failure.
    Skips (checkpoint) if the output file already exists.
    """
    out = output_path(output_dir, condition, date, ticker)
    if out.exists():
        logger.debug("SKIP (checkpoint) %s %s condition=%s", ticker, date, condition)
        return out

    db_path = str(Path(data_dir) / "knowledge.lance")

    from hifi.simulation.snapshot import build_minimal_snapshot
    snapshot_json = build_minimal_snapshot(ticker, date)

    try:
        output = _dispatch(
            ticker=ticker,
            date=date,
            condition=condition,
            snapshot_json=snapshot_json,
            data_dir=data_dir,
            db_path=db_path,
            _test_llms=_test_llms,
        )
    except Exception as exc:
        logger.error("FAIL %s %s condition=%s: %s", ticker, date, condition, exc)
        return None

    # Serialize and write atomically via temp file
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(output.model_dump_json(), encoding="utf-8")
    tmp.rename(out)
    logger.info("DONE %s %s condition=%s → %s", ticker, date, condition, out)
    return out


def _dispatch(
    ticker: str,
    date: str,
    condition: str,
    snapshot_json: str,
    data_dir: str,
    db_path: str,
    *,
    _test_llms: dict | None = None,
):
    """Route to the correct ensemble function based on condition."""
    if condition == "full":
        return _run_full(ticker, date, snapshot_json, data_dir, db_path, _test_llms)
    if condition == "parallel":
        return _run_parallel(ticker, date, snapshot_json, data_dir, db_path, _test_llms)
    if condition == "homogeneous":
        return _run_homogeneous(ticker, date, snapshot_json, data_dir, db_path, _test_llms)
    if condition == "no-memory":
        return _run_no_memory(ticker, date, snapshot_json, data_dir, db_path, _test_llms)
    raise ValueError(f"Unknown condition: {condition!r}")


def _run_full(ticker, date, snapshot_json, data_dir, db_path, _test_llms):
    """Full: sequential 5-org + episodic RAG from hifi-eval namespace."""
    from hifi.agents.ensemble_runner import run_sequential_ensemble

    regime = _get_regime(ticker, date, data_dir)
    sector = _get_sector(ticker)
    memory_prefixes = _build_episodic_prefixes(ticker, date, regime, sector, db_path)

    return run_sequential_ensemble(
        ticker=ticker,
        as_of_date=date,
        snapshot_json=snapshot_json,
        data_dir=data_dir,
        context_namespace=_EVAL_CONTEXT_NAMESPACE,
        memory_prefixes=memory_prefixes,
        _test_llms=_test_llms,
    )


def _run_parallel(ticker, date, snapshot_json, data_dir, db_path, _test_llms):
    """Parallel: independent 5-org, no inter-agent context sharing."""
    from hifi.agents.ensemble_runner import run_ensemble

    return run_ensemble(
        ticker=ticker,
        as_of_date=date,
        snapshot_json=snapshot_json,
        data_dir=data_dir,
        sequential=False,
        _test_llms=_test_llms,
    )


def _run_homogeneous(ticker, date, snapshot_json, data_dir, db_path, _test_llms):
    """Homogeneous: Phase 13 qwen-dominant config via env var injection."""
    from hifi.agents.ensemble_runner import run_sequential_ensemble

    saved = {k: os.environ.get(k) for k in _HOMOGENEOUS_ENV}
    try:
        os.environ.update(_HOMOGENEOUS_ENV)
        return run_sequential_ensemble(
            ticker=ticker,
            as_of_date=date,
            snapshot_json=snapshot_json,
            data_dir=data_dir,
            context_namespace=_EVAL_CONTEXT_NAMESPACE,
            _test_llms=_test_llms,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_no_memory(ticker, date, snapshot_json, data_dir, db_path, _test_llms):
    """No-memory: sequential 5-org, no episodic prefix injection."""
    from hifi.agents.ensemble_runner import run_sequential_ensemble

    return run_sequential_ensemble(
        ticker=ticker,
        as_of_date=date,
        snapshot_json=snapshot_json,
        data_dir=data_dir,
        context_namespace=_EVAL_CONTEXT_NAMESPACE,
        memory_prefixes={},
        _test_llms=_test_llms,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 15 walk-forward simulation harness"
    )
    p.add_argument(
        "--condition",
        choices=["full", "parallel", "homogeneous", "no-memory"],
        required=True,
        help="Ablation condition (DJ-096)",
    )
    p.add_argument(
        "--period",
        default="held-out-test",
        help=(
            "Evaluation period: training | validation | held-out-test | "
            "walk-forward | all (default: held-out-test)"
        ),
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="Override period start date (ISO 8601, e.g. 2022-01-31)",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="Override period end date (ISO 8601, e.g. 2022-03-31)",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Subset of tickers to evaluate (default: all 98 PHASE14_UNIVERSE tickers)",
    )
    p.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Data directory (default: HIFI_DATA_DIR env var or 'data')",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schedule without making any LLM calls",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print checkpoint progress and exit",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-item progress; print only summary",
    )
    return p.parse_args()


def _resolve_dates(args: argparse.Namespace) -> list[str]:
    from hifi.simulation.schedule import (
        WalkForwardPeriod,
        generate_month_ends,
        get_multi_period_dates,
        get_period_dates,
    )

    if args.start_date and args.end_date:
        return generate_month_ends(args.start_date, args.end_date)

    period = args.period
    if period == "all":
        return get_multi_period_dates([
            WalkForwardPeriod.VALIDATION,
            WalkForwardPeriod.HELD_OUT_TEST,
            WalkForwardPeriod.WALK_FORWARD,
        ])
    return get_period_dates(period)


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir or os.environ.get("HIFI_DATA_DIR", "data")

    # Resolve tickers
    if args.tickers:
        tickers = args.tickers
    else:
        from hifi.data.universe import PHASE14_UNIVERSE
        tickers = [row["ticker"] for row in PHASE14_UNIVERSE]

    # Resolve dates
    dates = _resolve_dates(args)

    total = len(dates) * len(tickers)

    if args.status:
        done = count_completed(args.output_dir, args.condition, tickers, dates)
        print(
            f"Condition: {args.condition}  Period: {args.period}\n"
            f"  Dates: {len(dates)}  Tickers: {len(tickers)}  "
            f"Total: {total}  Completed: {done}  Remaining: {total - done}"
        )
        return

    if args.dry_run:
        print(
            f"[dry-run] Condition: {args.condition}  Period: {args.period}\n"
            f"  Dates ({len(dates)}): {dates[0]} .. {dates[-1]}\n"
            f"  Tickers ({len(tickers)}): {tickers[:5]} ...\n"
            f"  Total runs: {total}"
        )
        return

    if not args.quiet:
        logger.info(
            "Starting walkforward: condition=%s dates=%d tickers=%d total=%d",
            args.condition, len(dates), len(tickers), total,
        )

    done = 0
    skipped = 0
    failed = 0
    t_start = time.monotonic()

    # Dates are processed in strictly ascending order (causal discipline)
    for date in dates:
        for ticker in tickers:
            out = output_path(args.output_dir, args.condition, date, ticker)
            if out.exists():
                skipped += 1
                continue
            result = run_one(
                ticker=ticker,
                date=date,
                condition=args.condition,
                data_dir=data_dir,
                output_dir=args.output_dir,
            )
            if result is not None:
                done += 1
            else:
                failed += 1

    elapsed = time.monotonic() - t_start
    print(
        f"Walkforward complete: condition={args.condition}\n"
        f"  done={done}  skipped={skipped}  failed={failed}  "
        f"elapsed={elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
