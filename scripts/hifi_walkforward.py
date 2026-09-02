#!/usr/bin/env python
"""CLI for the agent-first walk-forward sweep. All logic lives in ``hifi.live``.

The offline half of the experiment: N tickers x M month-ends x 4 ablation
conditions x 6 agents. The live cycle (scripts/hifi_live.py) shares this
package, so an ensemble composed tonight and one composed over the 2022-2023
held-out window are composed by the same code — which is the only reason the
two sets of results are comparable.

Modes are composable; at least one is required.

  --agent AGENT   Load the model, run that agent's passes over all
                  (date, ticker), unload. Checkpoint-resume: existing sidecars
                  are skipped, so an interrupted sweep resumes where it stopped.
  --aggregate     Turn stored per-agent sidecars into ensemble JSONs.
  --pipeline      Run the MCP pipeline on ensemble JSONs -> one PortfolioSnapshot
                  per date.
  --status        Report checkpoint progress. No LLM calls.
  --dry-run       Print the schedule without calling any model.

A full production run, one condition:

  uv run python scripts/hifi_walkforward.py --agent all --condition full --period held-out-test
  uv run python scripts/hifi_walkforward.py --aggregate --condition full --period held-out-test
  uv run python scripts/hifi_walkforward.py --pipeline  --condition full --period held-out-test

or in one call: make walkforward-orchestrate

Storage layout
--------------
  Agent sidecars:  {data_dir}/runs/{condition}-{date}-{ticker}/{ticker}_{agent}.json
  Ensemble JSONs:  {output_dir}/{condition}/{YYYY}/{MM}/{ticker}.json
  Portfolio JSONs: {output_dir}/{condition}/{YYYY}/{MM}/portfolio.json
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

from hifi.agents.roster import CANONICAL_ORDER  # noqa: E402
from hifi.live.ensemble import run_agent_mode, run_aggregate_mode  # noqa: E402
from hifi.live.paths import _resolve_dates, _resolve_tickers  # noqa: E402
from hifi.live.walkforward import run_pipeline_mode, run_status_mode  # noqa: E402

_DEFAULT_OUTPUT_DIR = "data/walkforward"
_DEFAULT_PERIOD = "held-out-test"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward sweep: agent-first passes + MCP pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Action flags (composable; at least one required)
    p.add_argument(
        "--agent",
        choices=CANONICAL_ORDER + ["all"],
        metavar="AGENT_TYPE",
        help=(
            "Run one agent's passes (fundamental|technical|risk|macro|sentiment|contrarian), "
            "or 'all' for the full sequential sweep"
        ),
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate stored per-agent sidecars into ensemble JSONs",
    )
    p.add_argument(
        "--pipeline",
        action="store_true",
        help="Run MCP pipeline on ensemble JSONs → PortfolioSnapshot per date",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Show checkpoint progress for all conditions (no LLM calls)",
    )

    # Scope
    p.add_argument(
        "--condition",
        choices=["full", "parallel", "homogeneous", "no-memory"],
        default="full",
    )
    p.add_argument(
        "--period",
        default=_DEFAULT_PERIOD,
        help=(
            "Evaluation period: training | validation | held-out-test | "
            "walk-forward (default: held-out-test)"
        ),
    )
    p.add_argument("--start-date", default=None, help="Override period start (ISO 8601)")
    p.add_argument("--end-date", default=None, help="Override period end (ISO 8601)")
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Subset of tickers (default: all 98 PHASE14_UNIVERSE tickers)",
    )

    # Output / behaviour
    p.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--data-dir",
        default=None,
        help="Data directory (default: $HIFI_DATA_DIR or 'data')",
    )
    p.add_argument("--dry-run", action="store_true", help="Show schedule without LLM calls")
    p.add_argument("--quiet", action="store_true", help="Suppress per-item log lines")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # At least one action required
    if not any([args.agent, args.aggregate, args.pipeline, args.status]):
        print(
            "ERROR: specify at least one action flag: --agent, --aggregate, --pipeline, --status"
        )
        sys.exit(1)

    data_dir = args.data_dir or os.environ.get("HIFI_DATA_DIR", "data")
    os.environ.setdefault("HIFI_DATA_DIR", data_dir)
    db_path = str(Path(data_dir) / "knowledge.lance")

    tickers = _resolve_tickers(args.tickers)
    dates = _resolve_dates(args.period, args.start_date, args.end_date)
    condition = args.condition

    if not args.quiet:
        logger.info(
            "Orchestrator: condition=%s period=%s dates=%d tickers=%d",
            condition, args.period, len(dates), len(tickers),
        )

    any_fail = False

    # ------------------------------------------------------------------
    # --status
    # ------------------------------------------------------------------
    if args.status:
        all_conditions = ["full", "parallel", "homogeneous", "no-memory"]
        run_status_mode(all_conditions, dates, tickers, data_dir, args.output_dir)

    # ------------------------------------------------------------------
    # --agent
    # ------------------------------------------------------------------
    if args.agent:
        agents_to_run = CANONICAL_ORDER if args.agent == "all" else [args.agent]
        t_start = time.monotonic()
        for at in agents_to_run:
            counts = run_agent_mode(
                agent_type=at,
                condition=condition,
                dates=dates,
                tickers=tickers,
                data_dir=data_dir,
                db_path=db_path,
                dry_run=args.dry_run,
                quiet=args.quiet,
            )
            if counts["fail"] > 0:
                any_fail = True
        elapsed = time.monotonic() - t_start
        print(f"Agent sweep complete ({elapsed:.0f}s)")

    # ------------------------------------------------------------------
    # --aggregate
    # ------------------------------------------------------------------
    if args.aggregate:
        counts = run_aggregate_mode(
            condition=condition,
            dates=dates,
            tickers=tickers,
            data_dir=data_dir,
            db_path=db_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )
        if counts["fail"] > 0:
            any_fail = True

    # ------------------------------------------------------------------
    # --pipeline
    # ------------------------------------------------------------------
    if args.pipeline:
        counts = run_pipeline_mode(
            condition=condition,
            dates=dates,
            tickers=tickers,
            data_dir=data_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )
        if counts["fail"] > 0:
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
