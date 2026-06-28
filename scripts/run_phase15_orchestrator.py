"""
Phase 15 Master Orchestrator — Agent-First Sequential Sweep (DJ-106, DJ-109).

The production script for the full scientific experiment:
  98 tickers × 24 month-ends × 4 ablation conditions × 6 agents.

Operation modes (mutually composable flags):
  --agent AGENT_TYPE   Load model, run agent passes for all (date, ticker), unload.
                       Checkpoint-resume: skips existing JSON sidecars.
  --aggregate          Aggregate stored per-agent sidecars → ensemble JSONs.
  --pipeline           Run MCP pipeline on ensemble JSONs → PortfolioSnapshot per date.
  --status             Show checkpoint progress (no LLM calls).
  --dry-run            Print schedule without calling LLMs.

Execution pattern for a full production run:
  uv run python scripts/run_phase15_orchestrator.py \\
      --agent fundamental --condition full --period held-out-test
  uv run python scripts/run_phase15_orchestrator.py \\
      --agent technical --condition full --period held-out-test
  ... (repeat for risk, macro, sentiment, contrarian) ...
  uv run python scripts/run_phase15_orchestrator.py \\
      --aggregate --condition full --period held-out-test
  uv run python scripts/run_phase15_orchestrator.py \\
      --pipeline --condition full --period held-out-test

Or in one call via the Makefile target:
  make walkforward-orchestrate

Storage layout
--------------
  Agent sidecars:  {data_dir}/runs/{condition}-{date}-{ticker}/{ticker}_{agent_type}.json
  Ensemble JSONs:  {output_dir}/{condition}/{YYYY}/{MM}/{ticker}.json
  Portfolio JSONs: {output_dir}/{condition}/{YYYY}/{MM}/portfolio.json
"""

from __future__ import annotations

import argparse
import json
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

_DEFAULT_OUTPUT_DIR = "data/walkforward"
_DEFAULT_PERIOD = "held-out-test"
_CAPITAL = 500_000.0
_EVAL_CONTEXT_NAMESPACE = "hifi-eval-context"
_EDGAR_NAMESPACE = "hifi-dev-sec"

_FINETUNE_MODEL = "qwen2.5-coder-32b"
_TECHNICAL_FINETUNE_URL = "http://localhost:1235/v1"
_FUNDAMENTAL_FINETUNE_URL = "http://localhost:1236/v1"
_FINETUNE_HEALTH_1235 = "http://localhost:1235/health"
_FINETUNE_HEALTH_1236 = "http://localhost:1236/health"

# Standard model config (full / parallel / no-memory conditions).
# Tuples: (agent_type, lms_model_id | None, env_var | None, load_timeout_s, ctx_len | None)
# ctx_len: override lms load -c <n>. Gemma 12B's default (~4096) is too small for
# tickers with long EDGAR passages (prompt + output ≈ 4,357 tokens for AAPL).
_AGENT_CONFIG: list[tuple[str, str | None, str | None, int, int | None]] = [
    # fmt: (agent_type, lms_model_id, env_var, load_timeout_s, ctx_len_override)
    ("fundamental", "llama-3.3-70b-instruct",      "HIFI_FUNDAMENTAL_MODEL", 600, None),
    ("technical",   None,                           None,                     0,   None),
    ("risk",        "mistral-small-3.2-24b-instruct-2506-mlx",
                                                    "HIFI_RISK_MODEL",        300, None),
    ("macro",       "deepseek-r1-distill-qwen-32b", "HIFI_MACRO_MODEL",       600, None),
    ("sentiment",   "gemma-3-12b-it",               "HIFI_SENTIMENT_MODEL",   300, 8192),
    ("contrarian",  "mlx-qwen3.5-35b-a3b",          "HIFI_CONTRARIAN_MODEL",  300, None),
]

# Homogeneous model config (Phase 13 qwen-dominant baseline, DJ-096).
_HOMOGENEOUS_AGENT_CONFIG: list[tuple[str, str | None, str | None, int, int | None]] = [
    ("fundamental", "qwen2.5-coder-32b-instruct-mlx",  "HIFI_FUNDAMENTAL_MODEL", 300, None),
    ("technical",   "qwen2.5-coder-32b-instruct-mlx",  "HIFI_TECHNICAL_MODEL",   300, None),
    ("risk",        "gemma-3-4b-it",                    "HIFI_RISK_MODEL",        120, None),
    ("macro",       "mlx-community-qwen3-235b-a22b",    "HIFI_MACRO_MODEL",       600, None),
    ("sentiment",   "qwen2.5-coder-32b-instruct-mlx",  "HIFI_SENTIMENT_MODEL",   300, None),
    ("contrarian",  "mlx-community-qwen3-235b-a22b",    "HIFI_CONTRARIAN_MODEL",  600, None),
]

CANONICAL_ORDER = ["fundamental", "technical", "risk", "macro", "sentiment", "contrarian"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _run_id(condition: str, date: str, ticker: str) -> str:
    return f"{condition}-{date}-{ticker}"


def _sidecar_path(data_dir: str, condition: str, date: str, ticker: str, agent_type: str) -> Path:
    rid = _run_id(condition, date, ticker)
    return Path(data_dir) / "runs" / rid / f"{ticker}_{agent_type}.json"


def _ensemble_path(output_dir: str, condition: str, date: str, ticker: str) -> Path:
    year, month, _ = date.split("-")
    return Path(output_dir) / condition / year / month / f"{ticker}.json"


def _portfolio_path(output_dir: str, condition: str, date: str) -> Path:
    year, month, _ = date.split("-")
    return Path(output_dir) / condition / year / month / "portfolio.json"


# ---------------------------------------------------------------------------
# Dates / tickers helpers
# ---------------------------------------------------------------------------


def _resolve_tickers(tickers_arg: list[str] | None) -> list[str]:
    if tickers_arg:
        return tickers_arg
    from hifi.data.universe import PHASE14_UNIVERSE  # noqa: PLC0415

    return [row["ticker"] for row in PHASE14_UNIVERSE]


def _resolve_dates(period: str, start_date: str | None, end_date: str | None) -> list[str]:
    from hifi.simulation.schedule import (  # noqa: PLC0415
        WalkForwardPeriod,
        generate_month_ends,
        get_multi_period_dates,
        get_period_dates,
    )

    if start_date and end_date:
        return generate_month_ends(start_date, end_date)
    if period == "all":
        return get_multi_period_dates([
            WalkForwardPeriod.VALIDATION,
            WalkForwardPeriod.HELD_OUT_TEST,
            WalkForwardPeriod.WALK_FORWARD,
        ])
    return get_period_dates(period)


def _agent_config_for_condition(
    condition: str,
) -> list[tuple[str, str | None, str | None, int, int | None]]:
    return _HOMOGENEOUS_AGENT_CONFIG if condition == "homogeneous" else _AGENT_CONFIG


# ---------------------------------------------------------------------------
# EDGAR context
# ---------------------------------------------------------------------------


def _fetch_edgar_context(ticker: str, date: str, db_path: str) -> str:
    try:
        from hifi.knowledge.edgar_retriever import retrieve_mda_context  # noqa: PLC0415

        return retrieve_mda_context(
            ticker=ticker,
            as_of_date=date,
            namespace=_EDGAR_NAMESPACE,
            db_path=db_path,
        )
    except Exception as exc:
        logger.debug("EDGAR miss %s/%s: %s", ticker, date, exc)
        return ""


# ---------------------------------------------------------------------------
# Model management helpers
# ---------------------------------------------------------------------------


def _port_is_listening(url: str, timeout_s: int = 3) -> bool:
    import urllib.request  # noqa: PLC0415

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _setup_agent_model(
    agent_type: str,
    lms_model_id: str | None,
    env_var: str | None,
    load_timeout: int,
    context_length: int | None = None,
) -> bool:
    """
    Load model and set env vars. Returns True if agent is ready to run.
    """
    import urllib.request as _ur  # noqa: PLC0415

    from hifi.simulation.model_manager import load_model, model_is_loaded  # noqa: PLC0415

    if agent_type == "technical" and lms_model_id is None:
        ok = _port_is_listening(_FINETUNE_HEALTH_1235)
        if ok:
            # mlx_lm server registers the model under its full local path, not the
            # short LM Studio name. Query /v1/models to get the actual registered ID
            # so requests are routed to the loaded model instead of triggering a
            # dynamic HuggingFace download (which would 404 for local-only models).
            try:
                with _ur.urlopen("http://localhost:1235/v1/models", timeout=5) as _r:
                    _registered_id = json.loads(_r.read())["data"][0]["id"]
            except Exception:
                _registered_id = _FINETUNE_MODEL  # fallback
            os.environ["HIFI_TECHNICAL_FINETUNE_URL"] = _TECHNICAL_FINETUNE_URL
            os.environ["HIFI_TECHNICAL_MODEL"] = _registered_id
            logger.info("Technical fine-tuned server ready (port 1235, model=%s)", _registered_id)
        else:
            logger.warning("port 1235 not healthy; technical passes will fail")
        return ok

    assert lms_model_id is not None
    assert env_var is not None

    if model_is_loaded(lms_model_id):
        os.environ[env_var] = lms_model_id
        logger.info("Model already loaded: %s", lms_model_id)
        return True

    # Evict stale models before loading (handles variant IDs that substring-match
    # at detection but fail to unload, e.g. mlx-qwen3.5-35b-a3b-claude-4.6-opus-
    # reasoning-distilled blocking Llama-70B at condition boundaries).
    from hifi.simulation.model_manager import unload_all  # noqa: PLC0415
    unload_all()

    logger.info("Loading %s ...", lms_model_id)
    t0 = time.monotonic()
    ok = load_model(lms_model_id, timeout_s=load_timeout, context_length=context_length)
    elapsed = int(time.monotonic() - t0)

    if ok:
        os.environ[env_var] = lms_model_id
        logger.info("Loaded %s (%ds)", lms_model_id, elapsed)
        return True

    # Fundamental-only fallback to fine-tuned server (port 1236)
    if agent_type == "fundamental":
        ft_ok = _port_is_listening(_FINETUNE_HEALTH_1236)
        if ft_ok:
            # Set BOTH env vars — old smoke test only set FINETUNE_URL (bug fix, DJ-109)
            os.environ["HIFI_FUNDAMENTAL_FINETUNE_URL"] = _FUNDAMENTAL_FINETUNE_URL
            os.environ["HIFI_FUNDAMENTAL_FINETUNE_MODEL"] = _FINETUNE_MODEL
            os.environ["HIFI_FUNDAMENTAL_MODEL"] = _FINETUNE_MODEL
            logger.info(
                "Fundamental FALLBACK: using fine-tuned server port 1236 (%s)",
                _FINETUNE_MODEL,
            )
            return True

    logger.warning(
        "Failed to load %s (%ds); passes for %s will fail",
        lms_model_id, elapsed, agent_type,
    )
    return False


# ---------------------------------------------------------------------------
# --agent mode: run one agent's passes across all (date, ticker)
# ---------------------------------------------------------------------------


def run_agent_mode(
    agent_type: str,
    condition: str,
    dates: list[str],
    tickers: list[str],
    data_dir: str,
    db_path: str,
    dry_run: bool,
    quiet: bool,
) -> dict[str, int]:
    """
    Load model, run agent passes for all (date, ticker), unload.
    Returns {"done": n, "skip": n, "fail": n}.
    """
    from hifi.simulation.agent_executor import run_agent_pass  # noqa: PLC0415
    from hifi.simulation.model_manager import unload_model  # noqa: PLC0415

    agent_cfg = {
        at: (mid, ev, lt, cl)
        for at, mid, ev, lt, cl in _agent_config_for_condition(condition)
    }
    lms_model_id, env_var, load_timeout, ctx_len = agent_cfg[agent_type]

    total = len(dates) * len(tickers)
    n_done = n_skip = n_fail = 0

    if dry_run:
        existing = sum(
            1 for d in dates for t in tickers
            if _sidecar_path(data_dir, condition, d, t, agent_type).exists()
        )
        print(
            f"[dry-run] --agent {agent_type} --condition {condition}\n"
            f"  dates={len(dates)}  tickers={len(tickers)}  total={total}"
            f"  already_done={existing}  remaining={total - existing}"
        )
        return {"done": 0, "skip": existing, "fail": 0}

    agent_ready = _setup_agent_model(agent_type, lms_model_id, env_var, load_timeout, ctx_len)

    for date in dates:
        for ticker in tickers:
            sidecar = _sidecar_path(data_dir, condition, date, ticker, agent_type)
            if sidecar.exists():
                n_skip += 1
                continue

            if not agent_ready:
                n_fail += 1
                continue

            try:
                extra_memory_prefix = ""
                if agent_type == "fundamental":
                    extra_memory_prefix = _fetch_edgar_context(ticker, date, db_path)

                run_agent_pass(
                    agent_type=agent_type,
                    ticker=ticker,
                    date=date,
                    condition=condition,
                    run_id=_run_id(condition, date, ticker),
                    data_dir=data_dir,
                    db_path=db_path,
                    context_namespace=_EVAL_CONTEXT_NAMESPACE,
                    extra_memory_prefix=extra_memory_prefix,
                )
                n_done += 1
                if not quiet:
                    logger.info("DONE %s %s %s", agent_type, ticker, date)
            except Exception as exc:
                n_fail += 1
                logger.error("FAIL %s %s %s: %s", agent_type, ticker, date, exc)

    # Unload LM Studio model (skip technical which uses external mlx_lm.server)
    if lms_model_id is not None:
        unload_model(lms_model_id)
        time.sleep(3)

    counts = {"done": n_done, "skip": n_skip, "fail": n_fail}
    print(f"Agent {agent_type}: done={n_done} skip={n_skip} fail={n_fail}")
    return counts


# ---------------------------------------------------------------------------
# --aggregate mode: sidecars → ensemble JSONs
# ---------------------------------------------------------------------------


def run_aggregate_mode(
    condition: str,
    dates: list[str],
    tickers: list[str],
    data_dir: str,
    db_path: str,
    output_dir: str,
    dry_run: bool,
    quiet: bool,
) -> dict[str, int]:
    """
    For each (date, ticker): read per-agent JSON sidecars → aggregate → write ensemble JSON.
    Checkpoint-resume: skips if ensemble JSON already exists.
    """
    from hifi.simulation.agent_executor import aggregate_agent_outputs  # noqa: PLC0415

    n_done = n_skip = n_fail = 0

    for date in dates:
        for ticker in tickers:
            out_path = _ensemble_path(output_dir, condition, date, ticker)
            if out_path.exists():
                n_skip += 1
                continue

            if dry_run:
                n_done += 1
                continue

            try:
                eo = aggregate_agent_outputs(
                    ticker=ticker,
                    date=date,
                    run_id=_run_id(condition, date, ticker),
                    db_path=db_path,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(".tmp")
                tmp.write_text(eo.model_dump_json(), encoding="utf-8")
                tmp.rename(out_path)
                n_done += 1
                if not quiet:
                    logger.info("AGGREGATE %s %s → %s", ticker, date, out_path)
            except Exception as exc:
                n_fail += 1
                logger.error("AGGREGATE FAIL %s %s: %s", ticker, date, exc)

    if dry_run:
        print(
            f"[dry-run] --aggregate --condition {condition}\n"
            f"  dates={len(dates)}  tickers={len(tickers)}  would_run={n_done}  skip={n_skip}"
        )
    else:
        print(f"Aggregate: done={n_done} skip={n_skip} fail={n_fail}")
    return {"done": n_done, "skip": n_skip, "fail": n_fail}


# ---------------------------------------------------------------------------
# --pipeline mode: ensemble JSONs → PortfolioSnapshot per date
# ---------------------------------------------------------------------------


def _load_ohlcv(
    data_dir: str, tickers: list[str], as_of_date: str
) -> dict[str, list[dict]]:
    try:
        import pandas as pd  # noqa: PLC0415

        result: dict[str, list[dict]] = {}
        for ticker in tickers:
            path = Path(data_dir) / "market" / ticker / "ohlcv.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df.columns = df.columns.str.lower()   # normalize 'Close' → 'close'
            df.index = pd.to_datetime(df.index)
            df = df[df.index <= as_of_date].tail(90)
            if df.empty:
                continue
            result[ticker] = [
                {"date": str(idx.date()), "close": float(row["close"])}
                for idx, row in df.iterrows()
            ]
        return result
    except Exception as exc:
        logger.warning("OHLCV load error: %s", exc)
        return {}


def run_pipeline_mode(
    condition: str,
    dates: list[str],
    tickers: list[str],
    data_dir: str,
    output_dir: str,
    dry_run: bool,
    quiet: bool,
) -> dict[str, int]:
    """
    For each date: collect signals from ensemble JSONs → run MCP pipeline → write portfolio.json.
    Checkpoint-resume: skips if portfolio.json already exists.
    """
    from hifi.data.universe import get_sector  # noqa: PLC0415
    from hifi.simulation.pipeline import run_pipeline  # noqa: PLC0415

    n_done = n_skip = n_fail = 0

    for date in dates:
        port_path = _portfolio_path(output_dir, condition, date)
        if port_path.exists():
            n_skip += 1
            continue

        if dry_run:
            n_done += 1
            continue

        # Collect signals from ensemble JSONs for this date
        signals: list[dict] = []
        for ticker in tickers:
            ens_path = _ensemble_path(output_dir, condition, date, ticker)
            if not ens_path.exists():
                continue
            try:
                data = json.loads(ens_path.read_text(encoding="utf-8"))
                ed = data.get("ensemble_decision", {})
                decision = ed.get("collective_decision") or "Hold"
                confidence = float(ed.get("collective_confidence") or 0.5)
                signals.append({
                    "ticker": ticker,
                    "decision": decision,
                    "confidence": confidence,
                    "sector": get_sector(ticker) or "Unknown",
                })
            except Exception as exc:
                logger.warning("Could not read ensemble JSON %s %s: %s", ticker, date, exc)

        if not signals:
            logger.warning("No signals for %s %s; skipping pipeline", condition, date)
            n_fail += 1
            continue

        try:
            ohlcv = _load_ohlcv(data_dir, tickers, date)
            prices = {t: rows[-1]["close"] for t, rows in ohlcv.items() if rows}

            portfolio_state = {
                "portfolio": {},
                "portfolio_value": _CAPITAL,
                "hwm_value": _CAPITAL,
                "holdings": {},
                "prices": prices,
            }
            constraints = {
                "max_single_stock": 0.05,
                "max_sector": 0.20,
                "min_position": 0.005,
                "capital": _CAPITAL,
                "current_capital": 0.0,
            }

            snapshot = run_pipeline(signals, ohlcv, portfolio_state, constraints)

            port_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = port_path.with_suffix(".tmp")
            tmp.write_text(snapshot.to_json(), encoding="utf-8")
            tmp.rename(port_path)
            n_done += 1
            if not quiet:
                logger.info(
                    "PIPELINE %s %s: buy=%d orders=%d notional=%.0f → %s",
                    condition, date,
                    snapshot.n_buy, len(snapshot.orders),
                    snapshot.total_estimated_value, port_path,
                )
        except Exception as exc:
            n_fail += 1
            logger.error("PIPELINE FAIL %s %s: %s", condition, date, exc)

    if dry_run:
        print(
            f"[dry-run] --pipeline --condition {condition}\n"
            f"  dates={len(dates)}  would_run={n_done}  skip={n_skip}"
        )
    else:
        print(f"Pipeline: done={n_done} skip={n_skip} fail={n_fail}")
    return {"done": n_done, "skip": n_skip, "fail": n_fail}


# ---------------------------------------------------------------------------
# --status mode
# ---------------------------------------------------------------------------


def run_status_mode(
    conditions: list[str],
    dates: list[str],
    tickers: list[str],
    data_dir: str,
    output_dir: str,
) -> None:
    total_passes = len(dates) * len(tickers)
    total_ensembles = len(dates) * len(tickers)
    total_portfolios = len(dates)

    print(f"\nStatus  |  dates={len(dates)}  tickers={len(tickers)}")
    print(f"{'Condition':<16}  {'Sidecars':>10}  {'Ensembles':>10}  {'Portfolios':>11}")
    print("  " + "-" * 58)

    for condition in conditions:
        # Count sidecars: 6 agents × N_dates × N_tickers
        n_sidecars = sum(
            1
            for at in CANONICAL_ORDER
            for d in dates
            for t in tickers
            if _sidecar_path(data_dir, condition, d, t, at).exists()
        )
        total_sidecars = len(CANONICAL_ORDER) * total_passes

        n_ensembles = sum(
            1 for d in dates for t in tickers
            if _ensemble_path(output_dir, condition, d, t).exists()
        )

        n_portfolios = sum(
            1 for d in dates
            if _portfolio_path(output_dir, condition, d).exists()
        )

        print(
            f"  {condition:<16}"
            f"  {n_sidecars:>5}/{total_sidecars:<5}"
            f"  {n_ensembles:>5}/{total_ensembles:<5}"
            f"  {n_portfolios:>5}/{total_portfolios:<5}"
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 15 master orchestrator: agent-first sweep + MCP pipeline",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
