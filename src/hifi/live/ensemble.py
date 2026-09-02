"""Running the six agent passes and aggregating them into one decision.

The same code serves the nightly cycle and the walk-forward sweep, which is the
point: the live arms must not be a re-implementation of the harness the offline
results were measured on, or the two stop being comparable. ``run_agent_mode``
runs one agent across every (date, ticker) with checkpoint-resume;
``run_aggregate_mode`` turns the stored sidecars into an ensemble decision;
``run_ensemble`` is the live wrapper that does both for a single date.
"""

from __future__ import annotations

import json
import logging
import os
import time

from hifi.agents.roster import CANONICAL_ORDER
from hifi.live import models, paths

logger = logging.getLogger(__name__)

#: LanceDB namespace holding per-run agent context for the offline sweep. The
#: live cycle uses "hifi-live-context" (paths._CONTEXT_NAMESPACE); the two are
#: kept separate so an eval run can never read a live run's prior context.
_EVAL_CONTEXT_NAMESPACE = "hifi-eval-context"


def _fetch_edgar_context(ticker: str, date: str, db_path: str) -> str:
    try:
        from hifi.knowledge.edgar_retriever import retrieve_mda_context  # noqa: PLC0415

        return retrieve_mda_context(
            ticker=ticker,
            as_of_date=date,
            namespace=paths._EDGAR_NAMESPACE,
            db_path=db_path,
        )
    except Exception as exc:
        logger.debug("EDGAR miss %s/%s: %s", ticker, date, exc)
        return ""


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
    from hifi.agents.context import (
        CONTEXT_ELIGIBLE_AGENTS as _CONTEXT_ELIGIBLE_AGENTS,  # noqa: PLC0415
    )
    from hifi.simulation.agent_executor import run_agent_pass  # noqa: PLC0415
    from hifi.simulation.model_manager import unload_model  # noqa: PLC0415

    agent_cfg = {
        at: (mid, ev, lt, cl)
        for at, mid, ev, lt, cl in models._agent_config_for_condition(condition)
    }
    lms_model_id, env_var, load_timeout, ctx_len = agent_cfg[agent_type]

    total = len(dates) * len(tickers)
    n_done = n_skip = n_fail = 0

    if dry_run:
        existing = sum(
            1 for d in dates for t in tickers
            if paths._sidecar_path(data_dir, condition, d, t, agent_type).exists()
        )
        print(
            f"[dry-run] --agent {agent_type} --condition {condition}\n"
            f"  dates={len(dates)}  tickers={len(tickers)}  total={total}"
            f"  already_done={existing}  remaining={total - existing}"
        )
        return {"done": 0, "skip": existing, "fail": 0}

    agent_ready = models._setup_agent_model(
        agent_type, lms_model_id, env_var, load_timeout, ctx_len)

    for date in dates:
        for ticker in tickers:
            sidecar = paths._sidecar_path(data_dir, condition, date, ticker, agent_type)
            if sidecar.exists():
                n_skip += 1
                continue

            if not agent_ready:
                n_fail += 1
                continue

            try:
                # DJ-130: standing-situation context for eligible agents. Only
                # active when the live orchestrator tagged an account via env —
                # evaluation harnesses never do, so historical replays stay
                # byte-identical to their original runs.
                extra_memory_prefix = ""
                _acct = os.environ.get("HIFI_ACTIVE_ACCOUNT")
                if agent_type in _CONTEXT_ELIGIBLE_AGENTS and _acct:
                    from hifi.agents.context import (  # noqa: PLC0415
                        build_market_block,
                        build_portfolio_context,
                        load_book_state,
                    )

                    _book = load_book_state(_acct, data_dir)
                    if _book:
                        extra_memory_prefix = build_portfolio_context(
                            _book, _acct, data_dir
                        )
                    _market = build_market_block(ticker, date, data_dir)
                    extra_memory_prefix = (
                        (_market + "\n\n" + extra_memory_prefix).strip()
                    )
                    logger.info(
                        "CONTEXT %s %s %s: portfolio+%d market+%d chars injected",
                        agent_type, ticker, date,
                        len(extra_memory_prefix) - len(_market) - 2, len(_market))

                if agent_type == "fundamental":
                    _edgar = _fetch_edgar_context(ticker, date, db_path)
                    extra_memory_prefix = (
                        _edgar + "\n\n" + extra_memory_prefix).strip()

                run_agent_pass(
                    agent_type=agent_type,
                    ticker=ticker,
                    date=date,
                    condition=condition,
                    run_id=paths._run_id(condition, date, ticker),
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
            out_path = paths._ensemble_path(output_dir, condition, date, ticker)
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
                    run_id=paths._run_id(condition, date, ticker),
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


def run_ensemble(
    tickers: list[str], date: str, condition: str, account: str, dry_run: bool
) -> None:
    """Run all 6 agents for the given condition, then aggregate.

    run_agent_mode and run_aggregate_mode are defined above in this module. They
    used to be reached from the live script via a sys.path insert into scripts/,
    which is how the production ensemble came to run out of a file named for a
    retracted evaluation phase (DJ-135).
    """
    # Date-partition the ensemble output (HiFi issue #2). The walk-forward
    # paths._ensemble_path is MONTH-keyed and the aggregate step skips-if-exists
    # (checkpoint-resume); in live trading multiple decision dates share a
    # month, so without this every run after the first-of-month reused the
    # first run's stale ensemble. One dir per date isolates each decision.
    output_dir = str(paths._account_dir(account) / "walkforward" / date)

    for agent_type in CANONICAL_ORDER:
        run_agent_mode(
            agent_type=agent_type,
            condition=condition,
            dates=[date],
            tickers=tickers,
            data_dir=paths._DATA_DIR,
            db_path=paths._DB_PATH,
            dry_run=dry_run,
            quiet=False,
        )

    if not dry_run:
        run_aggregate_mode(
            condition=condition,
            dates=[date],
            tickers=tickers,
            data_dir=paths._DATA_DIR,
            db_path=paths._DB_PATH,
            output_dir=output_dir,
            dry_run=False,
            quiet=False,
        )


def load_ensemble_signals(
    tickers: list[str], date: str, condition: str, account: str
) -> list[dict]:
    sectors = paths._get_sectors()
    year, month, _ = date.split("-")
    signals = []
    for ticker in tickers:
        ens_path = (
            paths._account_dir(account) / "walkforward" / date
            / condition / year / month / f"{ticker}.json"
        )
        if not ens_path.exists():
            continue
        with open(ens_path) as f:
            ens = json.load(f)
        ed = ens.get("ensemble_decision") or {}
        signals.append({
            "ticker": ticker,
            "decision": ed.get("collective_decision", "Hold"),
            "confidence": ed.get("collective_confidence", 0.5),
            "sector": sectors.get(ticker, "Unknown"),
        })
    return signals
