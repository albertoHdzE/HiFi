"""
Phase 15 Smoke Test — Agent-First Sequential Sweep (DJ-106, DJ-107, DJ-108, DJ-109).

Validates the full end-to-end pipeline on the 22-ticker SMOKE_UNIVERSE
(all 11 GICS sectors) before launching the full scientific run.

Agent-first sweep (memory-safe for 98 GB M3 Ultra):
  Pass 1: Load Llama 70B      → fundamental() for all 22 tickers → unload
  Pass 2: Technical (port 1235) → technical()  for all 22 tickers (no lms)
  Pass 3: Load Mistral 24B    → risk()         for all 22 tickers → unload
  Pass 4: Load DeepSeek 32B   → macro()        for all 22 tickers → unload
  Pass 5: Load Gemma 12B      → sentiment()    for all 22 tickers → unload
  Pass 6: Load Qwen3.5 MoE    → contrarian()   for all 22 tickers → unload
  Final:  CPU-only             → aggregate + MCP pipeline → PortfolioSnapshot

Peak VRAM at any point: ~35 GB (Llama 70B pass). OS overhead ~8 GB.
Total: ~43 GB — well within the 98 GB M3 Ultra budget.

Usage
-----
    uv run python scripts/run_phase15_smoke.py
    uv run python scripts/run_phase15_smoke.py --skip-load   # models already loaded
    uv run python scripts/run_phase15_smoke.py --cleanup
    uv run python scripts/run_phase15_smoke.py --condition parallel

Prerequisites
-------------
- LM Studio running (port 1234), no models pre-loaded
- Technical agent fine-tuned server on port 1235: make finetune-serve
- lms CLI: ~/.lmstudio/bin/lms
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SMOKE_DATE = "2022-01-31"
_CAPITAL = 500_000.0

_LM_BASE = "http://localhost:1234"
_FINETUNE_MODEL = "qwen2.5-coder-32b"
_TECHNICAL_FINETUNE_URL = "http://localhost:1235/v1"
_FUNDAMENTAL_FINETUNE_URL = "http://localhost:1236/v1"
_FINETUNE_HEALTH_1235 = "http://localhost:1235/health"
_FINETUNE_HEALTH_1236 = "http://localhost:1236/health"
_EDGAR_NAMESPACE = "hifi-dev-sec"  # all history, temporally filtered

# Agent-first sweep order.
# Tuples: (agent_type, lms_model_id | None, env_var | None, load_timeout_s | None, ctx_len | None)
# None model_id = external server (no lms load/unload).
# ctx_len: override lms load -c <n> when the model's built-in default is too small
#   for the intended prompts. Gemma 12B's default (~4096) is too small for AAPL
#   sentiment (prompt + output ≈ 4,357 tokens); 8192 gives comfortable headroom.
_AGENT_CONFIG: list[tuple[str, str | None, str | None, int | None, int | None]] = [
    # fmt: (agent_type, lms_model_id, env_var, load_timeout_s, ctx_len_override)
    ("fundamental", "llama-3.3-70b-instruct",      "HIFI_FUNDAMENTAL_MODEL", 600, None),
    ("technical",   None,                           None,                     None, None),
    ("risk",        "mistral-small-3.2-24b-instruct-2506-mlx",
                                                    "HIFI_RISK_MODEL",        300, None),
    ("macro",       "deepseek-r1-distill-qwen-32b", "HIFI_MACRO_MODEL",       600, None),
    ("sentiment",   "gemma-3-12b-it",               "HIFI_SENTIMENT_MODEL",   300, 8192),
    ("contrarian",  "mlx-qwen3.5-35b-a3b",          "HIFI_CONTRARIAN_MODEL",  300, None),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_is_listening(url: str, timeout_s: int = 3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _lm_api(path: str) -> dict:
    url = f"{_LM_BASE}{path}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=10) as resp:
        return json.loads(resp.read())


def _fetch_edgar_context(ticker: str, date: str, db_path: str) -> str:
    """Return EDGAR MD&A context, or '' on miss/error (fail-open)."""
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


def _load_ohlcv(
    data_dir: str, tickers: list[str], as_of_date: str
) -> dict[str, list[dict]]:
    """Load last 90 days of OHLCV (filtered to as_of_date) as list-of-dicts per ticker."""
    try:
        import pandas as pd  # noqa: PLC0415

        result: dict[str, list[dict]] = {}
        for ticker in tickers:
            path = Path(data_dir) / "market" / ticker / "ohlcv.parquet"
            if not path.exists():
                logger.warning("No OHLCV parquet for %s", ticker)
                continue
            df = pd.read_parquet(path)
            df.columns = df.columns.str.lower()
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


def _run_id(condition: str, ticker: str) -> str:
    """Deterministic per-ticker run_id for this smoke run."""
    return f"smoke-{condition}-{_SMOKE_DATE}-{ticker}"


def _sidecar_exists(data_dir: str, run_id: str, ticker: str, agent_type: str) -> bool:
    return (Path(data_dir) / "runs" / run_id / f"{ticker}_{agent_type}.json").exists()


# ---------------------------------------------------------------------------
# Agent-first sweep
# ---------------------------------------------------------------------------


def _setup_agent_env(
    agent_type: str,
    lms_model_id: str | None,
    env_var: str | None,
    load_timeout: int | None,
    skip_load: bool,
    context_length: int | None = None,
) -> bool:
    """
    Prepare env vars for an agent pass.
    Returns True if the agent is ready to run.
    """
    from hifi.simulation.model_manager import load_model, model_is_loaded  # noqa: PLC0415

    if agent_type == "technical":
        ok = _port_is_listening(_FINETUNE_HEALTH_1235)
        if ok:
            # mlx_lm server registers the model under its full local path, not the
            # short LM Studio name. Query /v1/models to get the actual registered ID
            # so requests are routed to the loaded model instead of triggering a
            # dynamic HuggingFace download (which would 404 for local-only models).
            try:
                with urllib.request.urlopen(
                    "http://localhost:1235/v1/models", timeout=5
                ) as _r:
                    _registered_id = json.loads(_r.read())["data"][0]["id"]
            except Exception:
                _registered_id = _FINETUNE_MODEL  # fallback
            os.environ["HIFI_TECHNICAL_FINETUNE_URL"] = _TECHNICAL_FINETUNE_URL
            os.environ["HIFI_TECHNICAL_MODEL"] = _registered_id
            print(f"  fine-tuned server port 1235: READY ({_registered_id})")
        else:
            print(
                "  WARNING: port 1235 not healthy. "
                "Technical passes will fail. Start with: make finetune-serve"
            )
        return ok

    # LM Studio agent
    assert lms_model_id is not None
    assert env_var is not None

    if skip_load or model_is_loaded(lms_model_id):
        if not model_is_loaded(lms_model_id):
            print(f"  --skip-load: assuming {lms_model_id} already loaded")
        else:
            print(f"  already loaded: {lms_model_id}")
        os.environ[env_var] = lms_model_id
        return True

    print(f"  loading {lms_model_id} ...", flush=True)
    t0 = time.monotonic()
    loaded_ok = load_model(
        lms_model_id, timeout_s=load_timeout or 600, context_length=context_length
    )
    elapsed = int(time.monotonic() - t0)

    if loaded_ok:
        os.environ[env_var] = lms_model_id
        print(f"  loaded ({elapsed}s)")
        return True

    # Fundamental-only fallback: fine-tuned server port 1236
    if agent_type == "fundamental":
        ft_ok = _port_is_listening(_FINETUNE_HEALTH_1236)
        if ft_ok:
            # BUG FIX (DJ-109): set BOTH FINETUNE_URL and FINETUNE_MODEL.
            # The old smoke script only set FINETUNE_URL, causing fundamental_agent.py
            # to fall through to its default model instead of the fine-tuned server.
            os.environ["HIFI_FUNDAMENTAL_FINETUNE_URL"] = _FUNDAMENTAL_FINETUNE_URL
            os.environ["HIFI_FUNDAMENTAL_FINETUNE_MODEL"] = _FINETUNE_MODEL
            os.environ["HIFI_FUNDAMENTAL_MODEL"] = _FINETUNE_MODEL
            print(
                f"  FALLBACK: Llama 70B OOM — using fine-tuned server "
                f"port 1236 ({_FINETUNE_MODEL} + fundamental_v1) [{elapsed}s]"
            )
            return True
        print(
            "  ERROR: Llama 70B failed AND port 1236 not healthy. "
            "Fundamental passes will fail."
        )
        return False

    print(f"  WARNING: {lms_model_id} failed to load ({elapsed}s). Passes will fail.")
    return False


def _run_sweep(
    smoke_tickers: list[str],
    condition: str,
    data_dir: str,
    db_path: str,
    skip_load: bool,
) -> dict[str, dict[str, bool]]:
    """
    Agent-first sequential sweep: for each agent, load model → run 22 tickers → unload.

    Returns {agent_type: {ticker: success_bool}}.
    """
    from hifi.simulation.agent_executor import run_agent_pass  # noqa: PLC0415
    from hifi.simulation.model_manager import unload_model  # noqa: PLC0415

    sweep_results: dict[str, dict[str, bool]] = {}

    for agent_type, lms_model_id, env_var, load_timeout, ctx_len in _AGENT_CONFIG:
        sweep_results[agent_type] = {}
        print(f"\n[{agent_type.upper()} PASS]", flush=True)

        agent_ready = _setup_agent_env(
            agent_type, lms_model_id, env_var, load_timeout, skip_load, ctx_len
        )

        n_ok = n_fail = n_skip = 0
        for ticker in smoke_tickers:
            rid = _run_id(condition, ticker)

            if _sidecar_exists(data_dir, rid, ticker, agent_type):
                sweep_results[agent_type][ticker] = True
                n_skip += 1
                continue

            if not agent_ready:
                sweep_results[agent_type][ticker] = False
                n_fail += 1
                continue

            try:
                extra_memory_prefix = ""
                if agent_type == "fundamental":
                    extra_memory_prefix = _fetch_edgar_context(ticker, _SMOKE_DATE, db_path)

                run_agent_pass(
                    agent_type=agent_type,
                    ticker=ticker,
                    date=_SMOKE_DATE,
                    condition=condition,
                    run_id=rid,
                    data_dir=data_dir,
                    db_path=db_path,
                    extra_memory_prefix=extra_memory_prefix,
                )
                sweep_results[agent_type][ticker] = True
                n_ok += 1
                print(f"  {ticker} OK", flush=True)
            except Exception as exc:
                sweep_results[agent_type][ticker] = False
                n_fail += 1
                print(f"  {ticker} FAIL: {exc}")

        print(f"  done={n_ok} skip={n_skip} fail={n_fail}")

        if agent_type != "technical" and lms_model_id is not None and not skip_load:
            unload_model(lms_model_id)
            time.sleep(3)  # let memory settle before next load

    return sweep_results


# ---------------------------------------------------------------------------
# Aggregate + MCP Pipeline
# ---------------------------------------------------------------------------


def _aggregate_and_pipeline(
    smoke_tickers: list[str],
    smoke_universe: list[dict],
    condition: str,
    data_dir: str,
    db_path: str,
) -> tuple[list[dict], object | None]:
    """
    Aggregate per-agent sidecars → EnsembleOutputs → signals list.
    Then run the full MCP pipeline (compose → risk → allocate).

    Returns (signals, PortfolioSnapshot | None).
    """
    from hifi.simulation.agent_executor import aggregate_agent_outputs  # noqa: PLC0415
    from hifi.simulation.pipeline import run_pipeline  # noqa: PLC0415

    sector_map = {e["ticker"]: e["sector"] for e in smoke_universe}
    signals: list[dict] = []

    print("\n[AGGREGATE]")
    for ticker in smoke_tickers:
        rid = _run_id(condition, ticker)
        try:
            eo = aggregate_agent_outputs(
                ticker=ticker,
                date=_SMOKE_DATE,
                run_id=rid,
                db_path=db_path,
            )
            ed = eo.ensemble_decision
            decision = ed.collective_decision or "Hold"
            confidence = float(ed.collective_confidence)
            n_sigs = len(eo.signals) if eo.signals else 0
            signals.append({
                "ticker": ticker,
                "decision": decision,
                "confidence": confidence,
                "sector": sector_map.get(ticker, "Unknown"),
            })
            print(f"  {ticker}: {decision} ({confidence:.3f}) signals={n_sigs}")
        except Exception as exc:
            print(f"  {ticker}: ERROR — {exc}")

    if not signals:
        print("  No valid signals — skipping pipeline.")
        return signals, None

    print("\n[PIPELINE]")
    ohlcv = _load_ohlcv(data_dir, smoke_tickers, _SMOKE_DATE)
    print(f"  OHLCV loaded for {len(ohlcv)}/{len(smoke_tickers)} tickers")

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

    try:
        snapshot = run_pipeline(signals, ohlcv, portfolio_state, constraints)
        print(
            f"  Pipeline OK: buy={snapshot.n_buy} hold={snapshot.n_hold} sell={snapshot.n_sell}"
            f"  orders={len(snapshot.orders)}  notional=${snapshot.total_estimated_value:,.0f}"
        )
        return signals, snapshot
    except Exception as exc:
        print(f"  Pipeline ERROR: {exc}")
        return signals, None


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(
    smoke_tickers: list[str],
    sweep_results: dict[str, dict[str, bool]],
    signals: list[dict],
    snapshot: object | None,
    condition: str,
) -> bool:
    sep = "=" * 78
    print(f"\n{sep}")
    print("  PHASE 15 SMOKE TEST — SUMMARY")
    print(sep)

    agent_types = [cfg[0] for cfg in _AGENT_CONFIG]
    signal_map = {s["ticker"]: s for s in signals}

    # Header row
    header = f"  {'Ticker':<8}"
    for at in agent_types:
        header += f"  {at[:4]:>4}"
    header += "  Decision    Conf  Sector"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    n_buy = n_hold = n_sell = 0
    sweep_fail_count = 0

    for ticker in smoke_tickers:
        row = f"  {ticker:<8}"
        for at in agent_types:
            ok = sweep_results.get(at, {}).get(ticker, False)
            row += f"  {'OK' if ok else 'XX':>4}"
            if not ok:
                sweep_fail_count += 1
        sig = signal_map.get(ticker, {})
        dec = sig.get("decision", "?")
        conf = sig.get("confidence", 0.0)
        sector = sig.get("sector", "?")[:18]
        row += f"  {dec:<8}  {conf:>5.3f}  {sector}"
        print(row)
        if dec == "Buy":
            n_buy += 1
        elif dec == "Hold":
            n_hold += 1
        elif dec == "Sell":
            n_sell += 1

    # Ensemble summary
    print(f"\n  Condition: {condition}  Date: {_SMOKE_DATE}")
    n_agg = len(signals)
    print(f"  Ensemble:  Buy={n_buy}  Hold={n_hold}  Sell={n_sell}  (from {n_agg} aggregated)")
    print(f"  Sweep:     {sum(sum(v.values()) for v in sweep_results.values())} OK  "
          f"{sweep_fail_count} FAIL")

    # Pipeline summary
    pipeline_ok = snapshot is not None
    if snapshot is not None:
        approved = snapshot.risk_report.get("approved_signals", [])
        blocked = snapshot.risk_report.get("blocked_tickers", [])
        print("\n  MCP Pipeline:")
        print(f"    Approved signals : {len(approved)}")
        print(f"    Blocked tickers  : {len(blocked)}")
        print(f"    Orders generated : {len(snapshot.orders)}")
        print(f"    Total notional   : ${snapshot.total_estimated_value:>12,.0f}")

        if snapshot.sector_exposure:
            print("\n  Sector exposure (max_sector=20%):")
            for sector, exp in sorted(
                snapshot.sector_exposure.items(), key=lambda x: -x[1]
            )[:8]:
                bar = "#" * int(exp * 50)
                cap_flag = " ← CAP" if exp >= 0.195 else ""
                print(f"    {sector[:28]:<28} {exp:>6.1%}  {bar}{cap_flag}")
    else:
        print("\n  MCP Pipeline: FAILED or skipped")

    # Verdict
    all_sweep_ok = sweep_fail_count == 0
    print(f"\n  Sweep:    {'PASS' if all_sweep_ok else 'FAIL'}")
    print(f"  Pipeline: {'PASS' if pipeline_ok else 'FAIL'}")
    verdict = "PASS" if (all_sweep_ok and pipeline_ok) else "FAIL"
    print(f"\n  {'─' * 30}  {verdict}  {'─' * 30}")
    if all_sweep_ok and pipeline_ok:
        print(
            "\n  NEXT: make walkforward-orchestrate"
            "\n        (full 98-ticker × 24-date × 4-condition scientific run)"
        )
    else:
        print("\n  Fix failures above before launching the full run.")
    print(sep)

    return all_sweep_ok and pipeline_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phase 15 smoke test: agent-first sequential sweep + MCP pipeline "
            "on 22-ticker SMOKE_UNIVERSE"
        )
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", "data"),
        help="Root data directory (default: $HIFI_DATA_DIR or 'data')",
    )
    parser.add_argument(
        "--condition",
        default="full",
        choices=["full", "parallel", "homogeneous", "no-memory"],
        help="Ablation condition (default: full)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Unload any remaining LM Studio models after the run",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip lms load commands (assume models already loaded in LM Studio)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    os.environ.setdefault("HIFI_DATA_DIR", data_dir)
    db_path = str(Path(data_dir) / "knowledge.lance")
    condition = args.condition

    from hifi.data.smoke_universe import SMOKE_UNIVERSE  # noqa: PLC0415

    smoke_tickers = [e["ticker"] for e in SMOKE_UNIVERSE]

    print(f"\nPhase 15 Smoke Test  |  condition={condition}  date={_SMOKE_DATE}")
    print(f"Universe : {len(smoke_tickers)} tickers (all 11 GICS sectors)")
    print(f"Data dir : {data_dir}")
    print(f"DB path  : {db_path}")

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------
    print("\n[PRE-FLIGHT]")
    lms_path = os.path.expanduser("~/.lmstudio/bin/lms")
    try:
        _lm_api("/v1/models")
        print("  LM Studio: OK (port 1234)")
    except Exception:
        print("  ERROR: LM Studio not responding at http://localhost:1234")
        print("  Start LM Studio first, then retry.")
        sys.exit(1)

    if not os.path.isfile(lms_path):
        print(f"  ERROR: lms CLI not found at {lms_path}")
        sys.exit(1)
    print(f"  lms CLI : {lms_path}")

    tech_alive = _port_is_listening(_FINETUNE_HEALTH_1235)
    print(f"  port 1235 (Technical): {'READY' if tech_alive else 'NOT RUNNING'}")
    if not tech_alive:
        print("    → Start with: make finetune-serve")

    # ------------------------------------------------------------------
    # Agent-first sweep
    # ------------------------------------------------------------------
    print(
        f"\nAgent-first sweep: {len(_AGENT_CONFIG)} agents × {len(smoke_tickers)} tickers"
        f" = {len(_AGENT_CONFIG) * len(smoke_tickers)} passes"
    )
    sweep_results = _run_sweep(
        smoke_tickers=smoke_tickers,
        condition=condition,
        data_dir=data_dir,
        db_path=db_path,
        skip_load=args.skip_load,
    )

    # ------------------------------------------------------------------
    # Aggregate + pipeline
    # ------------------------------------------------------------------
    signals, snapshot = _aggregate_and_pipeline(
        smoke_tickers=smoke_tickers,
        smoke_universe=SMOKE_UNIVERSE,
        condition=condition,
        data_dir=data_dir,
        db_path=db_path,
    )

    # ------------------------------------------------------------------
    # Summary + optional cleanup
    # ------------------------------------------------------------------
    all_ok = _print_summary(smoke_tickers, sweep_results, signals, snapshot, condition)

    if args.cleanup:
        print("\n[CLEANUP] Unloading any remaining LM Studio models...")
        from hifi.simulation.model_manager import get_loaded_ids, unload_model  # noqa: PLC0415

        for model_id in list(get_loaded_ids()):
            unload_model(model_id)
            time.sleep(2)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
