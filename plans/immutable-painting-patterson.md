# Surgical cleaning of HiFi

## Context

The repo reads as "dirty" and it is — but not where it looks. A full import-graph
and dead-definition sweep of `src/` found **exactly one** unreferenced function in
24,829 lines (`live_report.py:198 exposure_adjusted_returns`). The library layer is
sound. The mess is concentrated in four places, one of which is armed on the live
trading path:

1. **Production code lives in `scripts/`.** `run_phase16_live.py` (1,381 loc) and
   `run_phase15_orchestrator.py` (959 loc) hold the entire nightly cycle. The live
   script reaches the orchestrator through a `sys.path` insert
   (`scripts/run_phase16_live.py:349`), so the production ensemble runs out of a
   file named after a **retracted** evaluation phase. Three test files repeat the
   same `sys.path` trick to reach them.

2. **Dead branches on the live path.** `_setup_agent_model`
   (`run_phase15_orchestrator.py:406-418`) still silently reroutes the *fundamental*
   agent to a fine-tune server if anything answers on port 1236, overwriting
   `HIFI_FUNDAMENTAL_MODEL` and returning `True` so the run continues. That is
   DJ-124's exact failure mode — a rejected adapter live while the logs claim the
   base model — and it is still armed. A second branch (lines 321-366, port 1235)
   is unreachable because `_AGENT_CONFIG` now names a model for `technical`.

3. **The agent roster is defined six times**: `knowledge/agent_context.py:34`,
   `agents/ensemble_runner.py:124`, `analytics/decision_audit.py:33`,
   `run_phase15_orchestrator.py:116`, `scripts/verify_agent_repair.py:30`,
   `scripts/run_phase15_walkforward.py:94`. Adding a seventh agent makes five of
   them silently disagree.

4. **`scripts/` is 60 files / 16,523 loc**, of which ~26 are one-shot historical
   phase runs, 4 are unreferenced by anything, and one (`refresh_macro_store.py`) is
   superseded by `refresh_macro.py`. The Makefile carries ~60 targets, most of them
   Phase 3-13 baselines that will never run again.

Plus four orphan `src/` modules (1,170 loc) imported only by their own tests.

**Outcome:** one obvious entry point, production logic in `src/` under production
names, no armed dead branches, one definition per fact. Done **before** the
Genesis III reset so no refactor lands mid-experiment.

**Non-negotiable constraint:** this is behaviour-preserving. Genesis III must start
from code that produces the same decisions as the code verified on 2026-08-31.

---

## Step 1 — Disarm the live path (do this first, standalone commit)

Smallest change, highest safety value. In `scripts/run_phase15_orchestrator.py`:

- **Delete** the port-1236 fundamental fallback (lines ~406-418). Replace with a
  hard failure: if the named LM Studio model will not load, the pass must fail, not
  substitute a different model. Silent substitution is what DJ-124 was.
- **Delete** the unreachable `agent_type == "technical" and lms_model_id is None`
  branch (lines ~321-366) and the `_FINETUNE_*` constants it needs
  (`_TECHNICAL_FINETUNE_URL`, `_FUNDAMENTAL_FINETUNE_URL`, `_FINETUNE_HEALTH_1235/6`,
  `_FINETUNE_MODEL`). **Keep** the `HIFI_TECHNICAL_FINETUNE_URL` env-var *scrub* at
  lines 374-382 — that one is a guard, not a fallback.
- `tests/unit/test_served_model_selection.py` covers `_select_served_model` /
  `_probe_chat_model`, which stay (still used to pick the LM Studio model). Add one
  test asserting no code path can set `HIFI_FUNDAMENTAL_FINETUNE_URL`.

`_HOMOGENEOUS_AGENT_CONFIG` stays — the Phase 15 re-run still needs that condition.

## Step 2 — One source of truth for the agent roster

Promote `hifi.knowledge.agent_context.CANONICAL_ORDER` to `hifi/agents/roster.py`
exporting `CANONICAL_ORDER` (6, incl. contrarian) and `VOTING_AGENTS` (5, excl.
contrarian). Re-export from `agent_context` for compatibility. Replace the five
duplicate literals listed in Context §3 with imports. Add a test that asserts every
module reports the same roster.

## Step 3 — Extract the live path into `src/hifi/live/`

Pure code movement — copy bodies, do not rewrite logic. New package:

```
src/hifi/live/
  __init__.py     public surface: run_account_cycle, run_ensemble, guards
  models.py       <- orchestrator: _AGENT_CONFIG, _HOMOGENEOUS_AGENT_CONFIG,
                     _setup_agent_model, _select_served_model, _probe_chat_model,
                     _port_is_listening, _agent_config_for_condition
  paths.py        <- orchestrator: _run_id, _sidecar_path, _ensemble_path,
                     _portfolio_path, _resolve_tickers, _resolve_dates
                     + live: _account_dir, _decisions_log, _breaker_log, _hwm_path
  ensemble.py     <- orchestrator: run_agent_mode, run_aggregate_mode,
                     _fetch_edgar_context
                     + live: run_ensemble, load_ensemble_signals
  walkforward.py  <- orchestrator: run_pipeline_mode, run_status_mode, _load_ohlcv
  accounts.py     <- live: ACCOUNTS map, get_executor, _client_order_id,
                     _seed_hwm_from_history, update_hwm, already_decided, show_status
  market.py       <- live: update_data, _last_completed_session, _latest_prices
  guards.py       <- live: check_circuit_breakers, effective_halt_threshold,
                     _halt_before_submit, check_data_coverage, check_tradability,
                     _vanished_position_value, log_arm_invariance,
                     _log_circuit_breaker, _start_thread_watchdog
  strategies.py   <- live: run_control_strategy
  cycle.py        <- live: run_mcp_pipeline, execute_orders, log_episode,
                     run_account_cycle
```

Reuse, do not reimplement: `hifi.simulation.pipeline.run_pipeline` (already the
shared MCP chain — three call sites use it), `hifi.portfolio.PortfolioPolicy` for
constraints, `hifi.data.market_store` for OHLCV resolution.

Both scripts become thin CLIs that only parse args and dispatch:

- `scripts/run_phase16_live.py` → **`scripts/hifi_live.py`** (same flags:
  `--account --execute --dry-run --status --update-data --snapshot --smoke --date --force`)
- `scripts/run_phase15_orchestrator.py` → **`scripts/hifi_walkforward.py`** (same
  flags; the Phase 15 re-run and `scripts/watchdog_walkforward.sh` still work)

Update `scripts/nightly_live_execute.sh` (2 call sites), `scripts/watchdog_walkforward.sh`
(3 refs), and the Makefile `live-*` / `walkforward-*` targets. The three tests that
do `sys.path.insert` — `tests/unit/execution/test_phase16_live.py`,
`tests/unit/execution/test_phase19_idempotency.py`,
`tests/unit/test_served_model_selection.py` — become plain `from hifi.live import ...`.

## Step 4 — Orphan modules

**Delete** (no runtime path, no paper claim):
- `src/hifi/agents/graph.py` (367 loc) — an unused alternative to
  `run_sequential_ensemble`; the live path uses the runner. Remove the four graph
  tests in `tests/unit/test_sequential_ensemble.py:395-434`; keep the rest of that file.
- `src/hifi/mcp/indicators_server.py` (258 loc) + `tests/unit/test_indicators_server.py`
  + `scripts/setup_ta_venv.sh` + `venvs/ta/`. Nothing has ever launched it; it
  duplicates `market_store` resolution logic (per `tests/unit/test_market_store.py:121`).
  Remove `TestIndicatorsServerIntegration` from `test_market_store.py`.

**Wire in** (they back reproducibility claims in `doc/01_EVAL_HIFI_DAVID.md` §4.5 —
deleting them would remove a claim the paper needs):
- `src/hifi/data/versioning.py` — call `DatasetRegistry.register()` + `content_hash()`
  from `scripts/refresh_data.py` (Step 5) after each Parquet write, so every refresh
  records a hash. This closes the §4.5 gap rather than papering over it.
- `src/hifi/data/quality.py` — run `DataQualityChecker` inside the OHLCV refresh and
  log the completeness score; fail the refresh below the existing 98% threshold.
  Complements the 99% coverage gate already in `check_data_coverage`.

## Step 5 — Consolidate `scripts/`

**Delete:** `refresh_macro_store.py` (one-shot DJ-120 repair, superseded by the
merging `refresh_macro.py`), `notebooks/phase11_finetune_replication.py` (stray
export next to its `.ipynb`), `src/hifi/analytics/live_report.py:198`
`exposure_adjusted_returns` (the one dead function).

**Merge:** `refresh_fundamentals.py` + `refresh_macro.py` → **`scripts/refresh_data.py`**
with `--fundamentals --macro --ohlcv --all`. Preserve verbatim the `write_macro`
round-trip verification (the DJ-133c lesson: `df.to_parquet()` strips the schema
metadata `read_macro` requires) and the merge semantics (union of periods, fresh
wins on overlap).

**Move to `scripts/archive/`** the one-shot historical runs — every
`run_phase{3,4,5,6,7,8,9,10,11,12,13,14}_*.py`, `acquire_phase{1,10}_data.py`,
`acquire_macro_phase14.py`, `analyze_rank_sweep.py`, `calibrate_drift_monitors.py`,
`diag_sentiment_signals.py`, `diagnose_sentiment_sgr.py`,
`generate_compliance_examples.py` — with a `scripts/archive/README.md` mapping each
old path to its new one and naming the phase it belongs to.

**Stay in `scripts/`** (operational): `hifi_live.py`, `hifi_walkforward.py`,
`nightly_live_execute.sh`, `watchdog_walkforward.sh`, `run_phase15_smoke.py`,
`run_phase15_walkforward.py`, `compute_phase15_ic.py` (Phase 15 re-run is pending),
`refresh_data.py`, `verify_agent_repair.py`, `build_phase16_report_notebook.py`,
`run_personality_shadow.py`, `simulate_next_run.py`, `genesis2_reset.sh`,
`check_env.py`, `manage_namespaces.py`, `ingest_edgar_mda.py`, `ingest_episodes.py`,
`label_outcomes.py`, `run_label_outcomes.py`, `acquire_phase14_data.py`,
`record_sec_fixtures.py`, and the fine-tune setup scripts.

**Doc references:** 273 lines in `doc/` and `plans/` cite `scripts/…`. Rewrite paths
only in *active* documents — `README.md`, `plans/STATUS.md`, `plans/PHASE_2*`,
`doc/GENESIS_CHECKLIST.md`, `doc/PAPPERS/`. Leave `doc/bitacora/` and
`plans/PHASE_0*`–`PHASE_1*` **untouched**: they are the dated historical record and
rewriting them would falsify it. The archive README is what makes those paths
resolvable.

## Step 6 — Makefile and RUNBOOK

Collapse the Makefile to three sections — **Quality**, **Operations**, **Archive** —
with the historical `baseline-phase*` / `bootstrap*` / `eval-*` targets moved under a
single `archive-help` target that prints how to invoke them from `scripts/archive/`.

Live targets collapse to:

| target | meaning |
|---|---|
| `live-nightly` | the real thing (pre-flight, guards, orders) |
| `live-nightly DRY=1` | identical path, `--no-execute`, no orders |
| `live-plan` | prints the cycle, runs no agents (seconds) |
| `live-status` / `live-snapshot` / `live-update-data` | unchanged |

Delete `live-dry-run` and `live-execute` (both now reachable via `live-nightly`).

Write **`RUNBOOK.md`** at repo root: the nightly procedure, the pre-flight
dependencies (LM Studio :1234, LangFuse :3000), where decisions and sidecars land,
how to verify a run, how to reset for a Genesis, and a one-screen map of
`src/hifi/live/`. This is the file that stops the next agent getting lost.

---

## Verification

Behaviour preservation is the acceptance criterion; each rung is stronger than the last.

1. **`make test`** — 2,290 passing today. Expect ~2,270 after removing the graph and
   indicators-server tests. **Zero failures, zero new skips.**
2. **`make lint`** — clean on `src/` and `tests/` (pre-existing E501s in
   `scripts/archive/generate_reference_strategies.py` are acceptable).
3. **New characterization test** `tests/unit/live/test_extraction_is_behaviour_preserving.py`
   pinning the values that moved, against literals captured *before* Step 3:
   `_AGENT_CONFIG` (model ids, timeouts, the 8192 ctx for sentiment), the four path
   builders, `_client_order_id` format, `effective_halt_threshold` at several
   `(n_positions, exposure)` points, and the `ACCOUNTS` map.
4. **`make live-plan`** — capture the output before Step 3 and diff after. Must be
   byte-identical (no agents, no network, so any difference is a refactor bug).
5. **`bash scripts/nightly_live_execute.sh --no-execute`** — the full production path
   minus orders. Then `uv run python scripts/verify_agent_repair.py --date <run date>`
   must **PASS** the pre-declared thresholds (`MAX_MODAL_SHARE 0.95`,
   `MIN_UNIQUE_CONFIDENCES 3`, `MIN_RATIO_COVERAGE 0.90`) for both `parallel` and `full`.
6. **Deterministic-sidecar diff** — the strongest check. Compare new sidecars against
   the stored 2026-08-31 DJ-134 re-run for a 10-ticker sample. The **deterministic**
   payloads (ratios, growth metrics, indicators, risk metrics incl. beta) must be
   **byte-identical**; only LLM prose may differ. Any drift in a computed number means
   the extraction changed behaviour and the step is reverted.

Only after rung 6 passes: archive Genesis II, reset the four Alpaca accounts, run
`make live-nightly`.

## Git strategy

One commit per numbered step on `phase21/remediation-and-paper`, each independently
revertable, no co-author attribution. Steps 1 and 2 are safe to land immediately;
Step 3 is the one that must not be squashed with anything else.

## Explicitly out of scope

- Renaming the four experimental arms or any data layout under `data/`.
- Touching `doc/bitacora/` content.
- The 96 `except Exception` handlers in `src/` — all 96 log; **zero** are silent
  `pass`/`continue`. Not the emergency it looked like; leave for the T4 review.
- OSF amendment 002 (only Alberto can file it).
