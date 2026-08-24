# Phase 19 Plan: Genesis Hardening (DJ-129)

**Created:** 2026-08-23 ~17:50 CT · **Hard deadline:** GO/NO-GO 20:00 CT tonight
**Execution mode:** single session, scope-frozen, additive-only

See `PHASE_19_CONTEXT.md` for findings, evidence, and the recorded scope deviation
(broker-side idempotency instead of intent rows).

---

## Task List

### T1 — Executor: client order id support (`src/hifi/execution/alpaca_executor.py`)
- [ ] T1.1 `place_market_order(..., client_order_id: str | None = None)`: set on the
      `MarketOrderRequest` for both notional and share branches.
- [ ] T1.2 New `get_client_order_ids(limit=500) -> set[str]` (`@with_retry`): fetch
      open + closed recent orders, return the set of non-empty `client_order_id`s.
- [ ] T1.3 Echo the id in the returned `OrderResult.raw`.

### T2 — Orchestrator: deterministic ids + dedup gate (`scripts/run_phase16_live.py`)
- [ ] T2.1 `_client_order_id(account, date, ticker, side)` → `hifi{A|B|C|D}-{YYYY-MM-DD}-{buy|sell}-{TICKER}` (≤48 chars, Alpaca charset).
- [ ] T2.2 `execute_orders(..., account="A", date=None)`: when `date` given and not dry-run, prefetch ids once; **prefetch failure aborts the arm before any submit** (fail-closed). Per order: id already seen ⇒ record `status="skipped_duplicate"`, do NOT submit, do NOT count toward local spend; else submit with the id and add to the seen-set (intra-run dedup).
- [ ] T2.3 `run_control_strategy(..., account="C", date=None)`: same prefetch/dedup for its buy path.
- [ ] T2.4 All new parameters keyword-with-default; when absent (dry-run, tests), call signatures are byte-identical to today.

### T3 — Orchestrator: live high-water mark (A2)
- [ ] T3.1 `_hwm_path(account)` → `<account_dir>/hwm.json`.
- [ ] T3.2 `_seed_hwm_from_history(account)`: max equity over `equity.jsonl` rows (0.0 if none).
- [ ] T3.3 `update_hwm(account, current_equity) -> float`: `max(stored, seed, current)`; atomic tmp+rename persist (pattern: `performance_store.py:68–70`). Returns the value.
- [ ] T3.4 `run_account_cycle`: after the main breaker check, `hwm_value = update_hwm(...)` for live runs (persistence failure raises ⇒ arm skipped — fail-closed); pass into `run_mcp_pipeline(..., hwm_value=...)`; pipeline falls back to current behavior (`portfolio_value`) when None.
- [ ] T3.5 HWM updates are skipped on dry-run (no state mutation from inspection runs).

### T4 — Orchestrator: pre-submit breaker re-check (E2)
- [ ] T4.1 `_halt_before_submit(account, executor, is_dry, date) -> bool`: re-run
      `check_circuit_breakers`; on trip: log PRE-SUBMIT HALT, `record_account(...)`
      (observation preserved per DJ-119), disconnect, True.
- [ ] T4.2 Call before each submitting branch: control, riskbudget→execute, ensemble→execute.

### T5 — Tests (`tests/unit/execution/test_phase19_idempotency.py`)
- [ ] T5.1 Request carries the deterministic id (capture `submit_order` arg).
- [ ] T5.2 Crash-rerun replay: pre-existing id at broker ⇒ no second submit, `skipped_duplicate` recorded.
- [ ] T5.3 Prefetch failure ⇒ exception before any submit (fail-closed).
- [ ] T5.4 `get_client_order_ids` merges open + closed.
- [ ] T5.5 HWM: ratchet up only; persists across "restarts"; seeds from `equity.jsonl`; atomic file shape.
- [ ] T5.6 Pre-submit halt blocks submission and still records the account.
- [ ] T5.7 Legacy compatibility: without `date`, `execute_orders`/`run_control_strategy` issue exactly today's calls (existing suite stays green).

### T6 — Verification & landing
- [ ] T6.1 `pytest tests/unit/execution/ -q` green.
- [ ] T6.2 `ruff check` + `mypy` clean on touched files.
- [ ] T6.3 Commit (message cites DJ-129a/b/c). Tag `phase19-genesis-hardening`.
- [ ] T6.4 Bitácora entry drafted AFTER tonight's run (separate step, per protocol).

---

## Post-Run Verification (Monday pre-open, before genesis reset)

1. Broker positions ↔ `portfolio_history.json` ↔ `equity.jsonl` reconcile (invariant #4).
2. Alpaca order history: zero duplicate `client_order_id`s; every id matches `hifi{acct}-{date}-...`.
3. `hwm.json` exists per account, ≥ max historical equity.
4. Any `skipped_duplicate` row in episodes is investigated as a near-miss event.

## Deferred Queue (agreed order, post-genesis)

C15 VaR date-alignment + weight renormalization → C17 composer fixed-point loop +
output-cap assertion → D22 config `extra="forbid"` + SafetyConfig wiring → tighten
`simulate_next_run.py` drift tolerance → B7–B10 leakage cluster → Phase 15 re-run →
intent-row resume semantics (deferred from this phase) → M3 fail-open branch.
