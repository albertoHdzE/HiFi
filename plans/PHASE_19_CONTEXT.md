# Phase 19: Genesis Hardening — Idempotent Execution + Live High-Water Mark
## Context and Pre-Phase Decisions

**Gathered:** 2026-08-23 (Sunday), pre-genesis emergency scope
**Status:** IN PROGRESS — must land before tonight's nightly run
**Depends on:** Phase 16 live infrastructure (running); adversarial audit of 2026-08-23
**Blocks:** Clean genesis reset (Monday) — tonight's run is the genesis rehearsal

---

## Why This Phase Exists Now

An independent adversarial audit of the execution path (2026-08-23, cross-verified by a
second reviewer who reproduced every finding against source) confirmed three defects on
the live order path that corrupt either the broker book or the experimental record:

| ID | Defect | Evidence | Consequence |
|---|---|---|---|
| A1 / DJ-129a | **No idempotency on order submission.** `client_order_id` is never set (`alpaca_executor.py:165` uses SDK default); the dedup marker `already_decided()` reads `decisions.jsonl`, appended only *after* all submits succeed (`run_phase16_live.py:1052→1072`). | The repo's own incident comment at `run_phase16_live.py:1066`: arm A filled 37 orders then raised; no decision record was written. A same-evening rerun recomputes an identical order list (positions unchanged — orders queue for next open) and resubmits everything. | Every target position bought twice at next open; broker and record diverge (standing invariant #4). |
| A2 / DJ-129b | **Drawdown circuit breaker is dead code.** Caller passes `"hwm_value": portfolio_value` (`run_phase16_live.py:371`), so `(hwm−pv)/hwm > limit` at `risk_manager.py:148` is identically zero. No HWM is persisted anywhere. | The −15% control described in `risk_manager.py` and in the OSF pre-registration has never been capable of firing. | A pre-registered risk control that never existed — scientific-integrity disclosure required regardless of the fix (OSF amendment 002). |
| E2 / DJ-129c | **Breaker checked once, hours before submission.** Check at `run_phase16_live.py:1006`; submission after multi-hour LLM passes at `:1052/:1063`. | Nothing re-verifies account health between check and submit. | Orders can be submitted into a state that turned halt-worthy mid-cycle. |

## Scope Decision (recorded deviation from the original proposal)

The original Phase 19 sketch included an intent row written to `decisions.jsonl`
before submission, with `already_decided()` failing closed on intent-without-completion.
**Replaced by broker-side idempotency**, because:

1. Intent rows *block legitimate recovery* unless full resume semantics are built
   (crash before any submit ⇒ rerun blocked ⇒ cycle lost — worse than the disease).
2. A deterministic `client_order_id` deduplicates at the broker for *any* crash point,
   including mid-loop, with zero resume logic and no new failure modes.
3. Installed alpaca-py exposes `client_order_id` on `MarketOrderRequest` but has **no**
   `get_order_by_client_order_id` method — the pre-check scans recent orders
   (`get_orders`, open + closed, limit 500) and matches locally. Two GETs per
   account per night.

Intent rows with proper resume semantics are deferred to the post-genesis queue.

## What This Phase Deliberately Does NOT Touch

Strict scope freeze so tonight's rehearsal isolates one behavioral novelty (DJ-126
notional sizing, already queued) plus these additive changes:

- C15 VaR date-alignment (wrong today; gates research validity and risk approval — first fix after genesis)
- C17 composer cap self-breach (+ permanent output-contract assertion)
- D22 config `extra="forbid"` / wiring SafetyConfig
- B7–B13 temporal-leakage cluster (research-side only; live path is leakage-immune by construction)
- Fail-open breaker exception path (M3) — partially mitigated tonight by DJ-129c's re-check

All deferred items keep their agreed priority order for post-genesis work.

## Success Criteria (falsifiable)

1. Kill a run mid-submission, rerun same evening: **zero duplicate orders at the broker**.
2. HWM persists across restarts, ratchets monotonically, seeds from historical
   `equity.jsonl` max, and reaches `compute_risk_report` (wire-in verified through
   `pipeline.py:168` → `dd_breached` → `approved_signals`).
3. Breakers re-evaluate immediately before the first submit of each submitting branch.
4. All existing tests stay green; new tests pin each property above.
5. Existing exact-signature call sites (`place_market_order(...)`) unchanged when the
   new parameters are absent — dry-run and legacy callers behave byte-identically.
