# PHASE_19_GENESIS — Genesis II executed and verified

**Date:** 2026-08-25 (entry written day-of; events span 2026-08-23 → 2026-08-25)
**Status:** COMPLETE — Genesis II is live; all four arms trading from a clean baseline
**Scope:** DJ-129a–c hardening, Gate R reset, first two cycles, DJ-130 activation

---

## 1. What this genesis is

Second restart of the live ablation, governed by `doc/GENESIS_CHECKLIST.md`
§Genesis II. Unlike the first restart (A/B/D only, C preserved), **all four arms
reset** — including C — because DJ-129a introduced a permanent record-schema
break (`client_order_id` column exists only from 2026-08-21 episodes onward)
and DJ-126/127 changed order semantics. A uniform baseline on hardened code was
judged worth more than C's continuity across a format break.

## 2. Hardening landed before the clock restarted (DJ-129)

Commit `072c022`, tag `phase19-genesis-hardening`, PR #3 → main:

| ID | Defect closed | Mechanism |
|---|---|---|
| DJ-129a | crash-rerun double-fill window (fired 2026-08-17, arm A, 37 fills) | deterministic `client_order_id = hifi{acct}-{date}-{side}-{TICKER}`; broker-side dedup via recent-order scan; prefetch failure aborts arm pre-submit (fail-closed) |
| DJ-129b | −15% drawdown breaker was dead code (`hwm == pv`, identically zero) | persisted ratcheting HWM seeded from `equity.jsonl` history, wired through pipeline to `dd_breached` |
| DJ-129c | breaker checked hours before submission | `_halt_before_submit` re-checks on all three submitting branches |

Design deviation recorded in `plans/PHASE_19_CONTEXT.md`: broker-side
idempotency replaced the intent-row sketch — no resume semantics needed, no
recovery-blocking failure mode.

## 3. Gate 1 — shakedown (Sunday 2026-08-23 run) — PASSED

Evidence retained in `data/live/_genesis1_archive/nightly_20260823.log`:

- rc=0, **zero ERROR lines**, all four arms completed date=2026-08-21
- 23 orders submitted, every one carrying a well-formed id; broker-side scan:
  **0 duplicated ids, 0 malformed** (read-only GET, all accounts)
- `hwm.json` created ×4; account D demonstrated the ratchet holding a real
  historical peak ($100,612 vs equity $98,820) — impossible under old code
- Breakers: flags only, no halts; pre-submit recheck ran per arm

Monday-open fills confirmed: B 8/8 filled, A 6F+2P→filled intraday, D 4F+3P→filled.
Step 0 re-verified at 14:00-rule: **all 23 FILLED**, positions reconcile per arm.

## 4. Gate R — reset timeline (2026-08-24)

| Step | When | Result |
|---|---|---|
| Archive (`scripts/genesis2_reset.sh --archive`) | 08-24 08:06 | `data/live/_genesis1_archive/{A,B,C,D}` + shakedown log; originals intact; write-guarded against overwrite |
| Alpaca dashboard resets ×4 (manual, Alberto) | 08-24 morning | new API keys issued; verified $100,000.00 / 0 positions per arm |
| Clear state files (`--clear`) | after resets | `hwm.json` deleted (critical — stale peak would misalign breaker), logs recreated empty |

Script committed at `266bf4f`; guard behavior tested (refuses overwrite rc=65,
refuses clear-without-archive rc=66).

## 5. First genesis run (night of 2026-08-24, 16:17→23:51, rc=0, zero errors)

- Coverage 97/97, bars through the day's session; tradability clean
- **`High-water mark: $100000.00` for all four arms** — fresh-seeding proof
- Episodes logged ×4 dated 2026-08-24; every order carried a well-formed id
- Decision distributions (the honest ones): A 7B/84H/6S → 7 orders; B 11B/80H/6S
  → 11; C control rebuilt its full 97-name book; D calm_exposure 10B/12H/75S → 10
- Tuesday-open fills: **A 7, B 11, C 97, D 10 positions — broker ↔ record match
  exactly, arm by arm** (invariant 4 holds from bar one)
- Zero duplicate ids at the broker

Note for future readers: low order counts are recommendation counts, not drops
(orders == Buy signals 1:1; risk manager blocked nothing). LLM arms are
constitutionally Hold-heavy under 5-agent consensus + contrarian discount;
that conservatism is now measurable on a clean base.

## 6. DJ-130 disclosure — protocol change active from the 2026-08-25 cycle

Same session as this entry (commit `e35bee1`):

1. **Portfolio context injection** — eligible agents (fundamental/risk/macro/
   sentiment/contrarian) receive standing-situation block (equity/cash/exposure,
   top holdings, days since genesis, DEPLOYMENT-vs-STEADY phase, arm return vs C).
   Technical excluded (price-only purity). Injection activates only when the
   live orchestrator tags an account — evaluation replays stay byte-identical.
   Effect: A/B/D signal generation differs from pre-08-25 nights by design.
2. **Personality postures (SHADOW ONLY)** — baseline/aggressive/conservative/
   careful replayed nightly over stored votes into `shadow_personality.jsonl`.
   Validated on 2026-08-24 ensembles: 194 replays, **0 baseline mismatches**
   (deterministic arithmetic reproduces the stored collective exactly);
   divergences observed: aggressive +14/+15 entries (A/B), careful degrades 21
   of B's marginal holds. No personality trades until weeks of shadow evidence
   justify promotion via OSF amendment.

## 7. OSF amendment 002 items accrued by this phase

(In addition to the original Gate 3 list.) DJ-129 fixes and the client_order_id
format boundary; Genesis-II void period 2026-08-14 → 2026-08-24 for ALL arms
(supersedes "C unaffected"); production-line consolidation to `main`;
DJ-130 context injection as a second protocol boundary dated 2026-08-25.

## 8. Deferred queue (agreed order, unchanged)

C15 VaR date-alignment → C17 composer fixed-point loop + output-cap assertion →
D22 config strictness + SafetyConfig wiring → simulate_next_run tolerance →
B7–B10 leakage cluster → Phase 15 re-run → intent-row resume semantics → M3
fail-open branch → engine-computed context summaries (in flight,
`feature/context-engine-summaries`).

## 9. Standing invariants honored

#4 held from bar one (broker↔record exact match, both cycles); #6 exercised
nightly (tradability probe); new invariant #7 in force since first post-fix
order (no submit without a deterministic id).
