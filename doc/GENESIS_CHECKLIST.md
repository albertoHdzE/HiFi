# Phase 16 Genesis — restart checklist

**Purpose.** Between 2026-08-14 and 2026-08-18 five defects were found, each
hidden by the one above it. The live record for arms A, B and D is void as a
result. This checklist governs the restart: what must be true before the clock
starts again, and what must be recorded so the restart itself is auditable.

**Status:** prepared 2026-08-18. Gate 1 not yet passed.

---

## Why a restart rather than a continuation

| DJ | defect | effect on the record |
|---|---|---|
| 120 | three data paths starved 3/5 agents on 83/98 tickers | signals were about missing data, not securities |
| 121 | rebalance band ≥ max position size | A/B/D frozen at one holding for a month |
| 122 | position limits absolute, unrelated to universe size | capital stranded; the cap bound harder on diversified arms |
| 123 | EQR removed from the broker mid-run | A aborted after 37/39 fills with no decision record; D false-halted |
| 124 | technical agent ran a **rejected** LoRA | constant `Buy @ 0.70` on 98/98; ensemble entropy → 0 |

Arm **C** is unaffected throughout — it bypasses the pipeline — and its history
stays continuous and valid. Arms **A, B, D** restart.

---

## Gate 1 — shakedown run (must pass before reset)

Run one full cycle on the **current** accounts, after the close, and confirm
every line below. The point is to prove the stack on live infrastructure while
the accounts are still expendable.

    make live-nightly

- [ ] `Data coverage pre-flight: 97/97 tickers resolved (100.0%)`
- [ ] `Tradability pre-flight: 97/97 tickers tradable` — no untradable names
- [ ] Four `Allocation policy(n=…)` lines, one per pipeline arm
- [ ] **Zero** `ERROR` lines; no `cycle FAILED`
- [ ] Every arm logs an episode: A, B, C **and D** all get a 08-18 record
- [ ] `decision_audit.provenance_matrix` → tool-failure **0.000** for all six agents
- [ ] `decision_audit.degenerate_agents` → **empty** (technical no longer constant)
- [ ] Technical `model_id = qwen2.5-coder-32b-instruct-mlx` in the sidecars
- [ ] Orders placed on A, B, D; rejections (if any) recorded, not raised
- [ ] Next morning: fills confirm, positions >> 1 per arm

Verify with:

    uv run python scripts/simulate_next_run.py      # before: predicted orders
    uv run python -c "import sys;sys.path.insert(0,'src');\
    from hifi.analytics import decision_audit as da;\
    print(da.degenerate_agents('A')); print(da.degenerate_agents('B'))"

**If any line fails, fix and repeat. Do not reset on an unproven stack.**

---

## Gate 2 — reset to genesis

Only after Gate 1 passes cleanly.

- [ ] Reset A, B, D to $100,000 (Alpaca paper dashboard → *Reset Account*).
      **C is NOT reset** — its history is valid and continuous.
- [ ] Record the genesis date and each arm's opening equity in
      `doc/bitacora/` and in the OSF amendment.
- [ ] Archive the void period: move `data/live/{A,B,D}/` to
      `data/live/_pre_genesis/` rather than deleting it. It is evidence for the
      amendment and for the defect write-ups, not garbage.
- [ ] Re-create empty `decisions.jsonl` / `equity.jsonl` for A, B, D.
- [ ] Confirm `portfolio_history.json` starts fresh (Alpaca regenerates it).

---

## Gate 3 — pre-registration amendment (only Alberto can file)

**OSF amendment 002** must cover, in one document:

- [ ] The five defects above, with dates found and dates fixed
- [ ] **Retraction of the Phase 15 headline result.** IC +0.0642 (p=0.0019)
      decomposes to −0.1377 on the 16 data-bearing tickers and +0.0669 on the
      83 dark ones — the positive result lived entirely where the agents were
      blind. Phase 15 must be re-run on repaired data before any Page-theorem
      claim.
- [ ] The independent statistical problem: p-values treated 2,352 (ticker,
      date) pairs as independent when they are 25 dates × 99 tickers with
      overlapping 60-day forward windows. Effective n is far smaller.
- [ ] The universe change: 98 → 97 names (EQR retired, DJ-123)
- [ ] The position-limit change: absolute caps → universe-derived policy (DJ-122)
- [ ] The model change: technical agent off `technical_v2`, onto base
      qwen2.5-coder (DJ-124)
- [ ] Genesis date and the void period for A/B/D
- [ ] Still-unfiled **amendment 001** (circuit breaker, DJ-119) — fold in or file
      alongside

---

## Gate 4 — re-run Phase 15 on repaired data

Independent of the live restart, and required before any diversity claim.

- [ ] Re-run the walk-forward with the repaired data layer and base technical model
- [ ] Recompute IC with an honest effective sample size (block bootstrap or
      date-level aggregation, not pair-level p-values)
- [ ] Re-measure herding/entropy now that no agent is constant
- [ ] Compare against the retracted numbers explicitly — the delta is itself a
      result about how much of the original finding was artefact

---

## Standing invariants

Learned the hard way; each corresponds to a defect above.

1. **Never read a decision metric without its provenance.** Section 0 of the
   report notebook is a gate, not decoration. A dead tool call and a bearish
   opinion look identical downstream.
2. **A constant agent invalidates every diversity number.** Run
   `degenerate_agents` before quoting herding or entropy.
3. **Express portfolio constraints relative to the universe**, never as bare
   percentages, and check they cannot bind differently across arms — book width
   is downstream of the treatment.
4. **The broker and the record must never diverge.** An equity curve that moves
   with no decision attached is worse than a clean failure.
5. **A component rejected in research must be verified absent from production.**
   DJ-124's rejection lived in a bitácora for two months while the artifact
   stayed wired into the serving stack.
6. **The universe is perishable.** Names delist; `check_tradability` reports the
   drift nightly.

---

# Genesis II — production baseline (2026-08-24)

**Status:** Gate 1 PASSED 2026-08-23 (evidence below). Gate R (reset) pending —
scheduled for tonight, after Monday-open fills reach terminal state.
**Scope:** ALL FOUR arms restart, including C. Codebase consolidates to `main`
as the production line (branch `phase14/heterogeneous-ensemble` merged and retired).

## Why a second genesis

The first checklist restarted A/B/D after five defects voided their record while
deliberately keeping C continuous. Three things have changed since:

| Change | Effect on the record |
|---|---|
| **DJ-129a**: orders now carry deterministic `client_order_id`s | Every episode before 2026-08-21 has no id column — a permanent structural break in the record schema. Mixing both formats in one series poisons automated analysis. |
| **DJ-126/127**: notional buy sizing, explicit Sell exits | Order semantics differ from everything before 2026-08-18. |
| **DJ-129b/c**: live high-water mark, pre-submit breaker recheck | Risk controls that never fired before can fire now; a drawdown halt against a *seeded historical* peak is not comparable to one against a fresh $100k base. |

A clean, uniformly-formatted baseline on hardened code is worth more than C's
continuity across a format break. **C restarts too** — its prior validity was
relative to defects that never touched it; the schema break touches every arm.

## Gate 1 — shakedown: PASSED (2026-08-23 run, evidence retained in log)

The Sunday 2026-08-23 run (`data/live/logs/nightly_20260823.log`) was the
shakedown under Phase 19 hardening. Verified:

- [x] rc=0, **zero ERROR/CRITICAL lines**, all four arms completed date=08-21
- [x] 23 orders submitted, every one carrying a well-formed deterministic id
      (`hifi{acct}-2026-08-21-{side}-{TICKER}`)
- [x] Broker-side scan: **0 duplicated ids, 0 malformed** (read-only GET, all accounts)
- [x] `hwm.json` created for all arms; account D demonstrates the ratchet
      (HWM $100,612 vs equity $98,820 — a historical peak held above today)
- [x] Breakers: flags only (C's SRE −10.7%, UPS −12.8% correctly flagged at
      ~0.09–0.11% book impact); no halts; pre-submit recheck ran per arm
- [x] Fills confirmed at Monday open: B 8/8 filled, A 6F+2P, D 4F+3P (partials resolve intraday)

## Gate R — reset to genesis (tonight)

### Step 0 — confirm fills are terminal (~14:00 CT, after the open settles)

    # All PARTIALLY_FILLED from last night must now read FILLED (or cancelled):
    uv run python scripts/hifi_live.py --status

- [ ] Zero `PARTIALLY_FILLED` orders remain from the 08-21 cycle
- [ ] Broker positions ↔ `portfolio_history.json` ↔ last `decisions.jsonl` row
      reconcile per arm (invariant 4)

### Step 1 — archive, never delete

The pre-genesis record is **evidence, not garbage**: it documents the defect
period, the DJ-124 contamination, the EQR delisting handling, and the first
night of DJ-129 operation.

    mkdir -p data/live/_genesis1_archive
    for a in A B C D; do
      cp -r data/live/$a data/live/_genesis1_archive/$a
    done
    cp data/live/logs/nightly_20260823.log data/live/_genesis1_archive/

- [ ] `_genesis1_archive/` contains A, B, C, D and the shakedown log
- [ ] Note the archive path in the bitácora entry

### Step 2 — reset the paper accounts (Alpaca dashboard, manual)

For **each** of A, B, C, D: dashboard → *Reset Account* → back to $100,000.

- [ ] Four resets confirmed (`--status` shows $100,000.00 equity, no positions)

### Step 3 — clear local state files (the step that bites if skipped)

    for a in A B C D; do
      rm data/live/$a/hwm.json                     # CRITICAL — see note
      rm -f data/live/$a/decisions.jsonl \
            data/live/$a/equity.jsonl \
            data/live/$a/portfolio_history.json \
            data/live/$a/circuit_breakers.jsonl
      mkdir -p data/live/$a && touch data/live/$a/decisions.jsonl \
                                  data/live/$a/equity.jsonl
    done

> **Why deleting `hwm.json` is CRITICAL:** the ratchet keeps the highest equity
> ever seen. If it survives the reset, arm A starts life with HWM $101,858 on a
> $100,000 book and its drawdown breaker measures losses against a peak that no
> longer belongs to the account. Fresh account ⇒ fresh mark; the first post-
> genesis run re-seeds it from the fresh $100k.

- [ ] No `hwm.json`, no stale `portfolio_history.json` anywhere under `data/live/{A,B,C,D}`
- [ ] Empty `decisions.jsonl` + `equity.jsonl` recreated per arm

### Step 4 — first genesis run (tonight, post-close)

    make live-nightly          # LM Studio up first; Docker/Langfuse self-start

Verify in `data/live/logs/nightly_<today>.log`:

- [ ] `Decision date -> <today>` (Monday session)
- [ ] `High-water mark: $100000.00` (or within cents) for **each** arm — proves fresh seeding
- [ ] Data coverage 97/97; tradability 97/97; zero ERROR lines
- [ ] Episodes logged for A, B, C, D dated today; every order carries
      `client_order_id` starting `hifi<acct>-<today>-`
- [ ] Tuesday morning: fills terminal, positions >> 1 per arm, zero duplicate ids

### Step 5 — record keeping (only Alberto can file)

Bitácora entry `PHASE_19_GENESIS.md` must capture: Gate 1 evidence summary,
archive path, exact reset timestamps, opening equities, first-run verification
lines. **OSF amendment 002** (per original Gate 3 list above) additionally
discloses:

- [ ] DJ-129a–c fixes and the client_order_id record-format break as a
      protocol-change boundary (episodes before 2026-08-21 lack ids)
- [ ] Genesis-II date, void period = 2026-08-14 → 2026-08-24 for ALL arms,
      superseding the earlier "C unaffected" carve-out
- [ ] Production-line consolidation: repository reduced to `main`

## Standing invariants (amended)

7. **An order without a deterministic id is a double-fill waiting for a crash.**
   Every submit carries `client_order_id`; a rerun deduplicates at the broker,
   never in our discipline.
