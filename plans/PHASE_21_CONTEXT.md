# Phase 21: Allocation Remediation and Endpoint Re-specification
## Context and Pre-Phase Decisions

**Gathered:** 2026-08-27, day 4 of Genesis II
**Status:** PLANNED
**Depends on:** Phase 19 (Genesis Hardening, COMPLETE) — this phase drains its
deferred queue; Phase 20 (Situated Agents, COMPLETE except T6.6)
**Origin:** Owner question 2026-08-26 ("there were not any movement, is that
correct? find the errors") → allocation audit of the 2026-08-24 genesis run,
plus the still-open Phase 19 deferred queue and Phase 20's T6.6 follow-up.

---

## 0. Why this phase is not "Phase 19 again"

Phase 19 hardened **execution** — idempotency, high-water mark, pre-submit
breaker re-check. It succeeded: Gate 1 passed, Genesis II ran clean, invariant 4
has held from bar one. Nothing in this phase reopens that work.

Phase 21 hardens **allocation and measurement** — the layer between a conviction
and a dollar, and the layer between a dollar and a claim. Phase 19 closed the
deferred queue's preconditions; this phase closes the queue itself.

---

## 1. The finding that motivates the phase (DJ-131)

### 1.1 What was observed

Genesis II first cycle, 2026-08-24. Arm A emitted 7 Buys spanning conviction
0.4217–0.6800 (a 1.61x spread) and allocated **$10,000.00 ± $0.08 to every one
of them** — a dollar spread of 1.0000x. $29,999.73, or 30.0% of the book, was
left idle. Against the same night's reference: arm B emitted 11 Buys and
achieved a dollar spread of 1.5776x with only its top two names pinned.

So on the same night, under the same code, one arm transmitted conviction to the
portfolio and the other did not.

### 1.2 The mechanism

`PortfolioPolicy.ceil_max_single = 0.10` (`src/hifi/portfolio/policy.py:83`) is
an absolute constant. The deployment ceiling is therefore `min(1, n x 0.10)`,
where `n` is the arm's Buy count:

```
 n Buys    3     5     7     8     9    10    11+
deployed  30%   50%   70%   80%   90%  100%  100%
```

Below ten Buys an arm cannot invest its capital regardless of its convictions,
and above the point where `1/n < 0.10` every name receives the identical cap, so
conviction ordering is erased entirely.

### 1.3 Why it is disqualifying rather than merely suboptimal

Both effects are monotone functions of the Buy count, and **the Buy count is the
treatment**. Arm A is the conservative parallel ensemble; being selective is what
its condition does; and the allocator fines it for that in proportion to how
selective it is.

A simulation with skill held *identical* across arms (every bought name returns
+1%) yields A +0.700% and B +1.000% purely from idle cash. A reviewer reading the
equity curves cannot separate that from a real difference in signal quality —
and neither can we.

### 1.4 The part that makes this a repeat, not a discovery

`policy.py` was written for DJ-122 to eliminate precisely this failure. Its own
docstring, lines 11–14, names it:

> **Silent cash drag.** With a narrow or sector-concentrated buy list the caps
> bind long before the capital is deployed. ... Measured before this change:
> 8 Buys -> 20% deployed, 16 -> 40%, 30 -> 70%.

and lines 26–32 state the governing principle:

> **Design note — why not simply loosen the constants.** Because the binding
> constraint must not be a function of the *treatment*.

Line 83 violates the design note written 70 lines above it. DJ-122 replaced the
sector-driven cash drag with a relative cap and then bounded that cap with a new
absolute constant, which reintroduced the same failure at a different `n`.

This is the DJ-124 pattern in a second habitat: **the reasoning was recorded and
the artifact stayed wired up.** In DJ-124 the rejection lived in a bitácora while
the LoRA stayed loaded; here the principle lives in the docstring while the
constant stays in the dataclass.

### 1.5 The knob was never interior to its bracket

Applying the `datasaurus` G3 gate to our own fitted quantity — printing
`concentration = 3.0` against the bracket it operates in:

```
   n   raw=3/n    ceil   max_single   WHICH BINDS   deployable
   7    0.4286    0.10       0.1000   CEILING            70.0%  <-- STRANDS
  10    0.3000    0.10       0.1000   CEILING           100.0%
  30    0.1000    0.10       0.1000   CEILING           100.0%
  31    0.0968    0.10       0.0968   relative          100.0%
  97    0.0309    0.10       0.0309   relative          100.0%
```

The relative cap becomes the binding constraint only when `3/n < 0.10`, i.e.
**n > 30**. For n = 1..97 the cap sits exactly on its ceiling for 30 of 97
values, and the observed live Buy counts (A=7, B=11, D=10 on 2026-08-24) all
fall inside that pinned region.

**DJ-122's relative mechanism has never once bound in live operation.** Every
allocation this project has made was governed by the absolute constant DJ-122
was written to remove. That is the ledger entry: a fitted knob pinned at its own
ceiling, unexamined, while the surrounding machinery was believed to be working.

---

## 2. Design decision — what replaces the ceiling

**Rejected: raise the constant.** 0.15 moves the cliff from n<10 to n<7. It does
not remove the treatment-dependence, it relocates it, and it would be the third
time this defect is fixed by choosing a different number.

**Rejected: renormalise weights to 1.0 unconditionally.** This forces full
deployment always, which silently overrides a legitimate strategy expression: an
ensemble that finds three attractive names arguably *should* hold cash.

**Adopted: separate the two concerns the cap currently conflates.**

The cap must express *"do not concentrate"*. It must not also express *"do not
invest"*. Today it does both and only the first was intended.

1. **Concentration** stays governed by the relative policy, with the ceiling
   made a function of book width so full deployment is always *reachable*:
   `ceil = max(ceil_max_single, target_deployment / n)`.
2. **Deployment level** becomes an explicit, recorded decision rather than an
   arithmetic residue. Whatever cash an arm holds must be traceable to a rule
   with a name, not to `min(1, 0.1n)`.
3. **A fixed-point loop** replaces the current one-pass cap chain, so the stock
   cap still holds after the sector cap and min-position steps have run (this is
   audit finding C17 — the same fix closes both).

The invariance test is stated in advance and is binding on the implementation:
**an arm's deployable fraction must not be a function of its Buy count.**

---

## 3. Endpoint re-specification (DJ-132 — protocol change, disclosed)

### 3.1 The problem

The live arms produce one equity observation per arm per day. After six months
that is roughly 125 correlated daily observations across four arms. A
returns-based comparison at that sample size cannot reach significance for any
plausible effect size, and the same power failure has already occurred once in
this project — Phase 15's p-values ignored overlapping 60-day windows, making
effective n far below the nominal 2352.

### 3.2 The evidence that choosing later is not neutral

5,000 simulated universes, four arms, 125 days, **no arm given any skill**:

```
Endpoint declared IN ADVANCE (1 test):       4.64% false discoveries
Endpoint chosen AFTER looking (12 tests):   30.10% false discoveries
```

(A first run including an overlapping-window Sharpe test reached 93.26%; that
test is itself invalid and is the Phase 15 error, so the clean 30.10% is the
figure of record.)

Nothing about the second procedure is dishonest. It simply makes `p < 0.05` mean
something other than 5%, after which a real finding is indistinguishable from the
simulation above.

### 3.3 The decision

The primary endpoint is re-specified **before** any further inspection of live
outcomes, from an economic comparison to a **diversity-mechanism test**:

- **Primary:** does ensemble topology (parallel / sequential / homogeneous)
  change measured diversity — herding coefficient κ, disagreement entropy — and
  does measured diversity predict decision quality (IC)?
- **Secondary, exploratory, labelled as such:** arm-level economic performance,
  always reported with the exposure column.

This is not a retreat. It is the only formulation Page's theorem actually speaks
to, it is measured per decision rather than per day (raising usable n by roughly
two orders of magnitude), and it is the quantity no comparable project measures
at all.

### 3.4 Timing

Re-specification must be recorded in OSF **amendment 002 before the next live
cycle**. Recorded now it is pre-registration; recorded after further looking it
is post-hoc selection, and the demonstration in §3.2 is what it would then be
worth.

---

## 4. Inherited queue, verified still open at 2026-08-27

Each confirmed by reading the code today, not from the audit report:

| ID | Status | Evidence |
|---|---|---|
| **C17** composer cap self-breach | OPEN | `portfolio_composer.py` steps 3→4→5 apply stock cap, then sector cap, then min position, with no re-check of the stock cap |
| **C15** VaR misalignment | OPEN | `risk_manager.compute_portfolio_var` truncates every series to `min_len` **positionally** (`r[-min_len:]`), never by date; a ticker absent from `returns` is `continue`d and its weight silently vanishes, so the weights no longer sum to 1 |
| **C18** `var_95_20d` misnamed | OPEN | the quantity is a one-day VaR by construction; the corrected renderer exists only in `engines/market_summary.py` (Phase 20 T6.4) |
| **D22** config not strict | OPEN | no `ConfigDict` / `extra=` anywhere in `src/hifi/config/loader.py`; a typo'd key is silently ignored — the DJ-124 silent-misconfiguration class |
| **D23** SafetyConfig dead | OPEN | referenced nowhere |
| **T6.6** dead regime paths | OPEN | `ensemble_runner._get_regime_label:661-663` reads `data/market/SPY.parquet` and `data/macro/macro.parquet`; **both verified absent** — the post-DJ-120 layout is `data/market/SPY/`. The function has returned `"neutral"` unconditionally since the migration, and a bare `except Exception: return "neutral"` hides it |

T6.6 is live-affecting every night and is grouped with the allocation work rather
than the housekeeping tier.

---

## 5. Scope discipline

**In scope:** DJ-131 allocation, DJ-132 endpoint, C15, C17, C18, D22, D23, T6.6.

**Explicitly out of scope**, to be taken in the order Phase 19 recorded:
fill/slippage reconciliation (E2) — larger than this phase and deserves its own;
B7–B10 leakage cluster; Phase 15 re-run (blocked on B7–B10 and on the DJ-128
redesign); intent-row resume semantics; M3 fail-open branch.

**Non-negotiable sequencing:** DJ-131 and DJ-132 land *before the next live
cycle*. Every additional night multiplies the contaminated record — arm A carries
a 30% cash drag and fully erased conviction for as long as this runs. Four days
is a footnote in the paper; sixty is a retraction.

---

## 6. Numbering correction adopted in this phase

`doc/HIFI_PROTOCOL_V1.md` reserved Phase 17 = "Ablation Studies + Capstone
Deliverable" and Phase 18 = "Publication + Open Source Release". Neither was ever
started or given a plan file, and Phases 19 and 20 were subsequently built past
them, so protocol numbering and `plans/` numbering have diverged.

**Resolution: 19 and 20 keep their numbers; 17 and 18 are formally superseded.**

19 and 20 are immutable in practice — git tag `phase19-genesis-hardening`,
commit messages, and `doc/bitacora/PHASE_19_GENESIS.md` all cite them. Renaming
completed work to satisfy a stale plan would corrupt the audit trail this project
exists to keep.

| Old | New | Meaning |
|---|---|---|
| — | **21** | Allocation remediation + endpoint re-specification (this phase) |
| protocol 17 | **22** | Paper I — ablation analysis + capstone deliverable |
| protocol 18 | **23** | Paper II — publication + open-source release |

`plans/STATUS.md` (last updated 2026-07-13, still listing Phase 16 as NEXT) is
refreshed as part of T6 to record the remap and the true state of 19/20.
