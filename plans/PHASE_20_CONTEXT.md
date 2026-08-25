# Phase 20: Situated Agents — Portfolio Context + Strategy Personalities
## Context and Pre-Phase Decisions

**Gathered:** 2026-08-25, day 2 of Genesis II
**Status:** IN PROGRESS
**Depends on:** Genesis II baseline (running); DJ-129 hardened execution
**Origin:** Owner observation, 2026-08-25: arms answered "is this name attractive?"
without knowing the book was empty — 80 Holds from all-cash is an unanswered
entry question, not caution. Remedy adopted: situate the agents; do NOT override
the first move manually (manual orders break invariant 4 by definition).

---

## What this phase adds

### 1. Portfolio context injection (DJ-130 — protocol change, disclosed)
Agents currently reason decontextualized. From tonight, eligible agents receive a
rendered book-state block alongside their tool/RAG context:

- equity, cash, invested %, n positions
- top holdings with weight and unrealized P&L
- days since genesis, phase flag: `DEPLOYMENT` (exposure <25% and age ≤10 sessions)
  vs `STEADY`
- arm self-assessment: return since genesis vs control C (when computable)

**Eligibility:** fundamental, risk, macro, sentiment, contrarian.
**Excluded: technical** — its schema promises price-derived information only;
injecting book state would deepen the context-contamination class found in the
2026-08-23 audit (shared GraphRAG into technical).

**Injection point:** central, in `ensemble_runner` — appended ahead of each
eligible agent's `retrieved_context`. One edit point, uniform rendering, no
per-agent graph surgery. A native per-agent context field is the clean long-term
shape and is deferred deliberately.

### 2. Strategy personalities — SHADOW ONLY (no new accounts, no live effect)
Same-night agent votes are replayed deterministically through three postures at
the collective-decision layer (pure arithmetic on the recorded vote/confidence
vectors — zero extra LLM cost, fully reproducible from stored ensembles):

| Profile | Entry posture | Exit posture | Intent |
|---|---|---|---|
| AGGRESSIVE | half of Hold mass leans to entry when any Buy conviction exists | standard | owner pressing for deployment |
| CONSERVATIVE | Buy must exceed rival options by ≥25% margin, else Hold | standard | owner demanding justification |
| CAREFUL | Conservative's entry rule **plus** a quarter of Hold mass leans to Sell | amplified exits | drawdown-fearing owner |

Rationale for decision-layer (not prompt-level) personalities: deterministic,
unit-testable, replayable over the whole archived history from night one, and it
does not touch signal generation — so the running ablation is untouched.
Prompt-level personalities (changing how agents *reason*) are a possible later
phase and would require fresh arms + pre-registration.

## Non-goals / deferred
- New Alpaca accounts for personality arms (only after weeks of shadow evidence,
  promoted via OSF amendment)
- Per-agent memory decay (OQ-AG02) — untouched
- Native per-agent context fields in agent graphs — deferred cleanup
