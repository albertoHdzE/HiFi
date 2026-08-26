# Phase 20 Plan: Situated Agents (DJ-130)

**Created:** 2026-08-25 · **Mode:** full implementation, same session
See `PHASE_20_CONTEXT.md` for rationale and guardrails.

## Tasks

### T1 — Book-state artifact (`src/hifi/agents/context.py`)
- [ ] T1.1 `write_book_state(executor, account, data_dir)` → `data/live/<acct>/book_state.json`
      (equity, cash, invested %, positions with weight/pnl, n, updated-ts).
- [ ] T1.2 Genesis marker: `data/live/genesis_date.txt`; days-since computed by builder.
- [ ] T1.3 `build_portfolio_context(book) -> str`: rendered block with phase flag
      (`DEPLOYMENT` if exposure<25% and age≤10 sessions, else `STEADY`) + arm
      self-assessment vs control C (None-safe early days).
- [ ] T1.4 `CONTEXT_ELIGIBLE_AGENTS = {fundamental, risk, macro, sentiment, contrarian}`
      — technical excluded (price-only purity).

### T2 — Wiring
- [ ] T2.1 `run_phase16_live.run_account_cycle`: write book state each live cycle
      (after HWM, before signal generation).
- [ ] T2.2 `run_phase15_orchestrator.run_agent_mode`: prepend rendered context to
      `extra_memory_prefix` for eligible agents (existing channel — zero agent-code change).

### T3 — Personalities, shadow-only (`src/hifi/collective/personality.py`)
- [ ] T3.1 `PersonalityProfile` params + BASELINE / AGGRESSIVE / CONSERVATIVE / CAREFUL.
- [ ] T3.2 `posture_vote(votes, profile) -> dict` — deterministic arithmetic over
      (decision, confidence) vectors; tie→Hold convention matches voting.py;
      CONSERVATIVE/CAREFUL entry-margin rule.
- [ ] T3.3 Unit tests with hand-computed cases incl. boundary margins and all-Hold.

### T4 — Nightly shadow replay (`scripts/run_personality_shadow.py`)
- [ ] T4.1 Read stored ensemble JSONs (agent_decisions + confidences) per ticker/date.
- [ ] T4.2 Emit per-profile decisions to `data/live/<acct>/shadow_personality.jsonl`;
      baseline row cross-checked against stored collective decision (logged mismatch count).
- [ ] T4.3 LLM arms only (A/B); D excluded (single deterministic strategy, no vote vector).

### T5 — Verification & landing
- [ ] T5.1 New unit suites green; existing execution suite untouched-green.
- [ ] T5.2 ruff clean; commit citing DJ-130; push main.
- [ ] T5.3 Bitácora note folded into tonight's PHASE_19_GENESIS entry (context active
      from tonight's cycle; personalities shadow-only).

## Acceptance
Tonight's log shows book-state written per arm; tomorrow's signals carry the
context block in sidecars (spot-check one); shadow jsonl populates after next
aggregate with baseline≈stored decisions.

## T6 — Engine-computed market summaries (branch feature/context-engine-summaries)

- [x] T6.1 `engines/market_summary.py`: regime snapshot (live-wired DJ-089b),
      relative strength vs sector median, book VaR95 historical simulation.
- [x] T6.2 Point-in-time discipline: every series filtered `index <= as_of`
      before computing; test pins invisibility of post-as_of bars.
- [x] T6.3 C15-corrected VaR logic here: date-intersection alignment +
      weight renormalization (positional splice and silent weight-drop both
      test-pinned as forbidden behaviours). risk_manager's own flawed variant
      remains queued separately.
- [x] T6.4 Renderer `build_market_block` in agents/context.py; horizon stated
      in words ("one-day figure, horizon is NOT 60 days") per C18 lesson.
- [x] T6.5 SPY nightly benchmark refresh in update_data (regime input).
- [ ] T6.6 FOLLOW-UP (discovered): `ensemble_runner._get_regime_label` reads
      dead paths (`market/SPY.parquet`, `macro/macro.parquet`) → live regime
      has been silently "neutral" since DJ-120 migration; also debate prompts
      may consume it. Fix separately on main.
- [x] T6.7 Tests: 11 new, incl. no-lookahead edge, misaligned-delisting case,
      quantile-arithmetic fixture derived by hand ((77/100)^(1/5)−1).
