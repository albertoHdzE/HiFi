# Phase 16: Live Paper Trading — IBKR
## Context and Pre-Phase Decisions

**Gathered:** 2026-06-16
**Status:** NOT STARTED — awaits Phase 14 + Phase 15 completion
**Depends on:** Phase 14 (IBKR infrastructure, MCP tools), Phase 15 (validated ensemble)
**External dependency:** IBKR paper trading account credentials (user-provided)

---

## Why This Phase Exists

Phase 15 answered "what would the system have done?" Phase 16 answers "what does
the system do?" These are different questions. Historical simulation has no execution
risk, no market impact, no slippage, and no psychological challenge. Live paper trading
has all of these — even in paper mode, the discipline of committing to a position
before knowing the outcome is fundamentally different from backtesting.

More importantly: Phase 16 is where the episodic memory pipeline (DJ-092) matures.
Each day's live decision is logged, outcome-labeled after 60 days, and fed back into
the episodic RAG store. By week 12, each agent has ~60 outcome-labeled live episodes
to retrieve from. This is the first genuinely experience-based memory the system builds.

The WQU capstone requires paper trading (non-negotiable protocol requirement). This
phase fulfills that requirement with scientific rigor.

---

## DJ-098: IBKR Integration Design

**Problem:** How to connect HiFi's ensemble pipeline to IBKR paper trading execution.

**Decision:** `ib_insync` + custom async event loop. NOT Backtrader.

Rationale (DJ-094 established, restated here for Phase 16 context):
- LLM ensemble generates signals at 2-10 minutes per ticker × 100 tickers = overnight
  batch. Backtrader's event loop assumes millisecond-to-second signal generation.
- `ib_insync` wraps IBKR's asynchronous TWS API. A custom coroutine pipeline is
  natural: "await generate_signals() → await compose_portfolio() → await check_risk()
  → await place_orders() → await log_to_langfuse()"
- Same pipeline as Phase 15 simulation; only difference is the final step (real IBKR
  call vs. simulated fill).

**IBKR setup:**
- IB TWS or IB Gateway running locally on user's machine
- Paper trading port: 7497 (TWS) or 4002 (Gateway)
- Credentials stored in `.env` at repo root — NEVER committed to git
- `ib_insync` connects via localhost; no cloud exposure

**Order types:** MARKET only in Phase 16. Limit/stop optimization is Phase 17+ research.
Rationale: Phase 16 is about validating the signal quality, not execution quality.
Market orders minimize execution complexity; slippage is recorded but not optimized.

---

## DJ-099: Live Trading Operational Design

**Signal generation frequency:** Daily batch, running overnight (22:00-06:00 local time).
All 100 tickers processed sequentially (hardware constraint). Orders placed at next
market open (09:30 EST). No intraday rebalancing.

**Rebalancing logic (from capital allocator, DJ-091):**
- Only rebalance positions with > 5% weight drift from target
- Commission cost must be < 0.3% of trade notional (prevent over-trading)
- Monthly hard rebalance regardless of drift threshold

**Circuit breakers (from risk manager, DJ-091):**
- Daily portfolio loss > 2%: HALT all new positions, hold existing, alert user
- Single position loss > 10%: flag for review, reduce to half-size
- VaR 99% breach: reduce all positions by 20%, alert user

**Decision logging:**
Every ensemble call produces:
- LangFuse trace (Phase 6 infrastructure already exists)
- `data/live/{date}/{ticker}_ensemble.json` (local backup)
- EpisodeRecord in `hifi-live-episodes` namespace (Phase 14 episodic store)

**Outcome labeling:** `make label-outcomes` runs nightly. After 60 trading days from
any decision date, the outcome is labeled automatically. Labeled episodes become
retrievable in subsequent agent queries — the system learns from its own history.

---

## DJ-100: Phase 16 Duration and Completion Criteria

**Minimum duration:** 8 weeks (40 trading days). This provides at least 2 full
monthly rebalancing cycles and enough decisions for preliminary calibration analysis.

**Target duration:** 12-16 weeks. After 12 weeks, the first decisions (weeks 1-4)
have been outcome-labeled (60 trading days passed). The system now has a complete
generate-label-retrieve feedback loop operational.

**Completion criteria:**
1. 8+ weeks of live decisions accumulated
2. At least 480 outcome-labeled decisions (100 tickers × 2 rebalancing cycles × partial)
3. OQ-P14-04 answered: calibration curve at Phase 16 start vs. end — does episodic
   memory improve confidence-accuracy alignment?
4. Phase 15 vs. Phase 16 performance comparison: IC from live data vs. walk-forward

---

## Open Questions

| ID | Question | Resolution |
|---|---|---|
| OQ-P16-01 | Does live IC match walk-forward IC from Phase 15? | Phase 16 primary check |
| OQ-P14-04 | Does episodic RAG improve calibration over Phase 16 duration? | Phase 16 |
| OQ-P16-02 | What is realized slippage vs. simulated? | Phase 16 execution analysis |
| OQ-P16-03 | Which regime does Phase 16 fall in? Does performance match regime expectations? | Phase 16 |

---

## Phase 16 → Phase 17 Handoff

1. Live trading log: all decisions, executions, and outcomes
2. OQ-P14-04 answer: episodic RAG calibration improvement measured
3. IC/IR comparison: live vs. Phase 15 walk-forward
4. Episodic memory store: 480+ labeled episodes for Phase 17 ablation
5. Any operational anomalies documented (circuit breakers triggered, model failures)
