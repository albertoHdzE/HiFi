# Phase 16 Plan: Live Paper Trading — Multi-Broker

**Status:** IN PROGRESS
**Context:** plans/PHASE_16_CONTEXT.md (DJ-098, DJ-099, DJ-100)
**Depends on:** Phase 15 COMPLETE (IC results available)

---

## Objective

Transition from historical simulation to live paper trading.
Validate that walk-forward IC translates to live signal quality.
Fulfill WQU capstone paper trading requirement (non-negotiable).
Begin accumulating outcome-labeled live episodes for episodic RAG maturation.

**Broker strategy:** Alpaca (primary, active), IBKR (pending application), Binance (future).
Broker-agnostic `BrokerExecutor` protocol enables hot-swap.

---

## Experimental Design (DJ-111): 3-Account Live Ablation

Three Alpaca paper accounts, same 98-ticker universe, same $1M capital,
one decision cycle per day (evening after close, orders fill at open):

| Account | Condition | Rationale |
|---|---|---|
| A | `parallel` ensemble | Phase 15 champion (IC=+0.0642, herding=0.000) — the fundable system |
| B | `full` sequential ensemble | Herding contrast (IC=+0.0232, herding=0.361) — live Page-theorem replication |
| C | Equal-weight buy-and-hold, no LLM | Null model — separates intelligence from market beta |

**Why conditions, not universe sizes:** universe-size ablation measures
diversification mechanics (solved since Markowitz). The open question is
whether agent diversity produces live alpha with herding as the mechanism.
If A > B > C in live IC/Sharpe over 8-12 weeks, that is a live,
out-of-time replication of Page's diversity theorem — the paper's
strongest figure.

**Daily frequency rationale:** ~980 labeled episodes/week feed the
episodic RAG loop (vs 22/week under weekly rebalancing). The capital
allocator's 5% drift threshold suppresses churn when signals are stable,
so daily cadence costs nothing in commissions. Inference budget: 2 LLM
conditions × 588 passes ≈ 5 h/night on the Mac Studio — fits overnight.

**Statistical note:** daily decisions with 60-day forward labels give
overlapping return windows; use Newey-West / block bootstrap for live IC
inference, not naive t-tests.

**Account provisioning (user task):** two additional Alpaca signups →
`.env` keys `ALPACA_API_KEY_B`/`ALPACA_SECRET_B`, `ALPACA_API_KEY_C`/
`ALPACA_SECRET_C`; reset all three accounts to $1M. Account A falls back
to the existing unsuffixed keys. Missing accounts skip gracefully.

---

## Pre-Phase Checklist

### Infrastructure — Alpaca (DONE)
- [x] Alpaca paper trading account active (PA35PLMC2LMK)
- [x] Credentials in `.env`: `ALPACA_API_KEY`, `ALPACA_SECRET`, `ALPACA_END_POINT`
- [x] `alpaca-py` installed
- [x] Connection validated: $564 equity, $127 cash, BA position live
- [x] OHLCV data updated through 2026-07-13 (22 tickers, +132 bars each)

### Infrastructure — IBKR (PENDING)
- [ ] IBKR paper trading account active (TWS or IB Gateway)
- [ ] TWS/Gateway running locally, port 7497 (TWS) or 4002 (Gateway)
- [ ] IBKR credentials in `.env`: `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`
- [ ] `ib_insync` installed: `uv add ib_insync`

### Validation gates (must pass before first live order)
- [x] Paper account connects: `uv run python scripts/run_phase16_live.py --status`
- [ ] Single-ticker dry run: `uv run python scripts/run_phase16_live.py --dry-run`
- [ ] Full-universe dry run: verify all 22 tickers resolve to valid orders
- [ ] Circuit breaker test: verify halt logic triggers on simulated 2% daily loss

---

## Epics

### E0: Broker-Agnostic Execution Layer (DJ-098) — DONE

**Files:**
- `src/hifi/execution/broker.py` — `BrokerExecutor` protocol + `Position`/`OrderResult` dataclasses
- `src/hifi/execution/alpaca_executor.py` — Alpaca implementation (alpaca-py SDK)
- `src/hifi/execution/market_data.py` — Live OHLCV via Alpaca bars API + local parquet merge
- `tests/unit/execution/test_alpaca_executor.py` — 9 tests (mocked)

**Design:** Protocol-based. Future `IBKRExecutor`, `BinanceExecutor` implement same interface.

### E1: Live Orchestrator (DJ-099, DJ-111) — DONE

**File:** `scripts/run_phase16_live.py`

Daily batch pipeline, per account:
```
1. update_local_ohlcv(98 tickers)        # Alpaca bars API → extend parquets
2. check_circuit_breakers()              # daily loss 2%, position loss 10%
3. signals: run_ensemble(condition)      # A/B: 6 agents × 98 tickers, agent-first
           | run_control_strategy()      # C: equal-weight buy-and-hold, no LLM
4. run_mcp_pipeline(compose→risk→alloc)  # A/B only
5. execute_orders(snapshot)              # Alpaca paper orders (per-account keys)
6. log_episode(data/live/{account}/decisions.jsonl)
```

Flags: `--account A|B|C|all`, `--execute`, `--dry-run`, `--status`,
`--update-data`, `--smoke` (22-ticker test), `--date`

Makefile: `live-status`, `live-update-data`, `live-dry-run`, `live-execute`

**Universe:** 98-ticker PHASE14_UNIVERSE (all 11 GICS sectors)
**Tests:** `tests/unit/execution/test_phase16_live.py` (control strategy + account routing)

### E2: Outcome Labeling Pipeline (DJ-099)

**File:** `scripts/label_live_outcomes.py` (new or extend existing)

- Runs nightly after market close
- For each live episode with decision_date <= today - 60 trading days:
  - Fetch realized return from OHLCV
  - Update EpisodeRecord: `forward_return`, `outcome_correct`
  - Mark episode as retrievable in `hifi-live-episodes` namespace
- After 60 days: first live episodes enter episodic RAG loop

### E3: Live Performance Dashboard (DJ-100)

**File:** `notebooks/phase16_live_trading.ipynb` (new)

Sections:
1. Portfolio positions and P&L (live from IBKR)
2. Signal log: all decisions since Phase 16 start
3. IC comparison: live IC vs Phase 15 walk-forward IC
4. Episodic memory growth: labeled episodes over time
5. Calibration curve: confidence vs accuracy (OQ-P14-04)
6. Circuit breaker history

---

## Operational Schedule (8-week minimum)

| Week | Milestone |
|---|---|
| 1 | Infrastructure ready, first dry run passes, first live order placed |
| 2-4 | Daily batch running, monitoring active, first 60 decisions accumulated |
| 5-8 | First outcome labels available (week 1 decisions labeled), episodic RAG live |
| 8 | Minimum completion criteria met: 480 labeled decisions |
| 12-16 | Target: full generate-label-retrieve feedback loop operational |

---

## Completion Criteria (DJ-100)

1. 8+ weeks of live decisions accumulated
2. 480+ outcome-labeled decisions (100 tickers x 2 rebalancing cycles x partial)
3. OQ-P14-04 answered: does episodic memory improve confidence-accuracy alignment?
4. Phase 15 vs Phase 16 IC comparison documented
5. WQU capstone paper trading requirement fulfilled

---

## Circuit Breakers (DJ-099)

- Daily portfolio loss > 2%: HALT all new positions, hold existing, alert user
- Single position loss > 10%: flag for review, reduce to half-size
- VaR 99% breach: reduce all positions 20%, alert user
- All circuit breaker events logged to `data/live/circuit_breakers.jsonl`

---

## Open Questions

| OQ | Question | Expected resolution |
|---|---|---|
| OQ-P16-01 | Does live IC match walk-forward IC (+0.0642 parallel)? | Week 8+ |
| OQ-P14-04 | Does episodic RAG improve calibration over Phase 16? | Week 12+ |
| OQ-P16-02 | Realized slippage vs simulated? | Week 4+ |
| OQ-P16-03 | Which regime does Phase 16 fall in? | Week 1 (classify current regime) |
