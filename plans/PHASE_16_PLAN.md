# Phase 16 Plan: Live Paper Trading — IBKR

**Status:** NEXT
**Context:** plans/PHASE_16_CONTEXT.md (DJ-098, DJ-099, DJ-100)
**Depends on:** Phase 15 COMPLETE (IC results available)

---

## Objective

Transition from historical simulation to live paper trading on IBKR.
Validate that walk-forward IC translates to live signal quality.
Fulfill WQU capstone paper trading requirement (non-negotiable).
Begin accumulating outcome-labeled live episodes for episodic RAG maturation.

---

## Pre-Phase Checklist

### Infrastructure (from Phase 14)
- [ ] IBKR paper trading account active (TWS or IB Gateway)
- [ ] TWS/Gateway running locally, port 7497 (TWS) or 4002 (Gateway)
- [ ] IBKR credentials in `.env`: `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`
- [ ] `ib_insync` installed: `uv add ib_insync`
- [ ] `data/live/` directory created
- [ ] `hifi-live-episodes` LanceDB namespace initialized

### Validation gates (must pass before first live order)
- [ ] Paper account connects: `python -c "from ib_insync import IB; ib=IB(); ib.connect('127.0.0.1', 7497, clientId=1); print(ib.accountValues())"`
- [ ] Single-ticker dry run: `make live-dry-run TICKER=AAPL` (no order placed)
- [ ] Full-universe dry run: verify all 98 tickers resolve to valid IBKR contract IDs
- [ ] Circuit breaker test: verify halt logic triggers on simulated 2% daily loss

---

## Epics

### E0: IBKR Connection and Order Execution (DJ-098)

**File:** `src/hifi/execution/ibkr_executor.py` (new)

- `connect(host, port, client_id)` — ib_insync IB() wrapper
- `resolve_contract(ticker) -> Contract` — STK, SMART, USD
- `place_market_order(contract, action, quantity) -> Trade`
- `get_portfolio_positions() -> dict[str, Position]`
- `get_account_value() -> float`
- Async-safe: all calls within asyncio event loop

**Tests:** `tests/unit/execution/test_ibkr_executor.py` (mock ib_insync)

### E1: Live Orchestrator (DJ-099)

**File:** `scripts/run_phase16_live.py` (new)

Daily batch pipeline (runs 22:00-06:00 local):
```
1. load_ohlcv_through(today)           # update market data
2. run_phase15_orchestrator(condition=full, date=today)  # generate signals
3. run_pipeline(signals, ohlcv, portfolio_state)         # MCP pipeline
4. ibkr_executor.place_orders(snapshot.orders)           # paper execution
5. log_episode(decision, execution)    # write to hifi-live-episodes
6. langfuse.trace(full_pipeline)       # observability
```

Flags: `--dry-run`, `--condition`, `--no-execute` (signals only, no orders)

Makefile targets:
- `live-dry-run`: signals + pipeline, no orders
- `live-execute`: full daily batch
- `live-status`: portfolio positions + P&L

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
