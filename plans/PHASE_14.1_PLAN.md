# Phase 14.1: Pipeline Integration and Memory-Safe Orchestration
## Epic and Ticket Plan

**Phase status:** PLANNING
**Pre-phase decisions:** DJ-106 through DJ-110 (see PHASE_14.1_CONTEXT.md)
**David sections:** SS4.1 (Deterministic-First), SS6.2 (MCP as Nervous System), SS7.9 (Execution Layer), SS9.1 (Local Hardware), SS15 (Evaluation)
**Branch:** phase14/heterogeneous-ensemble (continues Phase 14 branch)
**Next DJ number at phase start:** DJ-111

---

## Central Objective

Integrate the six infrastructure components built in Phase 14 into an
orchestrated end-to-end pipeline that runs safely within 98 GB unified memory,
exercises all MCP tools at portfolio scale, and produces observable, reproducible
artifacts suitable for scientific evaluation and capstone documentation.

---

## Success Criteria

- [ ] Agent-first orchestrator: loads one model at a time, 6 sequential passes, deterministic run_id
- [ ] End-to-end pipeline: ensemble -> compose -> risk -> allocate producing PortfolioSnapshot per date
- [ ] Smoke test passing on 22-ticker stratified universe (all 11 GICS sectors), full pipeline
- [ ] Master orchestrator ready for 98-ticker, 4-condition, 24-date production runs
- [ ] Replication notebook: loads any completed run, exercises MCP pipeline, generates all visualizations
- [ ] OQ-P14-07 answered: agent-first sweep produces identical results to monolithic ensemble
- [ ] Tests: existing 1756+ green, new tests for pipeline and orchestrator
- [ ] Bitacora written: doc/bitacora/PHASE_14.1_PIPELINE_INTEGRATION.md

---

## Wave Structure

```
Wave 1 (no LLM required -- pure code, parallel):
  E0: Agent-first orchestrator core (deterministic run_id, per-agent execution)
  E1: End-to-end pipeline module (compose -> risk -> allocate chain)
  E2: Smoke universe definition (stratified 22-ticker sample)

Wave 2 (requires Wave 1 -- integration):
  E3: Smoke test rewrite (22 tickers, full pipeline, env var fix)
  E4: Production orchestrator (master script for real runs)

Wave 3 (requires Wave 2 + one successful smoke run):
  E5: Replication notebook (analyze, visualize, explain)

Wave 4 (at phase completion):
  E6: Documentation (bitacora, STATUS update, MEMORY update)
```

---

## E0: Agent-First Orchestrator Core

### E0-T1: Deterministic run_id in run_sequential_ensemble()

Add optional `run_id: str | None = None` parameter to `run_sequential_ensemble()`.
When provided, use it instead of `uuid.uuid4()`. Default remains uuid for
backward compatibility. All existing tests unaffected.

**File:** `src/hifi/agents/ensemble_runner.py`
**Change:** 1-line parameter addition + conditional in run_id assignment
**Tests:** New unit test confirming deterministic run_id produces reproducible
AgentContextStore lookups.

### E0-T2: Per-agent partial execution functions

Extract per-agent execution logic from the monolithic `run_sequential_ensemble()`
into callable functions that can be invoked independently per agent pass.

Each function:
1. Loads prior agent results from AgentContextStore (LanceDB)
2. Runs exactly ONE agent
3. Stores the result in AgentContextStore
4. Returns the agent's analysis object

**File:** `src/hifi/simulation/agent_executor.py` (new)
**Functions:**
- `run_agent_pass(agent_type, ticker, date, condition, run_id, data_dir, db_path)`
- `aggregate_agent_outputs(ticker, date, run_id, db_path) -> EnsembleOutput`

### E0-T3: Model loading/unloading manager

Reuse the LM Studio Management API pattern from `run_phase14_e0_full.py`:
- `load_model(model_id, timeout_s)` via `lms load -y`
- `unload_model(model_id)` via `lms unload`
- `model_is_loaded(model_id)` via `/api/v0/models`

**File:** `src/hifi/simulation/model_manager.py` (new)
**Rationale:** Extracted from `run_phase14_e0_full.py` to be reusable by both
smoke test and production orchestrator.

---

## E1: End-to-End Pipeline Module

### E1-T1: Pipeline implementation

Chain the three MCP tools into a single function that processes one evaluation
date across all tickers:

```python
def run_pipeline(
    signals: list[dict],     # per-ticker ensemble decisions
    ohlcv: dict[str, list],  # OHLCV data per ticker
    portfolio_state: dict,   # current holdings, capital, HWM
    constraints: dict,       # max_stock, max_sector, min_position
) -> PortfolioSnapshot:
```

Returns a `PortfolioSnapshot` dataclass containing:
- Input signals (with decisions, confidences, sectors)
- Portfolio weights (compose_portfolio output)
- Risk report (check_risk_limits output: approved/blocked, VaR, drawdown)
- Orders (allocate_capital output: sized positions with commissions)
- Summary statistics (n_buy, n_hold, n_sell, sector exposure, total estimated value)

**File:** `src/hifi/simulation/pipeline.py` (new)
**Imports:** `hifi.mcp.portfolio_composer`, `hifi.mcp.risk_manager`, `hifi.mcp.capital_allocator`

### E1-T2: Pipeline integration tests

Test the pipeline with synthetic signals spanning all 11 GICS sectors.
Verify: compose respects sector cap, risk blocks high-VaR positions,
allocator sizes with IBKR commissions, PortfolioSnapshot is JSON-serializable.

**File:** `tests/integration/test_simulation_pipeline.py` (new)

---

## E2: Smoke Universe Definition

### E2-T1: Stratified 22-ticker sample

Hardcoded, deterministic sample: 2 tickers per GICS sector (DJ-107).

**File:** `src/hifi/data/smoke_universe.py` (new)
**Exports:** `SMOKE_UNIVERSE: list[dict]` (same schema as `PHASE14_UNIVERSE`)
**Tests:** Verify all 11 sectors covered, 22 tickers, all in PHASE14_UNIVERSE.

---

## E3: Smoke Test Rewrite

### E3-T1: Rewrite run_phase15_smoke.py

Complete rewrite of the smoke test script:
1. Uses `SMOKE_UNIVERSE` (22 tickers, all sectors)
2. Agent-first sequential model loading (DJ-106)
3. Full pipeline: ensemble -> compose -> risk -> allocate
4. Fix `HIFI_FUNDAMENTAL_FINETUNE_MODEL` env var bug (root cause of 2026-06-19 failure)
5. Summary table: per-ticker decisions, portfolio composition, risk report, orders
6. Exit non-zero on any failure

**File:** `scripts/run_phase15_smoke.py` (rewrite)
**Dependencies:** E0 (orchestrator core), E1 (pipeline), E2 (smoke universe)

---

## E4: Production Orchestrator

### E4-T1: Master orchestrator script

The production script for the full scientific run (Phase 15).
Uses agent-first sequential loading, full pipeline, checkpoint-resume.

**File:** `scripts/run_phase15_orchestrator.py` (new)
**Parameters:**
- `--condition`: full | parallel | homogeneous | no-memory
- `--period`: held-out-test | validation | walk-forward | all
- `--agent`: Run only one agent's pass (for agent-first execution)
- `--aggregate`: Aggregate stored per-agent outputs into ensemble JSONs
- `--pipeline`: Run MCP pipeline on completed ensemble outputs
- `--status`: Show progress
- `--dry-run`: Show schedule

### E4-T2: Makefile targets

```makefile
walkforward-orchestrate:    Full orchestrated run (agent-first + pipeline)
walkforward-smoke-full:     22-ticker smoke test with full pipeline
walkforward-pipeline:       Run MCP pipeline on existing ensemble JSONs
walkforward-report:         Open replication notebook
```

---

## E5: Replication Notebook

### E5-T1: Phase 15 walkforward replication notebook

One Jupyter notebook following the established project pattern (Phase 11-13).

**File:** `notebooks/phase15_walkforward_replication.ipynb` (new)
**Sections:**
1. Configuration: data path, condition, date range (configurable cells)
2. Load ensemble outputs: read JSONs, build cross-sectional DataFrame
3. Decision analysis: Buy/Hold/Sell distribution by sector, confidence heatmap
4. Quality metrics: Shannon entropy, herding index, per-agent agreement rates
5. MCP Pipeline: compose_portfolio -> check_risk_limits -> allocate_capital
6. Risk dashboard: VaR 95/99, drawdown, sector exposure pie chart
7. Correlation analysis: top-N correlated pairs, blocked tickers
8. Capital allocation: position sizing table, commission estimates
9. Portfolio composition: final weights, sector breakdown, cash position
10. IC/IR computation: Spearman rank correlation with forward returns (when available)
11. Summary: key findings in markdown, suitable for capstone reference

Each section includes explanatory markdown cells.

---

## E6: Documentation (at phase completion)

### E6-T1: Bitacora

Write `doc/bitacora/PHASE_14.1_PIPELINE_INTEGRATION.md` at phase close.
Format follows established Phase 0-13 bitacoras.

### E6-T2: STATUS.md update

Update Phase 14.1 status to COMPLETE with test counts and key results.

### E6-T3: MEMORY.md update

Update auto-memory with Phase 14.1 completion state, DJ index, and next action.
