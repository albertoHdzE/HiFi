# Phase 14.1 Bitacora: Pipeline Integration and Memory-Safe Orchestration

**Phase status:** COMPLETE — 2026-06-20
**Tests at close:** 1809 passed, 3 skipped, 0 lint errors
**David sections:** SS4.1 (Deterministic-First), SS6.2 (MCP as Nervous System), SS7.9 (Execution Layer), SS9.1 (Local Hardware), SS15 (Evaluation)
**Branch:** phase14/heterogeneous-ensemble (continues Phase 14 branch)

---

## Objective

Phase 14 built six infrastructure components independently (sequential ensemble,
portfolio composer, risk manager, capital allocator, EDGAR MD&A retrieval, episodic RAG).
Each component was individually tested (1756 tests green). No script chained them together.

Phase 14.1 has three goals:

1. **Memory-safe execution:** Load all 6 models simultaneously (~95 GB) on the Mac Studio
   M3 Ultra (98 GB) crashed the system on the 2026-06-19 smoke test attempt. Redesign
   the execution strategy to stay within the hardware budget.

2. **End-to-end pipeline:** Chain the three MCP tools into a production-ready function
   that processes one evaluation date across all tickers.

3. **Production orchestrator:** Two-layer execution architecture: a validated smoke test
   (22 tickers, 1 date) and a master orchestrator for the full 98-ticker × 24-date
   × 4-condition scientific run.

---

## Architecture Decisions

### DJ-106: Memory-Safe Execution — Agent-First Sequential Sweep

**Problem:** 6 models totaling ~95 GB cannot fit simultaneously in 98 GB unified memory
with OS overhead (~8-10 GB). The 2026-06-19 attempt crashed the system.

**Analysis of alternatives:**

| Option | Description | Feasibility |
|--------|-------------|-------------|
| A: Ticker-first hot-swap | Per ticker: load 6 models | 56,448 swaps × 45s = ~700 hrs. Infeasible. |
| B: Fine-tuned 32B substitution | Replace Llama 70B with fine-tuned 32B | Fits in RAM but confounds scientific comparison |
| C: Agent-first sequential sweep | Per agent: load → all tickers → unload | 6 total swaps. Peak VRAM: 35 GB. Valid. |

**Decision:** Option C — Agent-First Sequential Sweep.

**Scientific rationale:** The sequential ensemble's inter-agent context is a **data
dependency** (stored in LanceDB AgentContextStore), not a **memory co-residence
dependency**. Each agent's weights are needed only during inference (~5-30 seconds
per ticker). After inference, the result is persisted to LanceDB. Subsequent agents
read prior results from LanceDB, not from co-resident weights.

**Execution structure:**
```
Pass 1: Load Llama 70B     → fundamental() for all (condition, date, ticker) → unload
Pass 2: Load Qwen 32B (ft) → technical()   for all (condition, date, ticker) → unload
Pass 3: Load Mistral 24B   → risk()        for all (condition, date, ticker) → unload
Pass 4: Load DeepSeek 32B  → macro()       for all (condition, date, ticker) → unload
Pass 5: Load Gemma 12B     → sentiment()   for all (condition, date, ticker) → unload
Pass 6: Load Qwen3.5 MoE   → contrarian()  for all (condition, date, ticker) → unload
Final:  CPU-only            → aggregate stored outputs → write ensemble JSONs
```

**Peak VRAM at any point:** 35 GB (Llama 70B pass). Total: ~43 GB. Within 98 GB budget
with 55 GB headroom.

### DJ-107: Stratified 22-Ticker Smoke Universe

**Problem:** The Phase 15 smoke test used 3 tickers (AAPL, JPM, XOM). At n=3:
- Correlation matrix is 3×3 (3 unique pairs) — insufficient for portfolio-level VaR
- One ticker per sector trivially satisfies any 20% sector cap
- Kelly criterion across 3 positions produces qualitatively different dynamics than n=15+
- Emergent system properties (cross-asset correlation, VaR, sector concentration) do not
  appear at micro-scale (single-ticker) observations

**Decision:** A stratified 22-ticker sample — 2 tickers per GICS sector:

| Sector | Bellwether | Mid-tier |
|--------|-----------|----------|
| Information Technology | AAPL | CRM |
| Health Care | UNH | ABT |
| Financials | JPM | BLK |
| Consumer Discretionary | AMZN | NKE |
| Communication Services | GOOGL | DIS |
| Industrials | HON | CAT |
| Consumer Staples | PG | COST |
| Energy | XOM | COP |
| Materials | LIN | FCX |
| Real Estate | PLD | AMT |
| Utilities | NEE | DUK |

22 tickers provide: meaningful 22×22 correlation matrix, sector concentration testing
(2 stocks can hit the 20% cap), portfolio-level VaR with real cross-asset covariance,
capital allocation across ~10-15 Buy positions.

### DJ-108: End-to-End Pipeline Integration

**Problem:** `run_phase15_walkforward.py` stops at EnsembleOutput JSONs. The MCP tools
exist as isolated functions. No production code chains them.

**Decision:** `src/hifi/simulation/pipeline.py` — a `run_pipeline()` function that
chains all three MCP tools:

```
signals → compose_portfolio() → check_risk_limits() → generate_orders() → PortfolioSnapshot
```

`PortfolioSnapshot` captures all intermediate and final results (signals, weights, risk
report, orders, sector exposure, total notional, constraints). JSON-serializable for
archiving.

### DJ-109: Two-Layer Execution Architecture

**Problem:** A smoke test and a production run have different requirements. One script
cannot serve both.

**Decision:**

| Script | Purpose | Universe | Frequency |
|--------|---------|----------|-----------|
| `run_phase15_smoke.py` | Technical correctness validation | SMOKE_UNIVERSE (22 tickers) | Once before full run |
| `run_phase15_orchestrator.py` | Full scientific experiment | PHASE14_UNIVERSE (98 tickers) | 4 conditions × 24 dates |

Both use agent-first sweep (DJ-106) and full MCP pipeline (DJ-108). The orchestrator
adds checkpoint-resume, status monitoring, and multi-condition support.

**Bug fix discovered during implementation:** The 2026-06-19 smoke failure also had a
second root cause beyond the OOM crash. The old script set `HIFI_FUNDAMENTAL_FINETUNE_URL`
for the fine-tuned fundamental fallback but NOT `HIFI_FUNDAMENTAL_FINETUNE_MODEL`.
The fundamental agent checks BOTH env vars (lines 319-320 of `fundamental_agent.py`).
Without `HIFI_FUNDAMENTAL_FINETUNE_MODEL`, it fell back to the default LM Studio model.
Fixed in the new smoke script: both vars are now set when using the port-1236 fallback.

### DJ-110: One Replication Notebook

**Problem:** Auto-generated notebooks (one per run) create file proliferation and
version control noise.

**Decision:** One living replication notebook:
`notebooks/phase15_walkforward_replication.ipynb`

Re-runnable against any condition's data via `CONDITION` and `PERIOD` configuration cells.
Covers: ensemble loading, decision analysis, quality metrics, MCP pipeline, risk dashboard,
correlation structure, capital allocation, portfolio composition, IC/IR computation.

---

## Epic E0: Agent-First Orchestrator Core

### E0-T1: Deterministic run_id

**File:** `src/hifi/agents/ensemble_runner.py`

Added `run_id: str | None = None` to `run_sequential_ensemble()`. When provided, the
caller's run_id is used instead of a new `uuid.uuid4()`. Backward compatible — existing
callers get a UUID as before.

**Design:** `run_id = f"{condition}-{date}-{ticker}"` scopes each context chain to one
(condition, date, ticker) triple. This allows Pass 2 (Technical) to read Pass 1
(Fundamental) results from AgentContextStore using the same run_id.

### E0-T2: Per-agent execution functions

**File:** `src/hifi/simulation/agent_executor.py` (new)

Key functions:
- `run_agent_pass(agent_type, ticker, date, condition, run_id, data_dir, db_path, ...)`:
  Runs exactly ONE agent, stores JSON sidecar at `{data_dir}/runs/{run_id}/{ticker}_{agent_type}.json`,
  and writes ≤300-char summary to AgentContextStore for subsequent agents.
- `aggregate_agent_outputs(ticker, date, run_id, db_path)`: Reads per-agent JSON sidecars,
  reconstructs via voting logic, returns `EnsembleOutput`.
- `extra_memory_prefix: str = ""` parameter on `run_agent_pass`: Caller-supplied prefix
  prepended to the store-derived memory context. Used by the fundamental agent for EDGAR
  MD&A context injection without modifying AgentContextStore records.

### E0-T3: Model manager

**File:** `src/hifi/simulation/model_manager.py` (new)

Encapsulates LM Studio Management API + `lms` CLI:
- `get_loaded_ids() → set[str]`: queries `/api/v0/models`
- `model_is_loaded(model_id) → bool`: exact + substring match
- `load_model(model_id, timeout_s=600) → bool`: skips if already loaded, then `lms load -y`
- `unload_model(model_id) → None`: `lms unload`, logs warning on failure

---

## Epic E1: End-to-End Pipeline Module

**File:** `src/hifi/simulation/pipeline.py` (new)

`PortfolioSnapshot` dataclass:
- `signals`: list of per-ticker decisions from EnsembleOutputs
- `weights`: compose_portfolio output (approved target weights)
- `risk_report`: check_risk_limits output (VaR, drawdown, sector cap, corr annotation)
- `orders`: generate_orders output (sized positions with IBKR commissions)
- `n_buy`, `n_hold`, `n_sell`, `sector_exposure`, `total_estimated_value`, `constraints`
- `to_json()` method for archiving

`run_pipeline(signals, ohlcv, portfolio_state, constraints) → PortfolioSnapshot`:
- Derives last-close prices from OHLCV if not in portfolio_state
- Filters weights to approved_set from risk report before order generation
- All intermediate results captured in snapshot

---

## Epic E2: Smoke Universe

**File:** `src/hifi/data/smoke_universe.py` (new)

`SMOKE_UNIVERSE`: list of 22 dicts, 2 per GICS sector, with keys `ticker`, `sector`,
`sub_industry`. All 22 tickers are also in `PHASE14_UNIVERSE`. Deterministic (hardcoded),
not randomized.

---

## Epic E3: Smoke Test Rewrite

**File:** `scripts/run_phase15_smoke.py` (complete rewrite)

Key improvements over the crashed 2026-06-19 version:
1. 22 tickers instead of 3 (exercises all 11 GICS sectors)
2. Agent-first sequential loading via `model_manager` (peak VRAM: 35 GB)
3. Full MCP pipeline after sweep
4. EDGAR MD&A context injected via `extra_memory_prefix` for fundamental agent
5. Bug fix: sets both `HIFI_FUNDAMENTAL_FINETUNE_URL` and `HIFI_FUNDAMENTAL_FINETUNE_MODEL`
6. Checkpoint-resume: skips existing JSON sidecars per (agent, ticker)
7. Comprehensive summary table: per-ticker decisions, sector exposure chart, pipeline metrics
8. `--skip-load`, `--cleanup`, `--condition` flags for operational flexibility

---

## Epic E4: Production Orchestrator

**File:** `scripts/run_phase15_orchestrator.py` (new)

Composable action flags:
- `--agent AGENT_TYPE`: loads model, runs all (date, ticker) passes, unloads
- `--agent all`: full 6-agent sequential sweep
- `--aggregate`: reads per-agent sidecars, writes ensemble JSONs
- `--pipeline`: reads ensemble JSONs, runs MCP pipeline, writes PortfolioSnapshot per date
- `--status`: shows sidecar/ensemble/portfolio counts across all 4 conditions
- `--dry-run`: prints schedule without LLM calls

Storage:
- Agent sidecars: `data/runs/{condition}-{date}-{ticker}/{ticker}_{agent_type}.json`
- Ensemble JSONs: `data/walkforward/{condition}/{YYYY}/{MM}/{ticker}.json`
- Portfolio JSONs: `data/walkforward/{condition}/{YYYY}/{MM}/portfolio.json`

Makefile targets:
- `walkforward-smoke-full`: 22-ticker smoke test
- `walkforward-orchestrate`: full agent sweep + aggregate + pipeline
- `walkforward-pipeline`: pipeline only on existing ensemble JSONs
- `walkforward-report`: status table + IC/IR metrics

---

## Epic E5: Replication Notebook

**File:** `notebooks/phase15_walkforward_replication.ipynb` (new)

10-section notebook:
1. Setup + configuration (CONDITION, PERIOD, DATA_DIR — all editable)
2. Ensemble output loading (reads `{OUTPUT_DIR}/{CONDITION}/{YYYY}/{MM}/*.json`)
3. Decision distribution (bar chart + sector breakdown)
4. Shannon entropy and herding index over time
5. MCP pipeline execution (compose → risk → allocate)
6. Risk dashboard (sector exposure, VaR 95/99, drawdown)
7. Return correlation matrix
8. Capital allocation table with IBKR commissions
9. Portfolio composition (weights bar chart + sector pie)
10. IC/IR computation (Spearman rank correlation with 21-day forward returns)

Fail-graceful throughout: all cells print `PENDING` if data is not yet available.

---

## Open Questions

| OQ | Question | Status |
|---|---|---|
| OQ-P14-07 | Does agent-first sweep produce identical results to monolithic ensemble? | DEFERRED — requires a completed smoke run for comparison |
| OQ-P14-08 | Is fine-tuned Qwen 32B an acceptable substitute for Llama 70B? | DEFERRED — requires smoke run with fallback active |

Both questions are answered by `make walkforward-smoke-full`.

---

## Phase 14.1 Outputs

### New code artifacts
- `src/hifi/simulation/agent_executor.py` — per-agent execution + aggregation
- `src/hifi/simulation/model_manager.py` — LM Studio load/unload/status
- `src/hifi/simulation/pipeline.py` — PortfolioSnapshot + run_pipeline()
- `src/hifi/data/smoke_universe.py` — 22-ticker stratified universe
- `src/hifi/agents/ensemble_runner.py` — `run_id` parameter added
- `scripts/run_phase15_smoke.py` — complete rewrite
- `scripts/run_phase15_orchestrator.py` — new production orchestrator

### New tests
- `tests/unit/simulation/test_agent_executor.py` (10 tests)
- `tests/unit/simulation/test_model_manager.py` (10 tests)
- `tests/unit/simulation/test_pipeline.py` (12 tests)
- `tests/unit/test_smoke_universe.py` (6 tests)
- `tests/unit/test_deterministic_run_id.py` (4 tests)
- `tests/integration/test_simulation_pipeline.py` (12 tests)
- **Total new tests: 54** (1756 → 1809 + 3 skipped at close)

### Documentation
- `notebooks/phase15_walkforward_replication.ipynb` (10 sections, fail-graceful)
- `plans/PHASE_14.1_CONTEXT.md` — DJ-106 through DJ-110 with full rationale
- `plans/PHASE_14.1_PLAN.md` — Epic/ticket plan (E0–E6, Waves 1–4)
- `doc/bitacora/PHASE_14.1_PIPELINE_INTEGRATION.md` — this document

---

## Complexity Science Notes

**Agent-first sweep as temporal decoupling:** The sequential ensemble produces causal
context chains (fundamental → technical → ... → contrarian). In a co-resident memory
model, this is a spatial constraint (all models must fit simultaneously). In the agent-first
model, this is a temporal constraint (agents run in sequence, context persists in LanceDB).
This decoupling demonstrates that the emergent property (causal reasoning chain) survives
disaggregation into separate inference steps — the intelligence is in the data flow
(LanceDB context records), not in shared memory.

**22-ticker smoke as meso-scale emergence test:** The original 3-ticker smoke test
operated at micro-scale where emergent portfolio properties (VaR, sector concentration,
correlation structure) are trivial or degenerate. The 22-ticker SMOKE_UNIVERSE operates
at meso-scale: 22×22 correlation matrix (231 unique pairs), two stocks per sector
(exercises sector concentration cap), portfolio-level VaR with real cross-asset covariance.
This is the minimum scale at which the system's collective intelligence properties
can be meaningfully observed.

**NEXT:** `make walkforward-smoke-full` — first real smoke run with live LLMs.
