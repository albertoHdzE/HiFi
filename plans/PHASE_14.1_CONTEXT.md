# Phase 14.1: Pipeline Integration and Memory-Safe Orchestration
## Context and Pre-Phase Decisions

**Gathered:** 2026-06-20
**Status:** PLANNING
**Depends on:** Phase 14 (5-org ensemble, MCP tools, EDGAR RAG, sequential ensemble)
**Branch:** phase14/heterogeneous-ensemble (continues Phase 14 branch)

---

## Why This Phase Exists

Phase 14 built six infrastructure components independently:
1. Sequential ensemble runner (agents produce signals)
2. Portfolio composer (signals to weights)
3. Risk manager (weights to approved/blocked)
4. Capital allocator (approved weights to orders)
5. EDGAR MD&A retrieval (grounding context)
6. Episodic RAG pipeline (memory)

These components are individually tested (1756 tests pass), but they are not
integrated into an end-to-end pipeline. The walkforward harness
(`run_phase15_walkforward.py`) stops at EnsembleOutput JSONs. The downstream
MCP tools exist as standalone functions. No script chains them together.

Additionally, the Mac Studio M3 Ultra (98 GB unified memory) cannot load all
6 models simultaneously (~95 GB model weights + OS overhead), causing a system
crash during the first smoke test attempt (2026-06-19). The execution strategy
must be redesigned for memory-safe operation.

Finally, the smoke test (3 tickers from 3 sectors) does not exercise emergent
system properties: cross-asset correlations, sector concentration limits,
portfolio-level VaR, or capital allocation dynamics. These properties emerge
only at multi-sector portfolio scale.

Phase 14.1 addresses all three gaps before Phase 15 can run the scientific
experiment.

---

## DJ-106: Memory-Safe Execution Strategy — Agent-First Sequential Sweep

**Problem:** The Mac Studio M3 Ultra has 98 GB unified memory. The Phase 14
ensemble requires 6 models:

| Agent | Model | Est. RAM (4-bit) |
|-------|-------|-----------------|
| Fundamental | Llama 3.3 70B | ~35 GB |
| Technical | Qwen 32B 8-bit + adapter | ~16 GB |
| Risk | Mistral Small 3.2 24B | ~12 GB |
| Macro | DeepSeek R1 Distill 32B | ~16 GB |
| Sentiment | Gemma 3 12B | ~6 GB |
| Contrarian | Qwen3.5-35B MoE | ~10 GB |
| **Total** | | **~95 GB** |

Loading all models simultaneously (95 GB) plus OS overhead (~8-10 GB) exceeds
the 98 GB budget. The 2026-06-19 smoke test attempted this and crashed the
system.

**Analysis of alternatives:**

| Option | Description | Feasibility |
|--------|-------------|-------------|
| A: Ticker-first hot-swap | Per ticker: load 6 models sequentially | 56,448 model swaps at ~45s each = ~700 hours overhead. Infeasible. |
| B: Fine-tuned 32B substitution | Replace Llama 70B with fine-tuned 32B | Fits in RAM (~76 GB) but confounds scientific comparison (unvalidated quality trade-off) |
| C: Agent-first sequential sweep | Per agent: load once, run all tickers, unload | 6 total model swaps. Max VRAM: 35 GB (Llama 70B). Scientifically valid. |

**Decision:** Option C — Agent-First Sequential Sweep.

**Scientific rationale:** The sequential ensemble's inter-agent context is a
**data dependency** (stored in LanceDB `AgentContextStore`), not a **memory
co-residence dependency**. Each agent's model weights are needed only during
inference (~5-30 seconds per ticker). After inference, the result is persisted
to LanceDB. Subsequent agents read prior results from LanceDB, not from
co-resident model weights. Therefore, temporal ordering of agent execution
can be decoupled from simultaneous memory allocation.

Execution structure:
```
Pass 1: Load Llama 70B     -> fundamental() for all (condition, date, ticker) -> unload
Pass 2: Load Qwen 32B (ft) -> technical()   for all (condition, date, ticker) -> unload
Pass 3: Load Mistral 24B   -> risk()        for all (condition, date, ticker) -> unload
Pass 4: Load DeepSeek 32B  -> macro()       for all (condition, date, ticker) -> unload
Pass 5: Load Gemma 12B     -> sentiment()   for all (condition, date, ticker) -> unload
Pass 6: Load Qwen3.5 MoE   -> contrarian()  for all (condition, date, ticker) -> unload
Final:  CPU-only            -> aggregate stored outputs -> write ensemble JSONs
```

Peak VRAM at any time: 35 GB (Llama 70B pass). Matches the `run_phase14_e0_full.py`
pattern exactly: one model at a time, all its work, then next model.

**Required code change:** `run_sequential_ensemble()` currently generates
`run_id = str(uuid.uuid4())` — ephemeral per invocation. To support agent-first
sweep, `run_id` must be deterministic: `f"{condition}-{date}-{ticker}"`. This
allows Pass 2 (Technical) to look up Pass 1 (Fundamental) results from LanceDB
using the same run_id. Backward compatible: default remains uuid.uuid4().

---

## DJ-107: Representative Smoke Universe — Stratified 22-Ticker Sample

**Problem:** The Phase 15 smoke test used 3 tickers (AAPL, JPM, XOM) from
3 sectors. This cannot exercise emergent system properties:

- **Correlation matrix:** 3x3 matrix has 3 unique pairs — insufficient for
  meaningful portfolio-level VaR or correlation-aware risk checks
- **Sector concentration:** 1 ticker per sector trivially satisfies any sector
  cap — the 20% cap is never tested
- **Capital allocation:** Kelly criterion across 3 positions vs. 15+ positions
  produces qualitatively different portfolio dynamics
- **Portfolio VaR:** Requires cross-asset covariance structure, which is
  degenerate at n=3

From complexity science: the emergent properties of a multi-agent portfolio
system cannot be inferred from micro-scale (single-ticker) observations. The
smoke test must operate at the meso scale.

**Decision:** A stratified 22-ticker sample — 2 tickers per GICS sector:

| Sector | Bellwether | Mid-tier | Rationale |
|--------|-----------|----------|-----------|
| Information Technology | AAPL | CRM | Hardware vs. SaaS |
| Health Care | UNH | ABT | Payer vs. devices |
| Financials | JPM | BLK | Bank vs. asset manager |
| Consumer Discretionary | AMZN | NKE | E-commerce vs. retail |
| Communication Services | GOOGL | DIS | Digital vs. media |
| Industrials | HON | CAT | Diversified vs. cyclical |
| Consumer Staples | PG | COST | Branded vs. retail |
| Energy | XOM | COP | Integrated vs. E&P |
| Materials | LIN | FCX | Industrial gas vs. mining |
| Real Estate | PLD | AMT | Logistics vs. towers |
| Utilities | NEE | DUK | Growth vs. income |

22 tickers: sufficient for a 22x22 correlation matrix, sector concentration
testing (2 stocks can hit the 20% cap), portfolio-level VaR with real
cross-asset covariance, and capital allocation across ~10-15 Buy positions.

The sample is deterministic (hardcoded, not randomized) for reproducibility.

---

## DJ-108: End-to-End Pipeline Integration

**Problem:** The walkforward harness (`run_phase15_walkforward.py`) produces
`EnsembleOutput` JSONs per (condition, date, ticker) and stops. The downstream
stages exist as isolated MCP tools:

- `hifi.mcp.portfolio_composer.compose_portfolio()` — signals to weights
- `hifi.mcp.risk_manager.check_risk_limits()` — portfolio risk checks
- `hifi.mcp.capital_allocator.allocate_capital()` — order generation

These tools are individually tested (50+ unit tests, 14 integration tests in
`test_portfolio_pipeline.py`). But no production code chains them together.

**Decision:** Create `src/hifi/simulation/pipeline.py` — a programmatic
pipeline that chains ensemble outputs through all three MCP tools:

```
collect_signals(tickers, date)
    -> compose_portfolio(signals, constraints)
    -> check_risk_limits(portfolio, ohlcv, portfolio_state)
    -> allocate_capital(approved_weights, prices, holdings, capital)
    -> PortfolioSnapshot (all intermediate and final results)
```

All functions are already importable from `hifi.mcp.*`. No MCP server
infrastructure needed at runtime.

The pipeline runs per evaluation date (after all tickers have been evaluated),
producing a cross-sectional portfolio view. This is the natural aggregation
level: individual agent signals are per-ticker, but portfolio construction is
cross-sectional.

---

## DJ-109: Two-Layer Execution Architecture

**Problem:** The smoke test and production run have fundamentally different
requirements. Conflating them into one script creates something too complex
for validation and too simple for production.

**Decision:** Two separate scripts with distinct responsibilities:

| Script | Purpose | Universe | Pipeline depth |
|--------|---------|----------|---------------|
| `run_phase15_smoke.py` | Technical correctness validation | SMOKE_UNIVERSE (22 tickers) | Full: ensemble + compose + risk + allocate |
| `run_phase15_orchestrator.py` | Production scientific run | PHASE14_UNIVERSE (98 tickers) | Full: ensemble + compose + risk + allocate |

The smoke test answers: "does the full pipeline work end-to-end?" (minutes).
The orchestrator answers: "run the scientific experiment" (hours/days).

Both use the agent-first sequential sweep (DJ-106). Both exercise the full
MCP pipeline (DJ-108). The orchestrator adds checkpoint-resume, status
monitoring, and multi-condition support.

---

## DJ-110: Replication Notebook Strategy

**Problem:** Should the system auto-generate a Jupyter notebook per run?

**Analysis:** Auto-generated notebooks create file proliferation (96+ notebooks
for 4 conditions x 24 dates), are unmaintainable (analysis logic frozen at
generation time), and produce version control noise (massive diffs).

The established project pattern (`notebooks/phase11_finetune_replication.ipynb`,
`phase12_*`, `phase13_replication.ipynb`) uses living replication notebooks
that demonstrate methodology, call proven code, and walk through results.

**Decision:** One replication notebook: `notebooks/phase15_walkforward_replication.ipynb`

This notebook:
1. Loads walkforward results from a configurable data path (any completed run)
2. Calls the MCP pipeline programmatically (compose, risk, allocate)
3. Generates visualizations (decision distributions, sector exposure, VaR,
   confidence heatmaps, correlation structure)
4. Prints quality metrics (Shannon entropy, herding, IC/IR)
5. Includes narrative markdown cells explaining methodology
6. Is re-runnable against any condition's data

This is reproducible, educational, and follows the established pattern.
It serves as both scientific evidence and learning artifact for the capstone.

---

## Open Questions

| ID | Question | Resolution |
|---|---|---|
| OQ-P14-07 | Does agent-first sequential sweep produce identical results to monolithic ensemble? | Phase 14.1 validation (deterministic run_id comparison) |
| OQ-P14-08 | Is the fine-tuned Qwen 32B (fundamental_v1) an acceptable substitute for Llama 70B? | Phase 14.1 smoke test quality comparison (if Llama 70B fits as sole model) |

---

## Phase 14.1 -> Phase 15 Handoff

1. Agent-first orchestrator operational and validated
2. End-to-end pipeline producing portfolio snapshots per date
3. Smoke test passing on 22-ticker stratified universe
4. Replication notebook generating all visualizations from completed data
5. Master orchestrator ready for the full 98-ticker scientific run
