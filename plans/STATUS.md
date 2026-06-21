# HiFi Project Status

**Last Updated:** 2026-06-21 (Phase 14.1 complete — awaiting smoke run)
**Current Phase:** Phase 14.1 (COMPLETE — branch: phase14/heterogeneous-ensemble)

---

## Quick Context for New Sessions

HiFi is a fully local multi-agent financial intelligence platform. Read these in order:

1. `doc/HIFI_DAVID.md` -- The ideal specification (the David)
2. `doc/HIFI_PROTOCOL_V1.md` -- The execution plan (18 phases)
3. `doc/HIFI_LEARNING_GUIDE.md` -- Learning tracker and David proximity matrix
4. `plans/PHASE_XX_PLAN.md` -- Epic/ticket plans per phase
5. `doc/bitacora/PHASE_XX_*.md` -- Scientific logbook per phase

---

## Phase Status

| Phase | Name | Status | Plan | Bitacora |
|---|---|---|---|---|
| 0 | Project Infrastructure | COMPLETE | plans/PHASE_00_PLAN.md | doc/bitacora/PHASE_00_INFRASTRUCTURE.md |
| 1 | Data Acquisition | COMPLETE | plans/PHASE_01_PLAN.md | doc/bitacora/PHASE_01_DATA_ACQUISITION.md |
| 2 | Deterministic Financial Engine | COMPLETE | plans/PHASE_02_PLAN.md | doc/bitacora/PHASE_02_DETERMINISTIC_ENGINE.md |
| 3 | First Agent (Baseline) | COMPLETE | plans/PHASE_03_PLAN.md | doc/bitacora/PHASE_03_FIRST_AGENT.md |
| 4 | Second Agent (First Ensemble) | COMPLETE | plans/PHASE_04_PLAN.md | doc/bitacora/PHASE_04_SECOND_AGENT.md |
| 5 | Verification Layer | COMPLETE | plans/PHASE_05_PLAN.md | doc/bitacora/PHASE_05_VERIFICATION.md |
| 6 | Observability (LangFuse) | COMPLETE | plans/PHASE_06_PLAN.md | doc/bitacora/PHASE_06_OBSERVABILITY.md |
| 7 | RAG Knowledge Systems | COMPLETE | plans/PHASE_07_PLAN.md | doc/bitacora/PHASE_07_RAG.md |
| 8 | Full Agent Population | COMPLETE | plans/PHASE_08_PLAN.md | doc/bitacora/PHASE_08_AGENT_POPULATION.md |
| 9 | Collective Decision Engine | COMPLETE | plans/PHASE_09_PLAN.md | doc/bitacora/PHASE_09_COLLECTIVE_ENGINE.md |
| 10 | Evaluation & Backtesting | COMPLETE | plans/PHASE_10_PLAN.md | doc/bitacora/PHASE_10_EVALUATION.md |
| 11 | Fine-Tuning | COMPLETE | plans/PHASE_11_PLAN.md | doc/bitacora/PHASE_11_FINE_TUNING.md |
| 12 | GraphRAG + Structured Debate | COMPLETE | plans/PHASE_12_PLAN.md | doc/bitacora/PHASE_12_GRAPHRAG_DEBATE.md |
| 12.1 | Completion and Correction | COMPLETE | plans/PHASE_12.1_PLAN.md | doc/bitacora/PHASE_12.1_COMPLETION.md |
| 13 | Verification Completeness, Sentiment Intelligence, System Resilience | COMPLETE | plans/PHASE_13_PLAN.md | doc/bitacora/PHASE_13_ADVANCED_FEATURES.md |
| 14 | Infrastructure: Model Diversity, Scale Expansion, MCP Tools (DJ-088) | COMPLETE | plans/PHASE_14_PLAN.md | -- |
| 14.1 | Pipeline Integration and Memory-Safe Orchestration (DJ-106) | COMPLETE | plans/PHASE_14.1_PLAN.md | doc/bitacora/PHASE_14.1_PIPELINE_INTEGRATION.md |
| 15 | Historical Walk-Forward Simulation (DJ-088) | NOT STARTED | -- | -- |
| 16 | Live Paper Trading — IBKR (DJ-088) | NOT STARTED | -- | -- |
| 17 | Ablation Studies + Capstone Deliverable | NOT STARTED | -- | -- |
| 18 | Publication + Open Source Release | NOT STARTED | -- | -- |

---

## Phase 10 Results (COMPLETE 2026-06-12)

- 939 tests, 0 skipped, 0 lint errors
- Accuracy on 2023-03-31 baseline (3 tickers, 4 methods): 0.0 all methods
  (agents voted BUY; market flat/negative in 2023-Q2 -- valid empirical result)
- Bootstrap accuracy (heuristics, 2018-2022): risk=0.349, technical=0.254, fundamental=0.079, macro=0.079
- Tear sheets: null metrics (3 tickers, 1 analysis date -- insufficient for QuantStats)
- Performance history: 255 bootstrap records (heuristic proxies only)
- 15-ticker expansion pending: run `make acquire-data-phase10` (requires internet)

## Phase 11 Results (COMPLETE 2026-06-13)

- 997 tests, 4 skipped, 0 lint errors
- Rank sweep: rank 4/8/16/32 at 300 iters, losses 0.314/0.299/0.296/0.298, optimal=rank 8
- technical_v1 adapter: rank 8, 1000 iters, 26,433 examples, 8202s, quality PASS
- fundamental_v1 adapter: rank 8, 1000 iters, 26,433 examples, 2767s, quality PASS
- Three-tier evaluation (AAPL/JPM/XOM, 2023-03-31):
  - Base Technical GR=1.000, Fine-tuned Technical GR=0.000 (NOT DEPLOYED -- GR degraded)
  - Base Fundamental GR=1.000, Fine-tuned Fundamental GR=1.000 (PASS)
  - Diversity pairwise=0.000 both runs (agents agreed on all tickers this date)
  - OQ-M01: rank 8 confirmed optimal
  - OQ-M02: diversity preserved (vacuously -- single date with no disagreement)
- Replication notebook: notebooks/phase11_finetune_replication.ipynb
- Bug fixes: serve_finetune_models.sh (log-level casing, deprecated module path),
  lm_client.py (base_url param), agent finetune URL routing, eval GR field path

## Phase 13 Status (IN PROGRESS — 2026-06-15 Wave 2 session 2)

**Tests:** 1271 passed, 0 skipped, 0 lint errors (src/ + tests/ only)
**Commit:** 9b344ee (Phase 13 Wave 2 partial), new scripts added this session

### Architecture decisions: DJ-071 through DJ-087 (see plans/PHASE_13_CONTEXT.md)

### Completed work

- E0: verify_agent() extended to Risk + Macro + Sentiment (SGR metric) ✓
- E1: ABORT — OQ-S01 NEGATIVE (0 Sell examples; FT deferred to Phase 14) ✓
- E2: run_debate_multi_round() + max_rounds in ensemble_runner ✓ (code)
- E4: AgentMemoryRecord + AgentMemoryStore + injection into all 5 agents ✓ (code)
- E5: DriftMonitor KS/chi-sq/CUSUM + calibration against 2022 regime ✓
- E6: ScenarioEvaluator + PHASE13_SCENARIOS (7 scenarios) ✓ (code)
- E7: Dataset Family E README + Dataset Family G MANIFEST.md ✓
- DJ-086/DJ-087: Gemma 4 E4B diagnosis → revert Sentiment to qwen2.5-coder + verbatim Rule 5 ✓

### E0 Baselines (2023-03-31)

- Risk: HR=0.000, GR=1.000 (max_drawdown verified), alias_coverage=38.9%
- Macro: HR=0.000, GR=0.000, n_claims=0-1 (FRED data absent/sparse)
- Sentiment (verbatim Rule 5, DJ-087): mean_SGR=0.667
  - AAPL: SGR=0.000 (8-K boilerplate context, no quotable signals)
  - JPM: SGR=1.000 (2/2 grounded)
  - XOM: SGR=1.000 (2/2 grounded)

### E5-T5 Drift Calibration (2022 regime)

- KS test (vol+RSI, 2020-21 vs 2022-23): p=0.000 ALERT ✓
- Chi-squared (momentum decisions): p=0.000 ALERT ✓
- CUSUM (frac < 50d MA): C_k=48.57 >> threshold=0.534 ALERT ✓
- OQ-DR01: YES — all three monitors detect 2022 rate-shock regime change

### LLM eval results (COMPLETE 2026-06-16)

- E2-T4 OQ-D04: NEGLIGIBLE — 2-round herding=0.929 vs 1-round=0.950, Δ=-0.021 (14/15 runs)
- E4-T4 OQ-M03: YES — memory changed 9/30 pairs (30%), fundamental most susceptible
- E6-T2 Dataset F: 4/7 aligned (57%); crash=67%, rate_shock=67%, earnings_beat=0%
  - Hold bias observed; F-001c/F-002c/F-003 misaligned (snapshot date mismatch + FRED sparsity)

### All Phase 13 OQ answers

| OQ | Answer |
|---|---|
| OQ-S01 | NEGATIVE — 0 Sell examples; Sentiment FT deferred to Phase 14 |
| OQ-D04 | NEGLIGIBLE — Δ=-0.021 (|Δ|<0.05); herding set by model diversity not round count |
| OQ-M03 | YES — 30% pairs changed; memory anchors Fundamental Agent toward prior decisions |
| OQ-DR01 | YES — all 3 monitors detect 2022 rate-shock (CUSUM C_k=48.57 >> threshold=0.534) |

### DJ Index (Phase 13)

| DJ# | Decision |
|---|---|
| DJ-071 | Phase 13 scope and wave structure |
| DJ-072 | Verification extension strategy (Risk + Macro branches) |
| DJ-073 | SGR metric for Sentiment verification |
| DJ-074 | Multi-round debate with vote-stability convergence |
| DJ-075 | OQ-D04 hypothesis pre-registration (NEGLIGIBLE) |
| DJ-076 | Agent memory: in-context prefix, JSON append-only store |
| DJ-077 | OQ-M03 experimental design (synthetic priors) |
| DJ-078 | Scenario methodology: historical events, not generative synthetic |
| DJ-079 | Drift monitor trio: KS + chi-sq + CUSUM |
| DJ-080 | Gemma 4 as Sentiment base (SUPERSEDED by DJ-087) |
| DJ-081 | Phase 12.1 technical_v2 compliance ratio fix |
| DJ-082 | Phase 12.1 retraining at 500 iters rank 8 |
| DJ-083 | Phase 12.1 factorial experiment design |
| DJ-084 | Phase 12.1 herding threshold definition |
| DJ-085 | Sentiment model → gemma-4-e4b (SUPERSEDED by DJ-087) |
| DJ-086 | E4B diagnosis: chat-template failure in LM Studio |
| DJ-087 | Revert Sentiment to qwen2.5-coder + verbatim Rule 5 |

---

## Phase 14 Status (COMPLETE — 2026-06-19)

**Tests:** 1756 passed, 0 lint errors
**Branch:** phase14/heterogeneous-ensemble

### Wave 1 — COMPLETE
- E2-T1: PHASE14_UNIVERSE (98 tickers, 11 GICS sectors) ✓
- E4-T1: hifi-portfolio-composer MCP server (deterministic) ✓
- E6-T1: NamespacedLanceDB + KnowledgeStore namespace param ✓

### Wave 2 — COMPLETE (E0 + E1)
- E0: diversity baseline OQ-P14-05 PASS (mean_entropy=0.7449) ✓
- E1: OQ-S01 NEGATIVE → permanently closed; FT deferred to Phase 16 ✓

### Wave 3 — COMPLETE (E3 sequential ensemble)
- E3-T1: AgentContextStore (LanceDB) + format_prior_context ✓
- E3-T2: run_sequential_ensemble() — causal context accumulation ✓
- E3-T3: graph.py — LangGraph StateGraph 6-node topology ✓
- E3-T4: run_ensemble(sequential=True) ✓

### Wave 4 — COMPLETE
- E2-T3: edgar_mda.py helpers + ingest_edgar_mda.py script ✓
- E2-T4: acquire_macro_phase14.py (FRED GS10/GS2 + spread) ✓
- E4-T2: risk_manager MCP server (VaR, drawdown, sector cap, corr) ✓
- E4-T3: capital_allocator MCP server (Kelly cap, IBKR commissions) ✓
- E4-T4: test_portfolio_pipeline.py (3-MCP integration) ✓
- E5-T1: regime.py (classify_regime, VIX fallback) ✓
- E5-T2: episodic_store.py (EpisodicStore + EpisodeRecord, LanceDB) ✓
- E5-T3: episodic_retriever.py (temporal-disciplined RAG) ✓
- E5-T4: label_outcomes.py (60-day forward return, yfinance) ✓
- E5-T5: episode creation in run_sequential_ensemble() ✓
- E6-T2: manage_namespaces.py + Makefile targets ✓
- E6-T3: ingest_episodes.py stub + temporal filter tests ✓

### Remaining (internet-dependent, non-blocking)
- E2-T2: acquire_phase14_data.py — bulk OHLCV 100 stocks × 21y (run separately)
- E2-T3 ingest: ingest_edgar_mda.py — EDGAR API calls (run separately, 4-8h)

### Phase 14 Wave 5 — Documentation (E7)
Deferred to run alongside Phase 14.1/15 (non-blocking).

---

## Phase 14.1 Status (COMPLETE — 2026-06-21)

**Tests:** 1809 passed, 3 skipped, 0 lint errors
**Branch:** phase14/heterogeneous-ensemble
**Decisions:** DJ-106 through DJ-110 (see plans/PHASE_14.1_CONTEXT.md)
**Bitacora:** doc/bitacora/PHASE_14.1_PIPELINE_INTEGRATION.md

Phase 14.1 integrates the six infrastructure components built in Phase 14
into an orchestrated end-to-end pipeline:

- **DJ-106:** Agent-first sequential sweep (one model in VRAM at a time; 35 GB peak vs 95 GB simultaneous) ✓
- **DJ-107:** Stratified 22-ticker smoke universe (2 per GICS sector, all 11 sectors) ✓
- **DJ-108:** End-to-end pipeline: ensemble → compose → risk → allocate → PortfolioSnapshot ✓
- **DJ-109:** Two-layer execution: smoke (22-ticker validation) + orchestrator (98-ticker production) ✓
- **DJ-110:** One replication notebook: notebooks/phase15_walkforward_replication.ipynb ✓

### Code artifacts
- `src/hifi/simulation/agent_executor.py` — run_agent_pass() + aggregate_agent_outputs()
- `src/hifi/simulation/model_manager.py` — load_model/unload_model/model_is_loaded via lms CLI
- `src/hifi/simulation/pipeline.py` — PortfolioSnapshot + run_pipeline() MCP chain
- `src/hifi/data/smoke_universe.py` — 22-ticker stratified universe (SMOKE_UNIVERSE)
- `src/hifi/agents/ensemble_runner.py` — run_id: str | None = None param added
- `scripts/run_phase15_smoke.py` — complete rewrite (agent-first + full pipeline)
- `scripts/run_phase15_orchestrator.py` — production orchestrator (--agent/--aggregate/--pipeline/--status)
- `notebooks/phase15_walkforward_replication.ipynb` — 10-section replication notebook

### Bug fix
- DJ-109: old smoke script set `HIFI_FUNDAMENTAL_FINETUNE_URL` but not `HIFI_FUNDAMENTAL_FINETUNE_MODEL`
  for the fine-tuned fallback. fundamental_agent.py checks both. Both now set correctly.

### NEXT: `make walkforward-smoke-full` (requires LM Studio + finetune server on port 1235)

---

## Phase 12.1 Results (COMPLETE 2026-06-15)

- 1197 tests, 0 skipped, 0 lint errors
- technical_v2: rank 8, 500 iters, loss 0.295, 26,630 examples (200 compliance augmented)
- Factorial 120/120 complete (A:30 B:30 C:30 D:30)
  - A (base, no debate):  entropy=0.367, herding=0.817
  - B (FT, no debate):    entropy=0.000, herding=1.000
  - C (base, debate):     entropy=0.100, herding=0.950, debate_rate=36.7%
  - D (FT, debate):       entropy=0.000, herding=1.000, debate_rate=0.0%
- OQ-M02: NEGATIVE — fine-tuning collapses diversity (entropy 0.367→0.000, 100% loss)
- OQ-D01: YES — debate increases herding A→C by +0.133 (threshold >0.10)
- OQ-D02: DEGENERATE — B=D (FT saturates ensemble pre-debate); interaction = -(C-A)
- Key finding: technical_v2+fundamental_v1 vote unanimously Buy; debate structurally inert
- DJ-085: Sentiment model → google/gemma-4-e4b (12B VLM incompatible with LM Studio)
- OQ-SGR01: NEGATIVE — Gemma 4 E4B SGR=0.000 (0 parseable signals on AAPL/JPM; 2 signals XOM, 0 grounded)
- Fixture: tests/fixtures/baseline/phase12_factorial_results.json

## Phase 11 Pre-Phase Decisions (DJ-053 to DJ-060)

Full rationale in `plans/PHASE_11_CONTEXT.md`.

- DJ-053: Scope = fine-tuning only. Structured debate deferred to Phase 12.
- DJ-054: Dataset Family C, heterogeneous labels per agent (Technical=max-return, Fundamental=risk-adjusted Sharpe).
- DJ-055: Fine-tune Technical (GR=0.667 target) + Fundamental (accuracy target).
- DJ-056: mlx_lm in venvs/finetune/ (Python 3.13); adapters in data/adapters/.
- DJ-057: Fine-tuned serving via mlx_lm.server ports 1235/1236 alongside LM Studio 1234.
- DJ-058: Three-tier evaluation: HR/GR + accuracy + diversity (answers OQ-M01, OQ-M02).
- DJ-059: New package src/hifi/models/ (training_data.py, fine_tune.py).
- DJ-060: label-outcomes Makefile target for incremental weight updates.

## Phase 12 Wave 1 Results (2026-06-14)

Wave 1 tickets: P12-E0-T1, P12-E1-T1, P12-E1-T2, P12-E3-T1 — all complete.

- **P12-E0-T1 (compliance fix):** `technical_compliance_v2.jsonl` generated.
  200 examples (6 extracted from Phase 4+9 fixtures, 194 synthetic from OHLCV at
  quarterly intervals, 16 tickers). Ratio vs domain training: ~0.75% (up from 0.19%).
  Re-training (E0-T2) and evaluation (E0-T3) are Wave 2 tasks (hardware-bound).

- **P12-E1-T1 (FinancialGraph):** `src/hifi/knowledge/graph_store.py` complete.
  NetworkX DiGraph, typed CRUD for Company/Sector/MacroFactor nodes and
  BELONGS_TO/COMPETES_WITH/SENSITIVE_TO edges. save/load via node_link_data JSON.
  expand_query_tickers: 1-hop competitors + 2-hop sector peers.

- **P12-E1-T2 (build_financial_graph):** `src/hifi/knowledge/graph_construction.py` complete.
  DEFAULT_COMPETITORS (11 tickers) and DEFAULT_MACRO_SENSITIVITY (3 sectors).
  ticker_metadata override for deterministic tests without yfinance.
  _TICKER_FALLBACK for all 17 known Phase 10 tickers.

- **P12-E3-T1 (debate schemas):** `src/hifi/collective/debate.py` complete.
  DebateTurn (phase: challenge/response/revision), DebateTranscript (full Oxford record).
  identify_minority(): plurality with Hold tie-break.
  compute_vote_delta(): converged/diverged/unchanged relative to initial majority.
  EnsembleOutput.debate_transcript: Optional[DebateTranscript] = None added to schemas.py.

- **Tests:** 1071 passed (up from 1001 at Phase 11 close), 0 skipped, 0 lint errors.
  +70 new tests: 30 graph_store, 14 graph_construction, 23 debate_schemas, 3 compliance_v2.
- **networkx 3.6.1** added to pyproject.toml as direct dependency.

Wave 2 readiness: E0-T2 (retrain) can start with E1-T3 (GraphRetriever) and E3-T2
(run_debate_round) in parallel once hardware is available.

## Phase 12 LLM Evaluation Results (2026-06-14)

- **E2 (Graph-expanded retrieval Precision@k, OQ-K02) -- CORRECTED 2026-06-15:**
  Initial run (2026-06-14) produced P@5=0.000 due to 3 compounding bugs (path mismatch,
  section metadata, naming). All fixed. Corrected results:
  Document P@5: RAG=0.3100, graph-expanded=0.3100, delta=0.0000.
  Section P@5: RAG=0.1000, graph-expanded=0.1000, delta=0.0000.
  n_retrieved=5 for all 20 queries. Zero delta because graph expansion adds competitor
  tickers with no indexed documents (only AAPL/JPM/XOM ingested).
  **OQ-K02 ANSWERED NEGATIVE. DJ-016 decision: KEEP plain RAG. Phase 13 E3 NOT triggered.**
  Fixture: `tests/fixtures/baseline/phase12_graphrag_precision.json`

- **Phase 13 E0-T6 (verification baseline, 2023-03-31):**
  - Risk: HR=0.000, GR=1.000, alias_coverage=38.9% (5 of 8 claims unresolvable — volatility alias gaps)
  - Macro: HR=0.000, GR=0.000 (FRED data absent for 2023-03-31 — data gap, not model failure)
  - Sentiment: mean_SGR=0.167 (1/6 signals grounded) — below Phase 13 E1 gate (SGR≥0.500)
  Fixture: `tests/fixtures/baseline/phase13_verification_baseline.json`

- **E4 (2x2 factorial, OQ-D03) -- UPDATED 2026-06-15:**
  Condition A COMPLETE (30/30). OQ-D03: 11/30 = 36.7% non-unanimous.
  XOM = 100% disagreement (10/10 dates), AAPL = 0%, JPM = 10%.
  Condition B PARTIAL (12/30, PID 7978 died at B_JPM_2020-06-30).
  technical_v1 signal=None in ALL 12 B runs (single-agent, only fundamental_v1).
  fundamental_v1 shows systematic Buy bias: AAPL Hold->Buy, JPM Hold->Buy (all dates).
  OQ-M02 NOT ANSWERABLE (technical_v1 failure confounds B). C/D NOT STARTED.
  Factorial BLOCKED on technical_v2 deployment.

## Phase 12 Pre-Phase Decisions (DJ-061 to DJ-069)

Full rationale in `plans/PHASE_12_CONTEXT.md`.

- DJ-061: Fix technical_v1 compliance (0.19% ratio root cause); augment to >=200 compliance examples, retrain at 500 iters.
- DJ-062: GraphRAG library = NetworkX + LanceDB extension (no new venv, no new external service).
- DJ-063: Graph schema = tight scope (~12 companies, 3 macro nodes, ~40 edges from curated seed + yfinance metadata).
- DJ-064: Query expansion = 2-hop graph traversal -> expanded ticker filter for LanceDB dense search.
- DJ-065: Structured debate = Oxford 1-round, all 5 voting agents, challenge/response/revision phases.
- DJ-066: DebateTranscript schema in src/hifi/collective/debate.py; Dataset Family D population.
- DJ-067: 2x2 factorial (base/FT x no-debate/debate), 10 dates x 3 tickers = 120 runs.
- DJ-068: use_graphrag parameter (mutually exclusive with use_rag) in run_ensemble().
- DJ-069: Remaining agent fine-tuning is staged: Sentiment (same model, highest priority) in Phase 13 pending verification layer extension; Risk/Macro deferred (different architectures).

---

## Source Package Map

| Package | Phase | Key Files |
|---|---|---|
| hifi.config | 0 | config.py |
| hifi.data | 1 | market.py, macro.py, storage.py, edgar.py, schemas.py |
| hifi.engines | 2 | fundamental.py, technical.py, risk.py, macro.py |
| hifi.mcp | 2,7 | financial_server.py, knowledge_server.py |
| hifi.agents | 3-8 | lm_client.py, ensemble_runner.py, 5 agents, prompts/ |
| hifi.collective | 4,9,10,12 | voting.py, metrics.py, performance_store.py, labeler.py, debate.py |
| hifi.verification | 5 | extractor.py, verifier.py, metrics.py |
| hifi.observability | 6 | tracing.py |
| hifi.knowledge | 7,12 | document_ingestion.py, vector_store.py, retrieval.py, graph_store.py, graph_construction.py |
| hifi.analytics | 10 | tearsheet.py |
| hifi.models | 11 | training_data.py, fine_tune.py |

---

## Environment Reference

| Service/Tool | Status | Start Command | Address |
|---|---|---|---|
| LM Studio | Required for live runs | Manual (GUI) | http://localhost:1234/v1 |
| mlx_lm.server (technical) | Phase 11 | make finetune-serve | http://localhost:1235/v1 |
| mlx_lm.server (fundamental) | Phase 11 | make finetune-serve | http://localhost:1236/v1 |
| venvs/ta/ | Exists | scripts/setup_ta_venv.sh | Python 3.12 |
| venvs/finetune/ | Phase 11 | scripts/setup_finetune_venv.sh | Python 3.13 |
| LangFuse web | Broken on macOS | make langfuse-start | http://localhost:3000 |
| ClickHouse | Unhealthy (macOS) | see STATUS.md note | -- |

### ClickHouse Fix (when needed for LangFuse)
Add to docker/langfuse/docker-compose.yml under the clickhouse service:
  security_opt:
    - seccomp:unconfined

### mlx / mlx_lm Location
Installed in pyenv Python 3.13.12 -- NOT in the project uv venv.
Path: /Users/alberto/.pyenv/versions/3.13.12/lib/python3.13/site-packages/
Versions: mlx 0.31.1, mlx_lm 0.31.1
Phase 11 creates venvs/finetune/ to pin these versions.

---

## Key Metrics

| Metric | Value |
|---|---|
| Tests passing | 1809 (3 skipped, 0 lint errors) |
| DJ decisions | DJ-000 through DJ-110 |
| Technical Agent GR (Phase 5) | 0.667 (improvement target Phase 11) |
| Fundamental Agent GR (Phase 5) | 1.000 |
| Bootstrap accuracy: risk | 0.349 |
| Bootstrap accuracy: technical | 0.254 |
| Bootstrap accuracy: fundamental | 0.079 |
| Performance history records | 255 |
| mlx / mlx_lm version | 0.31.1 / 0.31.1 |

## Non-Negotiable Principles

- No emojis or icons anywhere
- No mocks -- recorded fixtures and deterministic synthetic generators only
- Every feature: unit + integration + holistic tests
- Interface-first development
- Scientific bitacora per phase
- Isolated environments (venvs/{name}/) for incompatible dependencies
- Fine-tuned model not deployed unless it demonstrably outperforms base (DJ-058)
