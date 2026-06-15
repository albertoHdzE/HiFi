# HiFi Project Status

**Last Updated:** 2026-06-15
**Current Phase:** Phase 12.1 (IN PROGRESS — completion and correction)

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
| 12 | GraphRAG + Structured Debate | IN PROGRESS (infra complete, eval partial) | plans/PHASE_12_PLAN.md | doc/bitacora/PHASE_12_GRAPHRAG_DEBATE.md |
| 12.1 | Completion and Correction | IN PROGRESS | plans/PHASE_12.1_PLAN.md | doc/bitacora/PHASE_12.1_COMPLETION.md |
| 13 | Verification Completeness, Sentiment Intelligence, System Resilience | IN PROGRESS (E0 complete) | plans/PHASE_13_PLAN.md | -- |
| 14 | Paper Trading | NOT STARTED | -- | -- |
| 15 | Containerization | NOT STARTED | -- | -- |
| 16 | Open Source Release | NOT STARTED | -- | -- |
| 17 | Capstone Deliverable | NOT STARTED | -- | -- |
| 18 | Publication | NOT STARTED | -- | -- |

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
| Tests passing | 1197 (0 skipped, 0 lint errors) |
| DJ decisions | DJ-000 through DJ-084 (DJ-081: Phase 12.1 sub-phase; DJ-082: technical_v2 params; DJ-083: full factorial re-run; DJ-084: Gemma 4 12B variant) |
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
