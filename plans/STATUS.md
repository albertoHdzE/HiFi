# HiFi Project Status

**Last Updated:** 2026-06-12
**Current Phase:** Phase 12 (Phase 11 infrastructure complete; training pending hardware)

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
| 12 | GraphRAG + Structured Debate | NOT STARTED | -- | -- |
| 13 | Advanced Features | NOT STARTED | -- | -- |
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

---

## Source Package Map

| Package | Phase | Key Files |
|---|---|---|
| hifi.config | 0 | config.py |
| hifi.data | 1 | market.py, macro.py, storage.py, edgar.py, schemas.py |
| hifi.engines | 2 | fundamental.py, technical.py, risk.py, macro.py |
| hifi.mcp | 2,7 | financial_server.py, knowledge_server.py |
| hifi.agents | 3-8 | lm_client.py, ensemble_runner.py, 5 agents, prompts/ |
| hifi.collective | 4,9,10 | voting.py, metrics.py, performance_store.py, labeler.py |
| hifi.verification | 5 | extractor.py, verifier.py, metrics.py |
| hifi.observability | 6 | tracing.py |
| hifi.knowledge | 7 | document_ingestion.py, vector_store.py, retrieval.py |
| hifi.analytics | 10 | tearsheet.py |
| hifi.models | 11 | (planned) training_data.py, fine_tune.py |

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
| Tests passing | 997 (4 skipped, 0 lint) |
| DJ decisions | DJ-000 through DJ-060 |
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
