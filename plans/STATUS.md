# HiFi Project Status

**Last Updated:** 2026-06-09
**Current Phase:** Phase 0 COMPLETE, Phase 1 PLANNED (ready to implement)

---

## Quick Context for New Sessions

HiFi is a fully local multi-agent financial intelligence platform. Read these files in order to understand the project:

1. `doc/HIFI_DAVID.md` -- The ideal specification (what we aspire to build)
2. `doc/HIFI_PROTOCOL_V1.md` -- The execution plan (18 phases, which order)
3. `doc/HIFI_LEARNING_GUIDE.md` -- Learning tracker and David proximity matrix
4. `plans/PHASE_XX_PLAN.md` -- Epic/ticket plans per phase (detailed work breakdown)
5. `doc/bitacora/PHASE_XX_*.md` -- Scientific logbook per phase (narrative, insights, surprises)

---

## Phase Status

| Phase | Name | Status | Plan | Bitacora |
|---|---|---|---|---|
| 0 | Project Infrastructure | COMPLETE | plans/PHASE_00_PLAN.md | doc/bitacora/PHASE_00_INFRASTRUCTURE.md |
| 1 | Data Acquisition | PLANNED | plans/PHASE_01_PLAN.md | -- |
| 2 | Deterministic Financial Engine | NOT STARTED | -- | -- |
| 3 | First Agent (Baseline) | NOT STARTED | -- | -- |
| 4 | Second Agent (First Ensemble) | NOT STARTED | -- | -- |
| 5 | Verification Layer | NOT STARTED | -- | -- |
| 6 | Observability (LangFuse) | NOT STARTED | -- | -- |
| 7 | RAG Knowledge Systems | NOT STARTED | -- | -- |
| 8 | Full Agent Population | NOT STARTED | -- | -- |
| 9 | Collective Decision Engine | NOT STARTED | -- | -- |
| 10 | Evaluation & Backtesting | NOT STARTED | -- | -- |
| 11 | Fine-Tuning | NOT STARTED | -- | -- |
| 12 | GraphRAG | NOT STARTED | -- | -- |
| 13 | Advanced Features | NOT STARTED | -- | -- |
| 14 | Paper Trading | NOT STARTED | -- | -- |
| 15 | Containerization | NOT STARTED | -- | -- |
| 16 | Open Source Release | NOT STARTED | -- | -- |
| 17 | Capstone Deliverable | NOT STARTED | -- | -- |
| 18 | Publication | NOT STARTED | -- | -- |

---

## Current State of the Codebase

### What Exists

- **Project structure:** 12 source packages under src/hifi/
- **Configuration:** YAML loading with Pydantic validation (src/hifi/config/)
- **Tests:** 17 unit tests, all passing (tests/unit/test_config.py)
- **Synthetic data fixtures:** Deterministic OHLCV (GBM) and financials generators in tests/conftest.py
- **Dependencies:** pydantic, pyyaml, numpy (core); pytest, pytest-cov, ruff, mypy (dev)

### What Does Not Exist Yet

- No data acquisition code (Phase 1)
- No financial computation engines (Phase 2)
- No MCP servers (Phase 2)
- No agents (Phase 3)
- No RAG/knowledge systems (Phase 7)
- No observability (Phase 6)
- No verification (Phase 5)

---

## Tech Stack (Confirmed)

| Component | Choice | Decision ID |
|---|---|---|
| Language | Python >=3.11 (developing on 3.12.13) | -- |
| Package manager | uv 0.10.10 | DJ-006 |
| Validation | Pydantic 2.x | -- |
| Config format | YAML | -- |
| Testing | pytest + deterministic synthetic fixtures | -- |
| Linting/formatting | ruff | -- |
| Storage (initial) | Parquet | DJ-007 |
| Hardware | Apple M3 Ultra, arm64 | -- |

## Tech Stack (Pending Decisions)

| Component | Options | Decided In |
|---|---|---|
| Data acquisition | yfinance, fredapi | Phase 1 |
| Feature computation | pandas vs. polars | Phase 1 |
| MCP transport | stdio vs. SSE | Phase 2 |
| Local inference | Ollama vs. llama.cpp vs. MLX | Phase 3 |
| Agent orchestration | LangGraph vs. alternatives | Phase 3 |
| Vector store | Chroma vs. Qdrant vs. LanceDB | Phase 7 |
| Embedding model | nomic-embed vs. BGE vs. others | Phase 7 |
| Observability | LangFuse (self-hosted) | Phase 6 |

---

## Development Principles (Non-Negotiable)

- No emojis or icons in any output or file
- No mocks -- use recorded fixtures and deterministic synthetic generators
- Every feature has unit + integration tests; holistic tests per epic
- Atomic commits per epic; user makes final commit
- Interface-first development
- Scientific bitacora per phase
- JIRA-style epic/ticket planning with minimal cross-epic dependencies
- Hard constraints (tests) are deterministic; soft constraints (design) are adaptive

---

## Key Numbers (Updated Per Phase)

| Metric | Value |
|---|---|
| Total tests | 17 |
| Tests passing | 17 |
| Test duration | 0.02s |
| Source packages | 12 |
| Lines of production code | ~100 |
| Lines of test code | ~170 |
| Lint errors | 0 |
| David proximity | 0/53 sections (0%) |
| David coverage | 0/53 sections (0%) |
