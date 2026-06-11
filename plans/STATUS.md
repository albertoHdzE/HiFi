# HiFi Project Status

**Last Updated:** 2026-06-11
**Current Phase:** Phase 6 COMPLETE

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
| 1 | Data Acquisition | COMPLETE | plans/PHASE_01_PLAN.md | doc/bitacora/PHASE_01_DATA_ACQUISITION.md |
| 2 | Deterministic Financial Engine | COMPLETE | plans/PHASE_02_PLAN.md | doc/bitacora/PHASE_02_DETERMINISTIC_ENGINE.md |
| 3 | First Agent (Baseline) | COMPLETE | plans/PHASE_03_PLAN.md | doc/bitacora/PHASE_03_FIRST_AGENT.md |
| 4 | Second Agent (First Ensemble) | COMPLETE | plans/PHASE_04_PLAN.md | doc/bitacora/PHASE_04_SECOND_AGENT.md |
| 5 | Verification Layer | COMPLETE | plans/PHASE_05_PLAN.md | doc/bitacora/PHASE_05_VERIFICATION.md |
| 6 | Observability (LangFuse) | COMPLETE | plans/PHASE_06_PLAN.md | doc/bitacora/PHASE_06_OBSERVABILITY.md |
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

- **Project structure:** 13 source packages under src/hifi/
- **Configuration:** YAML loading with Pydantic validation (src/hifi/config/)
- **Data acquisition layer (Phase 1):**
  - `src/hifi/data/schemas.py` -- Pydantic schemas: OHLCVBar, OHLCVDataset, FundamentalsSnapshot, MacroIndicator, MacroDataset, ProvenanceRecord
  - `src/hifi/data/market.py` -- MarketDataFetcher, FundamentalsFetcher (yfinance)
  - `src/hifi/data/macro.py` -- MacroDataFetcher + forward_fill_to_daily (FRED/fredapi)
  - `src/hifi/data/storage.py` -- Parquet read/write with embedded schema metadata (pyarrow)
  - `src/hifi/data/quality.py` -- DataQualityChecker, QualityReport (completeness, gaps, price sanity)
  - `src/hifi/data/versioning.py` -- content_hash, DatasetRegistry (JSON-backed catalog)
  - `scripts/record_fixtures.py` -- one-time fixture recorder for test replay
  - `tests/fixtures/market/` -- recorded yfinance responses (AAPL, JPM, XOM, Q1 2023)
  - `tests/fixtures/macro/` -- recorded FRED XML responses (FEDFUNDS, CPIAUCSL, 2022)
- **Deterministic financial engine layer (Phase 2):**
  - `src/hifi/engines/types.py` -- Typed result models: FinancialRatioResult, GrowthMetricsResult, TechnicalIndicatorsResult, RiskMetricsResult, ValuationResult, MacroSnapshotResult; _NanToNoneBase validator
  - `src/hifi/engines/fundamental.py` -- compute_financial_ratios, compute_growth_metrics, compute_valuation_context (pure functions; P/E, P/B, P/S, ROE, ROA, debt/equity, trailing P/E percentile)
  - `src/hifi/engines/technical.py` -- compute_technical_indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR; custom numpy/pandas; primary sources cited; DJ-010)
  - `src/hifi/engines/risk.py` -- compute_risk_metrics (QuantStats adapter: hist vol 20/60/252d, beta, max drawdown, Sharpe, VaR 95%; DJ-011)
  - `src/hifi/engines/macro.py` -- compute_macro_snapshot (cross-section of macro indicators: FEDFUNDS, CPI YoY, UNRATE, GS10, GS2, VIXCLS, GDP growth)
  - `src/hifi/mcp/financial_server.py` -- FastMCP stdio server exposing 6 tools: get_financial_ratios, get_growth_metrics, get_technical_indicators, get_risk_metrics, get_valuation_context, get_macro_snapshot; call_id Phase 5 audit hook (DJ-009)
  - `venvs/ta/requirements.txt` -- pinned pandas 1.5.3 + pandas-ta 0.3.14b0 scaffold for Phase 8+ isolated TA venv (DJ-010)
  - `scripts/setup_ta_venv.sh` -- bootstrap script for venvs/ta/ virtual environment
- **Tests:** 377 tests passing, 8 skipped (baseline fixture tests await live LLM run); 0 lint errors
- **Synthetic data fixtures:** Deterministic OHLCV (GBM) and financials generators in tests/conftest.py; read_raw_ohlcv_fixture() for Phase 1 raw parquet replay
- **Agent layer (Phase 3):**
  - `src/hifi/agents/lm_client.py` -- make_llm() ChatOpenAI wrapper pointing at LM Studio (HIFI_LM_STUDIO_URL); DJ-013, DJ-014
  - `src/hifi/agents/schemas.py` -- AgentSignal, FundamentalAnalysis, TechnicalAnalysis; P3-E2, P4-E2
  - `src/hifi/agents/mcp_client.py` -- call_tool() synchronous subprocess MCP client; P3-E1
  - `src/hifi/agents/fundamental_agent.py` -- LangGraph graph (4 nodes: load_snapshot, call_mcp_tools, generate_analysis, parse_output); run_analysis() entrypoint; P3-E3, P3-E4
  - `src/hifi/agents/technical_agent.py` -- LangGraph graph (3 nodes: call_mcp_tools, generate_analysis, parse_output); run_technical_analysis() entrypoint; information-restricted to price-derived data; P4-E1
  - `src/hifi/agents/ensemble_runner.py` -- run_ensemble() sequential runner for both agents; P4-E4
  - `src/hifi/agents/baseline_metrics.py` -- compute_metrics(), count_hallucinated_numbers(), data_gap_acknowledged(), call_id_coverage(); P3-E4, P3-E5
  - `src/hifi/agents/prompts/fundamental_v1.md` -- versioned system+user prompt template; P3-E3
  - `src/hifi/agents/prompts/technical_v1.md` -- versioned system+user prompt template with indicator interpretation framework; P4-E1
  - `scripts/run_phase3_baseline.py` -- one-time baseline runner for AAPL/JPM/XOM; saves phase3_baseline.json; P3-E5
  - `scripts/run_phase4_ensemble.py` -- one-time ensemble runner for AAPL/JPM/XOM; saves phase4_ensemble.json; P4-E5
  - `tests/fixtures/baseline/` -- baseline fixture directory (generated after live LLM run); P3-E5, P4-E5
- **Collective decision engine (Phase 4):**
  - `src/hifi/collective/__init__.py` -- package stub
  - `src/hifi/collective/schemas.py` -- EnsembleDecision (confidence-weighted vote output + diversity metrics), EnsembleOutput (full 2-agent envelope); P4-E2
  - `src/hifi/collective/voting.py` -- confidence_weighted_vote() implementing David §12.2.2; P4-E3
  - `src/hifi/collective/metrics.py` -- disagreement_entropy() §5.6.1, opinion_dispersion() §5.6.2, pairwise_diversity() §5.6.5, compute_ensemble_metrics(); P4-E3
- **Verification layer (Phase 5):**
  - `src/hifi/verification/schemas.py` -- NumericalClaim, VerificationResult, Contradiction, AgentVerificationReport (auto-computes HR/GR via model_validator), EnsembleVerificationReport; P5-E1
  - `src/hifi/verification/extractor.py` -- FIELD_ALIAS_TABLE (160+ entries); _resolve_alias() progressive suffix stripping; extract_numerical_claims(); P5-E2
  - `src/hifi/verification/verifier.py` -- verify_claim() (1%/0.01 dual tolerance), verify_agent(), detect_contradictions(), verify_ensemble(); P5-E3
  - `src/hifi/verification/metrics.py` -- compute_verification_metrics() with alias_table_coverage; P5-E6
  - `scripts/run_phase5_verification.py` -- baseline runner (no LLM required); saves phase5_verification.json; P5-E5
  - `tests/fixtures/baseline/phase5_verification.json` -- Phase 5 baseline (AAPL/JPM/XOM, 2023-03-31)
- **Baseline results (Phase 5, 2026-06-10):**
  - Fundamental: HR=0.000, GR=1.000, alias_table_coverage=0.917
  - Technical: HR=0.067, GR=0.667, alias_table_coverage=1.000
  - 0 cross-agent contradictions; 0 triggered_by_disagreement
  - DJ-019 confirmed: regex+alias sufficient (both agents coverage >= 0.90)
- **Dependencies (production):** pydantic, pyyaml, numpy, pandas, pyarrow, yfinance, fredapi, mcp, quantstats, openai, langchain, langchain-openai, langgraph
- **Dependencies (dev):** pytest, pytest-cov, ruff, mypy, vcrpy

### What Does Not Exist Yet

- No RAG/knowledge systems (Phase 7)
- No full TA indicators server in venvs/ta/ (Phase 8+ trigger: >6 indicators or >50 tickers)
- No multi-agent population (Phase 8+)

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
| Data acquisition | yfinance + fredapi | DJ-008 |
| Intermediate computation | pandas (Phase 1; polars deferred) | DJ-008 |
| Parquet engine | pyarrow | DJ-008 |
| Test fixture replay | responses library | DJ-008 |
| MCP framework | FastMCP (mcp library) | DJ-009 |
| MCP transport | stdio (Phase 2-14); SSE/HTTP at Phase 15+ | DJ-009 |
| Risk metrics | QuantStats 0.0.62+ | DJ-011 |
| Technical indicators (Phase 2) | Custom numpy/pandas (6 indicators) | DJ-010 |
| Technical indicators (Phase 8+) | venvs/ta/ scaffold: pandas-ta 0.3.14b0 + pandas 1.5.3 pinned | DJ-010 |
| Hardware | Apple M3 Ultra, arm64 | -- |

## Tech Stack (Pending Decisions)

| Component | Options | Decided In |
|---|---|---|
| Local inference | LM Studio (DJ-013) | Phase 3 DONE |
| Baseline model | qwen2.5-coder-32b-instruct-mlx (DJ-014) | Phase 3 DONE |
| Agent orchestration | LangGraph (DJ-015) | Phase 3 DONE |
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
| Total tests | 582 (582 passing, 0 skipped) |
| Tests passing | 582 |
| Source packages | 17 (hifi.observability added in Phase 6) |
| Lines of production code | ~6800 |
| Lines of test code | ~9800 |
| Lint errors | 0 |
| Baseline HR (fundamental) | 0.000 (Phase 5 baseline, 2026-06-10) |
| Baseline HR (technical) | 0.067 (Phase 5 baseline, 2026-06-10) |
| Alias table coverage | fundamental=0.917, technical=1.000 |
| David sections addressed | 9/53 (~17%): §4.1, §4.3, §4.5, §6.2, §7.1 (substantially); §7.2, §8.2, §8.3, §10.1, §10.2, §10.3, §12.2, §5.6.1, §5.6.2, §5.6.5 (partial) |
