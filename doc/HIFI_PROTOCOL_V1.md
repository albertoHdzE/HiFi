# HiFi Execution Protocol v1.0

## A Realistic, Incremental Path from Zero to the David

**Author:** Alberto Espinosa

**Status:** Active Execution Plan

**Derived from:** HIFI_DAVID.md (The David — aspirational reference)

**Companion:** HIFI_LEARNING_GUIDE.md (learning tracker + David proximity)

---

## Governing Principles

### 1. Every layer earns its place with a measurement

No component is added because the David says so. A component is added when the previous layer is working and measured, and the new layer has a clear hypothesis about what it will improve. The measurement comes before the next layer, not after.

### 2. Vertical before horizontal

Build a thin end-to-end path first (one stock, one agent, one decision). Then widen. Do not build all of Layer 1 before touching Layer 2. A narrow slice through all layers teaches more than a broad implementation of one layer.

### 3. The simplest version that produces evidence

At every phase, ask: what is the minimum I must build to generate a measurable result? Build that. Measure. Then decide what comes next. The David describes the destination. This protocol describes the walk.

### 4. Educative journal as you go

Every phase produces not just code and data, but documented reasoning: what was tried, what worked, what failed, what was learned. These notes become thesis material, blog posts, and interview answers.

### 5. David reference at every phase

Each phase explicitly lists which David sections it addresses and which Learning Guide topics it exercises. After each phase, update the David Conformance Matrix and the Learning Readiness levels.

---

## Phase Overview

```
Phase 0:  Project Infrastructure
Phase 1:  Data Acquisition — First Vertical Slice
Phase 2:  Deterministic Financial Engine — First MCP Server
Phase 3:  First Agent — Baseline
Phase 4:  Second Agent — First Ensemble
Phase 5:  Verification Layer
Phase 6:  Observability — LangFuse
Phase 7:  Knowledge Systems — RAG
Phase 8:  Full Agent Population
Phase 9:  Collective Decision Engine
Phase 10: Evaluation Framework & Backtesting
Phase 11: Fine-Tuning
Phase 12: Knowledge Systems — GraphRAG
Phase 13: Advanced Features (Agent Memory, Synthetic Scenarios, Drift)
Phase 14: Paper Trading
Phase 15: Containerization & Deployment
Phase 16: Open Source Release
Phase 17: Capstone Deliverable
Phase 18: Publication Preparation (Post-Graduation)
```

**Critical path for capstone:** Phases 0–10 + 14 + 17 form the minimum viable thesis. Everything else strengthens the David but can be deferred if time requires it.

---

## Phase 0: Project Infrastructure

### Objective

Set up the development environment, repository structure, tooling, and conventions that every subsequent phase depends on.

### David Sections

- §4.5 Reproducibility
- §4.6 Modularity
- §7.10 Experiment Registry (foundation)

### Learning Guide Topics

- 10.1 Systems Design
- 10.2 Data Engineering (foundations)
- 6.3 Deployment & Containerization (foundations)

### Deliverables

**Repository structure:**

```
HiFi/
├── doc/                        # Project documents (David, Protocol, Learning Guide)
├── src/
│   ├── data/                   # Data acquisition and engineering
│   ├── engines/                # Deterministic financial computation
│   ├── mcp/                    # MCP server implementations
│   ├── knowledge/              # RAG, GraphRAG, knowledge graphs
│   ├── models/                 # Model management, fine-tuning
│   ├── agents/                 # Agent definitions and orchestration
│   ├── collective/             # Aggregation and collective decision
│   ├── verification/           # Claim extraction and verification
│   ├── observability/          # LangFuse integration, metrics
│   ├── execution/              # Paper trading, order management
│   └── evaluation/             # Backtesting, metrics, baselines
├── data/
│   ├── raw/                    # Raw ingested data (gitignored)
│   ├── processed/              # Cleaned and transformed data
│   ├── features/               # Feature datasets
│   ├── reference_strategies/   # Labelled reference datasets
│   └── evaluation/             # Fixed evaluation benchmarks
├── configs/                    # Configuration files
├── notebooks/                  # Exploration and analysis notebooks
├── tests/                      # Test suite
├── docker/                     # Dockerfiles and compose
└── scripts/                    # Utility scripts
```

**Environment:**

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Ecosystem breadth for ML/AI/finance |
| Package manager | uv or poetry | Reproducible dependency management |
| Formatting | ruff | Fast, comprehensive |
| Type checking | mypy (gradual) | Catch errors early, don't block progress |
| Testing | pytest | Standard |
| Git workflow | Feature branches, conventional commits | Clean history |
| Configuration | YAML + environment variables | Readable, overridable |

**Conventions:**

- Every function that computes a financial quantity is a pure function (same inputs → same outputs)
- Every experiment has a unique ID and records its configuration
- Every data transformation is logged with input/output hashes
- Configuration is never hardcoded

### Success Criteria

- [ ] Repository structure exists and is navigable
- [ ] Python environment is reproducible (lock file)
- [ ] A trivial test passes (`pytest` runs)
- [ ] Configuration loading works
- [ ] `.gitignore` properly excludes data/, model weights, secrets
- [ ] README exists with setup instructions

### Decision Journal Entry

**DJ-006: Python environment manager**
- Decision: [to be decided — uv vs. poetry vs. conda]
- Evaluate: dependency resolution speed, reproducibility, Apple Silicon compatibility
- Record the choice and why

---

## Phase 1: Data Acquisition — First Vertical Slice

### Objective

Ingest market data for a small universe of stocks. Not all data sources. Not all stocks. Just enough to feed one agent for one analysis.

### David Sections

- §7.1 Data Acquisition Layer
- §8.2 Dataset Family A (Market Observations) — partial

### Learning Guide Topics

- 10.2 Data Engineering
- 7.1 Quantitative Analysis (foundations — understanding what data is needed)

### Scope

**Initial universe:** 10 stocks from diverse sectors

| Stock | Sector | Rationale |
|---|---|---|
| AAPL | Technology | High-coverage baseline |
| NVDA | Technology/Semiconductors | High volatility, narrative-driven |
| JPM | Financials | Rate-sensitive |
| JNJ | Healthcare | Defensive |
| XOM | Energy | Commodity-sensitive |
| AMZN | Consumer/Technology | Growth narrative |
| PG | Consumer Staples | Defensive |
| TSLA | Consumer/Technology | High sentiment-driven |
| BRK-B | Financials/Conglomerate | Value investing proxy |
| META | Technology/Communication | Turnaround narrative |

**Data sources for Phase 1:**

| Source | Data | Library |
|---|---|---|
| Yahoo Finance | OHLCV, basic fundamentals | yfinance |
| FRED | Fed Funds Rate, CPI, Unemployment, VIX | fredapi |

**Period:** 2015-01-01 to present (10 years covers multiple regimes)

### Deliverables

- `src/data/market_data.py` — OHLCV acquisition and storage
- `src/data/fundamentals.py` — Basic financial statements acquisition
- `src/data/macro.py` — Macroeconomic indicators acquisition
- `data/raw/` — Downloaded raw data (Parquet format)
- Data quality report: completeness, gaps, date coverage per stock

### Success Criteria

- [ ] 10 stocks × 10 years of daily OHLCV data downloaded and stored
- [ ] Basic fundamentals (quarterly) for all 10 stocks
- [ ] Key macro indicators (monthly) for the full period
- [ ] Data quality report shows >99% completeness for OHLCV
- [ ] Data is stored in Parquet with consistent schema
- [ ] Provenance metadata recorded (download date, source, parameters)
- [ ] Data can be reloaded identically from stored files

### Open Questions to Resolve

- OQ-D01 (partial): What is the actual data quality from yfinance for our 10 stocks?
- DJ-007: Storage format — Parquet vs. DuckDB vs. SQLite for Phase 1

### Notes for Educative Journal

Document: What data is easy to get? What is surprisingly hard? What quality issues did you find? What corporate actions cause problems? These are the "production scars" questions interviewers ask.

---

## Phase 2: Deterministic Financial Engine — First MCP Server

### Objective

Build the first MCP server that computes financial metrics deterministically. This is the foundation of the deterministic-first principle. Agents will consume these computations rather than generating numbers themselves.

### David Sections

- §4.1 Deterministic-First Principle
- §6.2 MCP as the Nervous System
- §7.1 Deterministic Financial Layer (partial)

### Learning Guide Topics

- 9.1 Model Context Protocol
- 9.2 Tool Design Principles
- 7.1 Quantitative Analysis

### Deliverables

**MCP Server: `hifi-financial-calculator`**

Tools exposed:

| Tool | Input | Output |
|---|---|---|
| `get_financial_ratios` | ticker, date | P/E, P/B, P/S, EV/EBITDA, ROE, ROA, debt/equity, current ratio |
| `get_growth_metrics` | ticker, date | Revenue growth YoY, earnings growth YoY, margin trends |
| `get_technical_indicators` | ticker, date, window | SMA, EMA, RSI, MACD, Bollinger Bands, ATR |
| `get_risk_metrics` | ticker, date, window | Historical volatility, beta, max drawdown, Sharpe |
| `get_valuation_context` | ticker, date | Current valuation vs. 5-year range, vs. sector median |
| `get_macro_snapshot` | date | Fed rate, CPI, VIX, yield curve slope, unemployment |

**Implementation:**

- `src/engines/financial_calculator.py` — Pure computation functions
- `src/mcp/financial_server.py` — MCP server wrapping the calculator
- Tests for every computation against known values (e.g., verify P/E calculation against a manually computed example)

### Success Criteria

- [ ] MCP server starts and responds to tool discovery
- [ ] Each tool returns correct, verified results for test cases
- [ ] Computations are pure functions (same inputs → same outputs, always)
- [ ] All tools have clear JSON schemas with parameter descriptions
- [ ] Error handling: graceful response when data is missing or invalid
- [ ] Latency: each tool responds in <100ms for a single stock query
- [ ] At least 5 computations verified against manually computed values

### Design Decision to Record

**DJ-008: MCP transport** — stdio vs. SSE vs. HTTP for local server communication. For a single-machine system, stdio is simplest. Record the choice.

**DJ-009: Computation library** — pandas vs. polars vs. numpy for financial calculations. Measure performance on the 10-stock universe. Record.

### Notes for Educative Journal

Document: What is MCP in practice (not marketing)? How does tool schema design affect LLM usage? What happens when a tool returns an error — how does the agent handle it? This is the core of the deterministic-first philosophy. Every number here is auditable.

---

## Phase 3: First Agent — Baseline

### Objective

Deploy a single financial analysis agent that consumes MCP tools and produces a structured recommendation. This establishes the baseline: what can ONE agent do, with NO ensemble, NO fine-tuning, NO RAG?

This baseline is essential. Without it, we cannot attribute any later improvement to the architecture.

### David Sections

- §10.2 Agent Specifications (Fundamental Agent only)
- §10.1 Agent Design Philosophy

### Learning Guide Topics

- 2.1 Model Selection & Evaluation
- 2.2 Prompt Engineering
- 2.3 Hallucination Detection (baseline measurement)
- 3.1 Agent Architecture Fundamentals

### Deliverables

**Single agent: Fundamental Analyst**

- Model: Select one local model (start with Qwen 2.5 7B or Llama 3.x 8B, quantized)
- Information access: Financial Calculator MCP server only
- Prompt: Structured system prompt defining role, available tools, output format

**Structured output format:**

```json
{
  "ticker": "AAPL",
  "date": "2024-01-15",
  "decision": "BUY",
  "confidence": 0.72,
  "rationale": {
    "bull_case": "...",
    "bear_case": "...",
    "key_metrics": {
      "metric_name": "value cited",
      "metric_source": "MCP tool call ID"
    }
  },
  "key_risk": "..."
}
```

**Implementation:**

- `src/agents/base_agent.py` — Base agent class (tool calling, output parsing)
- `src/agents/fundamental_agent.py` — Fundamental agent specialization
- `src/agents/prompts/fundamental.md` — Prompt template (versioned)
- Model serving via Ollama (simplest local inference)

**Baseline evaluation:**

Run the agent on all 10 stocks for the EVAL-2022-2023 period (24 months of monthly decisions = 240 decisions).

Measure:
- Directional accuracy (did BUY/SELL align with actual 20-day forward return direction?)
- Confidence calibration (is 80% confidence right ~80% of the time?)
- Hallucination rate (how many numerical claims don't match MCP output?)
- Latency per analysis
- Output format compliance (does it produce valid JSON?)

### Success Criteria

- [ ] Agent loads a local model via Ollama
- [ ] Agent calls MCP tools correctly (verified through logs)
- [ ] Agent produces structured JSON output ≥90% of the time
- [ ] Baseline directional accuracy measured and recorded (even if low)
- [ ] Baseline hallucination rate measured and recorded
- [ ] Baseline confidence calibration measured and recorded
- [ ] All results stored in a structured format for future comparison
- [ ] The analysis of one stock completes in <60 seconds

### What This Phase Proves

This phase establishes whether the core loop works at all: data → MCP tools → agent → structured output. If a single agent cannot produce coherent analysis using deterministic tools, adding more agents will not help. Fix the foundation before expanding.

### Open Questions to Resolve

- OQ-M01 (partial): How well does a base model (no fine-tuning) perform on financial analysis?
- OQ-AG03 (baseline): How calibrated are LLM confidence estimates out of the box?
- DJ-010: Which model for the first agent? Evaluate Qwen 2.5 7B vs. Llama 3.1 8B on 5 sample analyses. Choose based on: output quality, tool-calling reliability, structured output compliance. Record.

### Notes for Educative Journal

Document: What does the agent get right? What does it get wrong? What kind of hallucinations appear? How does the prompt affect output quality? This is the raw material for understanding LLM capabilities and limitations in finance. These observations will be worth more than the architecture diagrams.

---

## Phase 4: Second Agent — First Ensemble

### Objective

Add a second agent with a DIFFERENT model and DIFFERENT information focus. Implement the simplest aggregation (majority vote of 2 is just agreement/disagreement, so use confidence-weighted). Measure whether two agents outperform one.

### David Sections

- §10.2 Agent Specifications (Valuation Agent or Technical Agent)
- §10.3 Diversity Requirements
- §12.2.2 Confidence-Weighted Voting
- §5.3 Ensemble Learning

### Learning Guide Topics

- 3.3 Collective Intelligence & Aggregation
- 5.2 Ensemble Learning (bias-variance in practice)

### Deliverables

**Second agent:** Technical Analyst OR Valuation Analyst (whichever provides maximum diversity from the Fundamental Agent)

- Different model family from Agent 1 (e.g., if Agent 1 is Qwen, Agent 2 is Llama or Gemma)
- Different information access (e.g., Technical Agent sees only technical indicators, not fundamentals)
- Different prompt structure

**Aggregation:**

- `src/collective/voting.py` — Confidence-weighted voting for 2+ agents
- Record: both individual decisions AND the collective decision

**Evaluation:**

Run both agents + ensemble on the same EVAL-2022-2023 period. Compare:

| Metric | Agent 1 | Agent 2 | Ensemble |
|---|---|---|---|
| Directional accuracy | ? | ? | ? |
| Confidence calibration | ? | ? | ? |
| Hallucination rate | ? | ? | ? |
| Agreement rate | — | — | ? |

### Success Criteria

- [ ] Two agents produce independent analyses (no shared state during reasoning)
- [ ] Agents use different model families
- [ ] Agents have different information access
- [ ] Confidence-weighted voting produces a collective decision
- [ ] Ensemble performance measured against both individuals
- [ ] Agreement rate measured (how often do agents agree?)
- [ ] Pairwise correlation of decisions measured (OQ-AG04 partial answer)

### What This Phase Proves

If the ensemble outperforms both individuals: the architecture has value, proceed.
If the ensemble matches the better individual: diversity may be insufficient, investigate.
If the ensemble underperforms: something is wrong — correlated errors, bad aggregation, or agents are not truly independent. Fix before adding more agents.

### Decision Journal Entry

**DJ-011: Second agent selection** — Which specialization (Valuation vs. Technical) provides maximum diversity from the Fundamental Agent? Measure decision correlation with each candidate. Choose the one with lower correlation. Record.

---

## Phase 5: Verification Layer

### Objective

Build the claim extraction and verification pipeline. Measure hallucination rate systematically. This is one of HiFi's central differentiators and one of the most publishable components.

### David Sections

- §13 Verification and Hallucination Control (full section)
- §4.3 Verifiability

### Learning Guide Topics

- 2.3 Hallucination Detection & Mitigation (deep dive)

### Deliverables

- `src/verification/claim_extractor.py` — Extract factual claims from agent outputs
- `src/verification/claim_classifier.py` — Classify claims as Objective vs. Interpretive
- `src/verification/verifier.py` — Check objective claims against MCP server results
- `src/verification/metrics.py` — Compute HR, GR, CCR (hallucination rate, grounding rate, cross-agent contradiction rate)

**Approach for claim extraction:**

Prefer structured output templates (agent output already in JSON with explicit `key_metrics` section) over free-text extraction. The structured format from Phase 3 was designed specifically to make verification straightforward.

For any free-text rationale, use pattern matching or a small local model to extract numerical claims.

### Success Criteria

- [ ] Objective claims are extracted from agent outputs
- [ ] Each claim is verified against MCP server
- [ ] Hallucination rate (HR) is computed per agent and overall
- [ ] Grounding rate (GR) is computed
- [ ] Cross-agent contradiction rate (CCR) is computed (if 2+ agents)
- [ ] Before/after comparison: does the verification layer reduce hallucination in system output?
- [ ] Contradicted claims are flagged and excluded from the final recommendation

### What This Phase Proves

That deterministic verification measurably reduces hallucination rate. This is a concrete, publishable result: "Verification against deterministic financial engines reduced hallucination rate from X% to Y%."

---

## Phase 6: Observability — LangFuse

### Objective

Instrument the existing pipeline with LangFuse tracing. From this point forward, every agent call, tool call, and decision is traced and auditable.

### David Sections

- §14 Observability (full section — foundation)
- §4.4 Observability principle

### Learning Guide Topics

- 6.1 LLM Observability
- 6.4 Experiment Tracking (foundations)

### Deliverables

- LangFuse self-hosted instance (Docker)
- `src/observability/tracing.py` — LangFuse integration for agent calls, MCP calls
- Traces for: full analysis pipeline (data → tools → agent → verification → decision)
- Dashboard showing: latency, token usage, hallucination rate over time

### Success Criteria

- [ ] LangFuse running locally (Docker)
- [ ] Every agent inference produces a trace
- [ ] Every MCP tool call is a span within the trace
- [ ] Verification results are logged as scores on traces
- [ ] Dashboard accessible showing system metrics
- [ ] Can trace any decision back to its inputs (full auditability)

### What This Phase Proves

That the system is fully auditable. Every recommendation can be traced back through: collective decision → individual agent votes → agent reasoning → MCP tool calls → raw data. This traceability is what "high-fidelity" means in practice.

---

## Phase 7: Knowledge Systems — RAG

### Objective

Add retrieval-augmented generation so agents can access information beyond what is in the MCP tools: earnings call transcripts, SEC filings, financial news. Start with basic RAG. Measure whether it improves agent analysis quality.

### David Sections

- §11.2 RAG
- §7.3 Knowledge Layer (partial)

### Learning Guide Topics

- 1.1 Chunking Strategies
- 1.2 Embedding Models
- 1.3 Vector Databases
- 1.4 Retrieval Strategies

### Deliverables

- `src/hifi/knowledge/document_ingestion.py` — Parse and chunk SEC EDGAR filings (10-K, 10-Q, 8-K)
- `src/hifi/knowledge/embeddings.py` — Embed chunks using nomic-embed-text-v1.5 via LM Studio (DJ-027)
- `src/hifi/knowledge/vector_store.py` — Store and query embeddings in LanceDB (DJ-026)
- `src/hifi/knowledge/retrieval.py` — Retrieve relevant context for agent queries; measure Precision@5
- `src/hifi/mcp/knowledge_server.py` — MCP server exposing retrieval as a tool (DJ-029)
- `scripts/record_sec_fixtures.py` — One-time SEC EDGAR fixture recorder for test replay (consistent with DJ-008 fixture philosophy)

**Document types for Phase 7:**

| Source | Format | Priority | Status |
|---|---|---|---|
| SEC 10-K annual reports | HTML/Text | High — authoritative annual financials | Phase 7 (DJ-028) |
| SEC 10-Q quarterly reports | HTML/Text | High — quarterly period data | Phase 7 (DJ-028) |
| SEC 8-K earnings releases | HTML/Text | High — earnings announcements | Phase 7 (DJ-028) |
| Earnings call transcripts | Text | High — rich qualitative information | Deferred Phase 8 (DJ-028) |
| Financial news | Text | Medium — recency and sentiment | Deferred Phase 8 |

**Chunking experiments:**

Test at least 3 chunking configurations:

| Config | Chunk Size | Overlap | Method |
|---|---|---|---|
| A | 512 tokens | 10% | Fixed-size |
| B | 1024 tokens | 20% | Fixed-size |
| C | Variable | N/A | Semantic (paragraph-based) |

Measure retrieval precision@5 for each on a set of 20 manually crafted test queries.

### Success Criteria

- [ ] At least 2 document types ingested, chunked, and embedded
- [ ] Vector store operational (local)
- [ ] Retrieval latency < 500ms for a query
- [ ] Retrieval precision@5 measured on test queries
- [ ] At least 3 chunking strategies compared with measured results
- [ ] MCP knowledge server responds to agent tool calls
- [ ] Agent analysis quality compared with and without RAG (before/after on same stocks)
- [ ] Embedding model selected with documented rationale

### Open Questions Resolved

- OQ-K01: Optimal chunking strategy (empirical answer from experiment)
- OQ-M03: Best embedding model for financial text (empirical answer)

### Pre-Phase Decisions (Resolved 2026-06-11)

The following decisions were made before Phase 7 planning to ensure architectural consistency:

- **DJ-026 (Vector store):** LanceDB — Arrow-native columnar format consistent with Parquet data layer; embedded mode; no server process. See DAVID.md §17.
- **DJ-027 (Embedding model baseline):** nomic-embed-text-v1.5 — already in LM Studio; 8192-token context; matryoshka dimensionality. OQ-M03 to be answered empirically. See DAVID.md §17.
- **DJ-028 (Document sources):** SEC EDGAR (10-K, 10-Q, 8-K) for AAPL/JPM/XOM at Q1 2023; earnings call transcripts deferred to Phase 8. See DAVID.md §17.
- **DJ-029 (Package path):** `src/hifi/knowledge/` — corrects Protocol draft which used `src/knowledge/`. See DAVID.md §17.

### Decision Journal Entries (Within-Phase Empirical)

**DJ-030: Chunk size configuration** — Record the 3 tested configurations (see Chunking Experiments table), Precision@5 for each, selected configuration and rationale. This resolves OQ-K01.

**DJ-031: Embedding model final selection** — Record Precision@5 under nomic-embed-text-v1.5 (DJ-027 baseline). If Precision@5 >= 0.6 on the 20-query financial test set, accept nomic-embed-text-v1.5. If below threshold, evaluate BGE-M3 and record comparison. This resolves OQ-M03.

### Notes for Educative Journal

This phase is RAG engineering in practice. Document everything: what chunk sizes you tested, what happened with each, what retrieval failures looked like, how you improved quality. This is the material for answering "How did you decide your chunk size?" in an interview — with numbers, not theory.

---

## Phase 8: Full Agent Population

### Objective

Expand from 2 agents to the full population defined in the David. Each agent is added one at a time, with diversity and marginal contribution measured after each addition.

### David Sections

- §10.2 All Agent Specifications
- §10.3 Diversity Requirements

### Learning Guide Topics

- 3.1 Agent Architecture Fundamentals (deep)
- 3.3 Collective Intelligence (measurement)
- 8.2 Collective Intelligence

### Deliverables

Add agents in this order (each addition is a sub-phase):

| Sub-Phase | Agent | Model Family | Information Focus |
|---|---|---|---|
| 8a | Risk Agent | (different from existing) | Risk metrics only |
| 8b | Macro Agent | (different from existing) | Macro indicators only |
| 8c | Sentiment Agent | (different from existing) | Earnings calls, filings via RAG |
| 8d | Contrarian Agent | (different from existing) | All information, adversarial role |

After each agent addition:

1. Run evaluation on EVAL-2022-2023
2. Measure ensemble performance (directional accuracy, Sharpe proxy)
3. Measure disagreement entropy H
4. Measure inter-agent correlation (pairwise)
5. Record marginal contribution of the new agent

### Success Criteria

- [ ] All agents operational with different model families
- [ ] Each agent has different information access
- [ ] Diversity verified through measured pairwise decision correlation
- [ ] Marginal contribution of each agent documented
- [ ] Ensemble performance improves (or a specific agent is identified as unhelpful)
- [ ] Disagreement entropy measured and tracked across additions
- [ ] Contrarian agent produces structured counter-theses (not just opposite votes)

### What This Phase Proves

Whether diversity actually helps. The marginal contribution curve (performance vs. number of agents) is one of the most interesting empirical results HiFi will produce. If adding the 5th agent does not improve performance, that is valuable evidence about the limits of ensemble diversity.

---

## Phase 9: Collective Decision Engine

### Objective

Implement and compare multiple aggregation mechanisms. Determine which produces the best collective decisions.

### David Sections

- §12 Collective Decision Engine (full section)
- §5.6 Formalization of Complexity Concepts

### Learning Guide Topics

- 3.3 Collective Intelligence & Aggregation (deep)
- 8.2 Collective Intelligence (measurement)
- 8.3 Emergence & Measurement

### Deliverables

- `src/collective/majority_vote.py` — Simple majority
- `src/collective/confidence_weighted.py` — Confidence-weighted (already exists from Phase 4, refine)
- `src/collective/performance_weighted.py` — Historical performance weighting
- `src/collective/debate.py` — Structured debate (experimental)
- `src/collective/contrarian_integration.py` — Contrarian influence mechanism
- `src/collective/metrics.py` — All complexity metrics: H, D, κ, S, Page diversity

**Aggregation comparison experiment:**

Run all methods on the same evaluation set. Compare:

| Method | Dir. Accuracy | Sharpe Proxy | Hallucination Rate | Entropy (H) |
|---|---|---|---|---|
| Majority Vote | ? | ? | ? | ? |
| Confidence-Weighted | ? | ? | ? | ? |
| Performance-Weighted | ? | ? | ? | ? |
| With Contrarian | ? | ? | ? | ? |
| Without Contrarian | ? | ? | ? | ? |

### Success Criteria

- [ ] At least 3 aggregation methods implemented and compared
- [ ] All complexity metrics (H, D, κ, S) computed and recorded
- [ ] Best aggregation method identified with statistical evidence
- [ ] Contrarian ablation study completed (with vs. without)
- [ ] Results reported with confidence intervals

---

## Phase 10: Evaluation Framework & Backtesting

### Objective

Implement the full evaluation framework: walk-forward backtesting, baseline comparisons, ablation studies, statistical significance testing. This is where the system's claims are either supported or refuted.

### David Sections

- §15 Evaluation Framework (full section)
- §8.8 Dataset Family G (Evaluation Datasets)

### Learning Guide Topics

- 7.2 Backtesting & Evaluation
- 7.3 Market Regimes
- 11.1 Classical ML (evaluation methodology)

### Deliverables

- `src/evaluation/walk_forward.py` — Walk-forward validation with purged CV
- `src/evaluation/baselines.py` — Random, buy-and-hold, momentum, single-best-agent
- `src/evaluation/statistical_tests.py` — Bootstrap confidence intervals, significance tests
- `src/evaluation/regime_analysis.py` — Performance segmented by market regime
- `src/evaluation/ablation.py` — Systematic ablation studies
- `src/evaluation/report_generator.py` — Generate evaluation report

**Evaluation datasets:**

Create immutable evaluation sets from the data acquired in Phase 1 (expanded universe if needed):

| Set | Period | Regime |
|---|---|---|
| EVAL-BULL | 2017-01-01 to 2019-12-31 | Late-cycle bull |
| EVAL-CRISIS | 2020-01-01 to 2020-12-31 | COVID crash + recovery |
| EVAL-BEAR | 2022-01-01 to 2022-12-31 | Rate hiking bear |
| EVAL-OOS | 2023-01-01 to 2024-12-31 | Out-of-sample |

**Ablation studies (from David §15.7):**

| Ablation | Removes | Measures |
|---|---|---|
| -1 Agent | Each agent removed in turn | Marginal contribution |
| -Contrarian | Contrarian agent | Value of dissent |
| -Verification | Verification layer | Hallucination impact |
| -RAG | Knowledge retrieval | Value of external knowledge |
| -Diversity | Homogenize to one model family | Value of model diversity |

### Success Criteria

- [ ] Walk-forward evaluation runs cleanly on all evaluation sets
- [ ] All baseline comparisons computed
- [ ] System beats random baseline with statistical significance (p < 0.05)
- [ ] Ensemble outperforms best individual agent (H1 tested)
- [ ] Ablation results documented
- [ ] Regime analysis shows performance variation across conditions
- [ ] Full evaluation report generated with tables and confidence intervals

### What This Phase Proves

Whether HiFi works. Not anecdotally, not on cherry-picked examples, but across market regimes with statistical rigor and controlled baselines. If the system cannot beat random on this evaluation, no amount of additional complexity will help. If it can, we have evidence worth presenting.

---

## Phase 11: Fine-Tuning

### Objective

Fine-tune agents on domain-specific data. Measure whether fine-tuning improves agent performance over base models. Measure whether fine-tuning reduces ensemble diversity.

### David Sections

- §9.4 Fine-Tuning Strategy
- §8.4 Dataset Family C (Reference Strategies — as training data)

### Learning Guide Topics

- 4.1 Parameter-Efficient Fine-Tuning
- 4.2 Training Data Engineering

### Deliverables

- `src/models/training_data.py` — Generate fine-tuning datasets from reference strategies
- `src/models/fine_tune.py` — LoRA/QLoRA fine-tuning pipeline via MLX
- Fine-tuned adapters for at least 2 agents
- Before/after comparison: base model vs. fine-tuned on held-out evaluation

### Success Criteria

- [ ] Training data generated from reference strategy datasets
- [ ] At least 2 agents fine-tuned with LoRA on Apple Silicon
- [ ] Each fine-tuned agent demonstrably outperforms its base model on held-out data
- [ ] Diversity metrics measured before and after fine-tuning (OQ-M02 answered)
- [ ] If fine-tuning reduces diversity excessively, document and adjust
- [ ] Training time and resource usage recorded (Learning Guide "Numbers")

### Decision Journal Entry

**DJ-015: Fine-tuning rank** — Test LoRA ranks 4, 8, 16, 32. Measure quality vs. training time vs. memory. Record optimal rank per agent.

---

## Phase 12: Knowledge Systems — GraphRAG

### Objective

Implement GraphRAG and compare against the standard RAG from Phase 7. Determine whether the additional complexity of a knowledge graph provides measurable improvement.

### David Sections

- §11.3 GraphRAG
- §11.1 Knowledge Graphs

### Learning Guide Topics

- 5.1 Knowledge Graphs
- 5.2 GraphRAG

### Deliverables

- `src/knowledge/graph_construction.py` — Build financial knowledge graph
- `src/knowledge/graph_rag.py` — GraphRAG retrieval pipeline
- Comparison experiment: RAG vs. GraphRAG on same evaluation queries

### Success Criteria

- [ ] Knowledge graph constructed with financial entities and relationships
- [ ] GraphRAG retrieval operational
- [ ] Measured comparison: RAG vs. GraphRAG on retrieval precision
- [ ] Measured comparison: agent quality with RAG vs. GraphRAG
- [ ] Cost analysis: additional latency and complexity of GraphRAG quantified
- [ ] Decision recorded: is GraphRAG worth the complexity? (OQ-K02 answered)

### Decision Journal Entry

**DJ-016: RAG vs. GraphRAG** — If GraphRAG does not measurably improve retrieval quality or agent performance, document and keep standard RAG. Do not add complexity without evidence.

---

## Phase 13: Advanced Features

### Objective

Implement the remaining David features that strengthen the system but are not on the critical path.

### Sub-phases (order flexible):

**13a: Agent Memory**
- David §10.4
- Agents remember past recommendations and outcomes
- Measure: does memory improve decision quality over time?

**13b: Synthetic Scenarios**
- David §8.7, Dataset Family F
- Generate stress-test scenarios (flash crash, liquidity crisis, etc.)
- Test agent behavior under extreme conditions
- Document methodology limitations honestly

**13c: Drift Detection**
- David §14.4
- Implement statistical tests for data drift, agent drift, collective drift
- Set up alerting thresholds

**13d: Dataset Families Completion**
- David §8.5 (Explanations), §8.6 (Interactions)
- Ensure all agent interactions are captured for future complexity analysis

### Success Criteria

- [ ] Agent memory implemented and measured
- [ ] At least 3 synthetic scenarios tested
- [ ] Drift detection operational with defined thresholds
- [ ] All dataset families populated (at least partially)

---

## Phase 14: Paper Trading

### Objective

Deploy HiFi in live paper trading on Alpaca. The system makes real decisions on real market data in real time. Results are recorded and analyzed.

### David Sections

- §7.9 Execution Layer
- §7.9 Safety Layer

### Learning Guide Topics

- 7.1 Quantitative Analysis (live application)
- 6.3 Deployment (operational)

### Prerequisites

- Phases 0-10 complete (minimum viable system evaluated)
- Safety layer implemented (position limits, circuit breakers)
- Observability fully operational (every decision traced)

### Deliverables

- `src/execution/paper_trader.py` — Alpaca paper trading integration
- `src/execution/risk_manager.py` — Safety limits enforcement
- Paper trading log: every decision, execution, and outcome recorded
- Daily performance dashboard

### Operational Parameters

| Parameter | Value | David Reference |
|---|---|---|
| Initial capital | $100,000 (paper) | — |
| Max position | 5% | §7.9 |
| Max daily loss | 2% | §7.9 |
| Max sector exposure | 25% | §7.9 |
| Decision frequency | Weekly | — |
| Universe | 10-50 stocks (expanded from initial 10) | — |
| Duration | Minimum 4 weeks, target 8-12 weeks | — |

### Success Criteria

- [ ] Paper trading operational on Alpaca
- [ ] Safety limits enforced and tested
- [ ] All decisions logged with full traceability
- [ ] No circuit breaker violations (or documented and analyzed if they occur)
- [ ] Performance report generated (return, Sharpe, max drawdown)
- [ ] Comparison against buy-and-hold over same period

---

## Phase 15: Containerization & Deployment

### Objective

Package HiFi as a containerized system that any user can deploy.

### David Sections

- §16 Deployment Strategy (full section)

### Learning Guide Topics

- 6.3 Deployment & Containerization

### Deliverables

- `docker/` — Dockerfiles for each component
- `docker-compose.yml` — Full system orchestration
- Deployment documentation: step-by-step from zero to running system
- Hardware requirements documentation

### Success Criteria

- [ ] `docker compose up` starts the complete system
- [ ] System passes smoke test after fresh deployment
- [ ] Documentation sufficient for someone who is NOT you to deploy
- [ ] Resource usage documented (RAM, disk, CPU per container)

---

## Phase 16: Open Source Release

### Objective

Prepare and release HiFi as an open-source project.

### David Sections

- §4.8 Open Research
- §8.9 Dataset Release

### Deliverables

- Clean repository with documentation
- Dataset release on Hugging Face with dataset cards
- README with architecture overview, setup, and usage
- LICENSE (MIT or Apache 2.0)
- Contributing guidelines

### Success Criteria

- [ ] Repository is clean and navigable
- [ ] README is sufficient for a new user to understand and deploy
- [ ] Datasets released with documentation
- [ ] License applied
- [ ] At least one external person has deployed successfully (if possible)

---

## Phase 17: Capstone Deliverable

### Objective

Package the work into a WQU MScFE capstone submission.

### Deliverables

- Capstone document (Jupyter notebook or report format, per WQU requirements)
- Demonstration of working system
- Evaluation results with visualizations
- Key architectural decisions documented

### Content Focus

| Section | Source |
|---|---|
| Introduction / Problem | David §2, §3 |
| Architecture | David §6, §7 |
| Data | Phase 1 results |
| Agents | Phases 3-8 results |
| Evaluation | Phase 10 results |
| Paper Trading | Phase 14 results |
| Conclusions | Honest assessment of what worked and what didn't |

### Notes

The capstone is a SUBSET of the David. It shows the functional system, its evaluation, and its results. The complexity science analysis, publication-quality statistical rigor, and full ablation studies are available but may be de-emphasized depending on WQU format requirements.

---

## Phase 18: Publication Preparation (Post-Graduation)

### Objective

Extract scientific insights from HiFi for publication in complexity science / computational finance venues.

### David Sections

- §5 Scientific Foundations (full rigor)
- §5.6 Formalization of Complexity Concepts

### Potential Publications

| Paper | Venue | Core Contribution |
|---|---|---|
| Collective intelligence in LLM agent ensembles | ICAIF, ACM Collective Intelligence | Empirical study of emergence in heterogeneous LLM populations |
| Deterministic verification for financial LLM systems | FinNLP workshop, EMNLP | Verification architecture and hallucination reduction results |
| Diversity vs. scale in local AI agent populations | Computational Economics | Evidence on Page diversity theorem applied to LLM agents |

### Notes

This phase uses the Agent Interaction Datasets (Family E) and complexity metrics collected throughout the project. The data is collected during normal operation — the publication requires analysis and writing, not new engineering.

---

## Critical Path Summary

If time forces prioritization, the minimum viable path is:

```
Phase 0  → Infrastructure
Phase 1  → Data (10 stocks, basic)
Phase 2  → First MCP server
Phase 3  → First agent (baseline)
Phase 4  → Second agent (first ensemble)
Phase 5  → Verification
Phase 6  → Observability
Phase 10 → Evaluation (can run on 2-agent system)
Phase 14 → Paper trading (minimum duration)
Phase 17 → Capstone
```

This produces a functional 2-agent system with verification, observability, evaluation, and paper trading. It is sufficient for the capstone. Every additional phase (RAG, full agents, fine-tuning, GraphRAG) strengthens the David proximity but is not strictly required for graduation.

The ambitious path includes all 18 phases. The realistic path includes at least the critical path. The learning path — which is the actual purpose — includes as many phases as dedication and time allow.

---

## Phase Dependency Graph

```
Phase 0 ─────────────────────────────────────────────────┐
    │                                                     │
    ▼                                                     │
Phase 1 (Data)                                            │
    │                                                     │
    ▼                                                     │
Phase 2 (MCP Engine)                                      │
    │                                                     │
    ▼                                                     │
Phase 3 (First Agent) ──────────────┐                     │
    │                               │                     │
    ▼                               │                     │
Phase 4 (Second Agent)              │                     │
    │                               │                     │
    ├──────────┐                    │                     │
    │          ▼                    ▼                     │
    │    Phase 5 (Verification)   Phase 6 (Observability)│
    │          │                    │                     │
    │          └────────┬───────────┘                     │
    │                   │                                 │
    ▼                   ▼                                 │
Phase 7 (RAG) ──► Phase 8 (Full Agents)                  │
                       │                                  │
                       ▼                                  │
                 Phase 9 (Collective Engine)               │
                       │                                  │
                       ▼                                  │
                 Phase 10 (Evaluation) ◄──────────────────┘
                       │
           ┌───────────┼───────────┐
           │           │           │
           ▼           ▼           ▼
     Phase 11    Phase 12    Phase 13
    (Fine-Tune)  (GraphRAG)  (Advanced)
           │           │           │
           └───────────┼───────────┘
                       │
                       ▼
                 Phase 14 (Paper Trading)
                       │
                       ▼
                 Phase 15 (Containers)
                       │
                       ▼
                 Phase 16 (Open Source)
                       │
                       ▼
                 Phase 17 (Capstone)
                       │
                       ▼
                 Phase 18 (Publication)
```

Note: Phases 5 and 6 can run in parallel. Phases 11, 12, and 13 can run in parallel. Phase 10 can begin as soon as Phase 4 is complete (with a 2-agent system) and be re-run as more agents are added.

---

*This protocol is derived from the David but is not the David. The David describes perfection. This protocol describes a realistic walk toward it. When a phase is completed, update the David Conformance Matrix in the Learning Guide. When a phase is deferred, record why. The protocol may be revised as reality teaches us what we did not know when we wrote it.*
