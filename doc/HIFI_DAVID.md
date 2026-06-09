# HiFi: High-Fidelity Financial Intelligence

## A Perennial Reference Specification

**Author:** Alberto Espinosa

**Status:** Living Document — Aspirational Reference (The David)

**Purpose:** This document describes the ideal, complete, and scientifically grounded vision of HiFi. It is not an execution plan. It is the Platonic form of the project — the reference against which all implementation decisions are measured, all trade-offs are justified, and all simplifications are acknowledged. It is also an educative journal: every architectural choice records not just *what* but *why*, and marks open questions to be resolved through construction.

---

## Table of Contents

1. [Preamble](#1-preamble)
2. [Problem Statement](#2-problem-statement)
3. [Vision](#3-vision)
4. [Core Principles](#4-core-principles)
5. [Scientific Foundations](#5-scientific-foundations)
6. [Intelligence Architecture](#6-intelligence-architecture)
7. [System Architecture](#7-system-architecture)
8. [Dataset Strategy](#8-dataset-strategy)
9. [Model Strategy](#9-model-strategy)
10. [Agent Architecture](#10-agent-architecture)
11. [Knowledge Systems](#11-knowledge-systems)
12. [Collective Decision Engine](#12-collective-decision-engine)
13. [Verification and Hallucination Control](#13-verification-and-hallucination-control)
14. [Observability](#14-observability)
15. [Evaluation Framework](#15-evaluation-framework)
16. [Deployment Strategy](#16-deployment-strategy)
17. [Decision Journal](#17-decision-journal)
18. [Open Questions](#18-open-questions)
19. [Success Criteria](#19-success-criteria)
20. [References](#20-references)
21. [Glossary](#21-glossary)

---

## 1. Preamble

### 1.1 What this document is

This is the idealized specification of HiFi. It describes the system as it would exist if every design choice were made perfectly, every layer were fully implemented, and every scientific question were rigorously answered.

No real system achieves its Platonic form. That is not the point.

The point is to have a reference that does not degrade under the pressure of deadlines, shortcuts, or expedience. When a decision is made to simplify, defer, or remove a component, that decision should be made *against* this document — explicitly, with recorded justification.

### 1.2 What this document is not

- It is not an execution plan (that will be derived separately)
- It is not a thesis proposal (the capstone is a subset of this)
- It is not a timeline (there are no dates here)
- It is not frozen (it evolves as understanding deepens)

### 1.3 The educative journal principle

Every section contains three kinds of content:

- **Specification:** What the component should be
- **Rationale:** Why this choice, grounded in theory or evidence
- **Open Questions:** What we do not yet know, to be resolved through building

This structure ensures that the document teaches, not just prescribes. When revisited months later — or by a different reader — the reasoning is preserved alongside the decision.

### 1.4 Epistemological position

HiFi is designed through the lens of complexity science. This does not mean the project is *about* complexity science. It means that the designers treat financial markets as complex adaptive systems, treat agent populations as ecologies, treat intelligence as an emergent property of interactions rather than a property of individual components, and treat robustness as more important than optimality.

This worldview is the operating system. The applications that run on top of it — financial analysis, trading decisions, agent coordination — are what users see. The operating system remains invisible but shapes every design decision.

---

## 2. Problem Statement

### 2.1 The state of financial AI

Modern financial AI systems exhibit several structural deficiencies.

**Centralization.** Most advanced financial AI depends on proprietary cloud infrastructure (OpenAI, Anthropic, Google APIs). This creates vendor lock-in, opacity, privacy exposure, and recurring costs that scale with usage. A retail investor, small fund, or researcher in an emerging market cannot afford — or may not trust — these dependencies.

**Opacity.** When a cloud-based system recommends "BUY NVDA," the user cannot inspect the reasoning chain, verify the calculations, audit the data sources, or reproduce the result. The recommendation is an oracle output. Oracles are useful until they fail, at which point their opacity becomes a liability.

**Hallucination.** Large language models generate plausible but fabricated claims. In conversational contexts this is inconvenient. In financial contexts it is dangerous. A hallucinated earnings number, a fabricated ratio, or a false sector comparison can lead to material losses. Current financial AI systems lack systematic mechanisms for detecting and preventing hallucinations.

**Irreproducibility.** Most published financial AI results cannot be reproduced because datasets are unavailable, prompts are undisclosed, model versions change, fine-tuning procedures are undocumented, and evaluation protocols are inconsistent. This undermines scientific progress and practical trust.

**Monoculture.** The dominant paradigm is "bigger model, better results." Scale is pursued at the expense of diversity, specialization, and structured collaboration. From a complexity science perspective, monocultures are fragile. A single model, however large, has a single set of biases, a single training distribution, and a single failure mode. There is no mechanism for self-correction through diverse perspectives.

### 2.2 The gap

There exists no fully local, open, auditable, and reproducible financial intelligence platform that:

1. Separates deterministic financial computation from language model interpretation
2. Deploys heterogeneous specialized agents with genuine diversity
3. Aggregates agent opinions through structured collective decision mechanisms
4. Verifies every claim against objective evidence
5. Provides full observability across all layers
6. Operates without cloud dependencies or recurring inference costs
7. Releases datasets, methods, and evaluation protocols openly

HiFi is designed to fill this gap.

### 2.3 Why this matters beyond engineering

Financial markets are among the most data-rich, adversarial, and non-stationary environments available for studying decision-making under uncertainty. A system that operates in this environment — and whose internal dynamics are fully observable — becomes a laboratory for studying questions that extend beyond finance:

- How does collective intelligence emerge from heterogeneous agents?
- What is the relationship between diversity and decision quality?
- How does verification affect reliability in generative AI systems?
- What retrieval architectures best support grounded reasoning?

These questions belong to complexity science, AI research, and computational social science. HiFi is designed so that the data to address them is collected as a byproduct of normal operation.

---

## 3. Vision

HiFi is a fully local, open-source, high-fidelity financial intelligence platform.

**High-fidelity** means:

- Every calculation is deterministic and verifiable
- Every recommendation is traceable to evidence
- Every agent opinion is recorded and auditable
- Every claim is checked against objective sources
- Every result is reproducible

**Financial intelligence** means:

- The system produces actionable investment analysis
- Analysis covers fundamental, technical, macro, risk, sentiment, and valuation dimensions
- Outputs are structured decisions (Buy/Hold/Sell) with confidence estimates and structured rationale
- The system operates under real market conditions through paper trading

**Fully local** means:

- All inference runs on consumer hardware (target: Apple Silicon M-series)
- No cloud API dependencies for production operation
- Data acquisition may use external APIs; reasoning does not
- The entire system is containerized and deployable by any user

**Open-source** means:

- All code is publicly available
- All datasets are released with documentation
- All evaluation protocols are reproducible
- All architectural decisions are documented with rationale

---

## 4. Core Principles

### 4.1 Deterministic-First

> Whenever a task can be solved through deterministic computation, HiFi shall prefer deterministic computation over language model generation.

Language models are reasoning and coordination components. They are not calculators, not databases, not optimizers. Financial ratios, risk metrics, technical indicators, portfolio analytics, statistical tests, and backtesting calculations are delegated to deterministic engines exposed through MCP servers.

**Rationale:** Deterministic computation is verifiable, reproducible, and free of hallucination by construction. Every task that can be moved out of the language model reduces the hallucination surface and increases auditability.

### 4.2 Diversity Over Scale

> Agent diversity is prioritized over individual model capability.

The system deploys multiple agents from different model families, with different information access, different prompt structures, and different specializations. The hypothesis is that collective intelligence emerges from diversity, not from scale.

**Rationale:** In ensemble learning, the benefit of aggregation depends on the diversity (decorrelation) of the components (Breiman, 2001). In collective intelligence research, groups outperform individuals when members are diverse, partially independent, and aggregated reasonably (Surowiecki, 2005; Page, 2007). A population of identical agents, however capable, provides no diversity benefit.

### 4.3 Verifiability

> Every claim made by any agent must be traceable to a deterministic source or explicitly flagged as interpretive.

There are two categories of statements in financial analysis:

- **Objective:** "AAPL P/E ratio is 28.3" — this is either correct or incorrect, verifiable against deterministic computation
- **Interpretive:** "AAPL is overvalued relative to sector peers" — this is a judgement that depends on methodology and context

HiFi must distinguish these categories explicitly. Objective claims are verified automatically. Interpretive claims are labelled as such and attributed to the agent that produced them.

### 4.4 Observability

> Every process, decision, interaction, and intermediate state must be measurable, traceable, and recordable.

Observability is not debugging. Observability is the scientific instrumentation of the system. Without it, the system is a black box whose outputs cannot be analyzed, improved, or trusted.

### 4.5 Reproducibility

> Every result produced by HiFi must be reproducible by any user with access to the same code, data, and configuration.

This requires:

- Dataset versioning with content hashing
- Model versioning with weight checksums
- Prompt versioning
- Configuration versioning
- Random seed control where applicable
- Deterministic execution paths where possible

### 4.6 Modularity

> Every component must be replaceable without requiring changes to other components.

Agent A can be replaced with Agent B. The voting mechanism can be swapped. The knowledge system can be upgraded. The observability backend can change. This requires well-defined interfaces between layers.

### 4.7 Local-First

> The system must operate without cloud dependencies for all inference and decision-making operations.

Data acquisition may use external APIs (market data providers, SEC EDGAR). But once data is ingested, all processing, reasoning, and decision-making runs locally.

### 4.8 Open Research

> Every methodology, dataset, evaluation protocol, and architectural decision must be publishable and inspectable.

HiFi is not a proprietary trading system. It is a research platform that happens to trade. The default is open. The exception is never.

---

## 5. Scientific Foundations

### 5.1 Complex Adaptive Systems

Financial markets are widely modelled as complex adaptive systems (Holland, 1992; Arthur, 2021): populations of heterogeneous agents interacting through nonlinear feedback mechanisms, producing emergent macro-level phenomena that cannot be predicted from individual agent behavior alone.

HiFi adopts this framing not as a research claim but as a design philosophy. The system is designed as an ecology of interacting components rather than a monolithic predictor.

**Relevance to design:**

- Agents are heterogeneous by construction (different models, different information, different roles)
- The system exhibits feedback (agent performance informs future weighting)
- Emergent properties (consensus, disagreement patterns, regime-dependent behavior) are observed and measured
- Robustness is valued over optimality (the system should degrade gracefully, not catastrophically)

### 5.2 Collective Intelligence

Research on collective intelligence establishes conditions under which groups outperform individuals (Surowiecki, 2005; Woolley et al., 2010):

1. **Diversity of opinion:** Each member holds private information or interpretation
2. **Independence:** Members' opinions are not determined by those around them
3. **Decentralization:** Members can specialize and draw on local knowledge
4. **Aggregation:** A mechanism exists for converting individual judgements into collective decisions

HiFi's agent architecture is explicitly designed around these four conditions.

**Critical challenge:** LLM agents may violate condition 2 (independence) because they share overlapping training data. This is an open research question, not an assumed property. HiFi must measure the actual independence of agent opinions empirically rather than assuming it.

### 5.3 Ensemble Learning

Ensemble methods in machine learning demonstrate that combining multiple models typically outperforms individual models when the component models have diverse error structures (Breiman, 2001; Dietterich, 2000).

The theoretical basis is the bias-variance decomposition. For an ensemble of M models with average bias b, average variance v, and average pairwise correlation ρ:

```
Ensemble_Error ≈ b² + ρv + (1-ρ)v/M
```

As M increases, the third term vanishes. But the second term — which depends on correlation ρ — persists. If agents are highly correlated (ρ → 1), the ensemble provides no benefit. Therefore:

**The value of the ensemble depends entirely on achieving low correlation between agents.**

This must be measured, not assumed. The diversity metrics defined in Section 15 are designed for this purpose.

### 5.4 Agent-Based Modelling

Agent-based models (ABMs) provide the natural framework for studying populations of interacting decision-makers under uncertainty (Farmer & Foley, 2009; Wooldridge, 2009). HiFi's agent layer is, in effect, a small-scale ABM where the agents are LLMs rather than rule-based automata.

### 5.5 Financial Decision Theory

Investment decisions operate under uncertainty, not risk. The distinction (Knight, 1921) is fundamental:

- **Risk:** Probabilities are known (e.g., rolling a die)
- **Uncertainty:** Probabilities are unknown or unknowable (e.g., will a trade war escalate?)

Financial markets exhibit Knightian uncertainty. Therefore, HiFi does not claim to produce "optimal" decisions. It claims to produce *well-reasoned, verifiable, and collectively informed* decisions. The quality of those decisions is evaluated empirically, not assumed a priori.

### 5.6 Formalization of Key Complexity Concepts

The following concepts are used throughout HiFi as design principles and measurement targets. They are formalized here to prevent them from becoming decorative vocabulary.

#### 5.6.1 Disagreement Entropy

Given N agents each producing a vote v_i ∈ {Buy, Hold, Sell}, let p_Buy, p_Hold, p_Sell be the proportions of each vote. The disagreement entropy is:

```
H = -Σ p_k · log₂(p_k)    for k ∈ {Buy, Hold, Sell}
```

H = 0 indicates perfect consensus (all agents agree). H = log₂(3) ≈ 1.585 indicates maximum disagreement (uniform distribution over three options).

**Interpretation:** High entropy before a decision may indicate genuine uncertainty in the market environment. The relationship between pre-decision entropy and subsequent decision quality (measured by financial outcome) is an empirical question that HiFi is instrumented to answer.

#### 5.6.2 Opinion Dispersion Index

Let c_i be the confidence score of agent i (normalized to [0,1]). The opinion dispersion is:

```
D = (1/N) · Σ |c_i - c̄|
```

where c̄ is the mean confidence. High dispersion indicates agents with very different levels of conviction, even if they agree on direction.

#### 5.6.3 Herding Coefficient

Over a sequence of T decisions, let a_t be the agreement rate at time t (proportion of agents voting with the majority). The herding coefficient is:

```
κ = (1/T) · Σ a_t
```

κ close to 1 indicates systematic herding. κ close to 1/3 (for three options) indicates independence. Values of κ significantly above random agreement, after controlling for market clarity, suggest agents are exhibiting correlated behavior that may reduce ensemble value.

#### 5.6.4 Consensus Stability

For a given stock over a window of W consecutive evaluation periods, let v_t be the majority decision at time t. The consensus stability is:

```
S = (1/(W-1)) · Σ 𝟙(v_t = v_{t+1})
```

S = 1 means the collective decision never changes. S = 0 means it changes every period. Excessively high stability may indicate insensitivity to new information. Excessively low stability may indicate noise-driven decisions.

#### 5.6.5 Diversity Index

Following Page (2007), the predictive diversity of an ensemble can be decomposed as:

```
Collective_Error = Average_Individual_Error - Diversity
```

where Diversity is measured as the average squared deviation of individual predictions from the collective prediction. This identity is exact (not approximate) and provides a direct empirical measure of how much value diversity adds.

**Open Question:** How should this be adapted for categorical predictions (Buy/Hold/Sell) rather than continuous predictions? Possible approaches include treating the problem as probabilistic classification and measuring diversity through KL divergence between individual agent probability distributions and the ensemble distribution.

---

## 6. Intelligence Architecture

### 6.1 The Hybrid Intelligence Principle

HiFi does not assume that intelligence resides within large language models. Intelligence in HiFi is an emergent property of the interaction between multiple subsystems:

```
┌──────────────────────────────────────────────────────┐
│                    HiFi Intelligence                  │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │Deterministic │  │  Knowledge  │  │  Retrieval   │ │
│  │   Engines    │  │   Graphs    │  │   Systems    │ │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘ │
│         │                 │                │         │
│         ▼                 ▼                ▼         │
│  ┌───────────────────────────────────────────────┐   │
│  │              MCP Server Layer                  │   │
│  └───────────────────┬───────────────────────────┘   │
│                      │                               │
│                      ▼                               │
│  ┌───────────────────────────────────────────────┐   │
│  │         Agent Orchestration Layer              │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌────────┐ │   │
│  │  │Fund.│ │Val. │ │Tech.│ │Risk │ │Contrar.│ │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └────────┘ │   │
│  └───────────────────┬───────────────────────────┘   │
│                      │                               │
│                      ▼                               │
│  ┌───────────────────────────────────────────────┐   │
│  │       Collective Decision Engine               │   │
│  └───────────────────┬───────────────────────────┘   │
│                      │                               │
│                      ▼                               │
│  ┌───────────────────────────────────────────────┐   │
│  │          Verification Engine                   │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

The agents are **consumers** of intelligence produced by deterministic engines, knowledge graphs, and retrieval systems. They are **producers** of interpretation, hypothesis, critique, and synthesis. The collective decision engine aggregates their outputs. The verification engine validates the result.

No single component is "the intelligence." The intelligence is the system.

### 6.2 MCP as the Nervous System

The Model Context Protocol (MCP) serves as the communication backbone between agents and deterministic tools.

Each deterministic capability is exposed as an MCP server:

| MCP Server | Responsibility |
|---|---|
| Financial Calculator | Ratios, metrics, valuations |
| Market Data | OHLCV, fundamentals, macro |
| Technical Analysis | Indicators, patterns, signals |
| Risk Analytics | VaR, drawdown, volatility, correlation |
| Portfolio Analytics | Allocation, optimization, attribution |
| Knowledge Graph | Entity relationships, sector maps |
| Retrieval | RAG/GraphRAG over financial documents |
| Backtesting | Historical strategy evaluation |
| Observability | Metrics, traces, logs |

Agents call these servers through standardized MCP tool interfaces. They never compute financial quantities directly. They request computations, receive results, and interpret them.

**Rationale:** This architecture makes every computation auditable. When an agent claims "AAPL's P/E is 28.3," the verification layer can confirm this by calling the same MCP server independently. The agent cannot fabricate a number that survives verification.

**Open Question:** What is the performance overhead of MCP inter-process communication for latency-sensitive operations? At what point does the overhead justify embedding computations directly rather than routing through MCP? This must be measured empirically.

---

## 7. System Architecture

### 7.1 Layer 1 — Data Acquisition

**Responsibility:** Ingest raw data from external sources into the local system.

**Sources:**

| Source | Data Type | Access Method |
|---|---|---|
| Yahoo Finance | OHLCV, fundamentals, dividends | yfinance API |
| SEC EDGAR | 10-K, 10-Q, 8-K filings | EDGAR API |
| FRED | Macroeconomic indicators | FRED API |
| Alpaca | Real-time market data, paper trading | Alpaca API |
| Earnings Call Transcripts | Unstructured text | Public sources / APIs |
| News | Headlines, articles | RSS / News APIs |

**Data Governance Requirements:**

- Every data point must carry provenance metadata (source, timestamp, version)
- All ingested data is stored locally in versioned datasets
- Data lineage is tracked from source through every transformation
- Ingestion failures are logged and do not silently produce gaps

**Open Question:** Which news/sentiment data sources provide sufficient coverage without prohibitive cost for a local-first system? Free-tier alternatives must be evaluated against coverage requirements.

### 7.2 Layer 2 — Data Engineering

**Responsibility:** Transform raw data into clean, versioned, machine-consumable representations.

**Components:**

- **Cleaning:** Handle missing values, outliers, corporate actions (splits, dividends)
- **Normalization:** Standardize formats across sources
- **Feature Engineering:** Compute derived features (see Section 8)
- **Versioning:** Every dataset release is content-hashed and immutable
- **Lineage:** Every feature traces to its raw source through documented transformations

**Technology Candidates:**

| Component | Options | Decision Status |
|---|---|---|
| Storage | Parquet, DuckDB, SQLite | Open |
| Versioning | DVC, custom content-hashing | Open |
| Feature computation | Pandas, Polars | Open |
| Orchestration | Prefect, custom DAG | Open |

**Open Question:** Is DuckDB sufficient for the scale of data HiFi handles (decades of daily data for S&P 500 constituents), or is a more specialized time-series store warranted? The answer depends on measured query performance, not architectural preference.

### 7.3 Layer 3 — Knowledge Layer

Detailed in Section 11 (Knowledge Systems).

### 7.4 Layer 4 — Model Layer

Detailed in Section 9 (Model Strategy).

### 7.5 Layer 5 — Agent Layer

Detailed in Section 10 (Agent Architecture).

### 7.6 Layer 6 — Collective Decision Layer

Detailed in Section 12 (Collective Decision Engine).

### 7.7 Layer 7 — Verification Layer

Detailed in Section 13 (Verification and Hallucination Control).

### 7.8 Layer 8 — Observability Layer

Detailed in Section 14 (Observability).

### 7.9 Layer 9 — Execution Layer

**Responsibility:** Translate verified decisions into paper trading orders and monitor execution.

**Components:**

- **Order Generation:** Convert Buy/Hold/Sell decisions into sized orders with stop-losses
- **Risk Limits:** Maximum position size, maximum daily loss, maximum portfolio concentration
- **Circuit Breakers:** Halt trading when anomalous conditions are detected
- **Execution Interface:** Alpaca paper trading API
- **Execution Logging:** Every order, fill, rejection, and modification is recorded

**Safety Layer:**

Even in paper trading, the system must enforce:

| Constraint | Default Value | Rationale |
|---|---|---|
| Max position size | 5% of portfolio | Prevent concentration risk |
| Max daily loss | 2% of portfolio | Prevent catastrophic days |
| Max sector exposure | 25% of portfolio | Prevent sector concentration |
| Max confidence override | 0.95 | Prevent overconfident actions |
| Min agents required | 4 of 7 voting | Prevent thin-quorum decisions |
| Circuit breaker | Halt if 3 consecutive losses exceed 1% each | Prevent cascading errors |

### 7.10 Layer 10 — Experiment Registry

**Responsibility:** Ensure every experiment is uniquely identified, versioned, and reproducible.

Every experiment records:

```
experiment:
  id: unique hash
  timestamp: ISO 8601
  dataset_version: content hash
  model_versions: {agent_name: model_hash}
  prompt_versions: {agent_name: prompt_hash}
  config_version: config hash
  random_seed: integer
  results: {metric_name: value}
  notes: freeform text
```

**Technology Candidates:** MLflow, custom registry, or integration with LangFuse experiment tracking.

**Open Question:** Should the experiment registry be a separate system or integrated into the observability layer? The trade-off is between simplicity (one system) and separation of concerns (experiments vs. operational monitoring).

---

## 8. Dataset Strategy

### 8.1 Foundational Principle

HiFi rejects the concept of a single "Golden Dataset."

Financial decision-making is inherently multi-objective and context-dependent. An action that is optimal under one objective function (maximize return) may be suboptimal under another (minimize drawdown). An action that is optimal for one risk tolerance is inappropriate for another.

Therefore, HiFi adopts a layered dataset architecture where each family serves a distinct purpose and carries explicit documentation of its assumptions and limitations.

### 8.2 Dataset Family A — Market Observation Datasets

**Purpose:** Represent reality as faithfully as possible.

**Contents:**

- OHLCV (Open, High, Low, Close, Volume) — daily and intraday where available
- Financial statements (quarterly, annual)
- Balance sheets, income statements, cash flow statements
- SEC filings (10-K, 10-Q, 8-K, proxy statements)
- Earnings call transcripts
- Macroeconomic indicators (rates, inflation, unemployment, GDP, PMI)
- Sector classifications

**Critical Requirements:**

- **Survivorship bias control:** Include delisted companies. Failing to do so inflates backtesting performance by excluding failures.
- **Point-in-time accuracy:** Use data as it was available at each historical date, not as it was later revised. Restated financials must be flagged.
- **Corporate action adjustment:** Splits, dividends, mergers must be handled consistently.

**Coverage Target:** S&P 500 constituents (including historical members), 1995–present.

**Open Question:** How far back is reliable point-in-time data available from free sources? SEC EDGAR full-text search is available from ~1996. Yahoo Finance historical data has known gaps and inconsistencies. The actual achievable coverage must be determined empirically.

### 8.3 Dataset Family B — Feature Datasets

**Purpose:** Transform raw observations into structured, machine-consumable representations.

**Categories:**

**Fundamental Features:**
- Profitability: ROE, ROA, ROIC, gross margin, operating margin, net margin
- Growth: Revenue growth (YoY, QoQ), earnings growth, book value growth
- Health: Current ratio, quick ratio, debt/equity, interest coverage
- Quality: Accruals ratio, earnings persistence, Piotroski F-Score

**Valuation Features:**
- P/E, P/B, P/S, EV/EBITDA, PEG ratio
- Relative valuation (vs. sector, vs. history)
- DCF-derived fair value estimates (under explicit assumptions)

**Technical Features:**
- Trend: SMA, EMA (multiple periods), MACD
- Momentum: RSI, Stochastic, Rate of Change
- Volatility: ATR, Bollinger Bands, historical volatility
- Volume: OBV, VWAP, volume profile
- Market structure: Support/resistance levels, 52-week range position

**Risk Features:**
- Historical volatility (multiple windows)
- Beta (market, sector)
- Value at Risk (parametric, historical)
- Maximum drawdown (trailing windows)
- Correlation with market index

**Macro Features:**
- Interest rate level and direction
- Yield curve slope
- Inflation rate and expectations
- VIX level and term structure
- Credit spreads
- Dollar index

**Embedding Features:**
- Dense vector representations of earnings calls
- Dense vector representations of SEC filings
- Dense vector representations of news articles

**Versioning:** Each feature set is versioned independently. Adding a new feature creates a new feature version, not a new dataset.

### 8.4 Dataset Family C — Reference Strategy Datasets

**Purpose:** Provide reproducible supervisory signals for training and benchmarking. These are NOT ground truth. They are reference strategies generated under explicit, documented assumptions.

**Labelling Methodologies:**

| Strategy | Objective | Labelling Rule |
|---|---|---|
| Max Return | Maximize forward return | BUY if forward N-day return > threshold; SELL if < -threshold; HOLD otherwise |
| Risk-Adjusted | Maximize forward Sharpe | Labels based on risk-adjusted forward returns |
| Drawdown-Constrained | Maximize return subject to max drawdown | Dynamic programming under drawdown constraint |
| Trend-Following | Capture sustained moves | Label based on confirmed trend changes |
| Mean-Reversion | Capture reversion to mean | Label based on deviation from moving average |

Each strategy is generated for multiple horizons (5, 10, 20, 60, 120 trading days).

**Critical Documentation:**

Every reference strategy dataset must explicitly document:

1. The objective function used
2. The look-ahead window
3. The threshold parameters
4. The assumptions about transaction costs
5. The known limitations and biases

**Warning on Look-Ahead Bias:**

Reference strategy labels are, by construction, computed using future information. This is acknowledged and intentional — they represent what *would have been* optimal under the stated objective, not what was knowable at the time. This distinction must be preserved in all downstream usage:

- Labels may be used for training agents to approximate reference strategies
- Labels may be used for benchmarking (how close does the agent come to the reference?)
- Labels must NOT be used to claim the system "predicts the future"
- Labels must NOT be conflated with ground truth

This is the "narrative look-ahead bias" identified in early project discussions. It remains one of the primary methodological risks.

### 8.5 Dataset Family D — Explanation Datasets

**Purpose:** Store reasoning and explanations generated by humans, analysts, or LLMs.

**Sources:**

- Analyst reports (where publicly available)
- LLM-generated explanations of market events
- Agent reasoning traces

**Critical constraint:** Explanations are NEVER treated as ground truth. They are used for:

- Interpretability research
- Agent alignment evaluation
- Prompt engineering
- Qualitative analysis

Explanations are inherently retrospective and subject to narrative look-ahead bias. This limitation must be documented in every use.

### 8.6 Dataset Family E — Agent Interaction Datasets

**Purpose:** Record the internal dynamics of the agent population.

**Contents:**

- Individual agent votes (Buy/Hold/Sell) per stock per evaluation period
- Individual agent confidence scores
- Agent reasoning traces
- Disagreement records
- Debate transcripts (when structured debate is used)
- Consensus formation trajectories

**Research Value:** These datasets enable the complexity science analysis that is HiFi's long-term research agenda. They capture the dynamics of collective decision-making: how consensus forms, how disagreement evolves, how contrarian opinions influence outcomes, how different market regimes affect agent behavior.

### 8.7 Dataset Family F — Synthetic Scenario Datasets

**Purpose:** Expose agents to rare and extreme situations that occur infrequently in historical data.

**Scenario Categories:**

| Scenario | Historical Analog | Purpose |
|---|---|---|
| Flash crash | May 2010, Aug 2015 | Test agent behavior under sudden price dislocation |
| Liquidity crisis | 2008 GFC, March 2020 | Test behavior when correlations spike to 1 |
| Inflation shock | 2022 rate cycle | Test macro agent under regime change |
| Earnings surprise | Various | Test fundamental agent under discontinuous information |
| Black swan | 9/11, COVID | Test system under unprecedented events |
| Slow decline | Dot-com bust 2000-2002 | Test recognition of extended bear markets |
| Sector rotation | Various | Test when leadership changes abruptly |

**Methodological Caution:**

Synthetic financial data generation is an open research problem. Most generators fail precisely when they are most needed: during tail events, regime changes, and contagion. Therefore:

- Synthetic data is used ONLY for stress testing, never for primary training
- The generation methodology must be documented and its limitations acknowledged
- Results on synthetic data are reported separately from results on historical data
- No claims of system robustness should rest solely on synthetic data performance

**Open Question:** What generation methodology is most appropriate? Options include historical bootstrapping, parametric models (GARCH family), generative adversarial networks, or scenario construction from historical templates. The choice should be informed by empirical evaluation of tail behavior fidelity.

### 8.8 Dataset Family G — Evaluation Datasets

**Purpose:** Fixed, immutable benchmarks for comparing system versions over time.

**Requirements:**

- Evaluation datasets are NEVER modified after initial release
- They contain held-out time periods not used in any training
- They include diverse market regimes (bull, bear, crisis, sideways)
- They are versioned with content hashes

**Proposed Structure:**

| Evaluation Set | Period | Regime Coverage |
|---|---|---|
| EVAL-2015-2017 | 2015-01-01 to 2017-12-31 | Mixed (includes 2015-2016 volatility) |
| EVAL-2018-2019 | 2018-01-01 to 2019-12-31 | Late-cycle bull, 2018 correction |
| EVAL-2020-2021 | 2020-01-01 to 2021-12-31 | Crisis + recovery |
| EVAL-2022-2023 | 2022-01-01 to 2023-12-31 | Bear market + rate cycle |
| EVAL-2024 | 2024-01-01 to 2024-12-31 | Out-of-sample current |

### 8.9 Dataset Release and Documentation

All datasets are released publicly (GitHub + Hugging Face) with:

- Dataset cards following the Datasheets for Datasets standard (Gebru et al., 2021)
- Generation code
- Provenance documentation
- Known limitations
- License (permissive open-source)
- Content hashes for integrity verification

---

## 9. Model Strategy

### 9.1 Primary Requirement

All inference runs locally on consumer hardware.

Target hardware: Apple Silicon M-series (M3 Ultra as reference platform, with M2/M3 Pro as minimum).

This constrains model selection to models that can run with acceptable latency under quantization on available memory and compute.

### 9.2 Model Selection Criteria

| Criterion | Weight | Rationale |
|---|---|---|
| Runs locally on target hardware | Required | Core principle |
| Financial reasoning quality | High | Primary task |
| Instruction following | High | Structured output generation |
| Diversity from other selected models | High | Ensemble decorrelation |
| Context window size | Medium | Affects document analysis capability |
| Fine-tuning ecosystem | Medium | LoRA/QLoRA support |
| Community and documentation | Medium | Practical sustainability |

### 9.3 Candidate Model Families

| Family | Representative Models | Rationale for Inclusion |
|---|---|---|
| Qwen | Qwen 2.5 (7B, 14B, 32B) | Strong multilingual, good reasoning |
| Llama | Llama 3.x (8B, 70B-quantized) | Largest open ecosystem |
| Gemma | Gemma 2 (9B, 27B) | Different architecture lineage (Google) |
| Mistral | Mistral (7B), Mixtral | Different training approach, MoE architecture |
| Phi | Phi-3/4 (small, medium) | Efficiency-focused, different scale regime |

**Diversity Principle:** No two agents should use models from the same family unless justified by specialization requirements.

### 9.4 Fine-Tuning Strategy

**Objective:** Improve agent performance on domain-specific tasks without sacrificing general reasoning ability.

**Technique:**

| Method | Description | When to Use |
|---|---|---|
| LoRA | Low-Rank Adaptation | Primary method — efficient, preserves base capabilities |
| QLoRA | Quantized LoRA | When memory is constrained |
| Full fine-tuning | Update all weights | Only if LoRA proves insufficient (unlikely for these model sizes) |

**Frameworks:**

| Framework | Platform | Status |
|---|---|---|
| MLX | Apple Silicon native | Primary for M-series hardware |
| Axolotl | Cross-platform | Fallback / comparison |
| Unsloth | Cross-platform | Efficiency candidate |

**Fine-Tuning Data:**

- Reference Strategy Datasets (Family C) for decision-making
- Financial Q&A pairs for domain knowledge
- Structured output examples for format compliance
- Agent-specific training sets tailored to each specialization

**Critical Requirement:** Fine-tuning effectiveness must be measured empirically. A fine-tuned model is only deployed if it demonstrably outperforms the base model on a held-out evaluation set. Fine-tuning for its own sake is not justified.

**Open Question:** How much fine-tuning data is needed per agent to achieve measurable improvement? The literature on LoRA suggests even small datasets (hundreds to low thousands of examples) can be effective, but this must be verified for financial reasoning tasks specifically.

**Open Question:** Does fine-tuning reduce diversity between agents (by moving them toward a common training signal)? This would undermine the ensemble benefit. Diversity metrics (Section 5.6) must be monitored before and after fine-tuning.

### 9.5 Inference Stack

| Component | Technology | Rationale |
|---|---|---|
| Model serving | Ollama, llama.cpp, MLX | Local inference engines |
| Quantization | GGUF (4-bit, 5-bit, 8-bit) | Memory efficiency |
| Embedding models | Local embedding model (e.g., nomic-embed, BGE) | Required for RAG/GraphRAG |
| Orchestration | LangGraph | Agent workflow management |

**Open Question:** What is the latency profile of running 5-7 agents sequentially vs. in parallel on an M3 Ultra? Can multiple models be loaded simultaneously, or must they be swapped? This determines whether the system processes one stock in seconds or minutes, which has implications for the size of the investable universe.

---

## 10. Agent Architecture

### 10.1 Agent Design Philosophy

Each agent is a specialized reasoner that:

1. Receives structured information from deterministic engines via MCP
2. Applies its specialization to interpret the information
3. Produces a structured output: decision, confidence, rationale
4. Has NO access to other agents' outputs during independent reasoning phase

The independence requirement (point 4) is critical for maintaining ensemble diversity. Agents interact only through the Collective Decision Engine, not through direct communication during the analysis phase.

### 10.2 Agent Specifications

#### Fundamental Agent

**Focus:** Business quality and financial health

**Information Access:**
- Revenue, earnings, margins, growth rates
- Balance sheet health metrics
- Cash flow quality
- Earnings persistence and quality indicators

**Output:**
- Decision: Buy / Hold / Sell
- Confidence: [0, 1]
- Rationale: Structured explanation referencing specific financial metrics
- Key concern: The single biggest risk identified

**Model Assignment:** To be determined (should be a model with strong numerical reasoning)

#### Valuation Agent

**Focus:** Whether the stock is cheap or expensive relative to intrinsic and relative benchmarks

**Information Access:**
- Valuation ratios (P/E, P/B, P/S, EV/EBITDA)
- Historical valuation range
- Sector peer comparison
- DCF inputs (if available)

**Output:**
- Decision: Buy / Hold / Sell
- Confidence: [0, 1]
- Rationale: Structured explanation with specific valuation comparisons
- Fair value estimate: Point estimate with uncertainty range

**Model Assignment:** Should differ from Fundamental Agent model family

#### Technical Agent

**Focus:** Price action, momentum, trend, and market structure

**Information Access:**
- Technical indicators (trend, momentum, volatility, volume)
- Price patterns
- Support and resistance levels
- Relative strength vs. market and sector

**Output:**
- Decision: Buy / Hold / Sell
- Confidence: [0, 1]
- Rationale: Based on specific indicator readings and price structure
- Time horizon: Expected duration of the signal

#### Risk Agent

**Focus:** Downside exposure and risk factors

**Information Access:**
- Historical volatility (multiple windows)
- Beta and correlation
- Maximum drawdown history
- Leverage and liquidity metrics
- Concentration risk (if portfolio context available)

**Output:**
- Decision: Buy / Hold / Sell (from a risk-management perspective)
- Confidence: [0, 1]
- Risk assessment: Structured risk profile
- Recommended position size: As fraction of portfolio, justified by risk metrics

#### Macro Agent

**Focus:** Economic environment and its implications for the stock/sector

**Information Access:**
- Interest rates (level, direction, curve shape)
- Inflation (actual, expectations)
- Employment data
- GDP and PMI
- Sector sensitivity to macro factors

**Output:**
- Decision: Buy / Hold / Sell (based on macro environment alignment)
- Confidence: [0, 1]
- Rationale: How macro environment supports or threatens the investment thesis
- Regime assessment: Current macro regime classification

#### Sentiment Agent

**Focus:** Market sentiment, narrative analysis, and qualitative information

**Information Access:**
- Earnings call transcripts (via RAG/GraphRAG)
- SEC filing analysis (via RAG/GraphRAG)
- News sentiment scores
- Analyst consensus (where available)

**Output:**
- Decision: Buy / Hold / Sell
- Confidence: [0, 1]
- Sentiment summary: Structured qualitative assessment
- Notable signals: Specific statements or events flagged as significant

#### Contrarian Agent

**Focus:** Challenge emerging consensus, identify risks the majority may be ignoring

**Information Access:**
- All information available to other agents (but processed independently)
- Historical examples of consensus failures
- Base rate of consensus being wrong by market regime

**Output:**
- Alternative thesis: What could go wrong with the majority view?
- Risk scenario: Specific adverse scenario with estimated probability
- Counterargument: Structured argument against the dominant position
- Confidence in contrarian view: [0, 1]

**Design Note:** The Contrarian Agent does not vote Buy/Hold/Sell in the same way as other agents. Its role is to stress-test the emerging consensus, not to add a vote. This is deliberate — a contrarian that simply votes the opposite is noise, not intelligence. A contrarian that articulates *why* the consensus might be wrong is valuable.

**Open Question:** Should the Contrarian Agent receive the other agents' outputs before formulating its counter-thesis? This would make it a second-pass agent rather than an independent first-pass agent. The trade-off is between informed critique (which requires seeing the consensus) and independence (which requires not seeing it). Both designs should be tested.

### 10.3 Diversity Requirements

Each agent must differ from every other agent along at least TWO of the following dimensions:

| Dimension | Description |
|---|---|
| Model Family | Different base model (Qwen vs. Llama vs. Gemma vs. Mistral) |
| Information Access | Different subset of available data |
| Prompt Structure | Different reasoning template |
| Fine-Tuning | Different training data or absence of fine-tuning |
| Role | Different analytical perspective |

**Measurement:** After the agent population is established, the actual diversity must be measured using the metrics from Section 5.6. If agents produce highly correlated outputs despite nominal diversity, the design has failed and must be revised.

### 10.4 Agent Memory

Agents are not stateless. Over time, each agent should accumulate:

- **Decision history:** What it recommended previously for the same stock
- **Outcome feedback:** Whether previous recommendations were correct
- **Calibration data:** How well its confidence estimates match actual outcomes
- **Error patterns:** Systematic biases identified in its own reasoning

This memory enables adaptation. An agent that recommended BUY on NVDA based on sentiment three months ago, and the result was negative, should incorporate that experience into future sentiment-based analyses.

**Implementation:** Agent memory is stored as structured records in the Agent Interaction Dataset (Family E) and made available to the agent through its context window or through retrieval.

**Open Question:** How should agent memory be weighted over time? Recent memories are more relevant but fewer. Distant memories provide more data but may reflect obsolete market conditions. A decay function is needed, but its parameters must be determined empirically.

---

## 11. Knowledge Systems

### 11.1 The Financial Memory Problem

Financial analysis requires access to large volumes of heterogeneous information: annual reports, earnings calls, macroeconomic analyses, sector studies, historical precedents. No language model's context window is sufficient to hold all relevant information simultaneously.

Therefore, HiFi requires a structured knowledge system that can:

1. Store large volumes of financial documents
2. Retrieve relevant information efficiently
3. Represent relationships between entities (companies, sectors, events, indicators)
4. Provide context-aware retrieval that understands financial relationships

### 11.2 RAG (Retrieval-Augmented Generation)

**Purpose:** Retrieve relevant text passages from a document store to augment agent context.

**Components:**

- **Document ingestion:** Parse and chunk financial documents
- **Embedding:** Convert chunks to dense vector representations
- **Vector store:** Index embeddings for similarity search
- **Retrieval:** Given an agent query, find the most relevant chunks
- **Augmentation:** Inject retrieved chunks into agent prompt

**Chunking Strategy:**

This is a critical design decision that affects retrieval quality.

| Parameter | Options | Trade-offs |
|---|---|---|
| Chunk size | 256, 512, 1024, 2048 tokens | Smaller = more precise retrieval, less context. Larger = more context, less precise. |
| Overlap | 0%, 10%, 20%, 50% | More overlap = better boundary handling, more storage and computation |
| Chunking method | Fixed-size, sentence-based, paragraph-based, semantic | Fixed = simple but may split concepts. Semantic = preserves meaning but more complex |

**Open Question:** What is the optimal chunking strategy for financial documents specifically? Earnings calls have a different structure than 10-K filings, which differ from news articles. A single chunking strategy may be suboptimal. This should be evaluated empirically by measuring retrieval precision and downstream agent performance across strategies.

**Open Question:** What embedding model provides the best retrieval quality for financial text? General-purpose embeddings (e.g., nomic-embed-text, BGE) may not capture domain-specific semantics well. A financial-domain embedding model or a fine-tuned embedding model may perform better. This must be measured.

### 11.3 GraphRAG

**Purpose:** Enrich retrieval with structural knowledge about relationships between entities.

Standard RAG retrieves text chunks based on semantic similarity. GraphRAG additionally leverages a knowledge graph that represents relationships:

- Company → Sector → Industry
- Company → Competitor
- Company → Supplier → Customer
- Event → Affected Companies
- Macro Factor → Sensitive Sectors
- Indicator → Correlated Indicators

**Why GraphRAG over plain RAG:**

Plain RAG may retrieve a passage about Apple's revenue without understanding that Apple is in the Technology sector, competes with Samsung, depends on TSMC for chip manufacturing, and is sensitive to consumer spending cycles. GraphRAG can traverse these relationships to provide more contextually relevant retrieval.

**Open Question:** Does GraphRAG provide measurable improvement over plain RAG for financial analysis tasks? The additional complexity (knowledge graph construction, maintenance, query expansion) is only justified if retrieval quality improves demonstrably. This is an empirical question.

**Open Question:** Should the knowledge graph be constructed manually (curated), semi-automatically (entity extraction + human validation), or fully automatically (LLM-extracted)? Each approach has different quality/cost trade-offs.

### 11.4 Technology Decision: LangGraph vs. LangChain

**Context:** LangGraph and LangChain serve different but overlapping roles.

| Aspect | LangChain | LangGraph |
|---|---|---|
| Primary purpose | Chain-based LLM application framework | Graph-based agent orchestration |
| Architecture | Sequential chains with tools | Stateful graphs with cycles |
| Agent support | ReAct, function-calling agents | Custom agent workflows with state |
| State management | Limited | First-class (checkpointing, branching) |
| Complexity | Higher abstraction, sometimes opaque | Lower abstraction, more explicit control |
| Suitability for HiFi | Could work but fights the multi-agent pattern | Natural fit for agent orchestration with state |

**Preliminary Rationale for LangGraph:** HiFi's agent orchestration requires:

- Multiple agents running independently then converging
- State that persists across analysis steps (agent votes, confidence values)
- Conditional branching (e.g., trigger contrarian only if consensus exceeds threshold)
- Checkpointing for reproducibility

LangGraph's graph-based model maps more naturally to these requirements than LangChain's chain-based model.

**Open Question:** This decision must be validated through a prototype. Build a minimal two-agent system in both LangChain and LangGraph. Measure: code complexity, debuggability, state management clarity, extensibility to N agents. Decide based on evidence, not marketing.

### 11.5 Long-Term Knowledge Store

Beyond document retrieval, HiFi requires persistent knowledge:

- **Trade history:** All past recommendations and their outcomes
- **Agent performance:** Calibration and accuracy records per agent
- **Market regime history:** Classified historical periods with key characteristics
- **Lessons learned:** Patterns identified from agent errors

This store is conceptually separate from the document retrieval system. It is closer to a structured database than a vector store.

**Open Question:** Should this be implemented as a relational database, a graph database, or a hybrid? The trade-off is between query flexibility (graph) and simplicity (relational). For a v1 system, a well-designed relational schema may be sufficient.

### 11.6 Spanner / Spanner Simulation

**Context:** Google Cloud Spanner is a globally distributed, strongly consistent relational database. It is relevant to HiFi in two ways:

1. **As a technology to learn:** Understanding distributed databases with strong consistency is valuable for the educational journey
2. **As an infrastructure option:** If HiFi ever scales beyond a single machine, Spanner's consistency guarantees would be valuable for maintaining data integrity across nodes

**For v1 (local):** Spanner itself is cloud-only. A Spanner emulator exists for local development. Alternatively, CockroachDB provides a Spanner-compatible open-source alternative.

**Open Question:** Is the Spanner emulator (or CockroachDB) worth the complexity for a local-first system? SQLite or PostgreSQL may be sufficient for all local use cases. Spanner becomes relevant only if HiFi is deployed as a distributed system. This decision should be deferred until the single-machine architecture is validated.

---

## 12. Collective Decision Engine

### 12.1 Purpose

The Collective Decision Engine aggregates individual agent opinions into a single collective decision. This is where HiFi's collective intelligence hypothesis is tested.

### 12.2 Aggregation Methods

#### 12.2.1 Majority Voting (Baseline)

Each agent casts one vote (Buy/Hold/Sell). The option with the most votes wins.

```
Decision = mode({v_1, v_2, ..., v_N})
```

**Properties:** Simple, transparent, equal-weight. Does not account for agent confidence or track record.

#### 12.2.2 Confidence-Weighted Voting

Each agent's vote is weighted by its confidence score:

```
Score(k) = Σ c_i · 𝟙(v_i = k)    for k ∈ {Buy, Hold, Sell}
Decision = argmax_k Score(k)
```

**Properties:** Accounts for conviction. An agent that is 90% confident has more influence than one at 55%. But confidence scores are self-reported by LLMs and may not be well-calibrated.

**Open Question:** How well-calibrated are LLM confidence estimates? If a model says "0.85 confidence," does it actually mean the model is correct 85% of the time? This must be measured. If calibration is poor, confidence-weighted voting may be worse than majority voting.

#### 12.2.3 Performance-Weighted Voting

Each agent's vote is weighted by its historical accuracy:

```
w_i = accuracy_i over last W decisions
Score(k) = Σ w_i · 𝟙(v_i = k)
Decision = argmax_k Score(k)
```

**Properties:** Adapts to demonstrated performance. Agents that are consistently wrong lose influence. Agents that are consistently right gain influence.

**Risk:** Performance weights may overfit to recent market regime. An agent that performed well in a bull market may be overweighted just as the regime changes.

#### 12.2.4 Structured Debate (Experimental)

After initial independent voting, agents engage in a structured debate:

1. Initial votes are collected
2. Agents with minority opinions present their arguments
3. All agents may revise their votes
4. Final votes are collected

**Properties:** Allows information exchange. May improve decision quality if minority agents hold valid private information. May degrade if debate causes herding.

**Open Question:** Does structured debate improve or degrade collective decision quality compared to independent voting? The literature on deliberation in groups is mixed (Sunstein, 2006). In some cases, deliberation produces "group polarization" — the group moves toward a more extreme position than the average individual. This must be measured.

#### 12.2.5 Adaptive Aggregation (Advanced)

Learn the optimal aggregation function from data:

```
Decision = f(v_1, c_1, v_2, c_2, ..., v_N, c_N, regime, ...)
```

where f is a learned function (e.g., logistic regression, small neural network) that takes agent votes, confidences, and contextual features as input.

**Properties:** Most flexible. Can capture nonlinear interactions between agents. But requires training data (agent votes + outcomes) that only becomes available after the system has operated for some time.

### 12.3 Contrarian Integration

The Contrarian Agent does not participate in voting. Instead, its output is used to:

1. **Flag risk:** If the contrarian identifies a plausible risk scenario, the collective confidence is reduced
2. **Trigger review:** If contrarian confidence exceeds a threshold, the decision is flagged for additional scrutiny
3. **Record dissent:** The contrarian's counter-thesis is always recorded alongside the decision for future analysis

**Open Question:** What is the optimal integration mechanism for contrarian information? A simple confidence discount? A veto power above a threshold? A Bayesian update on the collective probability? This must be evaluated empirically by comparing system performance with and without contrarian integration across different mechanisms.

---

## 13. Verification and Hallucination Control

### 13.1 The Hallucination Problem in Finance

When an LLM states "AAPL's P/E ratio is 28.3 and revenue grew 12% YoY," each claim is either correct or incorrect. In a general conversation, a wrong number is embarrassing. In a financial decision system, a wrong number can lead to material losses.

HiFi treats hallucination as a first-class engineering problem, not an unavoidable side effect.

### 13.2 Verification Architecture

```
Agent Output
    │
    ▼
┌──────────────────────┐
│  Claim Extractor     │ ← Identifies all factual claims in agent output
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Claim Classifier    │ ← Classifies each claim as Objective or Interpretive
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐  ┌──────────┐
│Verify  │  │  Label   │
│Against │  │  As      │
│MCP     │  │Interpret.│
│Server  │  └──────────┘
└───┬────┘
    │
    ▼
┌──────────────────────┐
│  Verification Result │
│  ✓ Confirmed         │
│  ✗ Contradicted      │
│  ? Unverifiable      │
└──────────────────────┘
```

### 13.3 Verification Categories

| Category | Definition | Example | Action |
|---|---|---|---|
| Confirmed | Claim matches deterministic computation | "P/E is 28.3" and MCP returns 28.3 | Accept |
| Contradicted | Claim conflicts with deterministic computation | "P/E is 28.3" but MCP returns 31.7 | Reject and flag |
| Unverifiable | Claim cannot be checked against available data | "Management seemed optimistic" | Label as interpretive |
| Fabricated | Claim references data that does not exist | "Q3 2024 revenue was $X" but Q3 hasn't been reported | Reject and flag |

### 13.4 Hallucination Metrics

**Hallucination Rate:**

```
HR = (Contradicted + Fabricated) / Total_Objective_Claims
```

**Grounding Rate:**

```
GR = Confirmed / Total_Objective_Claims
```

**Unverifiable Rate:**

```
UR = Unverifiable / Total_Claims
```

**Target:** HR < 0.02 (fewer than 2% of objective claims are hallucinated). GR > 0.95 (more than 95% of objective claims are confirmed).

### 13.5 Consistency Verification

Beyond individual claim verification, the system checks for inter-agent consistency:

- Do agents citing the same metric report the same value?
- Do agents make contradictory claims about the same entity?
- Does the collective rationale contain internal contradictions?

**Cross-Agent Contradiction Rate:**

```
CCR = Number_of_contradictions / Number_of_shared_claims
```

### 13.6 Implementation via MCP

The Verification Engine calls the same MCP servers that agents use for data access. This ensures verification uses the same source of truth as the original computation. A claim like "revenue grew 12% YoY" is verified by calling the Financial Calculator MCP server with the same parameters and comparing results.

**Open Question:** How should claims be extracted from free-text agent outputs? Options include: regex patterns for known metric formats, LLM-based claim extraction (using a separate small model), or structured output templates that make claims machine-parseable by construction. The latter (structured templates) is strongly preferred because it eliminates the extraction problem entirely.

---

## 14. Observability

### 14.1 Purpose

Observability in HiFi serves three functions:

1. **Operational monitoring:** Is the system working correctly?
2. **Scientific instrumentation:** What are the dynamics of the agent population?
3. **Auditability:** Can every decision be traced back to its inputs?

### 14.2 Observability Stack

**Primary Technology: LangFuse**

LangFuse provides tracing, evaluation, and monitoring for LLM applications. It is chosen because:

- Open-source and self-hostable (aligns with local-first principle)
- Supports traces, spans, and generations
- Provides evaluation and scoring
- Has a dashboard for visual analysis
- Integrates with LangChain/LangGraph

**Open Question:** Is LangFuse sufficient for all observability needs, or is additional infrastructure needed (e.g., Prometheus for system metrics, Grafana for dashboards, custom logging for agent interactions)? This should be determined by what LangFuse can and cannot capture.

### 14.3 What is Observed

#### System Level

| Metric | Description | Collection Frequency |
|---|---|---|
| Inference latency per agent | Time from prompt to response | Every inference |
| Memory usage per model | RAM/VRAM consumption | Every inference |
| Total analysis time per stock | End-to-end time | Every analysis |
| MCP server response time | Tool call latency | Every call |
| Error rate | Failed inferences, timeouts, MCP failures | Every event |

#### Agent Level

| Metric | Description | Collection Frequency |
|---|---|---|
| Decision distribution | Proportion of Buy/Hold/Sell per agent | Rolling window |
| Confidence distribution | Mean, std, min, max of confidence | Rolling window |
| Accuracy | Correctness of past decisions | Rolling (lagged by horizon) |
| Calibration | Confidence vs. actual accuracy | Rolling |
| Hallucination rate | Per-agent hallucination frequency | Every analysis |
| Rationale quality | Coherence and grounding of explanations | Sampled |

#### Collective Level

| Metric | Description | Collection Frequency |
|---|---|---|
| Disagreement entropy | H (see Section 5.6.1) | Every collective decision |
| Opinion dispersion | D (see Section 5.6.2) | Every collective decision |
| Herding coefficient | κ (see Section 5.6.3) | Rolling window |
| Consensus stability | S (see Section 5.6.4) | Rolling window |
| Diversity index | Page diversity (see Section 5.6.5) | Rolling window |
| Contrarian trigger rate | How often contrarian flags concerns | Rolling window |

#### Financial Level

| Metric | Description | Collection Frequency |
|---|---|---|
| Directional accuracy | Did Buy/Sell align with actual price movement? | Per decision (lagged) |
| Return attribution | Which agents contributed to profitable/unprofitable decisions? | Per decision (lagged) |
| Regime performance | Performance segmented by market regime | Periodic |

### 14.4 Drift Detection

Over time, both the data distribution and agent behavior may change. Drift detection monitors for:

- **Data drift:** Input feature distributions shift (e.g., volatility regime change)
- **Concept drift:** Relationships between features and outcomes change
- **Agent drift:** Individual agent behavior changes over time (e.g., increasing bias toward one decision)
- **Collective drift:** Ensemble properties change (e.g., increasing herding)

**Implementation:** Statistical tests comparing recent distributions to baseline distributions:

- Kolmogorov-Smirnov test for continuous features
- Chi-squared test for categorical distributions
- CUSUM (cumulative sum) for detecting change points in time series

**Open Question:** What window sizes and significance levels should trigger drift alerts? Too sensitive = false alarms. Too insensitive = missed regime changes. This requires calibration.

---

## 15. Evaluation Framework

### 15.1 Multi-Dimensional Evaluation

HiFi is evaluated across five dimensions. No single metric defines success.

### 15.2 Financial Metrics

| Metric | Definition | Purpose |
|---|---|---|
| Directional Accuracy | % of decisions where predicted direction matched actual | Primary predictive metric |
| Hit Rate by Regime | Directional accuracy segmented by market regime | Regime robustness |
| Annualized Return | Geometric mean return of portfolio following HiFi decisions | Absolute performance |
| Sharpe Ratio | (Mean return - risk-free rate) / std(returns) | Risk-adjusted performance |
| Sortino Ratio | (Mean return - risk-free rate) / std(downside returns) | Downside-risk-adjusted performance |
| Calmar Ratio | Annualized return / max drawdown | Return per unit of tail risk |
| Maximum Drawdown | Largest peak-to-trough decline | Worst-case loss |
| Turnover | Average position changes per period | Trading activity / transaction cost proxy |

**Evaluation Protocol:**

- Walk-forward validation with purged cross-validation (López de Prado, 2018)
- No overlap between training and evaluation periods
- Embargo period between train and test to prevent leakage
- Results reported with bootstrap confidence intervals
- Statistical comparison against baselines using paired bootstrap or Diebold-Mariano test

### 15.3 AI Quality Metrics

| Metric | Definition | Purpose |
|---|---|---|
| Hallucination Rate | Contradicted claims / total objective claims | Factual reliability |
| Grounding Rate | Confirmed claims / total objective claims | Verifiability |
| Cross-Agent Contradiction Rate | Contradictions / shared claims | Internal consistency |
| Retrieval Precision@K | Relevant retrieved chunks / K | RAG quality |
| Retrieval Recall | Relevant retrieved / total relevant | RAG completeness |
| Answer Faithfulness | Degree to which response uses retrieved context | RAG grounding |

### 15.4 Complexity Metrics

| Metric | Definition | Purpose |
|---|---|---|
| Disagreement Entropy (H) | Shannon entropy of vote distribution | Diversity of opinions |
| Opinion Dispersion (D) | Mean absolute deviation of confidence | Conviction spread |
| Herding Coefficient (κ) | Average agreement with majority | Independence measurement |
| Consensus Stability (S) | Proportion of unchanged consecutive decisions | Decision persistence |
| Page Diversity | Average squared deviation from collective | Ensemble diversity value |
| Contrarian Activation Rate | Proportion of decisions where contrarian flags concern | Minority voice frequency |
| Diversity-Performance Correlation | Correlation between H and decision quality | Core hypothesis test |

### 15.5 Engineering Metrics

| Metric | Definition | Purpose |
|---|---|---|
| End-to-End Latency | Time from data ingestion to decision | Operational viability |
| Per-Agent Latency | Inference time per agent | Bottleneck identification |
| Memory Footprint | Peak RAM/VRAM usage | Hardware feasibility |
| Cost per Decision | Compute cost (energy) per analysis | Economic viability |
| Reproducibility Score | % of results exactly reproduced from same inputs | Scientific reliability |
| System Uptime | Availability during paper trading | Operational reliability |

### 15.6 Baseline Comparisons

| Baseline | Description | Purpose |
|---|---|---|
| Random | Random Buy/Hold/Sell with equal probability | Null model (any system must beat this) |
| Single Best Agent | Best individual agent without ensemble | Ensemble value measurement |
| Equal-Weight Buy-and-Hold | Buy all stocks equally, hold | Market benchmark |
| Momentum Strategy | Simple momentum (buy winners, sell losers) | Quantitative baseline |
| Cloud LLM (GPT-4) | Same analysis pipeline with cloud model | Local vs. cloud comparison |
| FinRobot-like | Existing open-source financial AI | Competitive positioning |

### 15.7 Ablation Studies

| Ablation | What is Removed | What it Measures |
|---|---|---|
| Remove one agent | Each agent removed in turn | Marginal contribution of each agent |
| Remove contrarian | Contrarian agent | Value of contrarian challenge |
| Remove verification | Verification layer | Impact on hallucination rate and decision quality |
| Remove RAG/GraphRAG | Knowledge retrieval | Value of external knowledge |
| Homogenize models | All agents use same model | Value of model diversity |
| Remove fine-tuning | Use base models only | Value of domain fine-tuning |
| Random weighting | Replace aggregation with random weights | Is aggregation better than chance? |

---

## 16. Deployment Strategy

### 16.1 Containerization

The entire HiFi system is containerized for reproducible deployment.

**Architecture:**

```
docker-compose.yml
│
├── hifi-data          # Data acquisition and engineering
├── hifi-knowledge     # Vector store + knowledge graph
├── hifi-models        # Model serving (Ollama or llama.cpp)
├── hifi-agents        # Agent orchestration (LangGraph)
├── hifi-verification  # Verification engine
├── hifi-observability # LangFuse + monitoring
├── hifi-execution     # Paper trading interface
└── hifi-api           # User-facing API / dashboard
```

Each container has a well-defined interface. Containers communicate through internal networks. External access is limited to data APIs (inbound) and paper trading API (outbound).

### 16.2 Hardware Requirements

| Tier | Hardware | Capability |
|---|---|---|
| Minimum | M2 Pro, 16GB RAM | 1-2 agents (7B quantized), limited scope |
| Recommended | M3 Pro/Max, 32-64GB RAM | 3-5 agents (7B-14B), full scope |
| Ideal | M3 Ultra, 128-192GB RAM | 5-7 agents (14B-32B), parallel inference |

### 16.3 Deployment Modes

| Mode | Description | Use Case |
|---|---|---|
| Development | Single-machine, hot-reload, debug logging | Building and testing |
| Research | Full pipeline with observability, no execution | Evaluation and analysis |
| Paper Trading | Full pipeline with Alpaca paper execution | Live validation |
| Production | Hardened, monitored, with circuit breakers | Future (post-thesis) |

---

## 17. Decision Journal

This section records architectural decisions that require explicit justification. Each entry follows the format:

**Decision ID** — **Date** — **Decision** — **Rationale** — **Alternatives Considered** — **Status**

### DJ-001: Project Name

- **Decision:** HiFi (High-Fidelity Financial Intelligence)
- **Rationale:** "High-Fidelity" captures the deterministic-first, verifiable, reproducible philosophy. Short, memorable, professional. The audio association ("Hi-Fi") reinforces the idea of faithful signal reproduction.
- **Alternatives Considered:** LOCALFI (too infrastructure-focused), HiveFi (hive metaphor), CortexFi, AegisFI, FinEcos
- **Status:** Accepted

### DJ-002: Deterministic-First Principle

- **Decision:** All objective computations delegated to deterministic engines via MCP
- **Rationale:** Eliminates hallucination on verifiable claims by construction. Makes every numerical output auditable. Reduces the surface area where LLMs can introduce errors.
- **Alternatives Considered:** Allow LLMs to compute financial metrics directly (rejected: unverifiable, unreproducible)
- **Status:** Accepted

### DJ-003: MCP as Communication Backbone

- **Decision:** Use Model Context Protocol for agent-tool communication
- **Rationale:** Standardized protocol. Supports tool discovery. Clean separation between agent reasoning and tool execution. Growing ecosystem.
- **Alternatives Considered:** Direct function calls (tighter coupling, less auditable), REST APIs between services (more overhead, less standardized for LLM tools)
- **Status:** Accepted (pending prototype validation)

### DJ-004: LangGraph over LangChain

- **Decision:** Prefer LangGraph for agent orchestration
- **Rationale:** Graph-based state management maps naturally to multi-agent workflows with branching, convergence, and state persistence
- **Alternatives Considered:** LangChain (chain-based, less natural for multi-agent), CrewAI (higher abstraction, less control), custom orchestration (more work, less ecosystem)
- **Status:** Tentative (requires prototype comparison)

### DJ-005: Rejection of Golden Dataset

- **Decision:** Use layered dataset families with explicit assumptions instead of a single "golden" dataset
- **Rationale:** Financial decisions are multi-objective and context-dependent. No single labelling represents ground truth. Reference strategies are useful benchmarks, not Platonic ideals.
- **Alternatives Considered:** Single golden dataset (rejected: epistemologically unsound)
- **Status:** Accepted

*Additional entries to be added as decisions are made during implementation.*

---

## 18. Open Questions

This section consolidates all open questions raised throughout the document. Each question has an ID, a category, and a proposed method of resolution.

### Architecture

| ID | Question | Resolution Method |
|---|---|---|
| OQ-A01 | What is the MCP communication overhead? | Prototype measurement |
| OQ-A02 | LangGraph vs. LangChain for HiFi's specific needs? | Build minimal system in both, compare |
| OQ-A03 | Is Spanner emulator/CockroachDB warranted for local deployment? | Defer until single-machine architecture validated |
| OQ-A04 | Sequential vs. parallel agent inference on M-series hardware? | Benchmark with target models |

### Models

| ID | Question | Resolution Method |
|---|---|---|
| OQ-M01 | How much fine-tuning data is needed for measurable improvement? | Empirical evaluation with increasing dataset sizes |
| OQ-M02 | Does fine-tuning reduce inter-agent diversity? | Measure diversity metrics before and after fine-tuning |
| OQ-M03 | Which embedding model is best for financial text? | Retrieval benchmark across candidates |

### Knowledge

| ID | Question | Resolution Method |
|---|---|---|
| OQ-K01 | Optimal chunking strategy for financial documents? | A/B test across strategies measuring retrieval precision |
| OQ-K02 | Does GraphRAG improve over plain RAG for financial analysis? | Comparative evaluation on held-out questions |
| OQ-K03 | Manual vs. automatic knowledge graph construction? | Quality assessment of extracted entities/relations |

### Agents

| ID | Question | Resolution Method |
|---|---|---|
| OQ-AG01 | Should the Contrarian see other agents' outputs? | Test both designs, compare outcomes |
| OQ-AG02 | How should agent memory decay over time? | Evaluate different decay functions empirically |
| OQ-AG03 | How well-calibrated are LLM confidence estimates? | Calibration analysis on historical decisions |
| OQ-AG04 | Are LLM agents from different families truly independent? | Measure pairwise correlation of agent decisions |

### Evaluation

| ID | Question | Resolution Method |
|---|---|---|
| OQ-E01 | What window sizes and thresholds for drift detection? | Calibrate on historical data with known regime changes |
| OQ-E02 | How to adapt Page diversity for categorical predictions? | Compare KL divergence and other categorical diversity measures |

### Data

| ID | Question | Resolution Method |
|---|---|---|
| OQ-D01 | How far back is reliable point-in-time data available from free sources? | Empirical data quality assessment |
| OQ-D02 | What synthetic data methodology best captures tail events? | Compare generation methods against historical tails |
| OQ-D03 | What free news/sentiment sources provide adequate coverage? | Survey and coverage analysis |

---

## 19. Success Criteria

### 19.1 Capstone Success (WQU MScFE)

The project demonstrates:

1. A functional end-to-end pipeline from data to decision
2. Multiple agents producing structured analysis
3. Collective decision mechanism
4. Verification of agent claims
5. Paper trading execution with recorded results
6. Clear documentation of methodology and results

### 19.2 Engineering Success

1. The system runs end-to-end on target hardware without cloud dependencies
2. Results are reproducible (same inputs produce same outputs)
3. All decisions are traceable to their inputs
4. The system is containerized and deployable by external users
5. Hallucination rate < 2% on objective claims

### 19.3 Scientific Success

1. Ensemble outperforms average individual agent (H1 supported)
2. Diversity contribution is measurable and positive (H2 explored)
3. Verification layer demonstrably reduces hallucination (H4 supported)
4. Complexity metrics reveal meaningful patterns in agent dynamics
5. Results are reported with statistical confidence intervals

### 19.4 Community Success

1. Code is open-source with documentation
2. Datasets are released with dataset cards
3. Evaluation protocols are reproducible
4. At least one external user can deploy the system from documentation alone

### 19.5 Long-Term Success (Post-Graduation)

1. Complexity science analysis produces publishable insights
2. Agent ecology dynamics are characterized across market regimes
3. The system serves as a research platform for collective intelligence studies
4. Conference or journal submission based on HiFi research

---

## 20. References

Ordered alphabetically. All references are real and verifiable.

- Arthur, W. B. (2021). *Foundations of Complexity Economics*. Nature Reviews Physics, 3, 136-145.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Cont, R. (2001). Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues. *Quantitative Finance*, 1(2), 223-236.
- Dietterich, T. G. (2000). Ensemble Methods in Machine Learning. *Lecture Notes in Computer Science*, 1857, 1-15.
- Farmer, J. D., & Foley, D. (2009). The Economy Needs Agent-Based Modelling. *Nature*, 460, 685-686.
- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86-92.
- Holland, J. H. (1992). Complex Adaptive Systems. *Daedalus*, 121(1), 17-30.
- Knight, F. H. (1921). *Risk, Uncertainty and Profit*. Houghton Mifflin.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Mitchell, M. (2009). *Complexity: A Guided Tour*. Oxford University Press.
- Nemeth, C. J., Brown, K., & Rogers, J. (2001). Devil's Advocate versus Authentic Dissent: Stimulating Quantity and Quality. *European Journal of Social Psychology*, 31(6), 707-720.
- Page, S. E. (2007). *The Difference: How the Power of Diversity Creates Better Groups, Firms, Schools, and Societies*. Princeton University Press.
- Schwenk, C. R. (1990). Effects of Devil's Advocacy and Dialectical Inquiry on Decision Making: A Meta-Analysis. *Organizational Behavior and Human Decision Processes*, 47(1), 161-176.
- Sunstein, C. R. (2006). *Infotopia: How Many Minds Produce Knowledge*. Oxford University Press.
- Surowiecki, J. (2005). *The Wisdom of Crowds*. Anchor Books.
- Woolley, A. W., Chabris, C. F., Pentland, A., Hashmi, N., & Malone, T. W. (2010). Evidence for a Collective Intelligence Factor in the Performance of Human Groups. *Science*, 330(6004), 686-688.
- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems*. John Wiley & Sons.

---

## 21. Glossary

| Term | Definition |
|---|---|
| Agent | A specialized LLM instance configured to analyze financial data from a specific perspective |
| Collective Intelligence | The property of a group performing better than its individual members |
| Deterministic-First | Design principle: prefer verifiable computation over LLM generation |
| Disagreement Entropy | Shannon entropy of the agent vote distribution; measures opinion diversity |
| Ensemble | A collection of models whose outputs are aggregated |
| GraphRAG | Retrieval-Augmented Generation enhanced with knowledge graph traversal |
| Hallucination | A factual claim generated by an LLM that is not supported by available evidence |
| Herding | The tendency of agents to converge on the majority opinion regardless of private information |
| HiFi | High-Fidelity Financial Intelligence — the project name |
| Look-Ahead Bias | Using future information in a way that would not be possible in real-time |
| MCP | Model Context Protocol — standardized interface for LLM tool usage |
| Narrative Look-Ahead Bias | Creating explanations informed by known future outcomes |
| Point-in-Time | Data as it was available at a specific historical date, not as later revised |
| RAG | Retrieval-Augmented Generation — augmenting LLM context with retrieved documents |
| Reference Strategy | A labelling methodology based on explicit assumptions, not ground truth |
| Regime | A period of distinct market behavior (bull, bear, crisis, sideways) |
| Survivorship Bias | The error of analyzing only entities that survived to the present, ignoring failures |
| Walk-Forward Validation | Evaluation method that respects temporal ordering of data |

---

*This document is the David. It describes what HiFi aspires to be. The execution plan — what we build first, second, third — is a separate document derived from this one. The David does not change because reality is hard. Reality adapts to the David, one layer at a time.*
