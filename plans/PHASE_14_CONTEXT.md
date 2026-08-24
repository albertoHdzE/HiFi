# Phase 14: Infrastructure — Model Diversity, Scale Expansion, MCP Tools
## Context and Pre-Phase Decisions

**Gathered:** 2026-06-16
**Branch:** phase14/heterogeneous-ensemble
**Status:** PLANNING — DJ-088 through DJ-094 recorded; PHASE_14_PLAN.md not yet written

---

## Background: Why Phase 14 Is a Pivot

Phase 13 closed the homogeneous-ensemble chapter with two empirical findings that
demand architectural change before paper trading is meaningful:

1. **OQ-M02 (Phase 12.1):** Fine-tuning collapses diversity to entropy=0.000 when
   Technical + Fundamental share the same base model (qwen2.5-coder-32b). Herding
   reached 100% on every evaluation date. The ensemble is functionally a single agent.

2. **OQ-D04 (Phase 13):** Adding a second debate round reduces herding by only Δ=-0.021
   (below the 0.05 significance threshold). Multi-round deliberation cannot fix
   homogeneity. The diversity deficit is architectural, not procedural.

**Central scientific claim for Phase 14+:** A heterogeneous LLM ensemble — one where
each agent comes from a distinct model organization (different pre-training corpus,
architecture, RLHF procedure) — will produce higher Information Coefficient (IC) and
lower herding than a homogeneous ensemble, stable across market regimes.

This is the hypothesis Phase 15 will test quantitatively.

Phase 14 also corrects a scope problem: "Paper Trading" in HIFI_PROTOCOL_V1.md bundled
infrastructure, historical simulation, and live execution into one phase. These are
separated into Phases 14, 15, and 16.

---

## DJ-088: Phase Restructuring

**Problem:** Original Phase 14 (Paper Trading) conflates three distinct activities:
(a) infrastructure build-out, (b) walk-forward historical simulation, (c) live IBKR
execution. These have different timelines, risks, and scientific objectives. Bundling
them creates a phase that cannot be completed atomically.

**Decision:** Replace original Phase 14 with three phases:

```
Phase 14: Infrastructure + Model Diversity + Scale
Phase 15: Historical Walk-Forward Simulation
Phase 16: Live Paper Trading (IBKR)
```

Original phases 15-18 renumbered:

```
Phase 17: Ablation Studies + Capstone Deliverable  (was Phase 17)
Phase 18: Publication + Open Source Release         (merges old 15, 16, 18)
```

Containerization (old Phase 15) is absorbed into Phase 18 — it produces no scientific
output on its own and is a prerequisite for open source release, not a standalone phase.

**Critical path for capstone (minimum viable thesis):**
Phases 0–13 + 14 + 15 + 17. Phase 16 (live trading) strengthens the argument but
can be deferred if timeline requires it.

**Updated phase overview:**

| Phase | Name | Primary output |
|---|---|---|
| 14 | Infrastructure: Model Diversity, Scale, MCP Tools | 5-org ensemble; 100-stock pipeline; MCP tools; episodic RAG |
| 15 | Historical Walk-Forward Simulation | IC/IR/Sharpe across 2004-2025; 5-period walk-forward; clean-room validation |
| 16 | Live Paper Trading | IBKR execution; real-time signal loop; growing episodic memory |
| 17 | Ablation Studies + Capstone | Remove-one-agent ablation; thesis write-up; WQU submission |
| 18 | Publication + Open Source | Containerization; dataset cards; arXiv preprint; GitHub release |

---

## DJ-089: 5-Organization Ensemble + Sequential RAG Debate

### Part A: Model Diversity Upgrade

**Problem:** 5 of 6 agents are Alibaba/Qwen variants. Phase 12.1 confirmed this
produces herding=1.000 when fine-tuned agents are deployed. Architectural diversity
(distinct organizations, tokenizers, pre-training corpora, RLHF procedures) is the
only reliable path to genuine ensemble disagreement.

**Target assignments (confirm against LM Studio availability in Phase 14 planning):**

| Agent | Target Organization | Model candidate | Notes |
|---|---|---|---|
| Fundamental | Meta (Llama) | Llama 3.3 70B or Llama 3.1 70B | Dense autoregressive; strong instruction following |
| Technical | Alibaba (Qwen 2.5) | qwen2.5-coder-32b + technical_v2 | Keep FT adapter — only adapter retained |
| Risk | Mistral AI | Mistral Small 3.1 or Mistral 7B v0.3 | Sliding window attention; efficient; different RLHF |
| Macro | DeepSeek | DeepSeek-R1-Distill or DeepSeek-V3 | Chain-of-thought reasoning; economic competency |
| Sentiment | Google | Gemma 3 12B or 27B (NOT E4B — DJ-086) | Instruction-following quality; different architecture |
| Contrarian | Alibaba (Qwen 3.5) | qwen3.5-35b reasoning | Keep — already a different sub-family from qwen2.5 |

**Note on fundamental_v1 adapter:** The fundamental_v1 adapter (trained on qwen2.5-coder-32b)
is deprecated by this change. Llama 3.3 70B requires a new adapter if fine-tuning
is warranted. Phase 14 first establishes baselines on the new model; fine-tuning
decision made in Phase 14 after baseline HR/GR are measured.

Trade-off accepted: model-level diversity > adapter reuse. OQ-M02 evidence is clear —
the adapter collapses diversity; a new base model opens it.

**Phase 14 gate before implementation:** Run 1-ticker diagnostic (AAPL, 2023-03-31)
on each proposed model to confirm:
1. JSON schema compliance (structured output format reliability)
2. HR/GR ≥ Phase 13 baseline for Fundamental; GR ≥ 0.8 for Technical
3. SGR ≥ 0.5 for Sentiment (cite verbatim, not paraphrase)
4. LM Studio load/unload cycle works without memory errors
5. Per-decision latency acceptable (< 10 min for 6 agents in batch mode)

### Part B: Sequential RAG Debate Architecture

**Problem:** Current ensemble runs agents in parallel (no inter-agent information
sharing during generation). Oxford debate (Phase 12) partially addresses this —
but debate requires all agents to produce a first round, then a deliberation step
runs. Information sharing is only at the deliberation stage, not during analysis.

**Decision:** Introduce sequential context accumulation as the primary ensemble mode
for Phase 15 and 16. Each agent writes its analysis to a shared LanceDB context;
the next agent reads prior analyses before generating its own.

```
Agent 1: Fundamental
  → Analysis_1 written to LanceDB (eval namespace, ticker+date key)

Agent 2: Technical
  → Reads Analysis_1 via LanceDB query
  → Produces Analysis_2 informed by Fundamental's view (can agree or disagree)
  → Analysis_2 written to LanceDB

Agent 3: Risk
  → Reads Analysis_1 + Analysis_2
  → Produces Analysis_3
  → Written to LanceDB

Agent 4: Macro
  → Reads Analysis_1 + Analysis_2 + Analysis_3
  → Produces Analysis_4
  → Written to LanceDB

Agent 5: Sentiment
  → Reads all prior analyses + own RAG context (SEC filings)
  → Produces Analysis_5
  → Written to LanceDB

Agent 6: Contrarian  ← always last; has maximum context
  → Reads all 5 prior analyses
  → Produces contrarian brief arguing against the consensus view
  → Written to LanceDB

Ensemble Aggregator
  → Reads all 6 analyses from LanceDB
  → Produces EnsembleDecision (same aggregation methods as Phase 9)
```

**Why this architecture is scientifically defensible:**
- Each agent forms its own independent view (not copying); it is INFORMED by, not
  constrained by, prior analyses. Like a committee where each analyst has heard
  the prior presentations but must submit an independent written assessment.
- Contrarian's role is structurally reinforced: it always has the most context
  to argue against.
- Information flows causally, not cyclically — no oscillation risk.
- LM Studio sequential model loading (hardware constraint) becomes a feature:
  the forced sequence IS the information cascade. No workaround needed.

**Implementation:**
- `run_sequential_ensemble(ticker, date, snapshot_json, agent_order, namespace)`
  in `src/hifi/agents/ensemble_runner.py`.
- Phase 13 `run_ensemble()` (parallel, no sharing) preserved as default.
- Phase 14 introduces sequential mode; Phase 15 uses it exclusively.
- LanceDB namespace for inter-agent context: `hifi-eval-context/{date}/{ticker}/`
  (cleared between evaluation dates to prevent cross-date contamination).

**Open question OQ-P14-03:** Does sequential RAG debate improve IC vs. parallel
ensemble? Phase 15 ablation will test this by running the same walk-forward period
with both architectures.

---

## DJ-090: 100-Stock Universe Expansion

**Problem:** 3-stock evaluation (AAPL, JPM, XOM) cannot produce statistically
significant IC/IR metrics, cross-sector analysis, or a meaningful portfolio.

**Decision:** Phase 14 data acquisition target: ~100 stocks across all 11 GICS sectors.

**Data source strategy:**
- Primary: Yahoo Finance (yfinance) — free, 20+ year history, confirmed sufficient.
- Period: 2004-01-01 through 2025-12-31 (21 years, 5 market regimes).
- Fundamentals: yfinance quarterly for all 100 stocks.
- RAG corpus: EDGAR MD&A section parsing (targeted, not full filings) per ticker.
  Fixes the AAPL SGR=0.000 problem (Phase 13 E0) — boilerplate 8-K headers are
  excluded; MD&A earnings commentary and risk factors are the target sections.
- NOT using IBKR for historical data: IBKR TWS is for live execution only.
- Future upgrade (Phase 18+): Polygon.io or Refinitiv for production-grade data.

**Sector targets (~8-10 stocks each):**

| Sector | Representative tickers |
|---|---|
| Information Technology | AAPL, MSFT, NVDA, GOOGL, META, ORCL, AMD, INTC |
| Financials | JPM, BAC, GS, MS, WFC, BRK-B, C, AXP |
| Health Care | JNJ, UNH, PFE, ABBV, LLY, MRK, ABT, TMO |
| Consumer Discretionary | AMZN, TSLA, HD, MCD, NKE, TGT, BKNG, SBUX |
| Consumer Staples | PG, KO, PEP, WMT, COST, MDLZ, CL, K |
| Energy | XOM, CVX, COP, SLB, PSX, VLO, PXD, EOG |
| Industrials | GE, CAT, BA, HON, UPS, MMM, LMT, DE |
| Materials | LIN, APD, FCX, NEM, NUE, DD, PPG, VMC |
| Utilities | NEE, DUK, SO, D, SRE, AEP, EXC, XEL |
| Real Estate | AMT, PLD, CCI, EQIX, PSA, SPG, O, VICI |
| Communication Services | NFLX, DIS, T, VZ, CMCSA, EA, SNAP, PINS |

Final ticker list confirmed in Phase 14 plan. Constraint: EDGAR must cover the ticker
(all large-caps qualify). Small-cap additions deferred to Phase 16+.

**Walk-forward data split (Phase 15 methodology):**

| Period | Dates | Role |
|---|---|---|
| Training baseline | 2004-2019 | Agent calibration, initial label corpus |
| Validation | 2020-2021 | COVID regime; hyperparameter selection |
| Held-out test | 2022-2023 | Rate-shock regime (matches OQ-DR01 benchmark) |
| Walk-forward live-sim | 2024-2025 | Sequential monthly; ensemble sees data in causal order |
| Live paper trading | 2026+ | Phase 16 IBKR execution |

---

## DJ-091: Deterministic MCP Tools for Portfolio Management

**Problem:** Paper trading requires portfolio construction, risk management, and capital
allocation. Per the David (§4.1 Deterministic-First), these must be MCP tools, not
agent opinions. The ensemble is the debate engine; the deterministic layer handles math.

**Decision:** Three new MCP servers in `src/hifi/mcp/`:

### 1. `hifi-portfolio-composer`
Tool: `compose_portfolio(signals, constraints) → weights`
- Input: list of (ticker, decision, confidence) from EnsembleDecision per ticker
- Input: constraints (max_single_stock, max_sector, min_position)
- Output: dict(ticker → weight) summing to 1.0 for long positions
- Method: confidence-weighted signal strength; long-only (Phase 16 scope)
- Sector concentration: max 20% per GICS sector by default

### 2. `hifi-risk-manager`
Tool: `check_risk_limits(portfolio, new_signals, market_data) → risk_report`
- Checks: position limits, stop-loss triggers, correlation-aware concentration
- VaR/CVaR at 95% and 99% confidence (historical simulation method)
- Output: approved_trades, blocked_trades (with reasons), portfolio_risk_summary
- Fully deterministic: no LLM calls

### 3. `hifi-capital-allocator`
Tool: `allocate_capital(weights, capital, prices, commission_schedule) → orders`
- Converts target weights to share quantities given available capital
- Kelly Criterion (fractional, capped at 0.25) for position sizing
- IBKR commission model (tiered, per-share)
- Rebalancing threshold: only rebalance if drift > 5% from target weight
- Output: list of Order(ticker, side, quantity, order_type, estimated_cost)

These three tools are called AFTER the ensemble produces signals — they are downstream
of the agentic debate, never part of it. The agents discuss; the MCP tools decide sizes.

---

## DJ-092: Episodic RAG Pipeline

**Problem:** Phase 13 E4 (OQ-M03) showed memory has measurable influence (30% decision
change rate) with only 3 synthetic prior records. Real outcome-labeled episodic memory —
thousands of decisions with confirmed outcomes — could substantially improve agent
calibration in edge cases and rare regimes.

**Decision:** Build episodic RAG pipeline that grows continuously from Phase 15 onward.

```
Phase 15/16 ensemble decisions
     ↓ (logged to LangFuse — Phase 6 infrastructure already exists)
Nightly: label-outcomes job (60-day forward return → outcome_correct)
     ↓
EpisodeRecord stored in LanceDB (hifi-episodes namespace)
     fields: ticker, date, regime_label, agent_type, decision, confidence,
             collective_decision, forward_return, outcome_correct, reasoning_summary
     ↓
Agent RAG retrieval (during sequential ensemble)
     query: "similar regime + same sector + outcome_correct=True decisions"
     → injected as episodic context prefix alongside SEC RAG context
```

**Regime labels** (assigned deterministically from market data at decision date):
- `bull_low_vol`: SPY 52w return > 10%, VIX < 20
- `bear_high_vol`: SPY 52w return < -10%, VIX > 30
- `rate_shock`: Fed Funds Rate delta > 200bps trailing 6 months
- `recovery`: SPY 52w return > 20% following a bear period
- `neutral`: none of the above

Regime label is assigned at ingestion time; stored with the episode. This enables
"recall episodes from similar regimes" queries rather than only "recall same ticker" queries.

**Implementation:**
- `src/hifi/knowledge/episodic_store.py` — `EpisodicStore` wrapping LanceDB
- `src/hifi/knowledge/episodic_retriever.py` — query by (ticker, regime, sector)
- `make label-outcomes` — AUTOMATED nightly (was manual in Phase 10; automation
  is non-negotiable for Phase 16 live trading)
- Namespace: `hifi-episodes` (separate from `hifi-sec` and `hifi-graph`)

**Scientific question (OQ-P14-04):** Does episodic RAG improve individual agent
calibration (confidence vs. realized accuracy) over the Phase 16 paper trading period?
Measured by comparing calibration curves at Phase 16 start vs. Phase 16 end.

---

## DJ-093: Clean-Room Evaluation via Namespace Partitioning

**Problem:** Development accumulates LanceDB data (RAG corpus, episodes, inter-agent
context) that spans the entire historical period. Using the same namespace for Phase 15
evaluation means agents may retrieve context from future periods — data leakage.

**Decision:** Namespace partitioning (NOT full data wipe).

| Namespace prefix | Purpose | Populated by |
|---|---|---|
| `hifi-dev-*` | Development and debugging | All Phase 0-14 work |
| `hifi-eval-*` | Phase 15 walk-forward evaluation | `make eval-ingest-through DATE=` |
| `hifi-live-*` | Phase 16 live paper trading | Real-time ingestion pipeline |

**Temporal discipline for Phase 15:**
When evaluating on date D, the `hifi-eval` namespace contains ONLY:
- SEC filings published on or before D
- Episodes with decision_date < D (outcome labeled after D, but decision before D)
- Market data up to D (no look-ahead)

This is enforced by `make eval-ingest-through DATE=D`, which filters the ingestion
pipeline by publication date before writing to `hifi-eval-*`.

**Makefile targets:**
- `make eval-reset`: clears `hifi-eval-*` namespace; does NOT touch `hifi-dev-*`
- `make eval-ingest-through DATE=2022-03-31`: temporal-filtered ingest into eval namespace
- `make live-reset`: clears `hifi-live-*` at Phase 16 start
- `make label-outcomes`: labels outcomes for all episodes past 60-day horizon (automated)

---

## DJ-094: Live Paper Trading Platform

**Problem:** Phase 16 requires live execution. Options: Backtrader + IBKR vs. custom
event loop + `ib_insync`.

**Decision:** Custom async event loop + `ib_insync`. NOT Backtrader.

**Rationale:**
- Backtrader assumes millisecond-to-second signal generation. LLM ensemble takes
  2-10 minutes per ticker × 100 tickers. Backtrader's event loop architecture
  cannot accommodate this latency without deep workarounds that defeat its value.
- A custom async loop matches the actual workflow:
  ```
  every trading day morning:
    for ticker in universe:
      generate_signals()  # sequential ensemble, ~5 min/ticker for 100 tickers = overnight batch
    compose_portfolio()    # MCP deterministic
    check_risk_limits()    # MCP deterministic
    place_orders()         # ib_insync
    log_to_langfuse()      # observability
  ```
- Phase 15 (historical simulation) uses the same pipeline without IBKR calls.
  Reusing it for Phase 16 is architecturally clean — no second execution engine.
- `ib_insync` wraps IBKR API asynchronously; fits naturally in a coroutine pipeline.

**IBKR setup:**
- IB TWS or IB Gateway running locally (paper trading port 7497)
- `ib_insync` as Python client
- Order type: market orders only in Phase 16 (no limit/stop optimization)
- User to provide paper trading credentials; Claude Code accesses via config file
  (never hardcoded, never committed to git — stored in `.env` at repo root)

**Signal generation frequency:** Daily batch (overnight run for next-day orders).
Not intraday. LLM latency makes intraday impractical at 100-stock scale.

---

## What Phase 14 Produces (Completion Criteria)

Phase 14 is complete when ALL of the following pass:

1. 5-organization ensemble running: all 6 agents on confirmed models, 1271+ tests pass
2. Per-agent diagnostic baselines established (AAPL/JPM/XOM, 2023-03-31): HR/GR/SGR
3. Diversity confirmed: entropy > 0.3 on baseline dates (OQ-P14-05 answered)
4. 100-stock data pipeline: yfinance 2004-2025, all sectors, Parquet storage, tests pass
5. EDGAR MD&A ingestion: targeted section parsing for all 100 stocks → `hifi-eval` namespace
6. Sequential ensemble runner: `run_sequential_ensemble()` with inter-agent LanceDB context
7. 3 MCP tools: portfolio composer, risk manager, capital allocator — deterministic, tested
8. Episodic RAG: `EpisodicStore`, `EpisodicRetriever`, `make label-outcomes` automated
9. Namespace partitioning: dev/eval/live separation confirmed, `make eval-reset` works
10. Sentiment fine-tuning gate re-run (OQ-S01) on new Sentiment base model (if Gemma 3 12B/27B
    passes diagnostic): if POSITIVE, fine-tune; if NEGATIVE again, document and close permanently

Phase 14 produces ZERO simulation results. It produces the machinery Phase 15 needs.

---

## Open Questions Generated by Phase 14 Architecture

| ID | Question | Resolution target |
|---|---|---|
| OQ-P14-01 | Does Llama 3.3 70B comply with structured output schema reliably? | Phase 14 diagnostic |
| OQ-P14-02 | What is IC of the ensemble on 2022-2023 held-out test (100 stocks)? | Phase 15 |
| OQ-P14-03 | Does sequential RAG debate improve IC vs. parallel ensemble? | Phase 15 ablation |
| OQ-P14-04 | Does episodic RAG improve agent calibration over the Phase 16 period? | Phase 16 |
| OQ-P14-05 | Does 5-org ensemble reduce herding vs. qwen-dominant ensemble? | Phase 14 diversity test |
| OQ-P14-06 | What is Sharpe ratio of ensemble strategy vs. SPY buy-and-hold? | Phase 15 |
| OQ-AG03 | Is the LLM ensemble calibrated? (confidence vs. realized accuracy) | Phase 15 |

---

## DJ Index (Phase 14)

| DJ# | Decision |
|---|---|
| DJ-088 | Phase restructuring: 14=Infrastructure, 15=Historical sim, 16=Live trading |
| DJ-089 | 5-organization ensemble (Part A) + sequential RAG debate (Part B) |
| DJ-090 | 100-stock universe; yfinance 2004-2025; EDGAR MD&A ingestion |
| DJ-091 | Deterministic MCP tools: portfolio composer, risk manager, capital allocator |
| DJ-092 | Episodic RAG pipeline: LangFuse traces → LanceDB episodes → agent retrieval |
| DJ-093 | Clean-room evaluation: namespace partitioning (dev/eval/live) |
| DJ-094 | Platform: custom async event loop + ib_insync (not Backtrader) |

**Next DJ number: DJ-095**
