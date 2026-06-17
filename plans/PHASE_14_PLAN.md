# Phase 14: Infrastructure — Model Diversity, Scale Expansion, MCP Tools
## Epic and Ticket Plan

**Phase status:** PLANNING
**Pre-phase decisions:** DJ-088 through DJ-094 (see PHASE_14_CONTEXT.md)
**David sections:** §4.1 (Deterministic-First), §6.2 (MCP as Nervous System), §10.3 (Diversity Requirements), §10.4 (Agent Memory), §8.2 (Dataset Family A extension), §12.2 (Aggregation)
**Branch:** phase14/heterogeneous-ensemble
**Next DJ number at phase start:** DJ-095

---

## Central Scientific Claim

A heterogeneous LLM ensemble — where each agent comes from a distinct model organization
(different pre-training corpus, architecture, RLHF procedure) — produces higher
Information Coefficient and lower herding than the homogeneous Qwen-dominant ensemble
confirmed by Phase 12.1 (entropy=0.000, herding=1.000 under fine-tuning).

Phase 14 builds the infrastructure to test this claim. Phase 15 measures it.

---

## Success Criteria

- [ ] 5-organization ensemble: 6 agents across Meta/Alibaba/Mistral/DeepSeek/Google, all confirmed via diagnostic
- [ ] Diversity restored: ensemble entropy > 0.3 on Phase 12 baseline dates (vs. 0.000 with FT pair)
- [ ] 100-stock data pipeline: ~100 tickers, 2004-2025, all 11 GICS sectors, Parquet storage, >99% OHLCV completeness
- [ ] EDGAR MD&A ingestion: targeted section parsing (not full filings) for all 100 tickers → LanceDB
- [ ] Sequential ensemble: `run_sequential_ensemble()` with inter-agent LanceDB context accumulation
- [ ] 3 deterministic MCP tools: portfolio composer, risk manager, capital allocator — pure math, no LLMs
- [ ] Episodic RAG pipeline: EpisodicStore + EpisodicRetriever + automated `make label-outcomes`
- [ ] Namespace partitioning: dev/eval/live isolation confirmed, `make eval-reset` works
- [ ] Sentiment FT gate re-run (OQ-S01) on new base model; decision documented with evidence
- [ ] Tests: >= 1500 passing, 0 lint errors
- [ ] Replication notebook: frozen, no LLMs, < 60s runtime

---

## Wave Structure

```
Wave 1 (no LLM required — parallel, start immediately):
  E2: 100-stock data pipeline (yfinance bulk acquisition + EDGAR ingestion)
  E4: Deterministic MCP tools (portfolio composer, risk manager, capital allocator)
  E6: Namespace partitioning infrastructure

Wave 2 (requires LM Studio — sequential by model):
  E0: Model diagnostic + diversity upgrade (one model loaded at a time)

Wave 3 (requires Wave 2 complete):
  E1: Sentiment FT gate on new base model (needs E0 Gemma 3 diagnostic)
  E3: Sequential ensemble architecture (needs E0 confirmed model assignments)

Wave 4 (requires Wave 1 + Wave 3):
  E5: Episodic RAG pipeline (needs E3 sequential ensemble + E6 namespaces)

Wave 5 (integration, documentation, validation):
  E7: Bitacora, replication notebook, STATUS/MEMORY update
```

---

## Epic E0: Model Diagnostic + Diversity Upgrade (DJ-089)

**Objective:** Confirm each proposed model is functional, compliant, and latency-acceptable.
Update all agent model assignments. Establish new HR/GR/SGR baselines. Measure diversity
(OQ-P14-05: does 5-org ensemble restore entropy > 0.3?).

### Ticket E0-T1: Update lm_client.py with target model identifiers

File: `src/hifi/agents/lm_client.py`

Add model config entries for each Phase 14 target:
- `LLAMA_33_70B` — Meta Llama 3.3 70B (or Llama 3.1 70B if 3.3 unavailable)
- `MISTRAL_SMALL_31` — Mistral Small 3.1 (or Mistral 7B v0.3)
- `DEEPSEEK_R1_DISTILL` — DeepSeek-R1-Distill-Qwen-32B or DeepSeek-V3
- `GEMMA3_12B` — google/gemma-3-12b-it (NOT E4B — DJ-086)

Routing constants map agent_type → model_id. `_DEFAULT_*_MODEL` constants updated per
agent after E0-T3 confirms each model.

Tests: model config validation, routing logic for each agent type.

### Ticket E0-T2: Per-model diagnostic script

Script: `scripts/run_phase14_model_diagnostic.py`

For each proposed model (loaded sequentially in LM Studio):
1. Load model, run AAPL/JPM/XOM at 2023-03-31 with the agent role that model will fill
2. Check: JSON schema compliance (structured output reliability)
3. Check: HR/GR for Fundamental-type agents; SGR for Sentiment
4. Check: LM Studio load/unload cycle (no memory error after unload)
5. Record: latency_ms per call

Decision matrix per model:
- PASS: JSON valid, metric ≥ threshold, load/unload clean → proceed to E0-T3
- FAIL: document root cause (same pattern as DJ-086 diagnostic), choose fallback

Output: `tests/fixtures/baseline/phase14_model_diagnostic.json`

Thresholds:
- Fundamental (Llama): GR ≥ 0.8 (Phase 13 Risk baseline was GR=1.0 — set conservatively)
- Technical (qwen2.5 FT): GR ≥ 0.8 (inherits Phase 12.1 baseline)
- Risk (Mistral): GR ≥ 0.5 (Phase 13 E0 baseline: GR=1.0; use softer threshold for new model)
- Macro (DeepSeek): GR ≥ 0 (Phase 13 E0 Macro GR=0.0 — any claim generation counts)
- Sentiment (Gemma 3): SGR ≥ 0.5 (Phase 13 E0 target; verbatim quoting requirement)
- Contrarian (qwen3.5-35b): stays; already partially validated in Phase 13 E6

### Ticket E0-T3: Update agent model assignments

Files: `src/hifi/agents/fundamental_agent.py`, `risk_agent.py`, `macro_agent.py`,
       `sentiment_agent.py`

Update `_DEFAULT_*_MODEL` constant in each agent to the E0-T2 confirmed model.
Note: `technical_agent.py` and `contrarian_agent.py` unchanged.

The `fundamental_v1` fine-tuned adapter is deprecated. Document in Phase 14 bitacora:
"fundamental_v1 adapter trained on qwen2.5-coder-32b. Deprecated in Phase 14 when
Fundamental agent migrated to Llama 3.3 70B to restore ensemble diversity (DJ-089).
OQ-M02 evidence: adapter pair [fundamental_v1 + technical_v2] collapsed entropy to 0.000."

Tests: agent initialization with new model, prompt construction unchanged, model routing correct.

### Ticket E0-T4: Re-establish HR/GR/SGR baselines with new models

Script: `scripts/run_phase14_verification_baseline.py`

Re-run `verify_agent()` for all 5 voting agents on AAPL/JPM/XOM at 2023-03-31 with
the new model assignments confirmed in E0-T3.

Compare to Phase 13 E0 baselines:
- Risk: GR was 1.000; Mistral should maintain or improve
- Macro: GR was 0.000; DeepSeek reasoning may improve claim extraction
- Sentiment: SGR was 0.667 with qwen2.5; Gemma 3 with verbatim Rule 5 target ≥ 0.667

Output: `tests/fixtures/baseline/phase14_verification_baseline.json`

### Ticket E0-T5: Diversity validation (OQ-P14-05)

Script: `scripts/run_phase14_diversity_baseline.py`

Run full ensemble (all 6 agents, new models) on the 30 Phase 12 evaluation dates
(AAPL/JPM/XOM × 10 quarterly dates). Compute:
- entropy: Shannon entropy of collective vote distribution
- herding_coefficient: max_count / n_agents
- pairwise_diversity: fraction of agent pairs with different decisions

**OQ-P14-05:** Does 5-org ensemble restore entropy > 0.3?
- YES (entropy > 0.3): diversity hypothesis confirmed. Proceed to Phase 15.
- NO (entropy ≤ 0.3): investigate which agent pair is collapsing. Adjust model or
  prompt diversity (add explicit "form your own view" instruction). Document.

Output: `tests/fixtures/baseline/phase14_diversity_baseline.json`

---

## Epic E1: Sentiment Fine-Tuning Gate (deferred from Phase 13 OQ-S01)

**Objective:** Re-run the Phase 13 E1 gate on the new Sentiment base model (Gemma 3 12B/27B).
If Sell class ≥ 30 examples: fine-tune. If NEGATIVE again: close permanently.

**Gate (must pass before E1-T2):**
1. E0-T2 diagnostic: Gemma 3 12B/27B passes SGR ≥ 0.5 baseline
2. E1-T1: corpus gate passes (≥ 200 examples, ≥ 30 Sell)

### Ticket E1-T1: Re-run OQ-S01 corpus gate

Script: `scripts/validate_sentiment_corpus.py` (exists from Phase 13)

Re-run the same keyword-tone classifier on the Phase 7 EDGAR corpus PLUS the new
MD&A sections ingested in E2-T3. More documents = higher chance of Sell class.

If still NEGATIVE (< 30 Sell): close permanently. Sentiment fine-tuning is not
feasible with MD&A tone signals on this corpus. Document:
"OQ-S01 NEGATIVE (Phase 14 re-check): even with expanded 100-stock EDGAR corpus,
Sell class < 30. MD&A management tone is systematically optimistic. Sell signal
source must be event-driven (earnings miss, restatement, guidance cut) rather than
general tone. Deferred to live paper trading data (Phase 16)."

If POSITIVE: proceed to E1-T2.

### Ticket E1-T2: Sentiment fine-tuning on Gemma 3 base (if gate passes)

Investigate Gemma 3 MLX LoRA compatibility in `venvs/finetune/`.
If compatible: adapt `mlx_lm.lora` command for Gemma 3 architecture.
If not: document limitation, use PEFT/QLoRA via HuggingFace as alternative.
Deploy threshold: SGR ≥ 0.720 (same as Phase 13 target).

Makefile target: `finetune-sentiment-v2`

### Ticket E1-T3: Three-tier evaluation and deploy/close decision

Same three-tier structure as Phase 13 E1-T4:
- Tier 1 (SGR): compare base vs. fine-tuned on AAPL/JPM/XOM 2023-03-31
- Tier 2 (accuracy): forward accuracy on 10 evaluation dates
- Tier 3 (diversity): pairwise_diversity(sentiment_v2, others) vs. base

Output: `tests/fixtures/baseline/phase14_sentiment_evaluation.json`

---

## Epic E2: 100-Stock Data Pipeline (DJ-090)

**Objective:** Expand data universe from 3 to ~100 stocks across all 11 GICS sectors.
Acquire 21 years of OHLCV + fundamentals + EDGAR MD&A. Store in Parquet + LanceDB.

### Ticket E2-T1: Define and register ticker universe

File: `src/hifi/data/universe.py`

```python
PHASE14_UNIVERSE: list[dict] = [
    # Each entry: {"ticker": str, "sector": str, "sub_industry": str}
    # ~8-10 tickers per GICS sector, ~100 total
    {"ticker": "AAPL", "sector": "Information Technology", ...},
    {"ticker": "MSFT", "sector": "Information Technology", ...},
    ...
]
```

Full list covers all 11 GICS sectors. At minimum 8 per sector.
Constraint: ticker must have EDGAR filings coverage (all large-caps qualify).

Tests: all 11 sectors present, minimum 8 per sector, no duplicates, all tickers
have valid EDGAR identifiers.

### Ticket E2-T2: Bulk OHLCV + fundamentals acquisition (yfinance)

Script: `scripts/acquire_phase14_data.py`

Extension of Phase 1 data acquisition. For all 100 tickers:
- Daily OHLCV: 2004-01-01 to 2025-12-31
- Quarterly fundamentals (income statement, balance sheet, cash flow): same period
- Parquet storage: `data/market/{ticker}/ohlcv.parquet`, `data/fundamentals/{ticker}/quarterly.parquet`
- Provenance metadata: download date, source version, hash

Data quality checks:
- OHLCV completeness > 99% for all trading days per ticker
- Fundamentals: minimum 4 quarters per year (flag gaps)
- Adjusted close vs. close split-adjustment validation

Makefile target: `acquire-data-phase14` (expected runtime: 30-60 min, internet required)

Tests: schema validation on fixture data (not full 100-ticker run), provenance fields present,
date range completeness check on fixture.

### Ticket E2-T3: EDGAR MD&A targeted section ingestion

Script: `scripts/ingest_edgar_mda.py`
New module: `src/hifi/data/edgar_mda.py`

**Target sections (NOT full filings):**
- 10-K annual: Item 7 (MD&A) + Item 1A (Risk Factors)
- 10-Q quarterly: Item 2 (MD&A only)

**Approach:**
- SEC EDGAR full-text search API (`efts.sec.gov`) for CIK lookup and filing index
- Download section text using EDGAR filing viewer (deterministic section extraction)
- Parse filing headers to locate Item 7 and Item 2 boundaries
- Strip HTML, normalize whitespace, split into ~512-token chunks

**Period:** 2018-2025 (matches Phase 7 corpus; extends forward to 2025)

**Storage:** LanceDB namespace `hifi-dev-sec` (development default).
Phase 15 will use `hifi-eval-sec` (populated via `make eval-ingest-through DATE=`).

**Why this fixes AAPL SGR=0.000:** Phase 13 E0 retrieved an 8-K legal header (boilerplate)
because the LanceDB corpus contained only chunked full filings without section discrimination.
Targeted MD&A extraction ensures Sentiment agent retrieves earnings commentary, business
discussion, and risk language — quotable signals, not legal headers.

Makefile target: `ingest-edgar-mda` (expected runtime: 4-8 hrs for 100 tickers × 8 years)

Tests: section extraction on a fixture 10-K (AAPL 2022 10-K), chunk count,
no boilerplate leakage (legal header not present in extracted MD&A).

### Ticket E2-T4: Macro indicators expansion + regime labeling

Script: `scripts/acquire_macro_phase14.py`
New module: `src/hifi/data/regime.py`

Extend Phase 1 FRED acquisition:
- Period: 2004-01-01 to 2025-12-31
- Existing indicators: Fed Funds Rate, CPI, Unemployment, VIX
- New: 10Y Treasury yield, 10Y-2Y spread (yield curve inversion proxy)

Regime classification (deterministic):
```python
def classify_regime(date: str, market_data: dict, macro_data: dict) -> RegimeLabel:
    """Returns: bull_low_vol | bear_high_vol | rate_shock | recovery | neutral"""
```

Labels assigned at ingestion time; stored with episodic memory records (E5).

Tests: regime classification on known dates (2020-03-16 → bear_high_vol,
2022-06-30 → rate_shock, 2023-10-01 → bull_low_vol), deterministic (same inputs → same output).

---

## Epic E3: Sequential Ensemble Architecture (DJ-089b)

**Objective:** Implement causal context accumulation: each agent reads prior agents'
analyses from LanceDB before generating its own. Contrarian always last.

### Ticket E3-T1: AgentContextStore schema + LanceDB integration

File: `src/hifi/knowledge/agent_context.py`

```python
class AgentContextRecord(BaseModel):
    run_id: str           # unique per (ticker, date, ensemble_call)
    ticker: str
    date: str
    agent_type: str
    analysis_summary: str  # 200-400 token summary of agent's conclusion + key rationale
    decision: str
    confidence: float
    created_at: str

class AgentContextStore:
    def __init__(self, namespace: str = "hifi-dev-context"): ...
    def write(self, record: AgentContextRecord) -> None: ...
    def read_prior(self, run_id: str, before_agent: str) -> list[AgentContextRecord]: ...
    def clear_run(self, run_id: str) -> None: ...
```

`read_prior(run_id, before_agent)` returns all records for this run where agent_type
appears before `before_agent` in the canonical ordering.

LanceDB table: `agent_context` in the given namespace. Keyed by run_id.

Tests: write/read round-trip, read_prior returns only earlier agents, clear_run removes
all records for that run_id, empty store returns empty list.

### Ticket E3-T2: `run_sequential_ensemble()` implementation

File: `src/hifi/agents/ensemble_runner.py`

```python
def run_sequential_ensemble(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    agent_order: list[str] | None = None,  # None → canonical 6-agent order
    context_namespace: str = "hifi-dev-context",
    memory_prefixes: dict[str, str] | None = None,
) -> EnsembleOutput:
```

Canonical order: `["fundamental", "technical", "risk", "macro", "sentiment", "contrarian"]`

For each agent i in order:
1. `prior_ctx = store.read_prior(run_id, agent_type)`
2. Format `prior_ctx` as structured text block injected before agent's analytical prompt:
   ```
   [Prior Agent Analyses for {ticker} on {date}]
   Fundamental Agent: {decision} (conf={confidence:.2f}) — {analysis_summary}
   Technical Agent: ...
   ```
3. Call agent with augmented prompt (same LLM + tools as current `run_ensemble`)
4. `store.write(AgentContextRecord(run_id, ticker, date, agent_type, summary, decision, conf))`

`summary` extraction: first 300 characters of agent rationale + decision + confidence.
This is injected as context, not the full analysis — keeps prompt size controlled.

After all agents complete: run standard ensemble aggregation (same as `run_ensemble`).
Return `EnsembleOutput` — schema unchanged.

Fallback: if any individual agent call fails, log error, continue with remaining agents.
The failed agent contributes no context to later agents (equivalent to absent agent).

Tests:
- Mock LLM: verify Agent 2's prompt contains Agent 1's summary
- Mock LLM: Agent 1's prompt has no prior context block
- Contrarian's prompt contains all 5 prior summaries
- Single-agent run: no prior context injected
- Agent failure: later agents receive context from successful agents only

### Ticket E3-T3: LangGraph state update

File: `src/hifi/agents/graph.py` (or equivalent orchestration file)

Update LangGraph state graph to support sequential node execution:
- `AgentContextStore` added as shared mutable state in the graph
- Nodes execute in dependency order (not parallel map)
- Each node writes to context store before the next node runs

Tests: state graph topology (fundamental → technical → ... → contrarian), no cycles.

### Ticket E3-T4: `run_ensemble()` `sequential` parameter alias

File: `src/hifi/agents/ensemble_runner.py`

Add `sequential: bool = False` parameter to existing `run_ensemble()`. If True,
delegates to `run_sequential_ensemble()` with the same arguments. Default False
preserves all Phase 13 behavior exactly.

This ensures zero breakage of existing tests and scripts.

Tests: `sequential=False` → same output as before (mock LLM), `sequential=True` →
delegates to `run_sequential_ensemble` (mock both functions).

---

## Epic E4: Deterministic MCP Tools for Portfolio Management (DJ-091)

**Objective:** Three new MCP servers. Pure math, no LLMs. The ensemble provides signals;
these tools decide sizing, risk limits, and order quantities.

### Ticket E4-T1: `hifi-portfolio-composer` MCP server

File: `src/hifi/mcp/portfolio_composer.py`

MCP tool: `compose_portfolio`

Input schema:
```python
class SignalInput(BaseModel):
    ticker: str
    decision: Literal["Buy", "Hold", "Sell"]
    confidence: float  # [0, 1]
    sector: str        # GICS sector

class PortfolioConstraints(BaseModel):
    max_single_stock: float = 0.05   # 5% max per ticker
    max_sector: float = 0.20          # 20% max per sector
    min_position: float = 0.01        # 1% minimum if included
    long_only: bool = True
```

Algorithm:
1. Filter: keep only Buy signals (long-only mode)
2. Raw weight: `w_i = confidence_i / sum(confidence for Buy signals)`
3. Apply max_single_stock cap: if `w_i > max_single_stock`, cap and redistribute
4. Apply max_sector cap: sum weights per sector; if > max_sector, scale down pro-rata
5. Apply min_position: if `w_i < min_position`, set to 0 and redistribute
6. Normalize to sum = 1.0

Output: `dict[str, float]` (ticker → weight)

Edge cases: all Hold/Sell → return empty dict (no positions). Single Buy signal →
weight = min(confidence_i / 1.0, max_single_stock).

MCP server registration in `configs/mcp_servers.yaml`.

Tests (all deterministic — no LLMs):
- 3 Buy signals equal confidence → equal weights (1/3 each)
- 1 Buy signal: weight = max_single_stock (cap applied)
- Sector concentration: if 3 tech stocks → sector capped at 0.20, redistributed
- All Hold: returns empty dict
- Weights sum to 1.0 in all valid cases

### Ticket E4-T2: `hifi-risk-manager` MCP server

File: `src/hifi/mcp/risk_manager.py`

MCP tool: `check_risk_limits`

Input: current portfolio weights, proposed new signals, recent market data (OHLCV for
correlation and VaR computation).

Risk checks (all deterministic):
1. **VaR check (95%, 99%):** Historical simulation VaR from 252-day rolling window
   of portfolio returns. Alert if proposed portfolio VaR 95% > 5% or VaR 99% > 8%.
2. **Max drawdown limit:** If current portfolio is down > 15% from high-water mark,
   block all new Buy signals (capital preservation mode).
3. **Sector concentration:** Block if any sector > max_sector (20%) after proposed trades.
4. **Correlation-aware sizing:** Compute pairwise correlation matrix from 60-day OHLCV.
   If two positions have correlation > 0.85, reduce the lower-confidence position by 50%.

Output schema:
```python
class RiskReport(BaseModel):
    approved_signals: list[str]    # tickers with no limit breach
    blocked_signals: list[str]     # tickers blocked with reason
    block_reasons: dict[str, str]  # ticker → reason
    var_95: float
    var_99: float
    portfolio_risk_summary: str    # human-readable summary
```

Tests: VaR calculation on known fixture returns, max drawdown trigger at >15% loss,
correlation blocking at r>0.85, all clear returns all signals approved.

### Ticket E4-T3: `hifi-capital-allocator` MCP server

File: `src/hifi/mcp/capital_allocator.py`

MCP tool: `allocate_capital`

Input: target weights, available capital, current prices, current holdings, commission schedule.

Algorithm:
1. **Target shares:** `target_shares[i] = floor(target_weight[i] × capital / price[i])`
2. **Kelly cap:** Never allocate more than 25% of capital to a single position (fractional Kelly)
3. **Rebalancing threshold:** Only generate an order if `|current_weight - target_weight| > 0.05`
   (5% drift threshold prevents excessive trading)
4. **Commission model (IBKR tiered per-share):**
   - ≤ 300 shares: $0.0035/share, min $0.35
   - > 300 shares: $0.002/share
5. **Order type:** MARKET (Phase 16 scope; limit orders deferred to Phase 17+)

Output:
```python
class Order(BaseModel):
    ticker: str
    side: Literal["BUY", "SELL"]
    quantity: int
    order_type: Literal["MARKET"] = "MARKET"
    estimated_cost: float     # estimated commission
    estimated_value: float    # estimated notional

def allocate_capital(...) -> list[Order]
```

Tests: $100k capital, 10 tickers, known weights → correct share quantities, commissions
within IBKR schedule, drift threshold suppresses trivial rebalances, Kelly cap enforced.

### Ticket E4-T4: End-to-end integration test

Test file: `tests/integration/test_portfolio_pipeline.py`

Mock scenario: 10 tickers across 4 sectors, Buy/Hold/Sell mix, $250k capital.
- `compose_portfolio` → weights
- `check_risk_limits` → approved subset
- `allocate_capital` → order list
- Verify: orders sum to ≤ available capital, commission cost < 0.5% of notional,
  no blocked ticker appears in order list.

No LLMs, no network calls.

---

## Epic E5: Episodic RAG Pipeline (DJ-092)

**Objective:** Build the infrastructure that turns decision history into retrievable
episodic memory. Grows continuously from Phase 15 onward.

### Ticket E5-T1: RegimeLabel + regime classification

File: `src/hifi/data/regime.py`

```python
RegimeLabel = Literal["bull_low_vol", "bear_high_vol", "rate_shock", "recovery", "neutral"]

def classify_regime(
    date: str,
    ohlcv_series: pd.DataFrame,   # SPY daily OHLCV
    macro_series: pd.DataFrame,    # Fed Funds Rate daily
) -> RegimeLabel:
    """
    bull_low_vol:  SPY 52w return > 10% AND VIX trailing 20d avg < 20
    bear_high_vol: SPY 52w return < -10% AND VIX trailing 20d avg > 30
    rate_shock:    Fed Funds Rate delta > 2.0pp trailing 180d
    recovery:      SPY 52w return > 20% following a period where prior 52w < -10%
    neutral:       none of the above
    """
```

Deterministic. Tests: 2020-03-16 → bear_high_vol, 2022-06-30 → rate_shock,
2021-06-30 → bull_low_vol, determinism check (same inputs → same label).

### Ticket E5-T2: EpisodeRecord schema + EpisodicStore

File: `src/hifi/knowledge/episodic_store.py`

```python
class EpisodeRecord(BaseModel):
    episode_id: str    # uuid
    ticker: str
    decision_date: str
    regime_label: RegimeLabel
    sector: str
    agent_type: str    # "fundamental" | "technical" | "risk" | "macro" | "sentiment" | "ensemble"
    decision: Literal["Buy", "Hold", "Sell"]
    confidence: float
    collective_decision: str | None   # ensemble's final decision
    forward_return: float | None      # 60-day realized return (filled by label-outcomes)
    outcome_correct: bool | None      # filled by label-outcomes
    reasoning_summary: str            # 200-token excerpt from agent rationale
    labeled_at: str | None

class EpisodicStore:
    def __init__(self, namespace: str = "hifi-episodes"): ...
    def add(self, episode: EpisodeRecord) -> None: ...
    def search(
        self,
        ticker: str,
        regime: RegimeLabel,
        sector: str,
        outcome_correct: bool | None = True,
        n: int = 5,
    ) -> list[EpisodeRecord]: ...
    def get_unlabeled_past_horizon(self, horizon_days: int = 60) -> list[EpisodeRecord]: ...
```

Storage: LanceDB table `episodes` in the given namespace.
`search()` uses semantic similarity on `reasoning_summary` embeddings, post-filtered
by regime_label and outcome_correct flag.

Tests: add/search round-trip, outcome_correct filter, get_unlabeled_past_horizon
returns only episodes where `decision_date + 60 days < today AND labeled_at IS NULL`.

### Ticket E5-T3: EpisodicRetriever

File: `src/hifi/knowledge/episodic_retriever.py`

```python
class EpisodicRetriever:
    def __init__(self, store: EpisodicStore): ...

    def retrieve(
        self,
        ticker: str,
        date: str,
        agent_type: str,
        regime: RegimeLabel,
        sector: str,
        n: int = 3,
    ) -> str:
        """
        Returns formatted episodic prefix for injection into agent prompt.
        Retrieves n outcome_correct=True episodes with similar regime + sector.
        If no episodes found: returns empty string (no injection).
        """
```

Format:
```
[Episodic Memory — {n} successful past decisions in similar conditions]
Date: {d1}, Ticker: {t1}, Regime: {regime1}
Decision: {dec1} (conf={c1:.2f}) — {summary1}
Outcome: CORRECT (60d return: {r1:+.1%})
...
```

Tests: retrieval from populated store, empty store returns empty string, format
correctness, n capping, no future-date episodes returned (temporal discipline).

### Ticket E5-T4: label-outcomes automation

Script: `scripts/label_outcomes.py`

Automated (non-interactive). Called by `make label-outcomes`.

For each EpisodeRecord where `outcome_correct IS NULL AND decision_date + 60 days ≤ today`:
1. Fetch 60-day forward return from yfinance (close price at decision_date + 60 trading days)
2. Compute: `outcome_correct = (decision == "Buy" AND return > 0) OR (decision == "Sell" AND return < 0) OR (decision == "Hold" AND abs(return) < 0.05)`
3. Update EpisodeRecord in LanceDB: `forward_return`, `outcome_correct`, `labeled_at`

Makefile target: `label-outcomes` (idempotent — skip already labeled records)

Tests: labeling logic for Buy/Hold/Sell with known returns, idempotency (re-running
on already-labeled records does not change them), horizon enforcement (records < 60 days
old are not labeled even if listed as unlabeled).

### Ticket E5-T5: Episode creation in `run_sequential_ensemble()`

File: `src/hifi/agents/ensemble_runner.py`

After each successful agent call in `run_sequential_ensemble()`, create and store
an `EpisodeRecord` in the episodic store:
- `reasoning_summary`: first 200 chars of agent rationale
- `regime_label`: from `classify_regime()` at the date
- `sector`: from `PHASE14_UNIVERSE` lookup
- `forward_return`, `outcome_correct`: None at creation time (filled by label-outcomes)

`run_ensemble()` (parallel) also creates ensemble-level episodes for the collective decision
(`agent_type="ensemble"`).

Tests: episode created after each agent call, episode has correct regime, ensemble
episode created with collective_decision, forward_return=None at creation.

---

## Epic E6: Namespace Partitioning + Clean-Room Infrastructure (DJ-093)

**Objective:** Separate dev/eval/live data namespaces in LanceDB and AgentMemoryStore.
Enable temporal-filtered ingestion for Phase 15 evaluation isolation.

### Ticket E6-T1: NamespacedLanceDB abstraction

File: `src/hifi/knowledge/namespaced_store.py`

```python
class NamespacedLanceDB:
    """
    Wraps LanceDB client with a namespace prefix on all table names.
    Enables dev/eval/live separation without separate databases.
    """
    def __init__(self, db_path: str, namespace: str = "hifi-dev"): ...
    def open_table(self, table_name: str): ...  # opens {namespace}-{table_name}
    def create_table(self, table_name: str, schema): ...
    def drop_table(self, table_name: str): ...  # drops {namespace}-{table_name}
    def list_tables(self) -> list[str]: ...  # returns tables with this namespace prefix
```

All existing LanceDB wrappers (`SecStore`, `GraphStore`, `EpisodicStore`,
`AgentContextStore`) gain an optional `namespace: str = "hifi-dev"` parameter that
sets the prefix for all their tables.

Tests: table name construction, namespace isolation (dev tables not visible from eval
namespace), round-trip write/read across namespace switch.

### Ticket E6-T2: Makefile namespace management targets

File: `Makefile`

```makefile
eval-reset:
    @echo "Resetting hifi-eval-* namespace..."
    uv run python scripts/manage_namespaces.py --action reset --namespace hifi-eval

eval-ingest-through:
    @echo "Ingesting data through $(DATE) into hifi-eval namespace..."
    uv run python scripts/ingest_edgar_mda.py --namespace hifi-eval --through-date $(DATE)
    uv run python scripts/ingest_episodes.py --namespace hifi-eval --through-date $(DATE)

live-reset:
    @echo "Resetting hifi-live-* namespace..."
    uv run python scripts/manage_namespaces.py --action reset --namespace hifi-live

label-outcomes:
    uv run python scripts/label_outcomes.py
```

Script: `scripts/manage_namespaces.py --action [reset|list|status] --namespace [prefix]`

Tests: `manage_namespaces.py --action list` output, reset action drops correct tables,
no cross-namespace contamination after reset.

### Ticket E6-T3: Temporal filtering in ingestion scripts

Files: `scripts/ingest_edgar_mda.py`, `scripts/ingest_episodes.py`

Add `--through-date DATE` flag. When provided:
- EDGAR filings: only ingest filings with `period_of_report ≤ DATE`
- Episodes: only ingest episodes with `decision_date ≤ DATE`

This enforces temporal discipline for Phase 15 walk-forward evaluation.

Tests: filter excludes files with dates after through-date, files exactly on through-date
are included, no through-date flag includes all files.

---

## Epic E7: Documentation, Notebook, Status Update

### Ticket E7-T1: Phase 14 bitacora

File: `doc/bitacora/PHASE_14_INFRASTRUCTURE.md`

Sections:
- Objective and scientific claim
- Architecture decisions (DJ-088 through DJ-094) — hyperlinks to PHASE_14_CONTEXT.md
- Model diagnostic results (which models passed, which fallback used)
- OQ-P14-05 result (entropy restored?)
- OQ-S01 re-run result (Sentiment FT go/no-go final decision)
- Data pipeline statistics (100-stock, date ranges, completeness %)
- MCP tool validation results
- Lessons learned + open questions for Phase 15

### Ticket E7-T2: Phase 14 replication notebook

File: `notebooks/phase14_replication.ipynb`

No LLM calls. < 60s runtime. Loads from fixture files.

Sections:
1. Model diversity comparison: Phase 13 baselines vs. Phase 14 baselines (entropy, herding)
2. Data universe: sector distribution chart, OHLCV coverage heatmap
3. EDGAR MD&A coverage: docs per ticker, AAPL SGR comparison (Phase 13 vs. Phase 14)
4. Verification baselines: HR/GR/SGR by agent, Phase 13 vs. Phase 14 comparison table
5. Sequential ensemble: example inter-agent context flow diagram
6. MCP tool demo: mock signals → portfolio → risk → orders (no live data)
7. OQ answers summary: P14-05, S01 final decision

### Ticket E7-T3: STATUS.md + MEMORY.md update

Update STATUS.md:
- Phase 14 → COMPLETE
- Phase 15 → PLANNING
- Add Phase 14 results section (test count, key metrics)
- Add DJ-095+ index if new decisions made during Phase 14

Update MEMORY.md:
- Phase 14 model diversity table (confirmed models, not proposed)
- Phase 14 confirmed baselines (HR/GR/SGR per agent with new models)
- Next DJ number
- Phase 15 scope reminder

---

## Test Coverage Summary

| New test file | Epic | Scope |
|---|---|---|
| `tests/unit/test_universe.py` | E2-T1 | Ticker universe completeness, sector coverage |
| `tests/unit/test_edgar_mda.py` | E2-T3 | Section extraction, no boilerplate leakage |
| `tests/unit/test_regime.py` | E5-T1 | Regime classification on known dates |
| `tests/unit/test_agent_context.py` | E3-T1 | AgentContextStore write/read/clear |
| `tests/unit/test_sequential_ensemble.py` | E3-T2/T4 | Context accumulation, fallback |
| `tests/unit/test_portfolio_composer.py` | E4-T1 | Weight computation, caps, edge cases |
| `tests/unit/test_risk_manager.py` | E4-T2 | VaR, max drawdown, correlation blocking |
| `tests/unit/test_capital_allocator.py` | E4-T3 | Share quantities, Kelly cap, commission |
| `tests/integration/test_portfolio_pipeline.py` | E4-T4 | End-to-end mock portfolio |
| `tests/unit/test_episodic_store.py` | E5-T2 | EpisodeRecord, search, unlabeled query |
| `tests/unit/test_episodic_retriever.py` | E5-T3 | Retrieval format, temporal discipline |
| `tests/unit/test_label_outcomes.py` | E5-T4 | Labeling logic, idempotency, horizon |
| `tests/unit/test_namespaced_store.py` | E6-T1 | Namespace isolation, table naming |
| `tests/unit/test_temporal_filter.py` | E6-T3 | Through-date filtering in ingestion |

Target: **≥ 1500 tests, 0 lint errors** at phase close.

---

## Makefile Targets (Phase 14 additions)

| Target | Command | Requires LLM? | Estimated runtime |
|---|---|---|---|
| `acquire-data-phase14` | Bulk yfinance 100 stocks × 21 years | No | 30-60 min, internet |
| `ingest-edgar-mda` | EDGAR MD&A section parsing + LanceDB | No | 4-8 hrs, internet |
| `acquire-macro-phase14` | FRED indicators 2004-2025 | No | 5-10 min, internet |
| `diagnose-models` | E0-T2 per-model diagnostic | Yes (LM Studio) | 30-60 min per model |
| `diversity-baseline` | E0-T5 30-date ensemble run | Yes (LM Studio) | 2-4 hrs |
| `validate-sentiment-corpus-v2` | E1-T1 corpus gate re-run | No | 5 min |
| `finetune-sentiment-v2` | E1-T2 Gemma 3 fine-tuning (if gate passes) | Yes (mlx_lm) | 2-4 hrs |
| `eval-reset` | Clear hifi-eval-* namespace | No | <1 min |
| `eval-ingest-through` | Temporal-filtered ingest (DATE= required) | No | Varies |
| `live-reset` | Clear hifi-live-* namespace | No | <1 min |
| `label-outcomes` | Automated outcome labeling | No | 5-10 min |
| `test` | Full test suite (1500+ tests) | No | ~3-5 min |

---

## Phase 15 Handoff (produced by Phase 14)

1. **5-organization ensemble** — all models confirmed, baselines measured
2. **OQ-P14-05 answered** — entropy > 0.3 confirmed (or alternative documented)
3. **Sentiment FT final decision** — go (sentiment_v2 deployed) or permanently closed
4. **100-stock data pipeline** — yfinance 2004-2025 + EDGAR MD&A in LanceDB
5. **Sequential ensemble** — `run_sequential_ensemble()` tested and operational
6. **3 MCP tools** — portfolio composer, risk manager, capital allocator
7. **Episodic RAG pipeline** — EpisodicStore + EpisodicRetriever + label-outcomes
8. **Namespace partitioning** — dev/eval/live isolation working
9. **Open questions for Phase 15** — OQ-P14-02 (IC), OQ-P14-03 (sequential vs. parallel),
   OQ-P14-06 (Sharpe vs. SPY), OQ-AG03 (calibration)
