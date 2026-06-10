# Phase 4: Second Agent — First Ensemble

**Status:** COMPLETE (2026-06-10)

| Epic | Title | Status |
|---|---|---|
| P4-E1 | Technical Analyst Agent | DONE |
| P4-E2 | Ensemble output schemas (interface-first) | DONE |
| P4-E3 | Collective Decision Engine | DONE |
| P4-E4 | Ensemble runner | DONE |
| P4-E5 | Ensemble evaluation and baseline fixtures | DONE (fixture pending live LLM run) |
| P4-E6 | Holistic pipeline test + Phase 3 regression guard | DONE |

**David Sections:** §5.2 (Collective Intelligence), §5.3 (Ensemble Learning), §5.6 (Diversity Metrics), §10.2 (Agent Specifications), §10.3 (Diversity Requirements), §12.2 (Aggregation Methods)
**Learning Guide Topics:** 3.3 (Collective Intelligence & Aggregation), 5.2 (Ensemble Learning — bias-variance in practice)
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 4

---

## Governing Philosophy for This Phase

Phase 4 adds exactly one thing: a second agent. This is not an incremental improvement. It is a structural test of the central scientific hypothesis: that a population of diverse, independent agents produces better collective decisions than any individual agent.

The David's theoretical foundation (§5.3) states the relationship precisely. For an ensemble of M models with average bias b, average variance v, and average pairwise correlation ρ:

```
Ensemble_Error ≈ b² + ρv + (1-ρ)v/M
```

As M increases, the third term (variance reduction) vanishes. But the second term — which depends on pairwise correlation ρ — persists regardless of ensemble size. If agents are highly correlated (ρ → 1), the ensemble provides no benefit over a single agent. Therefore:

**The value of the ensemble depends entirely on achieving low correlation between agents.**

This is not assumed. It must be measured. Phase 4 produces the first empirical test of whether the architecture achieves genuine diversity or merely nominal diversity.

**What Phase 4 is NOT:**

Phase 4 does not attempt to achieve high financial accuracy. The agents are not fine-tuned, not verified, and not backed by RAG knowledge systems. Phase 4's sole purpose is to establish whether two architecturally diverse agents, when combined with a principled aggregation mechanism, produce different and independently valuable opinions. If they do not — if their outputs are highly correlated — the Phase 8 agent population expansion is compromised before it begins. This makes Phase 4 a prerequisite for the entire ensemble strategy, not just a feature addition.

**Diversity strategy:**

The second agent must differ from the Fundamental Agent along at least TWO dimensions (David §10.3):

| Dimension | Fundamental Agent | Technical Agent | Different? |
|---|---|---|---|
| Model family | qwen2.5-coder-32b (code-focused) | Claude Opus 4.6 reasoning-distilled | Yes (training objective) |
| Information access | Balance sheet + fundamentals + valuation + macro | Price action + momentum + volatility + risk metrics | Yes (orthogonal domains) |
| Prompt structure | Balance sheet quality focus | Price action and trend focus | Yes |
| Role | Business analyst | Market technician | Yes |

Information access is the most robust diversity mechanism because it is enforced architecturally: the Technical Agent's `call_mcp_tools_node` only calls `get_technical_indicators` and `get_risk_metrics`. No code path gives it access to fundamental data. This is diversity by construction, not by hope.

**Independence requirement:**

During the analysis phase, the two agents MUST NOT share state. They run in sequence but receive no information from each other during reasoning. Only the Collective Decision Engine sees both outputs. This is the independence condition from David §10.1, which is required for ensemble theory to apply. A system where Agent 2 sees Agent 1's reasoning is not an ensemble — it is a chain, and its errors are correlated by construction.

---

## Epic Dependency Graph

```
P4-E1 (Technical Agent)
    |
    +----------------------+
    |                      |
P4-E2 (Ensemble         (Phase 3 Fundamental
       Schemas)           Agent - already built)
    |
    v
P4-E3 (Collective Decision Engine)
    |
    v
P4-E4 (Ensemble Runner)
    |
    +------------------+
    |                  |
P4-E5 (Evaluation)   P4-E6 (Holistic Test)
```

E1 and Phase 3 (Fundamental Agent) are the two inputs to E2. E2 defines the schemas that E3 and E4 depend on. E5 and E6 are independent of each other and can be developed in parallel after E4.

---

## Key Decisions To Make in This Phase

**DJ-016: Technical Agent model selection**

The Phase 3 Fundamental Agent uses `qwen2.5-coder-32b-instruct-mlx`. For Phase 4, we need a model that is genuinely different in training objective. Two candidates are available in LM Studio:

- `mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled` — 35B Mixture-of-Experts (3.5B active parameters); distilled from Claude Opus 4.6; very fast inference on MLX due to sparse activation
- `qwen3.5-27b-claude-4.6-opus-reasoning-distilled-qx64-hi-mlx` — 27B dense model; distilled from Claude Opus 4.6; Q4 high-quality quantization

Both are Qwen 3.5 base models with Claude Opus 4.6 reasoning distillation. The key difference from Phase 3's model is the training objective: `qwen2.5-coder` is optimized for code generation and structured output; the distilled models are optimized for multi-step reasoning. For financial analysis interpretation, reasoning ability is the relevant capability.

**Decision procedure:** Run a structured JSON compliance test on both candidates (same test used for DJ-014). Confirm that the chosen model produces exact JSON on the first attempt with the Technical Agent's prompt. The MoE model (35B-A3.5B) is preferred if both pass because its inference speed is higher and will matter when running ensemble on 3+ tickers.

Decision to record as DJ-016 after P4-E1-T1 completes.

**DJ-017: Technical Agent prompt strategy**

The Technical Agent receives technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR) and risk metrics (historical volatility at 20/60/252-day windows, beta, max drawdown, Sharpe ratio, VaR 95%). These are numbers without price context unless the agent understands their meaning.

Two prompt strategies are possible:
1. **Indicator-state framing:** Describe what each indicator's current value implies (RSI > 70 = overbought, price > SMA = uptrend) in the system prompt; let the model apply these rules to the given data
2. **Raw-data framing:** Provide the numbers with minimal interpretation guidance; let the model apply its own domain knowledge

The risk of Strategy 1 is overfitting to the rules provided (the model may apply them mechanically without context). The risk of Strategy 2 is that different models have different levels of technical analysis knowledge, making interpretation inconsistent across Phase 8 agents.

Decision: Strategy 1 for Phase 4 baseline. The prompt should define the interpretation framework explicitly because this is the Phase 4 BASELINE — we want a controlled, reproducible interpretation methodology. Phase 10 can test Strategy 2 as an ablation.

Decision to record as DJ-017 after P4-E1-T3 completes (after testing the prompt on live data).

**DJ-018: Aggregation method for the 2-agent ensemble**

The Protocol specifies confidence-weighted voting for Phase 4 (majority vote of 2 requires agreement, so confidence-weighting is the appropriate first method for a 2-agent ensemble). The formula (David §12.2.2):

```
Score(k) = Σ c_i · 𝟙(v_i = k)    for k ∈ {Buy, Hold, Sell}
Decision = argmax_k Score(k)
```

For exactly 2 agents with votes v_1 ≠ v_2 (disagreement), confidence weighting selects the agent with higher confidence. For v_1 = v_2 (agreement), the collective decision matches both agents.

**Edge case:** With 2 agents on 3 options, it is possible for the collective confidence scores to be equal (tie). Resolution: the collective decision is "Hold" on a tie, with collective_confidence = 0.0, and a flag indicating disagreement. This is the conservative choice — when agents disagree with equal conviction, withhold a clear signal.

Decision to record as DJ-018 as CONFIRMED (confidence-weighted voting is the only principled 2-agent method). The open question is whether performance-weighted voting (Phase 9, David §12.2.3) outperforms it, but that requires outcome data that does not yet exist.

---

## Epic P4-E1: Technical Analyst Agent

**Objective:** Build the Technical Analyst Agent using the same LangGraph pattern as Phase 3's Fundamental Agent, but with different model, different tools, and different prompt. The information restriction is architectural: the Technical Agent's `call_mcp_tools_node` only calls `get_technical_indicators` and `get_risk_metrics`.

**Agent specification (David §10.2 Technical Agent):**
- Focus: price action, momentum, trend, risk-adjusted performance
- Information access: technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR) + risk metrics (volatility 20/60/252d, beta, max drawdown, Sharpe, VaR)
- Output: decision + confidence + rationale (citing specific indicator values) + time_horizon (short/medium/long term signal duration)
- Model: Claude Opus 4.6 reasoning-distilled (to be confirmed as DJ-016)

**LangGraph graph structure (identical pattern to Phase 3):**
```
[START]
   |
   v
call_mcp_tools_node      -- calls 2 tools; no snapshot required
   |
   v
generate_analysis_node   -- calls LM Studio with Technical Agent prompt
   |
   v
parse_output_node        -- extracts AgentSignal + time_horizon; retry once on failure
   |
   v
[END] --> TechnicalAnalysis
```

Note: no `load_snapshot_node` — the Technical Agent does not use a FundamentalsSnapshot. Its state starts with `ticker` and `as_of_date` only. The absence of the snapshot validation node is intentional: the Technical Agent's data comes from Parquet files in the data directory (accessed via MCP), not from a pre-fetched snapshot JSON.

**TechnicalAnalystState TypedDict:**
```python
class TechnicalAnalystState(TypedDict, total=False):
    ticker: str
    as_of_date: str
    data_dir: str
    tool_results: dict        # populated by call_mcp_tools_node
    llm_response: str
    signal: Optional[AgentSignal]
    time_horizon: Optional[str]   # "short-term", "medium-term", "long-term"
    error: Optional[str]
    start_time: float
```

**Tool calls:**
- `get_technical_indicators(ticker, date, window=20)` — returns SMA, EMA, RSI, MACD, Bollinger, ATR + call_id
- `get_risk_metrics(ticker, date)` — returns hist_vol_20d/60d/252d, beta, max_drawdown_252d, sharpe_252d, var_95_20d + call_id

The `window` parameter for technical indicators defaults to 20. This is the standard window for short-to-medium term analysis and is consistent with Phase 2's engine implementation.

**Technical Agent prompt (technical_v1.md):**
The system section defines: (a) the agent role as a technical analyst whose only input is price-derived data; (b) interpretation guidelines for each indicator class (RSI thresholds, MACD crossover meaning, Bollinger Band position interpretation, ATR regime classification); (c) strict grounding rule — only cite values present in the data; (d) time_horizon output specification.

The user section provides: raw indicator values + raw risk metrics + data gaps list + output JSON schema with decision, confidence, rationale, key_concern, and time_horizon.

**time_horizon extraction:** The LLM is instructed to include `time_horizon` in its JSON output. The parse function extracts it from the parsed dict before building AgentSignal (which does not include time_horizon). `time_horizon` is stored directly in TechnicalAnalysis.

| Ticket | Description | Status |
|---|---|---|
| P4-E1-T1 | Test both Claude-distilled candidates for structured JSON compliance; record DJ-016 | PLANNED |
| P4-E1-T2 | Write src/hifi/agents/prompts/technical_v1.md; system + user sections; indicator interpretation guidelines | PLANNED |
| P4-E1-T3 | Implement TechnicalAnalystState TypedDict in technical_agent.py | PLANNED |
| P4-E1-T4 | Implement call_mcp_tools_node for Technical Agent: get_technical_indicators + get_risk_metrics | PLANNED |
| P4-E1-T5 | Implement generate_analysis_node for Technical Agent (reuse make_llm(); load technical_v1.md) | PLANNED |
| P4-E1-T6 | Implement parse_output_node for Technical Agent: extract AgentSignal + time_horizon; retry pattern | PLANNED |
| P4-E1-T7 | Assemble LangGraph graph (no load_snapshot node; conditional edge on call_mcp_tools failure) | PLANNED |
| P4-E1-T8 | Implement run_technical_analysis(ticker, as_of_date, data_dir) -> TechnicalAnalysis; public entrypoint | PLANNED |
| P4-E1-T9 | Unit test: call_mcp_tools_node returns dict with technical_indicators and risk_metrics keys | PLANNED |
| P4-E1-T10 | Unit test: parse_output_node extracts AgentSignal + time_horizon from valid JSON; retry on invalid | PLANNED |
| P4-E1-T11 | Unit test: time_horizon is optional — if absent from LLM JSON, no error raised; TechnicalAnalysis.time_horizon = None | PLANNED |
| P4-E1-T12 | Integration test: full Technical Agent graph runs for AAPL 2023-03-31 using Phase 1 fixtures; TechnicalAnalysis returned | PLANNED |

**Files to create:**
- `src/hifi/agents/technical_agent.py` — graph nodes + graph assembly + run_technical_analysis()
- `src/hifi/agents/prompts/technical_v1.md` — versioned prompt template for Technical Agent

**Acceptance test:** `run_technical_analysis("AAPL", "2023-03-31", data_dir)` returns a TechnicalAnalysis with signal.decision in {"Buy", "Hold", "Sell"} and signal.call_ids non-empty.

---

## Epic P4-E2: Ensemble Output Schemas

**Objective:** Define the typed output contract for the 2-agent ensemble. These schemas are the output interface that Phase 9 (Collective Decision Engine) will extend to N agents, Phase 5 (Verification) will consume, and Phase 10 (Evaluation) will measure.

**TechnicalAnalysis — Phase 4 analysis envelope:**
```python
class TechnicalAnalysis(BaseModel):
    signal: Optional[AgentSignal]
    technical_indicators: dict    # raw result from get_technical_indicators
    risk_metrics: dict            # raw result from get_risk_metrics
    time_horizon: Optional[str]   # "short-term" | "medium-term" | "long-term" | None
    prompt_version: str           # "technical_v1"
    latency_ms: Optional[float] = None

    def tool_results_flat(self) -> dict: ...   # same pattern as FundamentalAnalysis
```

**EnsembleDecision — the collective output of the aggregation engine:**
```python
class EnsembleDecision(BaseModel):
    collective_decision: Optional[Literal["Buy", "Hold", "Sell"]]
    collective_confidence: float   # sum of winning-side confidence / total confidence
    n_valid_signals: int           # number of non-None AgentSignals contributed
    agreement: bool                # True if all agents voted identically
    disagreement_entropy: float    # David §5.6.1; 0.0 if unanimous, > 0 if split
    opinion_dispersion: float      # David §5.6.2; mean absolute deviation of confidences
    agent_decisions: list[str]     # individual decisions for audit trail
    agent_confidences: list[float] # individual confidence scores
    winning_score: float           # score of the winning option under confidence-weighting
    total_score: float             # sum of all scores (equals sum of all confidences)
```

**EnsembleOutput — the full Phase 4 analysis envelope:**
```python
class EnsembleOutput(BaseModel):
    ticker: str
    as_of_date: str
    fundamental_analysis: FundamentalAnalysis
    technical_analysis: TechnicalAnalysis
    ensemble_decision: EnsembleDecision
    latency_ms: float   # total wall-clock time for both agents + aggregation
```

**Design rationale for EnsembleDecision fields:**

- `collective_decision` is None when n_valid_signals == 0 (both agents failed to produce a signal). This is an explicit failure mode, not a default.
- `collective_confidence` is the normalized confidence of the winning option: `winning_score / total_score`. This is in [0, 1] and represents the fraction of total conviction that went to the winning option.
- `disagreement_entropy` uses the David §5.6.1 formula applied to the vote distribution. For 2 agents: if they agree, H = 0; if they split Buy/Sell, H = 1.0; if one votes Hold and one votes Buy, H is between 0 and 1 depending on proportion.
- `opinion_dispersion` uses the David §5.6.2 formula: `(1/N) * sum(|c_i - mean_c|)`. For 2 agents, this is `|c_1 - c_2| / 2`.

| Ticket | Description | Status |
|---|---|---|
| P4-E2-T1 | Define TechnicalAnalysis in src/hifi/agents/schemas.py; tool_results_flat() method | PLANNED |
| P4-E2-T2 | Define EnsembleDecision in src/hifi/collective/schemas.py; all diversity metric fields | PLANNED |
| P4-E2-T3 | Define EnsembleOutput in collective/schemas.py; wraps both agent analyses + collective decision | PLANNED |
| P4-E2-T4 | Unit test: TechnicalAnalysis serialises to JSON-safe dict (no NaN) | PLANNED |
| P4-E2-T5 | Unit test: EnsembleDecision with two agreeing agents has agreement=True, entropy=0.0 | PLANNED |
| P4-E2-T6 | Unit test: EnsembleDecision with two disagreeing agents has agreement=False, entropy > 0 | PLANNED |
| P4-E2-T7 | Unit test: EnsembleDecision with n_valid_signals=0 has collective_decision=None | PLANNED |
| P4-E2-T8 | Unit test: EnsembleOutput serialises to JSON-safe dict | PLANNED |

**Files to create:**
- Update `src/hifi/agents/schemas.py` — add TechnicalAnalysis
- `src/hifi/collective/__init__.py` — new package
- `src/hifi/collective/schemas.py` — EnsembleDecision, EnsembleOutput

**Acceptance test:** EnsembleDecision constructed from two agreeing AgentSignals has agreement=True, entropy=0.0, and collective_decision matching both signals. EnsembleDecision from disagreeing signals has collective_decision = higher-confidence agent's decision.

---

## Epic P4-E3: Collective Decision Engine

**Objective:** Implement confidence-weighted voting and the three diversity metrics from David §5.6 that are computable with a 2-agent system: disagreement entropy (§5.6.1), opinion dispersion (§5.6.2), and the diversity decomposition (§5.6.5). The herding coefficient (§5.6.3) and consensus stability (§5.6.4) require time-series data and are scaffolded but not populated until Phase 9.

**Confidence-weighted voting (David §12.2.2):**

```python
def confidence_weighted_vote(signals: list[AgentSignal]) -> EnsembleDecision:
```

Algorithm:
1. Filter out None signals; if none remain, return EnsembleDecision with n_valid_signals=0
2. Compute Score(k) = sum of confidence for each k in {Buy, Hold, Sell}
3. collective_decision = argmax Score(k)
4. On tie: collective_decision = "Hold"; collective_confidence = 0.0 (conservative default)
5. Compute disagreement_entropy using David §5.6.1 formula
6. Compute opinion_dispersion using David §5.6.2 formula
7. Return EnsembleDecision with all fields populated

**Disagreement entropy (exact formula from David §5.6.1):**
```
let p_k = (count of votes for k) / n_valid_signals, for k in {Buy, Hold, Sell}
H = -sum(p_k * log2(p_k) for p_k > 0)
```
For a 2-agent unanimous case: one p_k = 1.0, rest = 0 → H = 0.
For a 2-agent disagreement (Buy + Sell): p_Buy = 0.5, p_Sell = 0.5 → H = 1.0.
For a 2-agent disagreement (Buy + Hold): same → H = 1.0.

**Opinion dispersion (exact formula from David §5.6.2):**
```
c_bar = mean(c_i)
D = (1/N) * sum(|c_i - c_bar|)
```
For 2 agents: D = |c_1 - c_2| / 2.

**Diversity decomposition (David §5.6.5 adapted for categorical output):**

For categorical outputs, pairwise agreement is the natural diversity measure:
```
pairwise_agreement = 1 if signals[0].decision == signals[1].decision else 0
pairwise_diversity = 1 - pairwise_agreement
```

This is the Phase 4 version. Phase 9 will extend this to the full Page (2007) decomposition across N agents and T time periods.

**ensemble_metrics.py — Phase 4 ensemble-level metrics:**
```python
def compute_ensemble_metrics(outputs: dict[str, EnsembleOutput]) -> dict:
```
Returns: compliance_rate (fraction of tickers where both agents produced valid signals), agreement_rate (fraction of tickers where agents agreed), mean_disagreement_entropy, mean_opinion_dispersion, pairwise_diversity, mean_latency_ms.

| Ticket | Description | Status |
|---|---|---|
| P4-E3-T1 | Implement confidence_weighted_vote() in collective/voting.py; returns EnsembleDecision | PLANNED |
| P4-E3-T2 | Implement disagreement_entropy() in collective/metrics.py using David §5.6.1 formula | PLANNED |
| P4-E3-T3 | Implement opinion_dispersion() in collective/metrics.py using David §5.6.2 formula | PLANNED |
| P4-E3-T4 | Implement pairwise_diversity() in collective/metrics.py (categorical agreement rate) | PLANNED |
| P4-E3-T5 | Implement compute_ensemble_metrics() in collective/metrics.py | PLANNED |
| P4-E3-T6 | Unit test: confidence_weighted_vote() with unanimous signals → correct decision, entropy=0 | PLANNED |
| P4-E3-T7 | Unit test: confidence_weighted_vote() with disagreeing signals → higher-confidence agent wins | PLANNED |
| P4-E3-T8 | Unit test: confidence_weighted_vote() with tied scores → "Hold", confidence=0.0 | PLANNED |
| P4-E3-T9 | Unit test: confidence_weighted_vote() with zero valid signals → collective_decision=None | PLANNED |
| P4-E3-T10 | Unit test: disagreement_entropy() = 0.0 for unanimous; 1.0 for even 2-way split | PLANNED |
| P4-E3-T11 | Unit test: opinion_dispersion() = 0.0 for equal confidence; |c1-c2|/2 for unequal | PLANNED |
| P4-E3-T12 | Unit test: compute_ensemble_metrics() returns correct agreement_rate and pairwise_diversity | PLANNED |

**Files to create:**
- `src/hifi/collective/voting.py` — confidence_weighted_vote()
- `src/hifi/collective/metrics.py` — disagreement_entropy(), opinion_dispersion(), pairwise_diversity(), compute_ensemble_metrics()

**Acceptance test:** confidence_weighted_vote([Buy(0.8), Sell(0.6)]) → EnsembleDecision(collective_decision="Buy", agreement=False, entropy=1.0, dispersion=0.1). All math validated against the David §5.6 formulas.

---

## Epic P4-E4: Ensemble Runner

**Objective:** Wire both agents into an ensemble runner that runs them independently and aggregates their outputs. This is the first implementation of a multi-agent system in HiFi. The runner is designed to make independence explicit: no state is shared between agents during reasoning.

**run_ensemble() entrypoint:**
```python
def run_ensemble(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,    # FundamentalsSnapshot for Fundamental Agent
    data_dir: str | None = None,
) -> EnsembleOutput:
```

Execution sequence:
1. Record start time
2. `fundamental = run_analysis(ticker, as_of_date, snapshot_json, data_dir)` — Phase 3 entrypoint
3. `technical = run_technical_analysis(ticker, as_of_date, data_dir)` — Phase 4 entrypoint
4. Collect valid signals: `[a.signal for a in [fundamental, technical] if a.signal is not None]`
5. `decision = confidence_weighted_vote(valid_signals)`
6. Return EnsembleOutput(ticker, as_of_date, fundamental, technical, decision, latency)

**Independence guarantee:** Steps 2 and 3 are independent function calls with no shared state. The Fundamental Agent has no access to the Technical Agent's output and vice versa. This is enforced by the function call boundary — no mutable state, no shared queue, no communication channel between agents.

**Future scalability:** The sequential runner is correct for Phase 4 (2 agents). Phase 9 will parallelize agent runs using asyncio or concurrent.futures. The sequential order here should not be changed to parallel for Phase 4 — parallelism introduces complexity (thread safety, exception handling) that is not justified at this scale.

**scripts/run_phase4_ensemble.py:**
One-time runner that uses the same reference snapshots as `scripts/run_phase3_baseline.py` for AAPL, JPM, and XOM at Q1 2023. Saves full EnsembleOutput for each ticker to `tests/fixtures/baseline/phase4_ensemble.json`. Requires LM Studio running with both models loaded.

| Ticket | Description | Status |
|---|---|---|
| P4-E4-T1 | Implement run_ensemble() in src/hifi/agents/ensemble_runner.py | PLANNED |
| P4-E4-T2 | Integration test: run_ensemble() returns EnsembleOutput with both analyses populated | PLANNED |
| P4-E4-T3 | Integration test: fundamental and technical analyses run independently (no shared state via test isolation) | PLANNED |
| P4-E4-T4 | Integration test: run_ensemble() with ticker not in data dir fails with error in both analyses, not an exception | PLANNED |
| P4-E4-T5 | Write scripts/run_phase4_ensemble.py; runs AAPL/JPM/XOM; saves phase4_ensemble.json | PLANNED |

**Files to create:**
- `src/hifi/agents/ensemble_runner.py` — run_ensemble() public entrypoint
- `scripts/run_phase4_ensemble.py` — one-time ensemble evaluation runner

**Acceptance test:** `run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, data_dir)` returns an EnsembleOutput where both `fundamental_analysis.signal` and `technical_analysis.signal` are non-None and independently produced.

---

## Epic P4-E5: Ensemble Evaluation and Baseline Fixtures

**Objective:** Record baseline ensemble outputs as comparison fixtures. Measure whether the 2-agent ensemble produces genuinely diverse signals (low pairwise correlation). Establish the Phase 4 baseline metrics that Phase 5+ improvements will be measured against.

**phase4_ensemble.json fixture format:**
```json
{
  "metadata": {
    "phase": "4",
    "models": {
      "fundamental": "qwen2.5-coder-32b-instruct-mlx",
      "technical": "<DJ-016 confirmed model>"
    },
    "prompt_versions": {
      "fundamental": "fundamental_v1",
      "technical": "technical_v1"
    },
    "data_as_of": "2023-03-31",
    "run_date": "<ISO date>",
    "hifi_commit": "<git sha>"
  },
  "outputs": {
    "AAPL": { ...EnsembleOutput dict... },
    "JPM":  { ...EnsembleOutput dict... },
    "XOM":  { ...EnsembleOutput dict... }
  },
  "metrics": {
    "fundamental_compliance_rate": ...,
    "technical_compliance_rate": ...,
    "ensemble_agreement_rate": ...,
    "mean_disagreement_entropy": ...,
    "mean_opinion_dispersion": ...,
    "pairwise_diversity": ...,
    "mean_total_latency_ms": ...
  }
}
```

**Evaluation comparison table (to be filled by the live run):**

| Metric | Fundamental Agent | Technical Agent | Ensemble |
|---|---|---|---|
| Compliance rate | ? | ? | ? |
| Hallucination count | ? | ? | N/A |
| Mean latency (ms) | ? | ? | ? |
| Agreement rate | — | — | ? |
| Pairwise diversity | — | — | ? |
| Disagreement entropy | — | — | ? |

**What to interpret from the metrics:**
- `pairwise_diversity` > 0.3 → agents are genuinely diverse, ensemble design is valid
- `pairwise_diversity` < 0.1 → agents are highly correlated, review information restriction
- `ensemble_agreement_rate` consistently 1.0 → investigate whether both agents are receiving different information
- `disagreement_entropy` close to 1.0 on most tickers → agents frequently disagree, confidence-weighting is the right aggregation choice (performance-weighting would require historical outcomes)

| Ticket | Description | Status |
|---|---|---|
| P4-E5-T1 | Run ensemble baseline and save to tests/fixtures/baseline/phase4_ensemble.json | PLANNED |
| P4-E5-T2 | Unit test: phase4_ensemble.json exists and both fundamental and technical compliance_rate >= 0.90 | PLANNED |
| P4-E5-T3 | Unit test: all three tickers present in outputs dict | PLANNED |
| P4-E5-T4 | Unit test: each EnsembleDecision has a valid collective_decision or explicit None | PLANNED |
| P4-E5-T5 | Unit test: disagreement_entropy in [0, log2(3)] for every ticker in fixture | PLANNED |
| P4-E5-T6 | Unit test: pairwise_diversity in [0, 1] and documented in fixture metrics | PLANNED |

**Files to create:**
- `tests/fixtures/baseline/phase4_ensemble.json` — generated by scripts/run_phase4_ensemble.py
- `tests/unit/test_phase4_baseline.py` — structure and threshold tests; skip if fixture absent

**Acceptance test:** phase4_ensemble.json records ensemble outputs for 3 tickers. pairwise_diversity is documented (no minimum threshold enforced — this is a measurement, not a gate). Both agents achieve >= 90% compliance rate independently.

---

## Epic P4-E6: Holistic Pipeline Test + Phase 3 Regression Guard

**Objective:** Verify the full Phase 4 pipeline end-to-end using monkeypatched LLMs for both agents. Confirm that Phase 3's holistic test still passes. Validate the ensemble decision structure for all edge cases.

**What the holistic test validates:**
1. Both agents run independently (Fundamental uses snapshot_json, Technical uses only ticker/date)
2. `confidence_weighted_vote` produces a valid EnsembleDecision from the two monkeypatched signals
3. EnsembleOutput is structurally valid (all required fields, JSON-safe)
4. Disagreement case: two agents with different decisions → collective decision = higher-confidence agent's
5. Agreement case: both agents agree → collective_decision matches both
6. Phase 3 regression: Fundamental Agent alone still produces valid FundamentalAnalysis

**Monkeypatching strategy:**
Both agents use `make_llm()` in their `generate_analysis_node`. Two separate monkeypatches are needed: one for the Fundamental Agent's LLM call and one for the Technical Agent's LLM call. Since they import from the same `lm_client.py`, a single `monkeypatch.setattr(fa, "make_llm", ...)` and `monkeypatch.setattr(ta, "make_llm", ...)` applied to their respective modules is sufficient.

| Ticket | Description | Status |
|---|---|---|
| P4-E6-T1 | Write tests/holistic/test_phase4_ensemble_pipeline.py | PLANNED |
| P4-E6-T2 | Test: run_ensemble for AAPL with stubbed LLMs returns EnsembleOutput with both analyses | PLANNED |
| P4-E6-T3 | Test: EnsembleDecision is correct for mocked agreement scenario | PLANNED |
| P4-E6-T4 | Test: EnsembleDecision is correct for mocked disagreement scenario (confidence-weighted winner) | PLANNED |
| P4-E6-T5 | Test: EnsembleOutput is JSON-safe (model_dump() produces no NaN) | PLANNED |
| P4-E6-T6 | Test: Phase 3 Fundamental Agent holistic test still passes (regression guard) | PLANNED |

**Files to create:**
- `tests/holistic/test_phase4_ensemble_pipeline.py`

---

## New Dependencies

No new Python packages required. All infrastructure from Phase 3 is reused:
- `langchain`, `langchain-openai`, `langgraph` — already installed (Phase 3)
- `openai` — already installed (Phase 3; points to LM Studio)
- `mcp` — already installed (Phase 2; used by MCP client subprocess)

The `src/hifi/collective/` package is new production code (no new external dependencies).

No new LM Studio model downloads. Both candidate models for DJ-016 are already loaded in LM Studio.

---

## Phase 4 Quality Gates

| Gate | Criterion | Measured By |
|---|---|---|
| All unit tests pass | pytest -m unit, 0 failures | Manual run |
| All integration tests pass | pytest -m integration, 0 failures | Manual run |
| Holistic test passes | pytest tests/holistic/test_phase4_ensemble_pipeline.py | Manual run |
| Phase 3 regression | pytest tests/holistic/test_phase3_agent_pipeline.py still passes | Manual run |
| Technical Agent compliance | >= 90% on 3 tickers | Ensemble baseline metrics |
| Independence verified | Agents share no state during reasoning | Code review (static) |
| Diversity measured | pairwise_diversity documented in phase4_ensemble.json | Baseline fixture |
| DJ formulas correct | disagreement_entropy and opinion_dispersion match David §5.6 math | Unit tests |
| No live API calls | grep -r "api.openai\|api.anthropic" src/ returns nothing | Code review |
| Lint clean | ruff check src/ tests/ scripts/, 0 errors | Manual run |

---

## Commit Strategy

One commit per epic, in dependency order:

| Commit | Epic | Key Files |
|---|---|---|
| Phase 4 / E1: Technical Analyst Agent | P4-E1 | agents/technical_agent.py, agents/prompts/technical_v1.md, tests/unit/test_technical_agent_nodes.py, tests/integration/test_technical_agent.py |
| Phase 4 / E2: Ensemble output schemas | P4-E2 | agents/schemas.py (TechnicalAnalysis), collective/schemas.py, tests/unit/test_ensemble_schemas.py |
| Phase 4 / E3: Collective Decision Engine | P4-E3 | collective/voting.py, collective/metrics.py, tests/unit/test_voting.py, tests/unit/test_diversity_metrics.py |
| Phase 4 / E4: Ensemble runner | P4-E4 | agents/ensemble_runner.py, scripts/run_phase4_ensemble.py, tests/integration/test_ensemble_runner.py |
| Phase 4 / E5: Ensemble evaluation and baseline | P4-E5 | tests/fixtures/baseline/phase4_ensemble.json, tests/unit/test_phase4_baseline.py |
| Phase 4 / E6: Holistic pipeline test | P4-E6 | tests/holistic/test_phase4_ensemble_pipeline.py |

---

## Open Questions This Phase Will Answer

**OQ-P4-01: DJ-016 — Which Claude-distilled model for the Technical Agent?**
Resolved by P4-E1-T1. Test both candidates with the technical_v1 prompt. Choose the one with faster inference that passes structured JSON compliance. Expected: the MoE 35B-A3.5B model due to speed advantage with equal capability.

**OQ-P4-02: Is the Technical Agent genuinely diverse from the Fundamental Agent?**
Resolved by P4-E5 (live run required). The pairwise_diversity metric quantifies this. A diversity score above 0.3 confirms the information restriction strategy is working. A diversity score below 0.1 indicates the agents are producing correlated outputs despite different inputs — likely the LLMs have shared training priors that dominate over the specific prompt context.

**OQ-P4-03: Does the 2-agent ensemble outperform either individual agent?**
Cannot be resolved in Phase 4 alone — requires forward-in-time outcome data to measure directional accuracy. Phase 4 establishes structural validity; directional accuracy is a Phase 10 measurement. However, the disagreement patterns observed in Phase 4 provide a preview: if agents frequently disagree, the ensemble will sometimes be right when individuals are wrong.

**OQ-P4-04: How well-calibrated are the LLM confidence estimates for the Technical Agent?**
The Claude-distilled model may exhibit different calibration than the Phase 3 Qwen Coder model. Record all confidence values in the baseline fixture. Calibration analysis requires forward-looking outcome data (Phase 10), but the Phase 4 fixture captures the raw confidence distribution for future comparison.

**OQ-P4-05: Is confidence-weighted voting the right aggregation method for a 2-agent system?**
With only 2 agents, confidence-weighted voting reduces to "choose the agent with higher confidence when they disagree." This is principled only if the agents' confidence estimates are calibrated. If calibration is poor (a likely outcome at Phase 4), confidence-weighting may be worse than simple majority vote (which in a 2-agent tie would default to Hold). Phase 4 records both agents' decisions and confidences, enabling Phase 9 to retroactively compare aggregation methods on historical data.

---

## Connections to Earlier and Later Phases

**Depends on Phase 3:**
- The AgentSignal schema is unchanged — Technical Agent produces the same contract as Fundamental Agent
- The LangGraph graph pattern is directly reused (4-node structure with conditional abort edge)
- `make_llm()`, `call_tool()`, and the parse-and-retry pattern are all Phase 3 infrastructure
- Phase 2 MCP tools `get_technical_indicators` and `get_risk_metrics` are the Technical Agent's data sources

**Phase 5 (Verification) depends on this phase:**
- Both agents now produce `call_ids` that Phase 5's verifier will check
- The first cross-agent contradiction detection is possible when agents disagree in Phase 4
- `EnsembleDecision.disagreement_entropy` becomes a trigger condition for Phase 5's review flag

**Phase 9 (Collective Decision Engine) depends on this phase:**
- `confidence_weighted_vote()` is the Phase 4 implementation of Phase 9's final aggregation mechanism
- `EnsembleDecision` schema is the Phase 4 prototype of Phase 9's `CollectiveDecision` schema
- The diversity metrics (entropy, dispersion) scaffolded here become the Phase 9 monitoring dashboard inputs
- `pairwise_diversity` in Phase 4 becomes the multi-agent diversity matrix in Phase 9

**Phase 10 (Evaluation) uses this phase's baselines:**
- `phase4_ensemble.json` is the 2-agent baseline for all future ensemble improvement measurements
- The comparison table (Fundamental vs. Technical vs. Ensemble) becomes the standard evaluation format
- The diversity metrics (pairwise_diversity, agreement_rate) are the new columns added to the Phase 3 evaluation table

**Phase 8 (Full Agent Population) depends on this phase:**
- Phase 4 is the proof of concept for adding agents without breaking existing ones
- The independence design (no shared state, AgentSignal as the uniform interface) is validated here before scaling to 6 agents
- If Phase 4 shows high pairwise correlation (pairwise_diversity < 0.1), Phase 8's information restriction design must be revised before proceeding
