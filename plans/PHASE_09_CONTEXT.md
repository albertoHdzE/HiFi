# Phase 9: Collective Decision Engine — Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement and compare multiple aggregation mechanisms for the 6-agent ensemble.
Formalize contrarian integration (3 mechanisms). Implement the full complexity
metrics suite (H, D, κ, S, Page diversity). Build the historical bootstrap to
generate initial performance weights. Produce a comparison table with structural
metrics filled; accuracy columns are TBD until Phase 10 labels arrive.

This phase does NOT implement: fine-tuning, live trading, SEC data expansion,
new agent types, or rigorous statistical accuracy evaluation (Phase 10 scope).
</domain>

<decisions>
## Implementation Decisions

### D-01: EnsembleOutput Schema Refactor (do it now, before Phase 10)

Keep all six named analysis fields (they are traceability artifacts, not
aggregation inputs). Add three new fields to EnsembleOutput:

```python
signals: list[AgentSignal]            # voting inputs captured at ensemble time
aggregation_method: str               # which method produced ensemble_decision
method_comparison: dict[str, EnsembleDecision]  # all methods run simultaneously
```

EnsembleDecision gains two new fields for contrarian integration:
```python
contrarian_confidence_discount: float  # 1.0 if no contrarian; else 1 - α×c
review_flagged: bool                   # True when contrarian.confidence > θ
```

Rationale: N-generic means the aggregation pipeline works on `signals:
list[AgentSignal]` regardless of which agents produced them. The named fields
(fundamental_analysis, risk_analysis, etc.) remain for type safety and
Phase 5/6 verification traceability. Both concerns are served simultaneously.

### D-02: Aggregation Methods Scope (Phase 9)

Implement four methods in `src/hifi/collective/voting.py`:
1. `majority_vote(signals)` — mode of decisions, equal weight
2. `confidence_weighted_vote(signals)` — existing, refine to use `signals` list
3. `performance_weighted_vote(signals, weights: dict[str, float])` — weighted by
   historical accuracy per `agent_type`; falls back to equal weights when weights
   are uniform
4. `contrarian_adjusted_vote(signals, contrarian: ContrarianAnalysis | None)`
   — confidence-weighted base + contrarian discount + review flag

Structured debate (David §12.2.4) and adaptive aggregation (§12.2.5) are
deferred to Phase 11+ (require multi-turn LLM orchestration and training data
respectively).

`run_ensemble()` runs ALL four methods on every call and stores results in
`method_comparison`. The primary method for `ensemble_decision` remains
confidence-weighted (backward compatible). This can be changed via a parameter
in Phase 10 once accuracy data validates a better method.

### D-03: Contrarian Integration — All Three Mechanisms

All three David §12.3 mechanisms are implemented together (~30 lines):

1. **Record dissent:** already done (Phase 8 — `contrarian_analysis` field).

2. **Confidence discount (formula):**
   ```
   discounted = collective_confidence × (1 - α × contrarian_confidence)
   ```
   where α = 0.5 (provisional; calibrated from Phase 10 data).
   Stored in `EnsembleDecision.contrarian_confidence_discount`.

3. **Threshold review flag:**
   `review_flagged = contrarian_confidence > θ` where θ = 0.70 (provisional).
   Stored in `EnsembleDecision.review_flagged`.

These are not redundant: (1) is observability, (2) is signal compression under
epistemic uncertainty, (3) is anomaly detection / risk management. All three
are necessary for the Phase 10 ablation study (with vs. without contrarian).

When no contrarian agent ran: `contrarian_confidence_discount = 1.0`,
`review_flagged = False` (neutral/no-op).

### D-04: Historical Bootstrap for Performance Weights

Performance-weighted aggregation requires empirical priors — accuracy history
per agent_type. Generated via historical bootstrap:

**Approach:**
- 20 quarterly periods × 3 tickers (AAPL/JPM/XOM) = 60 analysis runs
- Run MCP tools deterministically at each historical quarter-end date
- Technical + risk agents: fully deterministic given indicator values
- Fundamental + macro agents: use stored quarterly snapshots
- Sentiment agent: fail-open default (Hold/0.0) throughout bootstrap
- Contrarian: skip during bootstrap (second-pass, no historical basis)
- Forward-return label: BUY correct if 60-day return > +2%; SELL correct if
  < −2%; HOLD correct if within ±2% (David §8.4 reference strategy)

**No GANs or adversarial NNs for Phase 9.** GAN-based synthetic scenario
augmentation belongs in Phase 12+ (Dataset Family F, §8.7) when the evaluation
universe has 50+ tickers and tail-behavior fidelity is needed.

**Initial performance weights:** computed from bootstrap accuracy per agent_type.
Stored in `data/agent_performance_history.json` (simple flat file at Phase 9
scale; Phase 10 graduates to a proper dataset).

**Script:** `scripts/run_phase9_bootstrap.py` — reproducible, seeded,
generates both the decision records and the forward-return labels.

### D-05: Complexity Metrics — Full Suite

H (disagreement entropy) and D (opinion dispersion) already implemented in
`collective/metrics.py`. Phase 9 adds:

| Metric | Formula | Window |
|---|---|---|
| κ (herding) | (1/T)·Σ agreement_rate_t | W = 5, 10, 20 periods |
| S (consensus stability) | (1/(W-1))·Σ 𝟙(v_t = v_{t+1}) | W = 5, 10, 20 periods |
| Page diversity | pairwise_diversity() (already implemented) | per decision |

Window sizes (measured in analysis periods, not calendar days):
- **Short: W = 5** (≈5 quarters ≈ 15 months)
- **Medium: W = 10** (≈10 quarters ≈ 2.5 years)
- **Long: W = 20** (≈20 quarters ≈ 5 years — bootstrap covers this exactly)

κ and S return `None` when fewer than W+1 records exist — this is correct
behavior, not an error. The system is epistemically honest about history depth.

Compute κ per aggregation method: if contrarian integration reduces κ relative
to confidence-weighted baseline, that is direct empirical evidence that
adversarial agents reduce collective herding (complexity science hypothesis).

### D-06: Comparison Experiment Design

**Structural comparison (Phase 9 — no accuracy labels needed):**

For each ticker × date, run all 4 aggregation methods on identical input
signals. Compare:
- Collective decision (do methods agree or diverge?)
- Collective confidence (method with lowest confidence = most uncertain)
- H, D per method
- κ, S across the bootstrap window

**Outcome-based comparison (Phase 10 scope — add later):**
Directional accuracy, Sharpe proxy, hit rate by regime. Fill accuracy columns
in the comparison table as Phase 10 labels arrive.

**EnsembleOutput.method_comparison structure:**
```python
{
  "majority": EnsembleDecision(...),
  "confidence_weighted": EnsembleDecision(...),  # = ensemble_decision (primary)
  "performance_weighted": EnsembleDecision(...),
  "contrarian_adjusted": EnsembleDecision(...)   # includes discount + flag
}
```

Divergence cases (methods produce different decisions) are the scientifically
interesting observations. Logged at INFO level for Phase 10 analysis.

### D-07: AgentPerformanceHistory Store

New schema and storage for performance weight bootstrapping and live updates:

```python
class DecisionRecord(BaseModel):
    ticker: str
    analysis_date: str       # ISO 8601 quarter-end
    agent_type: str          # "fundamental" | "technical" | "risk" | "macro"
    decision: str            # Buy/Hold/Sell
    confidence: float
    outcome_correct: bool | None  # None until forward date passes
    outcome_labeled_at: str | None
    horizon_days: int        # 60 (primary)
    forward_return: float | None

class AgentPerformanceHistory(BaseModel):
    records: list[DecisionRecord]
    weights: dict[str, float]  # agent_type -> accuracy (0 to 1)
    last_updated: str
    n_labeled: int
```

Storage: `data/agent_performance_history.json` (Phase 9). Graduates to
Parquet in Phase 10 when bootstrap covers 20+ tickers.

### Claude's Discretion

The following are left to implementation judgment:
- Exact file layout within `src/hifi/collective/` (new files vs. extending existing)
- How `run_ensemble()` exposes the aggregation method parameter
- Bootstrap script parallelization strategy
- Whether `method_comparison` stores full EnsembleDecision objects or a lighter
  summary struct (full preferred for Phase 10 forward compatibility)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Design Documents
- `doc/HIFI_DAVID.md` §5.6 — Formalization of complexity metrics (H, D, κ, S, Page diversity)
- `doc/HIFI_DAVID.md` §12 — Collective Decision Engine (all aggregation methods)
- `doc/HIFI_DAVID.md` §12.3 — Contrarian integration (3 mechanisms)
- `doc/HIFI_DAVID.md` §8.4 — Reference strategy labeling (forward-return labels, horizons)
- `doc/HIFI_DAVID.md` §8.6 — Agent Interaction Dataset (Family E)
- `doc/HIFI_DAVID.md` §8.7 — Synthetic Scenario Dataset (Family F) — Phase 12+ scope
- `doc/HIFI_PROTOCOL_V1.md` Phase 9 — Protocol deliverables and success criteria

### Existing Code to Extend
- `src/hifi/collective/voting.py` — `confidence_weighted_vote()` baseline
- `src/hifi/collective/metrics.py` — H, D, `pairwise_diversity()` already implemented
- `src/hifi/collective/schemas.py` — EnsembleDecision, EnsembleOutput (refactor target)
- `src/hifi/agents/schemas.py` — AgentSignal, ContrarianAnalysis
- `src/hifi/agents/ensemble_runner.py` — `run_ensemble()` (add method_comparison)

### Phase 8 Completed Artifacts
- `doc/bitacora/PHASE_08_AGENT_POPULATION.md` — DJ-032 through DJ-038
- `plans/PHASE_08_PLAN.md` — Phase 8 epic structure
- `tests/holistic/test_phase8_agent_population.py` — integration test pattern

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `collective/metrics.py`: `disagreement_entropy()`, `opinion_dispersion()`,
  `pairwise_diversity()` — used directly in new κ/S computation helpers
- `collective/voting.py`: `confidence_weighted_vote()` — base for all new methods
- `data/` directory: AAPL/JPM/XOM Parquet files from 2018–2023 — bootstrap input
- MCP servers (`market_server.py`, `macro_server.py`) — called at historical dates
  for bootstrap without LLM

### Established Patterns
- Agent output: `AgentSignal` with `decision`, `confidence`, `agent_type`, `call_ids`
- Test fixtures: `tests/fixtures/market/` + `tests/fixtures/macro/` Parquet files
- LLM stubbing: `_stub_llm()` pattern from Phase 4/8 holistic tests
- MCP call pattern: `call_tool(tool_name, params, data_dir=data_dir)`

### Integration Points
- `run_ensemble()` is the main entry point — gains `signals`, `method_comparison`
  fields in its output; backward-compatible (existing callers unaffected)
- `EnsembleDecision` gains two new fields — existing consumers (tests, fixtures)
  must be updated (same pattern as Phase 8 schema extension)
- `data/agent_performance_history.json` is new — created by bootstrap script,
  read by `performance_weighted_vote()`

</code_context>

<specifics>
## Specific Requirements

- Contrarian discount formula: `discounted = collective × (1 - 0.5 × contrarian_confidence)`
- Review flag threshold: `contrarian_confidence > 0.70`
- Bootstrap forward-return horizon: 60 trading days (primary); 20 days secondary
- Bootstrap label thresholds: ±2% 60-day return (HOLD within; BUY/SELL outside)
- Rolling window sizes: W ∈ {5, 10, 20} analysis periods
- Performance weights stored as `dict[agent_type, float]` initialized to uniform (1/N)
- All four aggregation methods run on every `run_ensemble()` call (stored in
  `method_comparison` regardless of which is the `aggregation_method`)
- No GANs, no adversarial NNs — deferred to Phase 12 synthetic scenarios

</specifics>

<deferred>
## Deferred Ideas

- **Structured debate (David §12.2.4):** multi-turn LLM debate protocol between
  minority/majority agents. Valuable but requires multi-turn orchestration.
  Deferred to Phase 11.
- **Adaptive aggregation (David §12.2.5):** learned aggregation function.
  Requires training data. Deferred to Phase 13 (post fine-tuning).
- **GAN synthetic scenarios (David §8.7):** tail-behavior augmentation.
  Deferred to Phase 12 when evaluation universe has 50+ tickers.
- **Walk-forward validation with purged cross-validation (López de Prado):**
  rigorous accuracy comparison. Phase 10 scope.
- **Drift detection (KS test, CUSUM):** Phase 14 scope.
- **Bootstrap expansion to 20+ tickers:** Phase 10 scope.

</deferred>

---

*Phase: 09-collective-decision-engine*
*Context gathered: 2026-06-12*
