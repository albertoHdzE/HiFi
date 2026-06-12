# Phase 9: Collective Decision Engine

**Status:** PLANNED

| Epic | Title | Status |
|---|---|---|
| P9-E0 | Schema extension (EnsembleDecision + EnsembleOutput + performance schemas) | PLANNED |
| P9-E1 | Aggregation methods (majority, performance-weighted, contrarian-adjusted) | PLANNED |
| P9-E2 | Rolling complexity metrics (κ, S) | PLANNED |
| P9-E3 | Ensemble runner extension (method_comparison + signals capture) | PLANNED |
| P9-E4 | Performance store + historical bootstrap | PLANNED |
| P9-E5 | Holistic test + baseline measurement + bitacora | PLANNED |

**David Sections:** §12 Collective Decision Engine, §12.3 Contrarian Integration, §5.6.3 Herding Coefficient, §5.6.4 Consensus Stability, §8.4 Reference Strategy Datasets
**Learning Guide Topics:** 3.3 Collective Intelligence & Aggregation (deep), 8.2 Collective Intelligence (measurement), 8.3 Emergence & Measurement
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 9
**Context Reference:** plans/PHASE_09_CONTEXT.md (D-01 through D-07)

---

## Governing Philosophy for This Phase

Phase 9 asks the second central empirical question of HiFi: **does the mechanism of aggregation matter?**

Phase 8 established that HiFi has six agents with measurably diverse outputs. Phase 9 asks whether that diversity is being harvested correctly. Majority voting, confidence-weighting, performance-weighting, and contrarian adjustment are four different answers to the same question: given N agents with heterogeneous signals, what is the most epistemically sound way to form a collective belief?

The answer is not obvious and cannot be assumed a priori. Confidence-weighted voting sounds reasonable, but LLM confidence scores are self-reported and may be poorly calibrated — if every agent says "0.85" regardless of actual correctness, confidence-weighting degrades to majority voting with extra steps. Performance-weighted voting is theoretically superior (it adapts to demonstrated accuracy), but requires historical outcome data that only now becomes available via the bootstrap. Contrarian adjustment is the most novel: it asks whether a single well-reasoned dissenting voice should compress the collective conviction below what the voting majority would otherwise produce.

**The structural comparison is the scientifically interesting result of Phase 9.** Even before Phase 10 provides outcome labels to measure accuracy, we can observe whether methods produce different decisions, whether the contrarian mechanism reduces herding (κ), and whether consensus stability (S) differs across methods. These are measurable structural properties of the collective — not predictions of the future, but descriptions of how the collective processes information under each aggregation regime. If contrarian integration reduces κ by a statistically meaningful amount, that is direct empirical evidence for the complexity-science hypothesis: adversarial agents reduce collective herding in multi-agent LLM ensembles.

**The bootstrap is a methodological commitment, not a convenience.** Using 20 quarterly periods × 3 tickers requires explicit decisions about point-in-time data handling, forward-return labeling conventions, and the treatment of sentiment (fail-open throughout) and contrarian (skipped) agents during initialization. These decisions are recorded in the bitacora and in `data/agent_performance_history.json` metadata so that Phase 10 can audit the bootstrap and either extend it or replace it with a more rigorous procedure.

**The `method_comparison` field is the core architectural change.** Every `run_ensemble()` call now runs all four aggregation methods on the same input signals and stores the results. This adds negligible latency (four in-process function calls, no LLM or MCP overhead) but creates a persistent structural record of how methods agree or diverge on every analysis. Phase 10 will add forward-return outcomes to this record and compute accuracy per method. Phase 11 will use the divergence patterns to trigger structured debates. The investment is made now.

---

## Pre-Phase Decisions (from plans/PHASE_09_CONTEXT.md)

All decisions below are locked. No re-discussion required.

### D-01: EnsembleOutput Schema Refactor

Keep all six named analysis fields (traceability artifacts for Phase 5/6 verification). Add three fields to `EnsembleOutput`:

```python
signals: list[AgentSignal]                          # voting inputs captured at ensemble time
aggregation_method: str                              # method that produced ensemble_decision
method_comparison: dict[str, EnsembleDecision]       # all four methods run simultaneously
```

Add two fields to `EnsembleDecision` with backward-compatible defaults:

```python
contrarian_confidence_discount: float = 1.0   # 1.0 = no contrarian; else 1 - 0.5×c
review_flagged: bool = False                  # True when contrarian.confidence > 0.70
```

### D-02: Four Aggregation Methods (all run on every call)

1. `majority_vote` — mode of decisions, equal weight
2. `confidence_weighted_vote` — existing; primary method for `ensemble_decision`
3. `performance_weighted_vote` — weighted by historical accuracy per `agent_type`
4. `contrarian_adjusted_vote` — confidence-weighted base + contrarian discount + review flag

All four run on every `run_ensemble()` call. Primary method remains `confidence_weighted` for `ensemble_decision` (backward compatible). Stored in `method_comparison` regardless.

### D-03: Contrarian Integration — All Three Mechanisms

1. **Record dissent:** already done (Phase 8, `contrarian_analysis` field). No change.
2. **Confidence discount:** `discounted = collective_confidence × (1 - 0.5 × contrarian_confidence)`. `contrarian_confidence_discount` stores the factor `1 - 0.5 × c`. `collective_confidence` in the returned `EnsembleDecision` is the discounted value.
3. **Threshold review flag:** `review_flagged = contrarian_confidence > 0.70`.

When no contrarian ran: `contrarian_confidence_discount = 1.0`, `review_flagged = False`.

### D-04: Historical Bootstrap

- 20 quarterly periods × 3 tickers (AAPL/JPM/XOM) = 60 analysis runs
- Technical + risk agents: fully deterministic MCP calls (no LLM)
- Fundamental + macro: stored quarterly snapshots (hardcoded per quarter)
- Sentiment: fail-open default (Hold/0.0) throughout bootstrap
- Contrarian: skipped during bootstrap (second-pass, no historical basis)
- Forward-return label: BUY correct if 60-day return > +2%; SELL correct if < −2%; HOLD correct within ±2%
- Output: `data/agent_performance_history.json`
- Script: `scripts/run_phase9_bootstrap.py` (deterministic, seeded, no LLM required)

### D-05: Complexity Metrics — Full Suite

Add to `collective/metrics.py`:

| Metric | Formula | Window sizes |
|---|---|---|
| κ (herding) | `(1/T)·Σ a_t` where `a_t` = fraction voting with majority at period t | W = 5, 10, 20 |
| S (consensus stability) | `(1/(W-1))·Σ 𝟙(v_t = v_{t+1})` | W = 5, 10, 20 |

Return `None` when fewer than W records exist. This is correct epistemic behavior.

### D-06: Structural Comparison (Phase 9) vs. Accuracy Comparison (Phase 10)

Phase 9 stores `method_comparison` with H, D, collective_decision per method. No accuracy columns yet. Phase 10 adds outcome labels.

### D-07: AgentPerformanceHistory Schemas

```python
class DecisionRecord(BaseModel):
    ticker: str
    analysis_date: str        # ISO 8601 quarter-end
    agent_type: str           # "fundamental" | "technical" | "risk" | "macro"
    decision: str             # "Buy" | "Hold" | "Sell"
    confidence: float
    outcome_correct: bool | None   # None until forward date passes
    outcome_labeled_at: str | None
    horizon_days: int         # 60 (primary)
    forward_return: float | None

class AgentPerformanceHistory(BaseModel):
    records: list[DecisionRecord]
    weights: dict[str, float]  # agent_type -> accuracy [0, 1]
    last_updated: str
    n_labeled: int
```

Storage: `data/agent_performance_history.json`. Uniform weights (1/N per agent type) before bootstrap labels arrive. After bootstrap: accuracy-derived weights.

---

## Interface Design

This section specifies exact signatures and field layouts before implementation begins. Implementation MUST match this spec; deviations require recorded justification.

### P9-E0: Modified Schemas

```python
# src/hifi/collective/schemas.py

class EnsembleDecision(BaseModel):
    # ... existing fields unchanged ...
    collective_decision: Literal["Buy", "Hold", "Sell"] | None
    collective_confidence: float
    n_valid_signals: int
    agreement: bool
    disagreement_entropy: float
    opinion_dispersion: float
    agent_decisions: list[str]
    agent_confidences: list[float]
    winning_score: float
    total_score: float
    # Phase 9 additions (backward-compatible defaults):
    contrarian_confidence_discount: float = 1.0
    review_flagged: bool = False


class EnsembleOutput(BaseModel):
    # ... existing fields unchanged ...
    ticker: str
    as_of_date: str
    fundamental_analysis: FundamentalAnalysis
    technical_analysis: TechnicalAnalysis
    ensemble_decision: EnsembleDecision
    latency_ms: float
    risk_analysis: RiskAnalysis | None = None
    macro_analysis: MacroAnalysis | None = None
    sentiment_analysis: SentimentAnalysis | None = None
    contrarian_analysis: ContrarianAnalysis | None = None
    # Phase 9 additions (backward-compatible defaults):
    signals: list[AgentSignal] = Field(default_factory=list)
    aggregation_method: str = "confidence_weighted"
    method_comparison: dict[str, EnsembleDecision] = Field(default_factory=dict)


class DecisionRecord(BaseModel):
    ticker: str
    analysis_date: str
    agent_type: str
    decision: str
    confidence: float
    outcome_correct: bool | None = None
    outcome_labeled_at: str | None = None
    horizon_days: int = 60
    forward_return: float | None = None


class AgentPerformanceHistory(BaseModel):
    records: list[DecisionRecord]
    weights: dict[str, float]
    last_updated: str
    n_labeled: int
```

### P9-E1: New Aggregation Functions

```python
# src/hifi/collective/voting.py — new additions

def majority_vote(
    signals: list[AgentSignal | None],
) -> EnsembleDecision:
    """
    Aggregate via simple majority (equal-weight). Mode of decisions wins.
    Tie-breaking: "Hold" with collective_confidence=0.0 (same convention as
    confidence_weighted_vote). Diversity metrics computed identically.
    """

def performance_weighted_vote(
    signals: list[AgentSignal | None],
    weights: dict[str, float],
) -> EnsembleDecision:
    """
    Aggregate via historical accuracy weights per agent_type.

    Each signal is weighted by weights[signal.agent_type]. Falls back to equal
    weight (1.0) for any agent_type not present in weights. When all weights are
    equal or weights is empty, behavior is identical to confidence_weighted_vote.
    """

def contrarian_adjusted_vote(
    signals: list[AgentSignal | None],
    contrarian: ContrarianAnalysis | None,
) -> EnsembleDecision:
    """
    Confidence-weighted base with contrarian discount and review flag.

    1. Run confidence_weighted_vote(signals) to get base decision.
    2. If contrarian is not None:
       discount = 1.0 - 0.5 * contrarian.confidence
       collective_confidence = base.collective_confidence * discount
       review_flagged = contrarian.confidence > 0.70
    3. Return EnsembleDecision with discounted collective_confidence,
       contrarian_confidence_discount, and review_flagged set.

    The winning direction (Buy/Hold/Sell) is not changed by discounting.
    contrarian_confidence_discount stores the factor (1 - 0.5*c), not the
    product. This lets Phase 10 reconstruct the undiscounted confidence.
    """

def run_all_methods(
    signals: list[AgentSignal | None],
    contrarian: ContrarianAnalysis | None,
    weights: dict[str, float],
) -> dict[str, EnsembleDecision]:
    """
    Run all four aggregation methods and return results keyed by method name.

    Keys: "majority", "confidence_weighted", "performance_weighted",
          "contrarian_adjusted"

    Called by ensemble_runner on every run_ensemble() invocation.
    """
```

### P9-E2: Rolling Metrics

```python
# src/hifi/collective/metrics.py — new additions

def herding_coefficient(
    agent_votes_per_period: list[list[str]],
    W: int,
) -> float | None:
    """
    Herding coefficient over the last W analysis periods (David §5.6.3).

    agent_votes_per_period[t] = list of all agent vote strings at period t.
    a_t = fraction of agents voting with the majority at period t.
    κ = mean(a_t) over the last W periods.

    Returns None when len(agent_votes_per_period) < W.
    Uses only the last W elements (rolling window).
    κ near 1/3 indicates independence; κ near 1.0 indicates systematic herding.
    """

def consensus_stability(
    collective_decisions: list[str],
    W: int,
) -> float | None:
    """
    Consensus stability over the last W analysis periods (David §5.6.4).

    collective_decisions[t] = collective decision (mode) at period t.
    S = (1/(W-1)) * sum(1 if v_t == v_{t+1} else 0) over the last W periods.

    Returns None when len(collective_decisions) < W.
    S = 1.0: collective decision never changed in window.
    S = 0.0: collective decision changed every period.
    """

def compute_rolling_metrics(
    agent_votes_per_period: list[list[str]],
    collective_decisions: list[str],
    W_values: list[int] = (5, 10, 20),
) -> dict[str, float | None]:
    """
    Compute κ and S for all specified window sizes.

    Returns dict with keys:
        "kappa_W5", "kappa_W10", "kappa_W20"
        "stability_W5", "stability_W10", "stability_W20"
    Values are float or None (when history is insufficient for that window).
    """
```

### P9-E3: Ensemble Runner Interface

```python
# src/hifi/agents/ensemble_runner.py — modified run_ensemble() return

# The run_ensemble() signature is UNCHANGED. The EnsembleOutput it returns
# gains three new fields:
#   signals: list[AgentSignal]               — all valid non-None signals
#   aggregation_method: str = "confidence_weighted"
#   method_comparison: dict[str, EnsembleDecision]  — all four methods

# Divergence logging (INFO level) when methods disagree:
#   "Method divergence for {ticker} {as_of_date}: majority={} cw={} pw={} ca={}"
```

### P9-E4: Performance Store Interface

```python
# src/hifi/collective/performance_store.py — new file

DEFAULT_HISTORY_PATH = "data/agent_performance_history.json"

def load_history(data_dir: str | None = None) -> AgentPerformanceHistory:
    """
    Load AgentPerformanceHistory from data_dir/agent_performance_history.json.
    If file does not exist, return a fresh history with uniform weights.
    data_dir defaults to HIFI_DATA_DIR env var, then current working directory.
    """

def save_history(history: AgentPerformanceHistory, data_dir: str | None = None) -> None:
    """
    Persist AgentPerformanceHistory to data_dir/agent_performance_history.json.
    Atomic write: write to .tmp, then rename.
    """

def compute_weights(records: list[DecisionRecord]) -> dict[str, float]:
    """
    Compute accuracy weights per agent_type from labeled DecisionRecords.

    accuracy(agent_type) = sum(outcome_correct) / len(labeled_records)
    Only records with outcome_correct is not None are included.
    Returns uniform weights (equal per agent_type) when no labeled records exist.
    """

def get_weights(data_dir: str | None = None) -> dict[str, float]:
    """
    Convenience: load history and return current weights dict.
    Called by ensemble_runner to pass to performance_weighted_vote().
    """
```

---

## Epic Details

### P9-E0: Schema Extension

**Rationale:** All downstream epics depend on the new field layout. Schema extension goes first to unblock parallel implementation of E1/E2.

**Modified files:**
- `src/hifi/collective/schemas.py`
- `tests/unit/test_ensemble_schemas.py` (extend existing)

**Tickets:**

**P9-E0-T1: EnsembleDecision — add contrarian fields**

Add to `EnsembleDecision`:
```python
contrarian_confidence_discount: float = 1.0
review_flagged: bool = False
```
Both have defaults — all existing EnsembleDecision constructors continue to work without modification.

Add a `@field_validator` for `contrarian_confidence_discount` to enforce `[0.0, 1.0]`.

**P9-E0-T2: EnsembleOutput — add aggregation fields**

Add to `EnsembleOutput`:
```python
signals: list[AgentSignal] = Field(default_factory=list)
aggregation_method: str = "confidence_weighted"
method_comparison: dict[str, EnsembleDecision] = Field(default_factory=dict)
```
All three have defaults — existing EnsembleOutput constructors and all Phase 4–8 callers unaffected.

Update the module docstring to document Phase 9 additions and cross-reference D-01.

**P9-E0-T3: Performance schemas — DecisionRecord + AgentPerformanceHistory**

Add `DecisionRecord` and `AgentPerformanceHistory` to `collective/schemas.py`.

`DecisionRecord` validator: `decision` must be in `{"Buy", "Hold", "Sell"}`. `outcome_correct` and `forward_return` may be None (unlabeled). `horizon_days` must be positive.

`AgentPerformanceHistory.n_labeled` is computed via `@model_validator(mode="after")` as `sum(1 for r in records if r.outcome_correct is not None)`. This mirrors the Phase 5 pattern for auto-computed fields.

**P9-E0-T4: Tests — schema validation**

Extend `tests/unit/test_ensemble_schemas.py`:
- `EnsembleDecision` with default new fields round-trips through JSON
- `EnsembleDecision` with explicit `contrarian_confidence_discount=0.6, review_flagged=True` round-trips
- `contrarian_confidence_discount` outside `[0, 1]` raises `ValidationError`
- `EnsembleOutput` with empty `signals` and `method_comparison` round-trips (existing behavior preserved)
- `EnsembleOutput` with populated `signals` and `method_comparison` round-trips
- `DecisionRecord` with `outcome_correct=None` is valid
- `DecisionRecord` with invalid `decision` raises `ValidationError`
- `AgentPerformanceHistory.n_labeled` auto-computed from records

Target: ~15 new test cases.

---

### P9-E1: Aggregation Methods

**Rationale:** Three new voting functions that implement the aggregation methods from David §12.2. All are pure functions operating on `list[AgentSignal | None]` — no I/O, no LLM, fully unit-testable.

**Modified files:**
- `src/hifi/collective/voting.py`
- `tests/unit/test_voting.py` (extend existing)

**New file:**
- `tests/unit/test_aggregation_methods.py`

**Tickets:**

**P9-E1-T1: majority_vote()**

Mode of decisions. Each agent casts one vote regardless of confidence.

```python
def majority_vote(signals: list[AgentSignal | None]) -> EnsembleDecision:
```

Rules:
- Filter None signals (same as confidence_weighted_vote)
- Count votes per option: `vote_counts = {"Buy": 0, "Hold": 0, "Sell": 0}`
- winning_decision = option with maximum count
- Tie: "Hold" with `collective_confidence=0.0` (conservative default, consistent with existing convention)
- `collective_confidence = winning_count / n_valid` (fraction of agents in majority)
- `winning_score = winning_count`, `total_score = n_valid` (in majority_vote, scores ARE counts)
- Diversity metrics (entropy, dispersion, agreement) computed identically to `confidence_weighted_vote`
- `contrarian_confidence_discount = 1.0`, `review_flagged = False` (neutral defaults)

**P9-E1-T2: performance_weighted_vote()**

Weight each agent's vote by its historical accuracy from `weights` dict.

```python
def performance_weighted_vote(
    signals: list[AgentSignal | None],
    weights: dict[str, float],
) -> EnsembleDecision:
```

Rules:
- `w_i = weights.get(signal.agent_type, 1.0)` — fall back to equal weight if not found
- `Score(k) = Σ w_i * c_i * 𝟙(v_i = k)` (weight × confidence for each option)
- Tie-breaking: "Hold" with `collective_confidence=0.0`
- `collective_confidence = winning_score / total_score` (same formula as confidence_weighted)
- Diversity metrics: computed from the unweighted votes (same formulas, consistent across methods)
- `contrarian_confidence_discount = 1.0`, `review_flagged = False`

When `weights` is empty `{}`: every `weights.get(agent_type, 1.0)` returns 1.0, which degrades to `confidence_weighted_vote`. This is the correct and documented fallback.

**P9-E1-T3: contrarian_adjusted_vote()**

Confidence-weighted base with contrarian discount applied post-vote.

```python
def contrarian_adjusted_vote(
    signals: list[AgentSignal | None],
    contrarian: ContrarianAnalysis | None,
) -> EnsembleDecision:
```

Rules:
1. Call `confidence_weighted_vote(signals)` → base
2. If `contrarian is None`:
   - Return base with `contrarian_confidence_discount=1.0`, `review_flagged=False`
3. If `contrarian is not None`:
   - `discount = 1.0 - 0.5 * contrarian.confidence`  (α = 0.5 per D-03)
   - `discounted_confidence = base.collective_confidence * discount`
   - `review_flagged = contrarian.confidence > 0.70`  (θ = 0.70 per D-03)
   - Return `EnsembleDecision` with:
     - `collective_confidence = round(discounted_confidence, 6)`
     - `contrarian_confidence_discount = round(discount, 6)`
     - `review_flagged = review_flagged`
     - All other fields identical to base (winning decision unchanged)

The winning direction (Buy/Hold/Sell) is never changed by the discount — discounting compresses conviction but does not reverse it. If that property is needed, it belongs in a Phase 11 veto mechanism.

**P9-E1-T4: run_all_methods()**

Convenience wrapper used by ensemble_runner to populate `method_comparison`.

```python
def run_all_methods(
    signals: list[AgentSignal | None],
    contrarian: ContrarianAnalysis | None,
    weights: dict[str, float],
) -> dict[str, EnsembleDecision]:
    return {
        "majority": majority_vote(signals),
        "confidence_weighted": confidence_weighted_vote(signals),
        "performance_weighted": performance_weighted_vote(signals, weights),
        "contrarian_adjusted": contrarian_adjusted_vote(signals, contrarian),
    }
```

Keys are canonical method names. `method_comparison["confidence_weighted"]` is always equal to `ensemble_decision` (same function, same inputs).

**P9-E1-T5: Tests**

New file `tests/unit/test_aggregation_methods.py`:

`majority_vote` cases:
- Three agents: 2 Buy, 1 Sell → Buy wins with `collective_confidence=2/3`
- Three agents: tie (1/1/1) → Hold, `collective_confidence=0.0`
- All Hold → Hold, `collective_confidence=1.0`, `agreement=True`
- No valid signals → same empty result as `confidence_weighted_vote`
- `contrarian_confidence_discount=1.0`, `review_flagged=False` always

`performance_weighted_vote` cases:
- Empty weights → identical result to `confidence_weighted_vote`
- weights={"fundamental": 0.9, "technical": 0.5} changes outcome vs. equal-weight when votes diverge
- agent_type not in weights → falls back to weight=1.0
- All-uniform weights (equal values) → same direction as `confidence_weighted_vote`

`contrarian_adjusted_vote` cases:
- No contrarian (None): result equals `confidence_weighted_vote` output; discount=1.0, flagged=False
- Contrarian confidence=0.5: discount=0.75; `collective_confidence` is 0.75× base
- Contrarian confidence=0.8: discount=0.60; `review_flagged=True`
- Contrarian confidence=0.70 (boundary): `review_flagged=False` (strictly greater than, not ≥)
- Contrarian confidence=0.701 (just over boundary): `review_flagged=True`
- Winning direction unchanged by discount

`run_all_methods` cases:
- Returns dict with exactly four keys
- `method_comparison["confidence_weighted"]` matches `confidence_weighted_vote()` output
- When `contrarian=None` and `weights={}`: "majority" and "confidence_weighted" may differ; "performance_weighted" equals "confidence_weighted"; "contrarian_adjusted" equals "confidence_weighted"

Target: ~30 test cases.

---

### P9-E2: Rolling Complexity Metrics

**Rationale:** κ and S formalize the temporal stability of collective behavior (David §5.6.3–5.6.4). These require sequences of historical decisions — conceptually different from per-snapshot metrics (H, D). Added to `metrics.py` as the natural home for all complexity measurement functions.

**Modified files:**
- `src/hifi/collective/metrics.py`
- `tests/unit/test_diversity_metrics.py` (extend existing; rename scope in docstring to reflect new metrics)

**New file:**
- `tests/unit/test_rolling_metrics.py`

**Tickets:**

**P9-E2-T1: herding_coefficient()**

```python
def herding_coefficient(
    agent_votes_per_period: list[list[str]],
    W: int,
) -> float | None:
```

Implementation:
- If `len(agent_votes_per_period) < W`: return `None`
- Use the last `W` elements: `window = agent_votes_per_period[-W:]`
- For each period `t` in window: `a_t = max_count / n_agents` where `max_count` is the count of the plurality option
- `κ = sum(a_t) / W`
- Return `round(κ, 6)`

Edge cases:
- Single agent per period: `a_t = 1.0` always → κ = 1.0 (mathematically correct; the concept requires N ≥ 2)
- Empty inner list: treat as no votes → skip that period (defensive)

**P9-E2-T2: consensus_stability()**

```python
def consensus_stability(
    collective_decisions: list[str],
    W: int,
) -> float | None:
```

Implementation:
- If `len(collective_decisions) < W`: return `None`
- Use the last `W` elements: `window = collective_decisions[-W:]`
- `n_stable = sum(1 for t in range(W-1) if window[t] == window[t+1])`
- `S = n_stable / (W - 1)`
- Return `round(S, 6)`
- Edge case `W = 1`: return `None` (W-1 = 0, undefined). Add guard.

**P9-E2-T3: compute_rolling_metrics()**

```python
def compute_rolling_metrics(
    agent_votes_per_period: list[list[str]],
    collective_decisions: list[str],
    W_values: list[int] = (5, 10, 20),
) -> dict[str, float | None]:
```

Returns dict with keys `"kappa_W{w}"` and `"stability_W{w}"` for each W. Values are float or None.

This is the primary function called in phase-level evaluation (e.g., in the baseline script and holistic test).

**P9-E2-T4: Tests**

New file `tests/unit/test_rolling_metrics.py`:

`herding_coefficient` cases:
- Unanimous agreement every period → κ = 1.0 (all a_t = 1.0)
- One agent per side per period (50/50 for 2 options) → κ = 0.5
- Insufficient history (len < W) → None
- Exactly W records → not None (boundary)
- W - 1 records → None (boundary)
- Different W values (5, 10, 20) from same sequence

`consensus_stability` cases:
- Same decision every period → S = 1.0
- Alternating Buy/Sell → S = 0.0
- Mixed: 3 stable + 1 change in 5 periods → S = 3/4 = 0.75
- Insufficient history → None
- Exactly W records → not None
- W = 1 → None (guard)

`compute_rolling_metrics` cases:
- 20-period sequence: all three windows produce float values
- 7-period sequence: W=5 → float, W=10 → None, W=20 → None
- Empty sequence: all None
- Result keys are exactly `{"kappa_W5", "kappa_W10", "kappa_W20", "stability_W5", "stability_W10", "stability_W20"}`

Target: ~25 test cases.

---

### P9-E3: Ensemble Runner Extension

**Rationale:** The runner is the integration point where all new capabilities converge. `run_ensemble()` gains three responsibilities: (1) capture valid signals into `EnsembleOutput.signals`, (2) run all four aggregation methods via `run_all_methods()`, and (3) log divergence when methods produce different decisions.

**Modified files:**
- `src/hifi/agents/ensemble_runner.py`
- `tests/integration/test_ensemble_runner.py` (extend)

**Tickets:**

**P9-E3-T1: Capture valid signals**

After the voting step (line 146 in current runner), the `valid_signals` list contains all non-None `AgentSignal` objects. Assign this directly to the `signals` field of `EnsembleOutput`.

```python
valid_signals = [s for s in candidate_signals if s is not None]
```

This list is already computed. The change is storing it in `EnsembleOutput.signals` instead of discarding it. No additional computation.

**P9-E3-T2: Load performance weights**

Before calling `run_all_methods()`, load current performance weights:

```python
from hifi.collective.performance_store import get_weights
perf_weights = get_weights(data_dir=data_dir)
```

`get_weights` is a fast file read (or returns uniform defaults if the file does not exist). No LLM or MCP overhead.

**P9-E3-T3: Run all four aggregation methods**

After `decision = confidence_weighted_vote(valid_signals)` (and after contrarian runs), call:

```python
from hifi.collective.voting import run_all_methods
method_comparison = run_all_methods(
    signals=candidate_signals,   # includes None entries (filtered inside each function)
    contrarian=contrarian_analysis,
    weights=perf_weights,
)
```

The primary `ensemble_decision` remains `confidence_weighted_vote(valid_signals)` for backward compatibility. `method_comparison["confidence_weighted"]` will be equal (same function, same inputs).

**P9-E3-T4: Log divergence**

After computing `method_comparison`:

```python
import logging
logger = logging.getLogger(__name__)

decisions = {k: v.collective_decision for k, v in method_comparison.items()}
unique = set(decisions.values()) - {None}
if len(unique) > 1:
    logger.info(
        "Method divergence for %s %s: %s",
        ticker,
        as_of_date,
        decisions,
    )
```

Divergence is the scientifically interesting case. INFO level (not WARNING) because divergence is expected and informative, not a failure.

**P9-E3-T5: Build EnsembleOutput with new fields**

```python
output = EnsembleOutput(
    ticker=ticker,
    as_of_date=as_of_date,
    fundamental_analysis=fundamental,
    technical_analysis=technical,
    ensemble_decision=decision,
    latency_ms=latency_ms,
    risk_analysis=risk_analysis,
    macro_analysis=macro_analysis,
    sentiment_analysis=sentiment_analysis,
    contrarian_analysis=contrarian_analysis,
    # Phase 9 new fields:
    signals=valid_signals,
    aggregation_method="confidence_weighted",
    method_comparison=method_comparison,
)
```

**P9-E3-T6: Tests**

Extend `tests/integration/test_ensemble_runner.py`:
- `output.signals` is non-empty after `run_ensemble()` with at least two agents
- `output.aggregation_method == "confidence_weighted"`
- `output.method_comparison` has exactly four keys
- `output.method_comparison["confidence_weighted"]` equals `output.ensemble_decision` (field-by-field equality)
- `output.signals` contains only non-None `AgentSignal` objects
- Backward compat: `agents=["fundamental", "technical"]` still produces valid output; `method_comparison` has four keys (methods still run on the two-agent signal list)
- `EnsembleOutput` round-trips through `model_dump_json()` → `model_validate_json()` with new fields populated

Target: ~15 new test cases in integration test.

---

### P9-E4: Performance Store + Historical Bootstrap

**Rationale:** `performance_weighted_vote()` requires empirical priors. The bootstrap generates these by running deterministic MCP calls (no LLM) at 20 historical quarter-ends per ticker and computing 60-day forward returns. This is a one-time offline computation that seeds `data/agent_performance_history.json`.

**New files:**
- `src/hifi/collective/performance_store.py`
- `scripts/run_phase9_bootstrap.py`
- `tests/unit/test_performance_store.py`

**Tickets:**

**P9-E4-T1: performance_store.py**

```python
# src/hifi/collective/performance_store.py

import json
import os
from pathlib import Path

from hifi.collective.schemas import AgentPerformanceHistory, DecisionRecord

_DEFAULT_FILENAME = "agent_performance_history.json"
_INITIAL_AGENT_TYPES = ["fundamental", "technical", "risk", "macro"]
_INITIAL_WEIGHT = 0.25  # 1/4 uniform


def _history_path(data_dir: str | None) -> Path:
    root = data_dir or os.environ.get("HIFI_DATA_DIR", ".")
    return Path(root) / _DEFAULT_FILENAME


def load_history(data_dir: str | None = None) -> AgentPerformanceHistory:
    path = _history_path(data_dir)
    if not path.exists():
        return AgentPerformanceHistory(
            records=[],
            weights={t: _INITIAL_WEIGHT for t in _INITIAL_AGENT_TYPES},
            last_updated="",
            n_labeled=0,
        )
    return AgentPerformanceHistory.model_validate_json(path.read_text())


def save_history(history: AgentPerformanceHistory, data_dir: str | None = None) -> None:
    path = _history_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(history.model_dump_json(indent=2))
    tmp.rename(path)


def compute_weights(records: list[DecisionRecord]) -> dict[str, float]:
    labeled = [r for r in records if r.outcome_correct is not None]
    if not labeled:
        return {t: _INITIAL_WEIGHT for t in _INITIAL_AGENT_TYPES}
    by_type: dict[str, list[bool]] = {}
    for r in labeled:
        by_type.setdefault(r.agent_type, []).append(r.outcome_correct)
    weights = {t: sum(correct) / len(correct) for t, correct in by_type.items()}
    # Fill any missing agent type with initial weight
    for t in _INITIAL_AGENT_TYPES:
        weights.setdefault(t, _INITIAL_WEIGHT)
    return weights


def get_weights(data_dir: str | None = None) -> dict[str, float]:
    return load_history(data_dir).weights
```

`save_history` uses atomic write (tmp → rename) to prevent corruption on crash.

`compute_weights` fills missing agent types with the initial uniform weight (0.25) so callers never get a sparse dict.

**P9-E4-T2: Bootstrap quarter definitions**

The 20 quarterly period quarter-ends for the bootstrap span 2018-Q1 through 2022-Q4 (20 quarters). This range is fully covered by the AAPL/JPM/XOM Parquet files in `data/market/` (Phase 1 data acquisition covers 2018–2023).

```python
QUARTER_ENDS = [
    "2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
    "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
    "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
    "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
]
TICKERS = ["AAPL", "JPM", "XOM"]
LABEL_HORIZON_DAYS = 60  # primary
LABEL_THRESHOLD = 0.02   # ±2%
```

For each (ticker, quarter_end):
1. Call `get_risk_metrics(ticker, quarter_end)` → extract risk agent "signal" deterministically using a simple rule: `Sharpe > 0.8` → Buy, `Sharpe < 0.3` → Sell, else Hold (this is a *proxy* signal, not an LLM call, for bootstrap purposes only).
2. Call `get_technical_indicators(ticker, quarter_end)` → extract technical signal using RSI threshold rule: `RSI < 40` → Buy, `RSI > 60` → Sell, else Hold.
3. Fundamental + macro signals: hardcoded per quarter using known historical consensus (or use the same deterministic rule based on available ratio values).
4. Sentiment: Hold/0.0 (fail-open throughout bootstrap).
5. Contrarian: skip.
6. Compute 60-day forward return from Parquet: `price at quarter_end + 60 trading days`.
7. Label: BUY correct if forward_return > +2%; SELL correct if < −2%; HOLD correct within ±2%.

The rule-based proxy signals are documented clearly as bootstrap-only heuristics. They are NOT LLM outputs. Their purpose is solely to generate an initial performance prior from observed market outcomes.

**P9-E4-T3: run_phase9_bootstrap.py script**

```
scripts/run_phase9_bootstrap.py
```

Structure:
- Imports: `call_tool`, Parquet readers, `performance_store`, `schemas`
- Iterates: `for ticker in TICKERS: for quarter in QUARTER_ENDS:`
  - Calls MCP tools deterministically (no LLM)
  - Produces `DecisionRecord` per agent per quarter-end
  - Computes forward return from existing Parquet data
  - Labels each record
- Calls `compute_weights(all_records)` → weights dict
- Saves `AgentPerformanceHistory` to `data/agent_performance_history.json`
- Prints summary: n_records, n_labeled, weights per agent_type

The script is fully deterministic given fixed Parquet data. Running it twice produces identical output.

**P9-E4-T4: Tests**

New file `tests/unit/test_performance_store.py`:
- `load_history` with non-existent file → returns uniform weights, empty records
- `load_history` with valid JSON file → parses correctly
- `save_history` → file exists, content is valid JSON, round-trips
- `save_history` → atomic: no partial writes on tmpfs (simulate by checking .tmp is removed)
- `compute_weights` with no labeled records → uniform weights for all four agent types
- `compute_weights` with labeled records: fundamental 3/4 correct, technical 2/4 → weights as expected
- `compute_weights` with missing agent type in records → filled with initial weight
- `get_weights` on non-existent file → returns uniform dict

Target: ~15 test cases.

---

### P9-E5: Holistic Test + Baseline + Bitacora

**Rationale:** End-to-end validation of Phase 9 capabilities in a single pipeline test. Baseline measurement documents the structural comparison results (per-method decisions, H, D, κ, S) at 2023-03-31 for AAPL/JPM/XOM.

**New files:**
- `tests/holistic/test_phase9_collective_engine.py`
- `tests/unit/test_phase9_baseline.py`
- `doc/bitacora/PHASE_09_COLLECTIVE_ENGINE.md`

**Modified files:**
- `tests/fixtures/baseline/` — add `phase9_collective.json`

**Tickets:**

**P9-E5-T1: Holistic test**

`tests/holistic/test_phase9_collective_engine.py`

Pattern mirrors Phase 8 holistic test: LLMs are stubbed (monkeypatched), MCP tools use Phase 1 Parquet fixtures, knowledge server is unavailable (sentinel fail-open).

What this test validates:
1. `run_ensemble()` with all 6 agents produces `EnsembleOutput` with non-empty `method_comparison`
2. `method_comparison` has exactly four keys: "majority", "confidence_weighted", "performance_weighted", "contrarian_adjusted"
3. `method_comparison["confidence_weighted"]` equals `output.ensemble_decision` field-by-field
4. `method_comparison["contrarian_adjusted"].contrarian_confidence_discount` < 1.0 when contrarian ran
5. `method_comparison["contrarian_adjusted"].review_flagged` is bool
6. `output.signals` contains exactly the non-None agent signals (no duplicates, no Nones)
7. `output.aggregation_method == "confidence_weighted"`
8. `EnsembleOutput` JSON round-trip is lossless
9. With `agents=["fundamental", "technical"]` (Phase 4 backward compat): `method_comparison` still has four keys (methods run on two-agent signals); `signals` has ≤ 2 entries
10. Rolling metrics computed over a synthetic 20-period sequence: κ and S are float for W=5, 10, 20

**P9-E5-T2: Baseline script**

`scripts/run_phase9_baseline.py` (following the Phase 8 pattern):
- Runs `run_ensemble()` for AAPL, JPM, XOM at "2023-03-31" (live LM Studio required)
- Computes `method_comparison` for all three tickers
- Records per-method decisions, collective_confidence, H, D
- Prints comparison table:

```
Ticker  | Majority | ConfWt  | PerfWt  | ContAdj | Contrarian Flag
AAPL    | Buy      | Buy     | Buy     | Buy     | False
JPM     | Hold     | Hold    | Hold    | Hold    | False
XOM     | Sell     | Sell    | Sell    | Sell    | False
```

(Actual values filled at baseline run time.)

Saves to `tests/fixtures/baseline/phase9_collective.json`.

**P9-E5-T3: Unit test for baseline fixture**

`tests/unit/test_phase9_baseline.py` (same pattern as `test_phase8_baseline.py`):
- `@pytest.mark.skipif(not os.path.exists(FIXTURE_PATH), reason="baseline fixture not generated")`
- Validates fixture schema: all required keys present, types correct
- `method_comparison` has four keys per ticker
- All `collective_decision` values are in `{"Buy", "Hold", "Sell", None}`

**P9-E5-T4: Scientific bitacora**

`doc/bitacora/PHASE_09_COLLECTIVE_ENGINE.md` follows the established bitacora template:
- Phase overview and scientific question
- Implementation decisions (DJ-039 through DJ-04X, continuing the DJ numbering from Phase 8)
- Surprises, deviations, calibration notes
- Baseline results (filled during execution)
- Structural comparison table
- Open questions for Phase 10

---

## Test Strategy

### Test Count Target

| Category | New tests | Notes |
|---|---|---|
| Unit: schema extension (E0) | ~15 | EnsembleDecision, EnsembleOutput, DecisionRecord, AgentPerformanceHistory |
| Unit: aggregation methods (E1) | ~30 | all 4 methods × multiple signal configurations |
| Unit: rolling metrics (E2) | ~25 | κ, S, compute_rolling_metrics, edge cases |
| Unit: performance store (E4) | ~15 | load, save, compute_weights, get_weights |
| Unit: baseline fixture (E5) | ~5 | schema validation, conditional on fixture existence |
| Integration: ensemble runner (E3) | ~15 | method_comparison, signals, backward compat |
| Holistic: Phase 9 pipeline (E5) | ~20 | full end-to-end with stubbed LLMs |
| **Total new** | **~125** | |

**Target total: ~870 tests** (745 Phase 8 baseline + ~125 new).

### No-Mock Policy

Consistent with HiFi conventions:
- LLMs: stub with objects that have `.model_name` and return canned JSON via monkeypatch (same pattern as Phase 8 holistic test)
- MCP tools: use Phase 1 Parquet fixtures in `tests/fixtures/market/` and `tests/fixtures/macro/`
- Performance history file: use `tmp_path` (pytest fixture) to avoid touching `data/`
- No `unittest.mock.MagicMock` or `pytest-mock` stubs for core logic

### Backward Compatibility Verification

The following existing tests must continue to pass without modification:
- `tests/unit/test_voting.py` — `confidence_weighted_vote()` is unchanged; `EnsembleDecision` gains fields with defaults
- `tests/unit/test_ensemble_schemas.py` — existing constructors still valid; new tests are additive
- `tests/holistic/test_phase4_ensemble_pipeline.py` through `test_phase8_agent_population.py` — `EnsembleOutput` new fields have defaults; no existing caller breaks
- `tests/integration/test_ensemble_runner.py` — existing assertions still valid; new test cases added

---

## Dependency Order

```
P9-E0 (schemas)
    │
    ├── P9-E1 (aggregation methods)  ─────────────────────────────────┐
    │                                                                  │
    ├── P9-E2 (rolling metrics)      ──────────────────────────────── │ ──┐
    │                                                                  │   │
    └── P9-E4-T1 (performance_store) ─────────────────────────────── │ ──┼──┐
                                                                       │   │  │
                                     P9-E3 (runner extension) ←────── ┘   │  │
                                          │                                │  │
                                          └──── P9-E5 (holistic) ←────────┘──┘
                                                    ↑
                                               P9-E4-T2,T3 (bootstrap) feeds
                                               data/agent_performance_history.json
                                               (offline, run manually)
```

E0 must be completed first. E1, E2, and E4-T1 can proceed in parallel after E0. E3 requires E0, E1, and E4-T1. E5 requires all of E0–E4.

The bootstrap script (E4-T2, T3) is an offline measurement tool. Tests do not depend on the bootstrap having been run (performance_store falls back to uniform weights when the file is absent). The bootstrap is run once by the developer to seed `data/agent_performance_history.json` before the baseline measurement.

---

## Success Criteria

From HIFI_PROTOCOL_V1.md Phase 9, adapted to D-01 through D-07 scope:

- [ ] Four aggregation methods implemented and exercised on every `run_ensemble()` call
- [ ] `EnsembleOutput.method_comparison` populated with all four decisions on each call
- [ ] Contrarian discount formula `1 - 0.5 × c` applied in `contrarian_adjusted_vote`
- [ ] `review_flagged = True` when contrarian confidence > 0.70
- [ ] κ and S implemented for W ∈ {5, 10, 20}; return None below minimum history
- [ ] Historical bootstrap produces `data/agent_performance_history.json` with labeled records
- [ ] `performance_weighted_vote()` uses bootstrap weights when file exists; uniform fallback when not
- [ ] `DecisionRecord` and `AgentPerformanceHistory` schemas implemented and persisted
- [ ] All new tests passing; total test count ≥ 860
- [ ] Zero lint errors (`ruff check src/ tests/`)
- [ ] Backward compatibility: all Phase 4–8 tests still pass without modification
- [ ] Structural comparison table (per-method decisions and metrics for AAPL/JPM/XOM) documented in bitacora

---

## Deferred to Later Phases

- **Structured debate (Phase 11):** multi-turn LLM debate after initial voting. Requires multi-turn orchestration and training data on when debate improves decisions.
- **Adaptive aggregation (Phase 13):** learned aggregation function. Requires labeled outcome data from Phase 10+.
- **GAN synthetic scenarios (Phase 12):** tail-behavior augmentation. Requires 50+ ticker evaluation universe.
- **Walk-forward validation with purged cross-validation (Phase 10):** rigorous accuracy comparison per aggregation method.
- **Bootstrap expansion to 20+ tickers (Phase 10):** AAPL/JPM/XOM coverage adequate for Phase 9 structural analysis.
- **Drift detection (Phase 14):** KS test, CUSUM for detecting when agent weights need recalibration.
- **Performance weight recalibration cadence (Phase 10):** policy for when to re-run bootstrap or apply live label updates.

---

*Phase: 09-collective-decision-engine*
*Plan authored: 2026-06-12*
*Locked decisions: D-01 through D-07 (plans/PHASE_09_CONTEXT.md)*
