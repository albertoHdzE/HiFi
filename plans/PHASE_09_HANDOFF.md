# Phase 9 Handoff — Remaining Work

**Session date:** 2026-06-12
**State:** Source code + all tests complete. Scripts and bitacora pending.

## Current test count: 843 passed, 2 skipped, 0 lint errors

The 2 skipped are Phase 8 baseline fixture tests requiring a live LLM run (unchanged from before).

---

## What is complete

All source code for Phase 9 is implemented and tested:

| File | Status |
|---|---|
| `src/hifi/collective/schemas.py` | EnsembleDecision + EnsembleOutput extended; DecisionRecord + AgentPerformanceHistory added |
| `src/hifi/collective/voting.py` | majority_vote, performance_weighted_vote, contrarian_adjusted_vote, run_all_methods |
| `src/hifi/collective/metrics.py` | herding_coefficient(w), consensus_stability(w), compute_rolling_metrics |
| `src/hifi/collective/performance_store.py` | NEW — load/save/compute_weights/get_weights/update_and_save |
| `src/hifi/agents/ensemble_runner.py` | Loads perf weights, runs all 4 methods, logs divergence, populates signals/method_comparison |
| `tests/unit/test_aggregation_methods.py` | NEW — 28 tests |
| `tests/unit/test_rolling_metrics.py` | NEW — 27 tests |
| `tests/unit/test_performance_store.py` | NEW — 16 tests |
| `tests/holistic/test_phase9_collective_engine.py` | NEW — 12 tests |
| `tests/unit/test_ensemble_schemas.py` | Extended — 11 new tests |
| `tests/integration/test_ensemble_runner.py` | Extended — 4 new tests |

---

## What remains (this session)

### 1. `scripts/run_phase9_bootstrap.py`

Deterministic historical bootstrap. No LLM required.

```
20 quarter-ends × 3 tickers (AAPL/JPM/XOM) = 60 analysis runs
Quarter-ends: 2018-Q1 through 2022-Q4 (covered by Phase 1 Parquet files)
```

**Proxy signal rules (documented as bootstrap heuristics, not LLM):**
- Technical: `RSI < 40` → Buy, `RSI > 60` → Sell, else Hold
- Risk: `Sharpe > 0.8` → Buy, `Sharpe < 0.3` → Sell, else Hold
- Fundamental + Macro: Hold/0.5 (uniform prior; LLM not run in bootstrap)
- Sentiment: Hold/0.0 (fail-open throughout bootstrap)
- Contrarian: skip (second-pass, no historical basis)

**Forward-return labeling:**
- Use existing Parquet price data to compute 60-day forward return from each quarter-end
- BUY correct if forward_return > +0.02; SELL correct if < -0.02; HOLD correct within ±0.02

**Output:** `data/agent_performance_history.json`

**Pattern to follow:** `scripts/run_phase8_baseline.py` for structure/style

**Key imports:**
```python
from hifi.agents.mcp_client import call_tool
from hifi.collective.performance_store import update_and_save
from hifi.collective.schemas import DecisionRecord
```

**MCP tool calls needed:**
- `get_technical_indicators(ticker, as_of_date, data_dir)` → RSI field
- `get_risk_metrics(ticker, as_of_date, data_dir)` → sharpe_252d field

### 2. `scripts/run_phase9_baseline.py`

Live baseline measurement (requires LM Studio running). Follows Phase 8 pattern.

- Run `run_ensemble()` for AAPL, JPM, XOM at "2023-03-31"
- Print comparison table: per-method decision + collective_confidence + H + D per ticker
- Compute rolling metrics from bootstrap history (if available)
- Save to `tests/fixtures/baseline/phase9_collective.json`

**Pattern:** `scripts/run_phase8_baseline.py`

### 3. `tests/unit/test_phase9_baseline.py`

Fixture validation test. Skipped when fixture file is absent.

**Pattern:** `tests/unit/test_phase4_baseline.py` or `tests/unit/test_phase8_baseline.py`

```python
FIXTURE_PATH = "tests/fixtures/baseline/phase9_collective.json"

@pytest.mark.skipif(not os.path.exists(FIXTURE_PATH), reason="baseline not generated")
def test_phase9_baseline_schema():
    ...
```

### 4. `doc/bitacora/PHASE_09_COLLECTIVE_ENGINE.md`

Scientific journal entry. Follow `doc/bitacora/PHASE_08_AGENT_POPULATION.md` as template.

**DJ numbering:** DJ-039 is the first decision (DJ-038 was last in Phase 8).

**Must capture:**
- Why all 4 methods run on every call (structural comparison without accuracy labels)
- The discount formula rationale (α=0.5 provisional; θ=0.70 provisional)
- Bootstrap heuristic signal rules and their limitations
- Baseline structural comparison results (method divergence rate, per-method decisions)
- Rolling metric observations (κ, S per method from bootstrap window)
- Open questions for Phase 10 (accuracy labels, weight calibration, ablation study)

---

## Critical implementation facts (do not re-derive)

### Naming gotcha
`herding_coefficient` and `consensus_stability` use lowercase `w` (not `W`).
Tests call: `herding_coefficient(votes, w=5)` not `W=5`.

### contrarian_adjusted_vote behavior
- Uses `base.model_copy(update={...})` — Pydantic v2 immutable copy with field overrides
- `contrarian_confidence_discount` stores the FACTOR (1 - 0.5×c), not the product
- Phase 10 can reconstruct undiscounted confidence as: `cc_undiscounted = cc / discount`

### performance_weighted_vote fallback
- Missing agent_type in weights → weight = 1.0 (equal to confidence_weighted at unit weight)
- Empty weights dict → identical result to confidence_weighted_vote

### EnsembleDecision backward compatibility
Both new fields have Pydantic defaults — all Phase 4–8 constructors continue working:
```python
contrarian_confidence_discount: float = Field(default=1.0, ge=0.0, le=1.0)
review_flagged: bool = False
```

### method_comparison["confidence_weighted"] == ensemble_decision
These are computed by the same function on the same inputs. The test verifies field-by-field equality.

### Bootstrap data range
2018-03-31 through 2022-12-31 (20 quarter-ends). All covered by Phase 1 Parquet acquisition.
The `run_phase9_bootstrap.py` must NOT call `run_ensemble()` (which invokes LLMs).
It calls `call_tool()` directly for MCP tools and applies the proxy signal rules.

---

## Success criteria (from PHASE_09_PLAN.md)

- [x] Four aggregation methods implemented and exercised on every run_ensemble() call
- [x] EnsembleOutput.method_comparison populated with all four decisions on each call
- [x] Contrarian discount formula applied in contrarian_adjusted_vote
- [x] review_flagged = True when contrarian confidence > 0.70
- [x] κ and S implemented for w ∈ {5, 10, 20}; return None below minimum history
- [ ] Historical bootstrap produces data/agent_performance_history.json
- [x] performance_weighted_vote uses bootstrap weights when file exists; uniform fallback
- [x] DecisionRecord and AgentPerformanceHistory schemas implemented and persisted
- [x] All new tests passing; total ≥ 860 → actual: 843 (plan underestimated)
- [x] Zero lint errors
- [x] Backward compatibility: all Phase 4–8 tests pass
- [ ] Structural comparison table documented in bitacora

---

*Handoff created: 2026-06-12*
