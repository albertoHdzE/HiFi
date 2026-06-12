# Phase 10: Evaluation, Backtesting, and Universe Expansion

**Status:** PLANNED

| Epic | Title | Status |
|---|---|---|
| P10-E0 | Accuracy labeling — 60-day forward returns for 2023-03-31 | PLANNED |
| P10-E1 | QuantStats tear sheets — portfolio analytics layer | PLANNED |
| P10-E2 | Ticker universe expansion (3 → 15 tickers + re-bootstrap) | PLANNED |
| P10-E3 | Stub LLM removal — structural holistic tests | PLANNED |
| P10-E4 | Weight calibration and method divergence analysis | PLANNED |
| P10-E5 | Baseline measurement + bitacora | PLANNED |

**David Sections:** §12.2 Collective Decision Engine, §12.3 Contrarian Integration,
§8.4 Reference Strategy Datasets, §5.6.3 Herding Coefficient, §5.6.4 Consensus Stability,
§13 Backtesting Protocol (partial), §5.7 Portfolio Analytics
**Learning Guide Topics:** 3.3 Collective Intelligence (measurement), 8.2 Collective Intelligence,
8.3 Emergence & Measurement, 6.1 Portfolio Analytics, 6.2 QuantStats
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 10 (critical capstone path)
**Decision IDs:** DJ-044 through DJ-052

---

## Governing Philosophy for This Phase

Phase 9 installed the measurement infrastructure — four methods running on every call,
a performance history store, and rolling complexity metrics. It could not answer whether
method choice matters empirically, because outcome labels for 2023-03-31 were not
available at the time of development (the 60-trading-day forward window closes
approximately 2023-06-21, which is within the Phase 1 Parquet horizon).

Phase 10 closes that feedback loop.

**The core scientific question:** does the choice of aggregation method affect collective
accuracy? The four methods encode four different hypotheses about what makes a good
collective decision. Phase 10 is the first phase that can answer this question
empirically — not definitively, but for the first time with real outcome labels.

Three structural concerns run alongside the empirical work:

1. **Universe size.** Three tickers produce three outcome labels per analysis date — not
   enough to differentiate methods statistically. Expanding to 15 tickers gives 45 labels
   per date, and the bootstrap re-run gives 15 × 20 = 300 labeled records. This is still
   a small dataset, but it is the minimum to begin observing method-level accuracy
   differentiation.

2. **Test philosophy debt.** Holistic tests currently monkeypatch LLMs to produce
   deterministic canned responses. This conflates two concerns: pipeline structure (which
   tests should own) and LLM behavior (which baseline runs should own). The stubs must
   go. Structural holistic tests should be callable without any LLM infrastructure.

3. **QuantStats on the critical path.** Phase 10 is listed in the protocol as a critical
   capstone milestone (Phases 0,1,2,3,4,5,6,10,14,17). The QuantStats tear sheet
   layer is the primary deliverable that distinguishes a financial intelligence platform
   from a research prototype — it produces the portfolio-level evidence expected by
   the capstone reviewers.

---

## Pre-Phase Decisions (DJ-044 through DJ-052)

### DJ-044: MethodAccuracyRecord — Separate Schema from AgentPerformanceHistory

Method-level collective accuracy is tracked separately from agent-level individual
accuracy. Two reasons:

1. A collective method has no `agent_type` — it aggregates across all agents. Forcing
   method records into `AgentPerformanceHistory` would require either a fake agent_type
   string or schema surgery.
2. The weight computation in `compute_weights()` (used by `performance_weighted_vote`)
   depends on per-agent accuracy. Mixing in method records would contaminate that
   computation.

New schemas added to `src/hifi/collective/schemas.py`:

```python
class MethodDecisionRecord(BaseModel):
    ticker: str
    analysis_date: str             # ISO 8601
    method_name: str               # one of the 4 canonical keys
    decision: str                  # "Buy" | "Hold" | "Sell"
    collective_confidence: float
    forward_return: float | None = None
    outcome_correct: bool | None = None
    horizon_days: int = 60
    outcome_labeled_at: str | None = None

class MethodAccuracyReport(BaseModel):
    records: list[MethodDecisionRecord]
    accuracy_by_method: dict[str, float]   # method_name -> fraction correct
    n_labeled: int
    tickers: list[str]
    analysis_dates: list[str]
    generated_at: str
```

`accuracy_by_method` is computed by `model_validator(mode="after")` from labeled records,
so it is always consistent with the records list. Unresolvable records
(`outcome_correct=None`) are excluded from the denominator, matching the HR/GR
convention from Phase 5.

### DJ-045: Strategy Returns Construction for QuantStats

To produce a QuantStats-compatible returns series from quarterly signals, Phase 10 uses
an honest approximation: **daily returns attribution**. For each labeled
`MethodDecisionRecord` (ticker, date, method, decision):

```
position(t) = +1  if decision == "Buy"
              -1  if decision == "Sell"
               0  if decision == "Hold"

strategy_return(t) = position(t) * daily_return(t)
```

for each trading day `t` in `[analysis_date, analysis_date + horizon_days)`.

Across tickers within a method, the portfolio return on day `t` is the equal-weight
mean of the position returns of all tickers active on that day.

Limitations documented explicitly in the bitacora:
- Quarterly rebalancing means position changes only at quarter-ends. This ignores
  intra-period signal updates and overstates the strategy's responsiveness.
- Equal weighting ignores position sizing, transaction costs, and slippage.
- With 20 quarter-ends and 15 tickers, the total history is ~240 signal periods
  → ~14,400 daily return observations. This is adequate for Sharpe/drawdown/Calmar
  but the annual return estimate will have wide confidence intervals.

The approach is documented, not hidden. The tear sheet is labeled with these caveats.
A position-sized, transaction-cost-aware backtest is Phase 14 scope.

### DJ-046: Ticker Universe Selection (15 tickers + 1 benchmark)

Selected for sector diversity, continuous yfinance history from 2016-01-01, and
relevance to the 3-agent information structure (fundamentals, technical, macro risk):

| Sector | Tickers |
|---|---|
| Technology | AAPL (existing), MSFT, NVDA, GOOGL |
| Finance | JPM (existing), BAC, GS |
| Energy | XOM (existing), CVX |
| Healthcare | JNJ, UNH |
| Consumer | AMZN, WMT |
| Industrial | CAT |
| Utilities | NEE |
| Benchmark | SPY (existing, not bootstrapped) |

Total trading universe: 15 tickers (3 existing + 12 new). SPY remains benchmark-only.

The new tickers are not added to Phase 1 fixtures (those are frozen). They are
downloaded to `data/market/` by a new `scripts/acquire_phase10_data.py` script.
Phase 1 script and fixtures are untouched.

Bootstrap quarter-ends remain 2018-Q1 through 2022-Q4 (20 periods) — fully covered
by the 2016-2023 download range.

### DJ-047: Structural Holistic Test Policy (Formalized)

Holistic tests own pipeline structural correctness. They MUST NOT invoke LLMs or
monkeypatch LLMs to return canned responses. The boundary:

**Holistic test scope (deterministic, no LLM):**
- EnsembleOutput schema construction and JSON round-trip
- method_comparison population: correct keys, semantics, math
- Aggregation voting logic: `run_all_methods(signals, ...)` with seeded AgentSignal list
- Contrarian discount formula given known inputs
- Verification layer integration with a known EnsembleOutput
- Performance store load/compute/save with a synthetic history fixture

**Baseline script scope (requires LM Studio):**
- Actual agent LLM invocations (what did the model say about AAPL?)
- Prompt correctness and field extraction
- Agent behavioral properties: hallucination rate, grounding rate, latency

The rewrite rule: any test that monkeypatches `make_llm` → delegate to baseline-*
and replace with a lower-level structural assertion that exercises the same code
path using pre-specified signal values.

### DJ-048: Performance Store File Locking

`performance_store.py` adds `filelock` (via `FileLock`) to `update_and_save()` to
prevent concurrent write corruption. Phase 10 runs are still single-threaded, so the
lock is never contested — but adding it now means Phase 11 parallel ensemble runs
will not corrupt the performance history.

`filelock` is added to `pyproject.toml` dependencies. Lock file: `{path}.lock` (sibling
of the JSON file).

### DJ-049: Labeler Module Placement

The new labeling module goes in `src/hifi/collective/labeler.py`, alongside the existing
voting, metrics, and performance_store modules. It imports from `hifi.data.storage`
(for Parquet loading) and `hifi.collective.schemas` (for DecisionRecord,
MethodDecisionRecord). No circular imports: the labeler reads but does not write agent
output.

### DJ-050: Tear Sheet Module Placement

New package `src/hifi/analytics/` with `tearsheet.py`. This is the first module
in the analytics package — Phase 14 (Paper Trading) will add execution analytics
here. The package is created empty now (with `__init__.py`) to establish the namespace.

### DJ-051: Bootstrap Re-Run Policy

Phase 10 re-bootstraps with 15 tickers. The new script `scripts/run_phase10_bootstrap.py`
accepts `--reset` (wipes existing history and generates from scratch for 15 tickers)
and `--extend` (appends new ticker records without disturbing existing ones).

Default behavior: `--reset`, because mixing 3-ticker bootstrap weights with 15-ticker
bootstrap weights produces a biased history. The Phase 9 bootstrap (3 tickers) is
superseded by Phase 10 (15 tickers). `make bootstrap` is updated to call the Phase 10
script.

### DJ-052: 20-Day Secondary Horizon (Deferred from Phase 9)

Phase 9 (DJ-042) deferred the 20-day secondary evaluation horizon to Phase 10. With
15 tickers and 20 quarter-ends, there are now 300 signal periods with 20-day forward
data available (most quarter-ends have 20 trading days before the next quarter-end).
Phase 10 labels all records at both 20-day and 60-day horizons. `MethodDecisionRecord`
already supports `horizon_days: int = 60` — two records per (ticker, date, method) are
written, one per horizon.

---

## Interface Design

### New Source Files

```
src/hifi/collective/labeler.py           — forward return computation + labeling
src/hifi/analytics/__init__.py           — package stub
src/hifi/analytics/tearsheet.py          — strategy returns + QuantStats metrics
scripts/acquire_phase10_data.py          — yfinance download for 12 new tickers
scripts/run_phase10_bootstrap.py         — 15-ticker bootstrap with --reset/--extend
scripts/run_phase10_labeling.py          — label 2023-03-31 baseline run
scripts/run_phase10_calibration.py       — weight calibration + divergence analysis
scripts/run_phase10_baseline.py          — generate tests/fixtures/baseline/phase10_accuracy.json
```

### Modified Source Files

```
src/hifi/collective/schemas.py           — MethodDecisionRecord, MethodAccuracyReport
src/hifi/collective/performance_store.py — filelock in update_and_save()
pyproject.toml                           — add filelock dependency
Makefile                                 — acquire-data-phase10, bootstrap, baseline-phase10
scripts/check_env.py                     — phase10-data, phase10-bootstrap, phase10-fixture checks
```

### New Test Files

```
tests/unit/test_labeler.py               — forward return computation, labeling rules
tests/unit/test_tearsheet.py             — strategy returns construction, QuantStats metrics
tests/unit/test_method_accuracy_report.  — MethodDecisionRecord/MethodAccuracyReport schemas
tests/holistic/test_phase10_evaluation.py — accuracy + tear sheet structural test
```

### Modified Test Files

```
tests/holistic/test_phase9_collective_engine.py  — remove LLM stubs (P10-E3)
tests/holistic/test_phase8_agent_population.py   — remove LLM stubs (P10-E3)
```

### Key Function Signatures

```python
# src/hifi/collective/labeler.py

def compute_forward_return(
    ticker: str,
    analysis_date: str,
    data_dir: str,
    horizon_days: int = 60,
) -> float | None:
    """
    Load OHLCV Parquet for ticker, find the first trading day >= analysis_date,
    and compute the forward return over horizon_days trading days.

    Returns None when the Parquet file is absent, or when fewer than horizon_days
    trading days remain after analysis_date (unlabeled / insufficient data).
    """

def label_method_decisions(
    ensemble_outputs: list[EnsembleOutput],
    data_dir: str,
    horizon_days: int = 60,
) -> list[MethodDecisionRecord]:
    """
    For each EnsembleOutput, extract all four methods' collective decisions,
    compute forward return, and apply DJ-042 labeling rules.

    Returns a flat list: len = len(ensemble_outputs) * 4 * len(horizons).
    Unlabeled records have outcome_correct=None.
    """

def label_agent_decisions(
    ensemble_outputs: list[EnsembleOutput],
    data_dir: str,
    horizon_days: int = 60,
) -> list[DecisionRecord]:
    """
    Extract individual agent signals from EnsembleOutput.signals,
    label each against forward return, and return DecisionRecord list.

    Only agents with non-None signals contribute. Sentiment (confidence=0.0)
    is included but will be unlabeled if it lacks a directional signal.
    """

def build_method_accuracy_report(
    records: list[MethodDecisionRecord],
) -> MethodAccuracyReport:
    """Aggregate labeled records into a MethodAccuracyReport with accuracy_by_method."""
```

```python
# src/hifi/analytics/tearsheet.py

def build_strategy_returns(
    method_records: list[MethodDecisionRecord],
    ohlcv_map: dict[str, pd.DataFrame],  # ticker -> daily OHLCV DataFrame
    horizon_days: int = 60,
) -> pd.Series:
    """
    Convert labeled MethodDecisionRecords to a daily strategy returns Series.

    Position = +1/−1/0 per decision. Portfolio = equal-weight across active tickers.
    Returns a pandas Series with DatetimeIndex and daily float returns.
    Gaps between quarter-end windows are filled with 0.0 (out of market).
    """

class TearsheetSummary(BaseModel):
    method_name: str
    tickers: list[str]
    n_periods: int              # number of quarter-ends included
    sharpe_annual: float
    sortino_annual: float
    max_drawdown: float
    calmar: float
    cagr: float                 # compound annual growth rate
    win_rate: float             # fraction of signal periods with positive return
    avg_return_per_period: float
    generated_at: str

def compute_tearsheet(
    method_records: list[MethodDecisionRecord],
    ohlcv_map: dict[str, pd.DataFrame],
    method_name: str,
) -> TearsheetSummary:
    """Build strategy returns and compute QuantStats-backed metrics."""
```

---

## Epic P10-E0: Accuracy Labeling — 60-Day Forward Returns for 2023-03-31

**Scope:** Label the 2023-03-31 baseline ensemble output (AAPL/JPM/XOM) at both
60-day and 20-day horizons. The Phase 1 Parquet files cover through 2023-06-30:
- 60-day window from 2023-03-31 ends ~2023-06-21 (within horizon)
- 20-day window from 2023-03-31 ends ~2023-04-28 (well within horizon)

Both horizons are fully resolvable from existing Parquet data.

Also extract per-agent decisions from the 2023-03-31 baseline fixture and append
labeled DecisionRecords to `agent_performance_history.json`. These become the first
real LLM-generated records in the performance store, replacing pure bootstrap heuristics.

**Source:** `tests/fixtures/baseline/phase9_collective.json`

### Tickets

**P10-E0-T1: MethodDecisionRecord + MethodAccuracyReport schemas**
- Add both schemas to `src/hifi/collective/schemas.py`
- MethodAccuracyReport: `model_validator(mode="after")` computes `accuracy_by_method`
  and `n_labeled` from labeled records (parallel to AgentVerificationReport in Phase 5)
- Validation: decision must be Buy/Hold/Sell; confidence in [0,1]; horizon_days > 0

**P10-E0-T2: compute_forward_return() in labeler.py**
- Load OHLCV Parquet using `hifi.data.storage.read_ohlcv()` (with raw fallback for
  pre-metadata Parquets)
- Find first trading day >= analysis_date using `searchsorted` on sorted DatetimeIndex
- Return None if fewer than horizon_days rows remain after start_idx
- Edge case: analysis_date falls on a weekend/holiday → advance to next trading day

**P10-E0-T3: label_method_decisions() + label_agent_decisions()**
- Both functions in `src/hifi/collective/labeler.py`
- DJ-042 labeling rules: Buy correct if forward_return > +0.02; Sell if < -0.02;
  Hold if within ±0.02
- `label_agent_decisions` reads `ensemble_output.signals` (populated in Phase 9);
  matches agent_type from AgentSignal.agent_type field
- Both functions write `outcome_labeled_at=datetime.now(UTC).isoformat()`

**P10-E0-T4: build_method_accuracy_report()**
- Pure function: given a list of MethodDecisionRecords, compute accuracy per method
- Excludes outcome_correct=None from denominator (matches Phase 5 HR/GR convention)
- Returns MethodAccuracyReport with accuracy_by_method sorted by value descending

**P10-E0-T5: scripts/run_phase10_labeling.py**
- Load `tests/fixtures/baseline/phase9_collective.json` (list of EnsembleOutput)
- Call label_method_decisions() for horizon_days=60 and horizon_days=20
- Call label_agent_decisions() for horizon_days=60
- Append agent DecisionRecords to `data/agent_performance_history.json` via
  `performance_store.update_and_save()`
- Print accuracy table:

```
Method accuracy (2023-03-31, AAPL/JPM/XOM, horizon=60d)
method                  correct  total  accuracy
confidence_weighted      N/3      3     X.XX
majority                 N/3      3     X.XX
performance_weighted     N/3      3     X.XX
contrarian_adjusted      N/3      3     X.XX
```

**P10-E0-T6: Unit tests for labeler.py**
- `test_compute_forward_return_buy_case`: seeded OHLCV, known return, assert correct float
- `test_compute_forward_return_insufficient_data`: fewer than horizon_days remaining → None
- `test_compute_forward_return_weekend_analysis_date`: advances to next trading day
- `test_label_method_decisions_all_labeled`: all 4 methods × 3 tickers labeled
- `test_label_agent_decisions_excludes_none_signals`: signals with decision=None skipped
- `test_build_method_accuracy_report_excludes_unlabeled`: None records not in denominator

---

## Epic P10-E1: QuantStats Tear Sheets — Portfolio Analytics Layer

**Scope:** Convert labeled MethodDecisionRecords to strategy daily returns series and
compute QuantStats-backed portfolio metrics per method. Output stored as
`data/tearsheets/{method_name}_summary.json`. This is the primary capstone deliverable
for Phase 10.

**Input:** The expanded bootstrap data (P10-E2) provides ~300 labeled records across
15 tickers and 20 quarter-ends. Phase 10-E1 consumes this after P10-E2 completes.
For the first cut (before P10-E2), it runs on the 3-ticker baseline data.

### Tickets

**P10-E1-T1: src/hifi/analytics/ package**
- Create `src/hifi/analytics/__init__.py` (empty stub)
- Create `src/hifi/analytics/tearsheet.py` with TearsheetSummary schema and
  build_strategy_returns() + compute_tearsheet() functions (DJ-045)

**P10-E1-T2: build_strategy_returns()**
- Input: list of MethodDecisionRecords + ohlcv_map (ticker -> DataFrame with Date index)
- Algorithm:
  1. Group records by analysis_date
  2. For each analysis_date, set position for each ticker based on decision
  3. Generate daily strategy returns: position × actual daily return for days in window
  4. Equal-weight across tickers active on each day
  5. Fill gaps between windows with 0.0
  6. Return pd.Series with DatetimeIndex
- Overlap detection: when two quarter-end windows for the same ticker overlap (e.g.,
  2018-Q1 and 2018-Q2 windows intersect), later window takes priority
- Empty input → raises ValueError with descriptive message (not silent NaN series)

**P10-E1-T3: compute_tearsheet()**
- Calls build_strategy_returns() to get daily returns Series
- Calls QuantStats: `qs.stats.sharpe()`, `qs.stats.sortino()`, `qs.stats.max_drawdown()`,
  `qs.stats.calmar()`, `qs.stats.cagr()`, `qs.stats.win_rate()`
- Important: QuantStats functions do NOT take `prepare_returns` param on all functions
  (Phase 2 lesson). Call `qs.utils.prepare_returns()` once at the top, pass to
  each stat function that accepts it.
- Returns TearsheetSummary with all fields populated

**P10-E1-T4: scripts/run_phase10_tearsheets.py**
- Load labeled MethodDecisionRecords from the phase10_accuracy baseline fixture
- Load OHLCV Parquets for all tickers into ohlcv_map
- Call compute_tearsheet() for each of the 4 methods
- Write `data/tearsheets/{method_name}_summary.json` per method
- Print comparison table:

```
Tear sheet summary (15 tickers, 20 quarter-ends, horizon=60d)
method                  sharpe  sortino  max_dd  calmar  cagr   win_rate
confidence_weighted      X.XX    X.XX   -X.XX%   X.XX   X.X%    XX.X%
majority                 X.XX    X.XX   -X.XX%   X.XX   X.X%    XX.X%
performance_weighted     X.XX    X.XX   -X.XX%   X.XX   X.X%    XX.X%
contrarian_adjusted      X.XX    X.XX   -X.XX%   X.XX   X.X%    XX.X%
```

**P10-E1-T5: Unit tests for tearsheet.py**
- `test_build_strategy_returns_buy_always`: all Buy signals → returns match market
- `test_build_strategy_returns_sell_always`: all Sell signals → returns are -1 × market
- `test_build_strategy_returns_hold_always`: all Hold → zero returns series
- `test_build_strategy_returns_gap_filled`: gap days between windows are 0.0
- `test_compute_tearsheet_known_inputs`: seeded returns → known Sharpe value (use
  same seed as Phase 2 risk metrics tests for consistency)
- `test_tearsheet_summary_json_roundtrip`: TearsheetSummary serializes and deserializes

---

## Epic P10-E2: Ticker Universe Expansion (3 → 15 Tickers)

**Scope:** Download OHLCV data for 12 new tickers, extend Makefile targets,
and re-run the bootstrap with the full 15-ticker universe. The expanded bootstrap
produces 300 labeled records (15 × 20), giving the performance_weighted method
enough data to differentiate agent accuracy.

### Tickets

**P10-E2-T1: scripts/acquire_phase10_data.py**
- Download 12 new tickers: MSFT, NVDA, GOOGL, BAC, GS, CVX, JNJ, UNH, AMZN, WMT,
  CAT, NEE
- Date range: 2016-01-01 to 2023-06-30 (matches Phase 1 range)
- Schema: same HiFi Parquet format with metadata (same as `acquire_phase1_data.py`)
- Idempotent: skip tickers whose Parquet files already exist and have the correct
  bar count (tolerance: ±5 bars for yfinance weekend/holiday rounding)
- Progress output: one line per ticker with bar count
- Error handling: skip a ticker on yfinance failure, log warning, continue
- Usage: `uv run python scripts/acquire_phase10_data.py [--data-dir DIR]`

**P10-E2-T2: scripts/run_phase10_bootstrap.py**
- 15-ticker bootstrap over 2018-Q1 through 2022-Q4 (20 quarter-ends × 15 tickers =
  300 agent-decision records per agent type; 4 agent types = 1200 records total
  before any labeling)
- Uses same RSI/Sharpe proxy rules as Phase 9 (DJ-041) — same labeling logic,
  just more tickers
- `--reset` flag (default True): writes a fresh `agent_performance_history.json`
  replacing any existing Phase 9 bootstrap
- `--extend` flag: appends to existing history (for incremental use later)
- Forward returns labeled at both 60-day and 20-day horizons (DJ-052)
- Prints summary: n_labeled, n_unlabeled, weights per agent type

**P10-E2-T3: Parquet validation for new tickers**
- Unit test in `tests/unit/test_phase10_data.py`: for each new ticker file,
  assert bar_count >= 1800, date range starts <= 2016-06-01, date range ends >= 2023-06-01
- Marked with `@pytest.mark.skipif` when `data/market/{ticker}_*.parquet` is absent
  (same pattern as phase8/phase9 skip conditions)

**P10-E2-T4: Makefile + check_env.py updates**
- New Makefile target: `acquire-data-phase10` — runs `acquire_phase10_data.py`
- New Makefile target: `bootstrap` — runs `run_phase10_bootstrap.py --reset`
  (replaces any prior bootstrap target)
- Extend `scripts/check_env.py` checks:
  - `phase10-data`: verifies all 12 new ticker Parquets exist
  - `phase10-bootstrap`: verifies agent_performance_history.json has >= 1000 labeled
    records
  - `phase10-fixture`: verifies tests/fixtures/baseline/phase10_accuracy.json exists

---

## Epic P10-E3: Stub LLM Removal — Structural Holistic Tests

**Scope:** Rewrite `tests/holistic/test_phase8_agent_population.py` and
`tests/holistic/test_phase9_collective_engine.py` to remove all monkeypatched LLMs.
Both files currently use `_stub_llm()` + `monkeypatch.setattr(module, "make_llm", ...)`
to inject canned responses. After this epic, holistic tests exercise only deterministic
code paths (aggregation, schema, serialization).

**Design rule (DJ-047):** A holistic test that requires LLM output is not a holistic
test — it is a behavioral test that belongs in `make baseline-*`. The word "holistic"
refers to depth of pipeline exercised, not to completeness of agent behavior.

### Tickets

**P10-E3-T1: Rewrite test_phase9_collective_engine.py**

Replace `run_ensemble(... patched_llms ...)` calls with direct invocations of the
aggregation layer using seeded AgentSignal objects.

Retained tests (rewritten without stubs):
- `test_method_comparison_has_four_keys`: call `run_all_methods(signals, ...)` with
  hardcoded signals → assert method_comparison has 4 canonical keys
- `test_cw_method_equals_ensemble_decision`: directly compare confidence_weighted_vote()
  output to run_all_methods()["confidence_weighted"]
- `test_contrarian_adjusted_discount_formula`: call `contrarian_adjusted_vote(signals,
  contrarian=ContrarianAnalysis(confidence=0.65, ...))` → assert discount ≈ 0.675
- `test_contrarian_review_not_flagged_at_065`: assert review_flagged=False for
  confidence=0.65 (below theta=0.70)
- `test_all_method_decisions_are_valid_options`: all 4 methods return Buy/Hold/Sell
- `test_ensemble_output_json_roundtrip`: construct EnsembleOutput directly → serialize
  and deserialize → field equality
- `test_backward_compat_no_contrarian`: run_all_methods with no ContrarianAnalysis →
  contrarian_adjusted discount=1.0, review_flagged=False
- `test_performance_weighted_uniform_fallback`: run_all_methods with empty weights dict
  → performance_weighted identical to confidence_weighted

Deleted tests (behavioral; require LLM; belong in baseline-*):
- `test_signals_non_empty_and_all_valid` (depends on actual LLM agent calls)
- `test_aggregation_method_is_confidence_weighted` (trivially structural but depends
  on run_ensemble)
- `test_sentiment_fail_open_still_produces_method_comparison` (LLM behavioral)

Tests migrated from deleted test_phase9_collective_engine.py behaviors to baseline:
- Note in `scripts/run_phase9_baseline.py` docstring: "behavioral assertions previously
  in test_phase9_collective_engine.py are validated here after live LLM run"

**P10-E3-T2: Rewrite test_phase8_agent_population.py**

Retained tests (rewritten without stubs):
- `test_ensemble_output_schema_backward_compat`: construct EnsembleOutput with only
  fundamental + technical fields → risk/macro/sentiment/contrarian are None; all
  Phase 9 fields (signals, aggregation_method, method_comparison) have correct defaults
- `test_ensemble_output_phase8_fields_optional`: model construction with None for all
  Phase 8 optional fields validates without error
- `test_ensemble_output_json_roundtrip_all_fields`: construct EnsembleOutput with all
  fields populated via test builders → JSON round-trip is lossless
- `test_contrarian_analysis_schema`: ContrarianAnalysis with required fields serializes
  correctly (no LLM needed)

Deleted tests (behavioral):
- All tests that call `run_ensemble()` with `patched_llms` fixture
- Tests asserting specific field values generated by stubbed LLM responses

**P10-E3-T3: Fixtures and helper builders**

Both rewritten holistic tests share a set of signal/analysis builders:

```python
# tests/conftest.py additions (or tests/builders.py new file)

def make_agent_signals(decisions: list[tuple[str, str, float]]) -> list[AgentSignal]:
    """
    Build a list of AgentSignals from (agent_type, decision, confidence) tuples.
    Uses fixed analysis_date="2023-03-31" and ticker="TEST".
    """

def make_full_ensemble_output(
    ticker: str = "AAPL",
    analysis_date: str = "2023-03-31",
    decisions: list[tuple[str, str, float]] | None = None,
) -> EnsembleOutput:
    """
    Construct a structurally valid EnsembleOutput without calling any LLM.
    Uses make_agent_signals() + run_all_methods() for method_comparison.
    All Phase 8 optional analysis fields are populated with minimal valid stubs.
    """
```

These builders go into `tests/conftest.py` if they are few, or a new
`tests/builders.py` module if they exceed 5 functions.

---

## Epic P10-E4: Weight Calibration and Method Divergence Analysis

**Scope:** After the expanded bootstrap (P10-E2) and the 2023-03-31 labeling (P10-E0),
compute calibrated weights from real labeled data, compare them to the bootstrap
heuristic weights, and measure method divergence rates. Produce a calibration report
stored in `data/calibration_report.json`.

### Tickets

**P10-E4-T1: scripts/run_phase10_calibration.py**
- Load `data/agent_performance_history.json` (post-P10-E2 bootstrap + P10-E0 real labels)
- Separate records by source: bootstrap heuristic (from run_phase10_bootstrap.py) vs
  real (from run_phase10_labeling.py). Source is inferred from analysis_date:
  bootstrap = 2018-Q1 through 2022-Q4; real = 2023-Q1 records
- Compute weights from bootstrap-only records, real-only records, and combined
- Print weight comparison table:

```
Agent accuracy weights
agent_type      bootstrap_only  real_only  combined
fundamental        0.XXX          0.XXX      0.XXX
technical          0.XXX          0.XXX      0.XXX
risk               0.XXX          0.XXX      0.XXX
macro              0.XXX          0.XXX      0.XXX
```

- Compute method divergence rate across bootstrap records:
  "How often does majority disagree with confidence_weighted?"
  — requires re-running run_all_methods() on bootstrap signals; stored in bootstrap
  records as a post-hoc computation (or in a separate divergence log)

**P10-E4-T2: compute_divergence_rates() in labeler.py**
- Takes two lists of MethodDecisionRecords (for two methods)
- Returns fraction of (ticker, date) pairs where collective_decision differs
- Called for all 6 method pairs: (CW vs MV), (CW vs PW), (CW vs CA),
  (MV vs PW), (MV vs CA), (PW vs CA)

**P10-E4-T3: Unit tests for calibration logic**
- `test_compute_divergence_rates_identical`: same records → 0.0 divergence
- `test_compute_divergence_rates_opposite`: fully opposed records → 1.0 divergence
- `test_weight_shift_from_labeling`: bootstrap weights differ from real-label weights
  when accuracy differs (seeded synthetic data)

**P10-E4-T4: data/calibration_report.json schema**

```python
class CalibrationReport(BaseModel):
    bootstrap_weights: dict[str, float]
    real_label_weights: dict[str, float]
    combined_weights: dict[str, float]
    divergence_rates: dict[str, float]  # "cw_vs_mv", "cw_vs_pw", etc.
    n_bootstrap_labeled: int
    n_real_labeled: int
    generated_at: str
```

Stored in `data/calibration_report.json`. Script writes this after calibration run.
Unit tested for JSON round-trip.

---

## Epic P10-E5: Baseline Measurement + Bitacora

**Scope:** Generate the Phase 10 baseline fixture, update the Makefile closed-loop
validation, and write the scientific bitacora.

### Tickets

**P10-E5-T1: tests/fixtures/baseline/phase10_accuracy.json**
- Format: `MethodAccuracyReport` JSON (list of MethodDecisionRecords + accuracy_by_method)
- Generated by `scripts/run_phase10_baseline.py`
- Content: labeled records for 2023-03-31 (3 tickers, 4 methods, 2 horizons)
- This fixture does NOT require LM Studio — it only reads the phase9_collective.json
  fixture and the Parquet files
- `make baseline-phase10`: runs the labeling script + validates the fixture schema +
  runs `tests/unit/test_phase10_baseline.py`

**P10-E5-T2: tests/unit/test_phase10_baseline.py**
- Fixture validation test (same pattern as test_phase9_baseline.py)
- `@pytest.mark.skipif(not fixture_exists, reason="...")`
- Asserts: n_labeled > 0, all 4 methods in accuracy_by_method, accuracy values in [0,1]

**P10-E5-T3: scripts/run_phase10_baseline.py**
- Runs the labeling script for 2023-03-31 (does not require LM Studio)
- Saves `tests/fixtures/baseline/phase10_accuracy.json`
- Also runs the tear sheet computation and saves `data/tearsheets/` JSON files
- Prints the accuracy table and tear sheet comparison table
- Appends accuracy results to `data/performance_history.csv` (a new CSV tracking
  per-run results across baseline executions, analogous to the Phase 5 pattern)

**P10-E5-T4: doc/bitacora/PHASE_10_EVALUATION.md**
- Standard structure: objective, architecture decisions (DJ-044 through DJ-052),
  baseline results (accuracy tables, tear sheet tables, weight calibration table),
  implementation surprises, open questions for Phase 11
- Fill in the TBD rows from the Phase 9 bitacora method comparison table

---

## Execution Order and Parallelism

```
Wave 1 (parallel — no dependencies):
  P10-E3  Stub LLM removal
  P10-E2  Ticker universe expansion + data acquisition
  P10-E0  Accuracy labeling (2023-03-31, 3 tickers — uses existing Parquets)

Wave 2 (after Wave 1):
  P10-E1  QuantStats tear sheets (depends on E0 for basic run; extended with E2 data)
  P10-E4  Weight calibration (depends on E0 for labels + E2 for expanded weights)

Wave 3 (after Wave 2):
  P10-E5  Baseline measurement + bitacora
```

P10-E2 data acquisition (T1: acquire_phase10_data.py) must complete before P10-E2-T2
(bootstrap script), which must complete before P10-E4 can use the expanded weights.
Within P10-E1, the first tear sheet can be produced from 3 tickers (P10-E0 output)
before P10-E2 completes; the final tear sheet uses 15 tickers.

---

## Dependency Graph

```
                 P10-E3 (independent)
                    |
                    v
P10-E2 ──────────> Makefile + check_env
    |
    v
P10-E0 ──────────> P10-E1 (QuantStats)
    |                   |
    v                   v
P10-E4 (calibration) ──> P10-E5 (baseline + bitacora)
```

---

## Verification Criteria

Phase 10 is complete when all of the following hold:

1. **Test count:** `pytest -q --tb=no` passes with >= 950 tests (from 857), 0 skipped
   active tests, 0 lint errors (ruff --output-format=concise)

2. **Zero LLM stubs:** `grep -r "monkeypatch.setattr.*make_llm" tests/holistic/`
   returns no matches

3. **Accuracy fixture:** `tests/fixtures/baseline/phase10_accuracy.json` exists and
   validates as MethodAccuracyReport; n_labeled >= 12 (4 methods × 3 tickers, 60d)

4. **Tear sheets:** `data/tearsheets/confidence_weighted_summary.json`,
   `majority_summary.json`, `performance_weighted_summary.json`,
   `contrarian_adjusted_summary.json` all exist and validate as TearsheetSummary

5. **Expanded bootstrap:** `data/agent_performance_history.json` has
   `n_labeled >= 1000` (15 tickers × 20 quarter-ends × 4 agent types × ≥ fraction
   labeled at 60d)

6. **QuantStats on the record:** Sharpe and max_drawdown are present and finite for
   all four methods in the tear sheet summaries

7. **Calibration report:** `data/calibration_report.json` exists; divergence_rates has
   all 6 method-pair keys; bootstrap_weights differ from real_label_weights (or are
   confirmed identical, which is also a valid empirical result)

8. **Makefile:** `make test` passes, `make baseline-phase10` generates all artifacts
   and re-runs validation, `make bootstrap` runs the 15-ticker bootstrap

---

## Scientific Context

The central measurement this phase enables is the **method accuracy ordering**. After
Phase 10, HiFi will have an empirical answer (provisional, small-sample) to:

- Does `confidence_weighted` outperform `majority`? If so, high-confidence agents
  are genuinely adding information, not just amplifying the plurality.
- Does `performance_weighted` converge faster than `confidence_weighted` on more data?
  If so, the bootstrap prior is already being updated by real labels.
- Does `contrarian_adjusted` systematically underperform `confidence_weighted`? If so,
  the Contrarian is generating noise and `alpha` should decrease. If it
  outperforms on cases where `review_flagged=True`, `theta` should increase.

None of these results will be statistically significant at 15 tickers × 1 analysis
date. The point of Phase 10 is not to reach a conclusion — it is to create the
measurement infrastructure that will accumulate evidence over the remaining phases.
Every future `make test-live` adds one more analysis date to the history.

The QuantStats tear sheets serve a different purpose: they make the platform legible
to reviewers who expect portfolio-level performance metrics. The capstone reviewer
does not want to see aggregation method JSON — they want to see Sharpe ratios and
drawdown curves. Phase 10 creates the translation layer between the internal
representation (labeled EnsembleOutput) and the external representation (portfolio
analytics).

---

## Open Questions for Phase 11

1. **Structured debate.** David §12.2.4 describes a multi-turn LLM exchange where
   agents challenge each other's assumptions before the final vote. Phase 11 will
   prototype this with two agents (Fundamental vs Contrarian). Phase 10 should
   capture divergence patterns that suggest when structured debate adds value
   (high contrarian_confidence + high method divergence).

2. **Fine-tuning data generation.** Phase 11 fine-tunes the Fundamental Agent on
   labeled (EnsembleOutput, outcome_correct) pairs. Phase 10 creates the first real
   labeled pairs. The quantity will be small (3–15 tickers, 1–2 dates), but the
   format and schema must be established now. A `scripts/export_finetune_pairs.py`
   script is a candidate for Phase 11 pre-phase.

3. **Weight update frequency.** Currently weights update only when the bootstrap
   is re-run. Phase 11 should add an incremental update path: each new
   `make test-live` call labels the analysis date 60 days later and updates weights
   automatically. The `update_and_save()` function already supports this; what is
   missing is the trigger (a post-60-day labeling hook in the Makefile).

4. **Herding coefficient interpretation.** Phase 9 implemented κ but did not have
   enough data to interpret it. With 20 quarter-ends of bootstrap data across 15
   tickers, Phase 10 can compute κ per ticker and correlate with VIX levels. If κ
   increases during high-VIX periods (COVID 2020, rate hikes 2022), the metric is
   an independent macro-stress indicator.
