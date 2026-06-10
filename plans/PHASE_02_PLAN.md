# Phase 2: Deterministic Financial Engine -- Epic Plan

**Status:** COMPLETE — All epics delivered; 309 tests passing; 0 lint errors

| Epic | Title | Status |
|---|---|---|
| P2-E1 | Engine interfaces and result types | DONE |
| P2-E2 | Fundamental and valuation engine | DONE |
| P2-E3 | Technical indicator engine (numpy/pandas) | DONE |
| P2-E4 | Risk metrics engine (QuantStats) | DONE |
| P2-E5 | MCP server -- hifi-financial-calculator | DONE |
| P2-E6 | Holistic pipeline test | DONE |
**David Sections:** 4.1 (Deterministic-First), 6.2 (MCP as the Nervous System), 7.2 (Data Engineering), 8.3 (Feature Datasets -- partial)
**Learning Guide Topics:** 9.1 (Model Context Protocol), 9.2 (Tool Design Principles), 7.1 (Quantitative Analysis)
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 2

---

## Governing Philosophy for This Phase

This phase implements the deterministic-first principle (David §4.1) in its most
concrete form: every financial quantity an agent will ever cite is computed here,
deterministically, as a pure function. The MCP server is the interface. Agents are
consumers. They do not compute; they request and interpret.

- **Interface-first:** Engine result types are defined and tested before any
  computation code exists. A Phase 3 agent will depend on these types, not on
  engine internals.
- **Pure functions:** Every engine function is a pure function. Same inputs always
  produce the same outputs. No database access, no network calls, no side effects.
  This is what makes the computation auditable.
- **Minimal custom numpy for Phase 2 indicators (DJ-010):** Six specific indicators
  (SMA, EMA, RSI, MACD, Bollinger, ATR) implemented directly in numpy/pandas (~40
  lines). No TA library is added in Phase 2: all evaluated libraries have blockers
  against pandas 3.x. The chosen long-term architecture for Phase 8+ is not "pick a
  library and import it" but rather "deploy a dedicated MCP server in an isolated venv
  that carries its own compatible stack." The MCP boundary is already a process
  boundary; giving each computation server its own virtual environment is a zero-cost
  extension of the existing pattern. QuantStats for risk metrics (DJ-011) is justified
  by scale and Phase 10 reuse.
- **Data loaded, not fetched:** Engines load data from Parquet files written by Phase 1.
  They never call yfinance or FRED. The data layer (Phase 1) and the computation layer
  (Phase 2) are strictly separated.
- **Typed results over dicts:** Every engine returns a typed result object, not a raw
  dictionary. This makes it impossible to silently drop a field or misname a key.
- **None over fabrication:** If a required field is missing (e.g., eps is None from
  the fundamentals snapshot), the ratio is None. The engine never substitutes a default
  value or guesses. Downstream consumers must handle None explicitly.

---

## Epic Dependency Graph

```
P2-E1 (Engine Interfaces and Result Types)
    |
    +------------------+------------------+
    |                  |                  |
P2-E2 (Fundamental    P2-E3 (Technical   P2-E4 (Risk
      and Valuation         Indicator          Metrics
      Engine)               Engine)            Engine)
    |                  |                  |
    +------------------+------------------+
                       |
                 P2-E5 (MCP Server)
                       |
              Holistic Test (full pipeline)
```

E2, E3, and E4 are fully independent of each other: they share only the result
types from E1. They can be developed and tested in parallel.
E5 requires all three engines to be complete before the server can be assembled.
The holistic test requires E5.

---

## Epic P2-E1: Engine Interfaces and Result Types

**Objective:** Define the typed result objects that all engines produce and that all
downstream consumers (MCP server, agents, evaluation) depend on. These are the output
contracts, not the implementations.

**David rationale:** Section 4.6 (Modularity) requires every component to be
replaceable without changing other components. Result types are the interface layer
that makes this possible. An improved RSI implementation in Phase 8 replaces the
internal calculation but preserves the TechnicalIndicatorsResult contract.

**Why Pydantic v2 for result types:** The same validation model used for data schemas
(Phase 1) is used for engine results. Consistency means agents use one import for all
structured data -- no mixing of Pydantic models and dataclasses.

| Ticket | Description | Status |
|---|---|---|
| P2-E1-T1 | Define FinancialRatioResult: pe, pb, ps, ev_ebitda, roe, roa, debt_equity, current_ratio -- all Optional[float] | DONE |
| P2-E1-T2 | Define GrowthMetricsResult: revenue_growth_yoy, earnings_growth_yoy, gross_margin, operating_margin, net_margin -- all Optional[float] | DONE |
| P2-E1-T3 | Define TechnicalIndicatorsResult: sma, ema, rsi, macd, macd_signal, macd_hist, bb_upper, bb_mid, bb_lower, atr -- all Optional[float] | DONE |
| P2-E1-T4 | Define RiskMetricsResult: hist_vol_20d, hist_vol_60d, hist_vol_252d, beta, max_drawdown_252d, sharpe_252d, var_95_20d -- all Optional[float] | DONE |
| P2-E1-T5 | Define ValuationResult: current_pe, pe_1y_min, pe_1y_max, pe_1y_percentile, price_to_52w_high, price_to_52w_low -- all Optional[float] | DONE |
| P2-E1-T6 | Define MacroSnapshotResult: fed_funds_rate, cpi_yoy, unemployment_rate, yield_10y, yield_2y, yield_curve_slope, vix, gdp_growth -- all Optional[float] | DONE |
| P2-E1-T7 | Unit tests: each result type accepts all-None construction without error | DONE |
| P2-E1-T8 | Unit tests: each result type serialises to JSON-safe dict (no float('nan'), no None-to-null issues) | DONE |

**Files to create:**
- `src/hifi/engines/__init__.py`
- `src/hifi/engines/types.py` -- all result types

**Acceptance test:** Every result type can be constructed with all-None fields.
Every result type serialises to a dict that is JSON-serialisable (json.dumps succeeds).

---

## Epic P2-E2: Fundamental and Valuation Engine

**Objective:** Pure functions that take a FundamentalsSnapshot (from Phase 1) and
an OHLCVDataset (for current price context) and return typed financial ratio and
valuation results.

**Scoping note on Phase 1 data depth:** FundamentalsSnapshot contains only one
period's data per ticker (the most recently available annual snapshot from yfinance).
This limits what we can compute:

- **Available now (Phase 2):** Ratios that require only one snapshot (P/E, ROE, ROA,
  debt/equity, margins) and valuation context using price history vs. a fixed eps.
- **Deferred to Phase 7+:** YoY growth rates (require two periods), detailed quarterly
  analysis, SEC EDGAR data. Growth metric fields (revenue_growth_yoy, earnings_growth_yoy)
  will return None in Phase 2 with a documented explanation.

This limitation is documented in the engine's module docstring and in the MCP tool
schema descriptions. An agent that receives None for growth metrics must interpret this
as "insufficient data" and proceed accordingly, not as "zero growth."

**Formulas used:**

| Ratio | Formula | None condition |
|---|---|---|
| P/E | close_price / eps | eps is None or eps <= 0 |
| P/B | market_cap / total_equity | either field None or equity <= 0 |
| P/S | market_cap / revenue | either field None or revenue <= 0 |
| ROE | net_income / total_equity | either field None or equity == 0 |
| ROA | net_income / total_assets | either field None or assets == 0 |
| Debt/Equity | total_liabilities / total_equity | either None or equity == 0 |
| Gross Margin | (revenue - cogs) / revenue | Phase 1: cogs not in snapshot; returns None |
| Operating Margin | from pe_ratio proxy | Phase 1: limited; returns None if unavailable |
| Net Margin | net_income / revenue | either field None or revenue <= 0 |

**Valuation context:** Computes a trailing P/E series using the snapshot eps against
each daily close in the OHLCVDataset. Reports where the current P/E sits within the
trailing 252-day (or available) range. Does NOT compare to sector peers (requires
multi-ticker data; deferred to Phase 8+).

| Ticket | Description | Status |
|---|---|---|
| P2-E2-T1 | Implement compute_financial_ratios(snapshot, current_price) -> FinancialRatioResult | DONE |
| P2-E2-T2 | Implement compute_growth_metrics(snapshot) -> GrowthMetricsResult (partial; growth fields return None) | DONE |
| P2-E2-T3 | Implement compute_valuation_context(snapshot, dataset, as_of_date) -> ValuationResult | DONE |
| P2-E2-T4 | Unit test: P/E = 25.0 when close=250.0 and eps=10.0 | DONE |
| P2-E2-T5 | Unit test: P/E is None when eps is None, eps == 0, or eps < 0 | DONE |
| P2-E2-T6 | Unit test: ROE = 0.15 when net_income=150, total_equity=1000 | DONE |
| P2-E2-T7 | Unit test: all ratios are None when snapshot has all-None financial fields | DONE |
| P2-E2-T8 | Unit test: net_margin computed correctly from net_income and revenue | DONE |
| P2-E2-T9 | Unit test: valuation_context pe_1y_percentile == 0.5 when current P/E is the median of the trailing range | DONE |
| P2-E2-T10 | Unit test: valuation_context returns None fields gracefully when eps is None | DONE |

**Files to create:**
- `src/hifi/engines/fundamental.py`
- `tests/unit/test_fundamental_engine.py`

**Acceptance test:** compute_financial_ratios() on a snapshot with known values
(eps=10, market_cap=1000, total_equity=500, revenue=400, net_income=50) produces:
P/E = close/10, ROE = 50/500 = 0.10, net_margin = 50/400 = 0.125. All confirmed by
manual computation.

---

## Epic P2-E3: Technical Indicator Engine

**Objective:** Six technical indicators implemented directly in numpy/pandas, exposing
a single `compute_technical_indicators()` function that returns TechnicalIndicatorsResult.

**Library decision (DJ-010): Custom numpy for Phase 2; TA library deferred to Phase 8.**

All evaluated libraries have blockers at this stack level:
- **pandas-ta:** Incompatible with `pandas>=2.0` — calls `DataFrame.append()` which
  was removed in pandas 2.0. Project uses pandas 3.0.3. Fails at runtime.
- **ta (bukosabino):** Works with pandas 3.x but unmaintained since 2022.
- **TA-Lib (C extension):** Native build required; fragile on arm64/macOS.

Custom numpy is the correct choice for Phase 2 because: (1) only 6 indicators are
needed — the implementation is ~40 lines, not a "TA library from scratch"; (2) every
formula cites its primary source in the docstring, satisfying §4.3 verifiability;
(3) zero new dependencies.

**Phase 8+ long-term architecture: MCP server in a dedicated virtual environment.**

The compatibility problem (pandas-ta requires pandas<2.0; project uses pandas 3.x)
reveals a general solution that is architecturally superior to "wait for a library to
update." MCP servers already run as subprocesses communicating via stdio JSON-RPC.
The main process never imports the child's dependencies. Giving the child process its
own virtual environment is a zero-cost extension of the existing MCP pattern:

```
Main HiFi process (pandas 3.x, .venv)
      |  stdio / MCP JSON-RPC
      v
venvs/ta/bin/python -m hifi.mcp.indicators_server
      (pandas 1.x pinned, pandas-ta, TA-Lib, or any future library)
```

This pattern:
- Eliminates all cross-dependency conflicts permanently
- Maps directly to Phase 15 (containerization): each venv becomes a Docker service
- Preserves reproducibility: each venv has its own pinned lockfile
- Allows multiple TA libraries to coexist for comparison (scientific value)
- Requires no changes to the main codebase or any existing MCP server

**Phase 8 trigger:** When more than 6 indicators are required or the universe exceeds
~50 stocks. At that point, create `venvs/ta/` with pinned compatible dependencies and
implement `src/hifi/mcp/indicators_server.py` consuming that environment. The main
agent layer sees only an additional MCP tool — it has no knowledge of the isolation.

**Indicators and formulas (primary source cited in docstring):**

| Indicator | Formula | Source |
|---|---|---|
| SMA(n) | `series.rolling(n).mean()` | Standard |
| EMA(n) | `series.ewm(span=n, adjust=False).mean()` | Standard |
| RSI(n) | Wilder smoothing: `ewm(alpha=1/n, adjust=False)` on gains/losses | Wilder (1978) |
| MACD | EMA(12) - EMA(26); signal = EMA(9) of MACD; hist = MACD - signal | Appel (1979) |
| Bollinger Bands | `rolling(20).mean()` ± 2 × `rolling(20).std()` | Bollinger (1992) |
| ATR | `mean(max(H-L, |H-C_prev|, |L-C_prev|))` over 14 bars | Wilder (1978) |

When a series is shorter than the required window, the function returns None.
None is propagated to the MCP response as JSON null.

| Ticket | Description | Status |
|---|---|---|
| P2-E3-T1 | Implement _to_dataframe(bars, as_of_date) -> pd.DataFrame: OHLCVBar list to OHLCV DataFrame, filtered to as_of_date | DONE |
| P2-E3-T2 | Implement _last(series) -> float | None: extract last non-NaN scalar from a pandas Series | DONE |
| P2-E3-T3 | Implement compute_technical_indicators(bars, as_of_date, window) -> TechnicalIndicatorsResult using pandas-ta | DONE |
| P2-E3-T4 | Unit test: SMA(window=3) on a 5-bar series with known closes equals the manually computed mean | DONE |
| P2-E3-T5 | Unit test: EMA of a constant-price series equals the constant | DONE |
| P2-E3-T6 | Unit test: RSI is in [0.0, 100.0] for any valid price series with >= 15 bars | DONE |
| P2-E3-T7 | Unit test: RSI is near 100 for a monotonically rising series, near 0 for falling | DONE |
| P2-E3-T8 | Unit test: Bollinger upper >= mid >= lower for all valid inputs | DONE |
| P2-E3-T9 | Unit test: ATR is strictly positive for any valid OHLCV series | DONE |
| P2-E3-T10 | Unit test: all indicators return None when bars count < required window | DONE |
| P2-E3-T11 | Unit test: adjusted_close is used as price column when present; close used as fallback | DONE |

**Files to create:**
- `src/hifi/engines/technical.py`
- `tests/unit/test_technical_engine.py`

**Acceptance test:** SMA and RSI verified against manually computed values on a
20-bar synthetic series. RSI verified to be in [0, 100] across 1000 iterations of
GBM price series (using the conftest synthetic generator). None returned correctly
when series length < window.

---

## Epic P2-E4: Risk Metrics Engine

**Objective:** An adapter layer over QuantStats that computes portfolio and risk
metrics from OHLCVBars and returns RiskMetricsResult.

**Library decision (DJ-011): QuantStats for risk metrics; Pyfolio explicitly rejected.**

Three options were evaluated:

- **Pyfolio:** Rejected. Pyfolio was maintained by Quantopian, which shut down in 2020.
  The library received no meaningful updates after 2021, has known incompatibilities
  with modern pandas versions, and is no longer maintained. Using a deprecated library
  in an open-source research project and capstone submission is indefensible. Pyfolio
  is excluded from all phases of HiFi.

- **Custom numpy implementation:** Rejected for the same reasons as in P2-E3: rebuilding
  standard formulas produces no scientific value and introduces subtle divergences from
  community norms (annualization conventions, Sharpe denominator choices, drawdown
  sign conventions) that are hard to detect and complicate comparisons with published
  research.

- **QuantStats:** Accepted. `quantstats.stats` provides Sharpe, Sortino, max drawdown,
  VaR, annualized volatility, and many more metrics through a clean pandas-based API.
  Beyond Phase 2, QuantStats generates standardized performance tear sheets that are
  directly useful in Phase 10 (Evaluation & Backtesting) — adopting it here gives
  Phase 10 professional-grade analytics at no additional integration cost.

**What the adapter does:**

QuantStats operates on a `pd.Series` of prices or returns. The adapter:
1. Converts `list[OHLCVBar]` → `pd.Series` (adjusted_close or close, indexed by date)
2. Slices to the trailing `window` bars ending at `as_of_date`
3. Calls the appropriate `quantstats.stats` function
4. Maps results to RiskMetricsResult fields; converts NaN to None

**Note on benchmark for beta:** Requires a second price Series (benchmark, typically
SPY). The adapter accepts optional `benchmark_bars`. If None, beta returns None.
The MCP tool loads the benchmark from the registry if a "SPY_yfinance" entry exists.

**Note on risk-free rate for Sharpe:** Passed as `rf` to `qs.stats.sharpe()`.
Taken from FEDFUNDS macro snapshot if available; otherwise 0.0 with a WARNING log.

**Metrics and QuantStats calls:**

| Metric | QuantStats call | Notes |
|---|---|---|
| Historical vol (20d, 60d, 252d) | qs.stats.volatility(prices, periods=252) on sliced windows | Annualized |
| Beta | qs.stats.greeks(prices, benchmark).beta | Requires benchmark Series |
| Max drawdown | qs.stats.max_drawdown(prices) | Stored as positive float (abs value) |
| Sharpe ratio | qs.stats.sharpe(prices, rf=risk_free_rate) | Annualized |
| VaR 95% (historical) | qs.stats.value_at_risk(prices, confidence=0.95) | Daily figure |

| Ticket | Description | Status |
|---|---|---|
| P2-E4-T1 | Implement _to_price_series(bars, as_of_date, window) -> pd.Series: OHLCVBars to dated price Series, sliced to window | DONE |
| P2-E4-T2 | Implement compute_hist_vol(bars, as_of_date, windows=[20, 60, 252]) -> dict[int, float | None] via QuantStats | DONE |
| P2-E4-T3 | Implement compute_beta(stock_bars, benchmark_bars, as_of_date, window=252) -> float | None via QuantStats | DONE |
| P2-E4-T4 | Implement compute_max_drawdown(bars, as_of_date, window=252) -> float via QuantStats | DONE |
| P2-E4-T5 | Implement compute_sharpe(bars, as_of_date, risk_free_rate=0.0, window=252) -> float | None via QuantStats | DONE |
| P2-E4-T6 | Implement compute_var(bars, as_of_date, confidence=0.95, window=20) -> float | None via QuantStats | DONE |
| P2-E4-T7 | Implement compute_risk_metrics(bars, as_of_date, window, benchmark_bars=None, risk_free=0.0) -> RiskMetricsResult | DONE |
| P2-E4-T8 | Unit test: annualized vol of a flat-price series is 0.0 | DONE |
| P2-E4-T9 | Unit test: annualized vol of GBM series with sigma=0.20 is within [0.17, 0.23] over 252 bars | DONE |
| P2-E4-T10 | Unit test: max drawdown on a monotonically declining series equals (peak - final) / peak | DONE |
| P2-E4-T11 | Unit test: max drawdown of a monotonically rising series equals 0.0 | DONE |
| P2-E4-T12 | Unit test: Sharpe of a zero-variance series returns None, not ZeroDivisionError | DONE |
| P2-E4-T13 | Unit test: beta of a series identical to benchmark equals 1.0 | DONE |
| P2-E4-T14 | Unit test: all metrics return None gracefully when window exceeds available bars | DONE |

**Files to create:**
- `src/hifi/engines/risk.py`
- `tests/unit/test_risk_engine.py`

**Acceptance test:** Annualized volatility of a GBM series with known sigma=0.20 falls
within [0.17, 0.23] over a 252-bar window. Max drawdown of a series that falls 30%
from peak returns 0.30. QuantStats tear sheet generation previewed in the holistic
test as a forward demonstration for Phase 10.

---

## Epic P2-E5: MCP Server -- hifi-financial-calculator

**Objective:** Wrap all engine functions in an MCP server that exposes 6 tools via
stdio transport. This is the interface that all agents in Phase 3+ will call.

**Transport decision (DJ-009): stdio over SSE/HTTP.**

Rationale: HiFi is a single-machine local system in Phases 2-14. stdio transport
is the simplest possible MCP transport: the server reads JSON-RPC messages from stdin
and writes to stdout. No sockets, no authentication, no service discovery. A Phase 3
agent starts the server as a subprocess and communicates through pipes.

The performance overhead of stdio JSON-RPC is negligible for our use case:
each tool call takes O(disk I/O + computation) time, not O(network) time. The
bottleneck is disk I/O for loading Parquet files, not the transport protocol.

SSE or HTTP transport is appropriate for multi-machine deployments (Phase 15+).
This decision is documented here and will be revisited at Phase 15.

**Data access pattern:** Each tool handler follows the same pattern:
1. Validate input parameters (ticker format, date format, window range)
2. Look up the relevant dataset in the DatasetRegistry
3. Load the dataset from Parquet using read_ohlcv() or read_macro()
4. Call the appropriate engine function
5. Return the result serialised to a JSON-safe dict (None fields included; NaN is
   converted to null, not left as float('nan') which is not valid JSON)

**Error handling contract:** MCP tools return structured errors, not exceptions.
A tool call for an unknown ticker returns `{"error": "TICKER_NOT_FOUND", "detail": "..."}`.
A date out of range returns `{"error": "DATE_OUT_OF_RANGE", "detail": "..."}`.
This contract is documented in the tool schema descriptions.

| Tool | Parameters | Data Required |
|---|---|---|
| get_financial_ratios | ticker: str, date: str (ISO 8601) | OHLCVDataset (for price on date), FundamentalsSnapshot |
| get_growth_metrics | ticker: str, date: str | FundamentalsSnapshot |
| get_technical_indicators | ticker: str, date: str, window: int (default 20) | OHLCVDataset (bars up to date) |
| get_risk_metrics | ticker: str, date: str, window: int (default 252) | OHLCVDataset (bars up to date), optional benchmark dataset |
| get_valuation_context | ticker: str, date: str | OHLCVDataset, FundamentalsSnapshot |
| get_macro_snapshot | date: str | All MacroDatasets in registry |

| Ticket | Description | Status |
|---|---|---|
| P2-E5-T1 | Set up src/hifi/mcp/financial_server.py: mcp.Server with stdio transport | DONE |
| P2-E5-T2 | Implement DataLoader helper: loads OHLCVDataset and FundamentalsSnapshot from registry+Parquet | DONE |
| P2-E5-T3 | Register and implement tool handler: get_financial_ratios | DONE |
| P2-E5-T4 | Register and implement tool handler: get_growth_metrics | DONE |
| P2-E5-T5 | Register and implement tool handler: get_technical_indicators | DONE |
| P2-E5-T6 | Register and implement tool handler: get_risk_metrics | DONE |
| P2-E5-T7 | Register and implement tool handler: get_valuation_context | DONE |
| P2-E5-T8 | Register and implement tool handler: get_macro_snapshot | DONE |
| P2-E5-T9 | Integration test: server starts via subprocess; tools/list response lists 6 tools | DONE |
| P2-E5-T10 | Integration test: get_technical_indicators for AAPL Q1 2023 fixture returns RSI in [0, 100] | DONE |
| P2-E5-T11 | Integration test: get_macro_snapshot for 2022-06-15 returns non-None fed_funds_rate | DONE |
| P2-E5-T12 | Integration test: call for unknown ticker "ZZZZZ" returns error dict, not exception | DONE |
| P2-E5-T13 | Integration test: all 6 tools return JSON-serialisable dicts (json.dumps succeeds) | DONE |

**Files to create:**
- `src/hifi/mcp/__init__.py`
- `src/hifi/mcp/financial_server.py`
- `tests/integration/test_mcp_server.py`

**Acceptance test:** Server starts, tools/list returns exactly 6 tools with valid
parameter schemas. get_macro_snapshot for 2022-06-15 returns the correct FEDFUNDS rate
for that date (forward-filled from the June 2022 FRED reading).

---

## Holistic Test: Phase 2 Financial Engine Pipeline

**File:** `tests/holistic/test_phase2_engine_pipeline.py`

This test runs after all epics are complete. It validates the full Phase 2 pipeline:
load data (from Phase 1 fixtures) → compute metrics (via engines) → serve via MCP → receive result.

**Scenario:** For AAPL Q1 2023 fixture:
1. Load OHLCVDataset and FundamentalsSnapshot via the storage API
2. Compute all six result types via direct engine calls
3. Start the MCP server subprocess
4. Call all 6 MCP tools for AAPL / 2023-03-31 via the MCP client
5. Verify results from MCP match the results from direct engine calls
6. Verify the Phase 1 holistic test still passes (regression guard)

**Assertions:**
- All result types are valid (no validation errors)
- RSI is in [0, 100]
- Bollinger upper >= mid >= lower
- Historical vol is positive
- Max drawdown is in [0, 1]
- MCP responses are JSON-serialisable
- Phase 1 pipeline test still passes (data layer regression guard)

---

## New Dependencies to Add

```toml
# Production
mcp>=1.0            # MCP Python SDK (stdio server and client infrastructure)
quantstats>=0.0.62  # Risk and portfolio metrics; also used for Phase 10 tear sheets
```

No TA indicator library is added to the main environment. The 6 indicators in P2-E3
use numpy and pandas already present from Phase 1. The Phase 8+ TA library will live
in a dedicated isolated virtual environment, not in the main project environment.

**Decision to record (DJ-009): MCP transport**

stdio chosen over SSE and HTTP for Phase 2. Rationale: single-machine deployment,
zero network overhead, zero authentication complexity, subprocess model is natural
for agent frameworks (LangGraph, LlamaIndex all support subprocess MCP servers).
SSE/HTTP will be evaluated for Phase 15 (containerization) where multi-container
deployments may require network transport.

**Decision to record (DJ-010): Technical indicator library and isolation architecture**

Custom numpy for Phase 2; dedicated MCP venv for Phase 8+.

Libraries evaluated and rejected for the main environment:
- pandas-ta: incompatible with pandas>=2.0; project uses pandas 3.0.3; fails at runtime.
- ta (bukosabino): unmaintained since 2022.
- TA-Lib: C build required; fragile on arm64/macOS without Homebrew.

Phase 2 resolution: custom numpy for 6 indicators (~40 lines, primary sources cited).

Phase 8+ architecture (chosen, not deferred): MCP-server-in-dedicated-venv. The MCP
subprocess boundary already isolates the child process from the main environment. A
dedicated virtual environment at `venvs/ta/` carries any TA library at any python/pandas
version without touching the main stack. The main process consumes MCP tool results —
it is agnostic to what runs behind the server. This is not a workaround; it is the
natural extension of the deterministic-first / MCP-backbone architecture to the
dependency management problem. See P2-E3 for the full design note.

**Decision to record (DJ-011): Risk metrics library**

QuantStats chosen over Pyfolio and custom numpy implementation.
- Pyfolio rejected and excluded from all phases: maintained by Quantopian (shutdown
  2020), last meaningful release 2021, known pandas incompatibilities, unmaintained.
  Using a deprecated library in an open-source capstone is academically indefensible.
- Custom implementation rejected: same reasoning as DJ-010 for technical indicators.
- QuantStats chosen: actively maintained, standard metrics (Sharpe, Sortino, drawdown,
  VaR, volatility), pandas-based API, generates professional tear sheets that will
  be reused directly in Phase 10 (Evaluation & Backtesting). Adopting now gives
  Phase 10 analytics infrastructure at no additional integration cost.

**Decision to record (DJ-012): FinanceToolkit deferred to Phase 7+**

FinanceToolkit provides deep fundamental analysis (Piotroski F-Score, Altman Z-Score,
full DuPont decomposition, DCF inputs). Phase 2 only needs a small set of ratios from
one FundamentalsSnapshot; the custom implementation in P2-E2 covers this at low cost.
FinanceToolkit's value grows when multi-period quarterly data is available (Phase 7+,
SEC EDGAR). Integration complexity with our Parquet data model is unresolved. Deferred.

---

## Phase 2 Quality Gates

| Gate | Criterion | Measured By |
|---|---|---|
| All unit tests pass | pytest -m unit, 0 failures | Manual run |
| All integration tests pass | pytest -m integration, 0 failures | Manual run |
| Holistic test passes | pytest tests/holistic/test_phase2_engine_pipeline.py | Manual run |
| Phase 1 regression | pytest tests/holistic/test_phase1_pipeline.py still passes | Manual run |
| Linting clean | ruff check src/ tests/, 0 errors | Manual run |
| No live API calls | grep -r "yfinance.download\|fredapi" tests/ returns nothing | Code review |
| Pure function verification | At least 5 computations verified against manually computed values | Test review |
| MCP tool schemas | All 6 tools have JSON schemas with correct parameter types | Integration test |
| Latency (informal) | Each tool completes in <1s for single ticker on M-series hardware | Manual observation |

---

## Commit Strategy

One commit per epic, in dependency order:

| Commit | Epic | Files |
|---|---|---|
| Phase 2 / E1: Engine interfaces and result types | P2-E1 | src/hifi/engines/__init__.py, engines/types.py, tests/unit/test_engine_types.py |
| Phase 2 / E2: Fundamental and valuation engine | P2-E2 | engines/fundamental.py, tests/unit/test_fundamental_engine.py |
| Phase 2 / E3: Technical indicator engine | P2-E3 | engines/technical.py, tests/unit/test_technical_engine.py |
| Phase 2 / E4: Risk metrics engine | P2-E4 | engines/risk.py, tests/unit/test_risk_engine.py |
| Phase 2 / E5: MCP server | P2-E5 | mcp/__init__.py, mcp/financial_server.py, tests/integration/test_mcp_server.py, tests/holistic/test_phase2_engine_pipeline.py |

E2, E3, and E4 commits can be made in any order since they are independent.

---

## Open Questions This Phase Will Answer

**OQ-P2-01 (DJ-009 resolved):** What MCP transport is right for Phase 2?
This phase uses stdio and measures its practical performance. If a single analysis
request takes >1 second due to Parquet I/O + computation, it will be documented here
as a Phase 3 concern (caching layer).

**OQ-P2-02:** How do engines handle missing fundamentals gracefully?
Phase 1 showed that yfinance fundamentals have gaps (EV/EBITDA, current ratio not
available in `.info`). This phase will document exactly which fields are reliably
available and which return None. That knowledge directly informs agent prompt design
in Phase 3.

**OQ-P2-03:** What is the practical performance of indicator computation on M-series hardware?
The holistic test will measure wall-clock time for a full analysis of one ticker.
This establishes a latency baseline before any caching or optimisation is added.

**OQ-P2-04 (from Phase 1 OQ-P1-01):** Should we add a US trading calendar in Phase 2?
The quality checker's approximate completeness threshold (weekday count, no holidays)
was identified as a limitation in Phase 1. Phase 2 will evaluate pandas_market_calendars
as a lightweight dependency. If it adds <50 lines of integration code and resolves the
false-positive quality failures, it will be added in this phase.

**OQ-P2-05:** What is the serialization overhead of passing OHLCV arrays across the
MCP subprocess boundary at Phase 8+ scale? The venv isolation architecture assumes
JSON serialization of ~252 bars is negligible. This needs to be measured empirically
when the Phase 8 indicators server is first built. If overhead is significant, binary
serialization (msgpack, Arrow IPC) over stdio is the mitigation — no change to the
MCP interface, only to the transport encoding inside the server.

---

## Connections to Earlier and Later Phases

**Depends on Phase 1:**
- All engine inputs are Phase 1 schema types (OHLCVDataset, FundamentalsSnapshot,
  MacroDataset, OHLCVBar)
- Data is loaded via Phase 1 storage API (read_ohlcv, read_macro)
- Test fixtures are Phase 1 recorded API responses (AAPL/JPM/XOM Q1 2023,
  FEDFUNDS/CPIAUCSL 2022)

**Phase 3 depends on this phase:**
- The Fundamental Agent calls get_financial_ratios, get_growth_metrics, get_valuation_context
- The Technical Agent (Phase 4) calls get_technical_indicators
- All agents call get_macro_snapshot for market context
- The MCP server must be running before any agent can produce an analysis

**Phase 5 (Verification) depends on this phase:**
- The verifier checks agent-cited numbers against MCP tool outputs
- The MCP tool call ID (returned in the tool result) is the audit trail
- This phase must return a unique call identifier in every response for Phase 5 to work

**Phase 8 (Full Agent Population) enables the venv isolation architecture:**
- When more than 6 indicators are required, `venvs/ta/` is created alongside `.venv/`
- `src/hifi/mcp/indicators_server.py` is implemented as a new MCP server launched
  from `venvs/ta/bin/python` — invisible to the agent layer, which sees only new tools
- Phase 15 (Containerization) maps this directly: each venv becomes a Docker service
  with its own base image, dependency set, and pinned lockfile
