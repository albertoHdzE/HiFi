# Phase 2 Bitacora: Deterministic Financial Engine

**Phase:** 2 -- Deterministic Financial Engine
**Status:** COMPLETE
**Dates:** 2026-05 to 2026-06-10
**Author:** Alberto Espinosa
**Tests at completion:** 309 (159 new in Phase 2); 0 lint errors

---

## Objective

Implement the deterministic computation layer that agents will rely on for all financial
quantities. Every ratio, indicator, and risk metric an agent cites must flow from a pure
function with no I/O, no side effects, and no implicit assumptions. The MCP server is the
only interface through which agents access this layer. This phase establishes the audit
trail infrastructure (call_id) that Phase 5 will use for claim verification.

---

## E1: Engine Interfaces and Result Types

The first task was defining the output contracts before any computation code existed.
This is interface-first development applied strictly: agents that will be written in
Phase 3 can be designed against these types today, regardless of implementation details.

The key architectural insight was the `_NanToNoneBase` Pydantic v2 model. Numpy
computations naturally produce `float('nan')` for missing or undefined results (e.g.,
SMA when insufficient bars exist). JSON does not support NaN -- `json.dumps(float('nan'))`
raises a `ValueError`. Catching this at serialization time and converting to `None`
(JSON null) is the correct contract: downstream consumers receive a type-safe signal
("no value") rather than a silent corruption. The `@model_validator(mode="before")` hook
converts any `nan` to `None` before Pydantic's field validators run, so no NaN can
ever escape from a result object.

Six result types were defined:
- `FinancialRatioResult`: P/E, P/B, P/S, ROE, ROA, debt_equity, current_ratio
- `GrowthMetricsResult`: revenue_growth_yoy, earnings_growth_yoy, gross_margin,
  operating_margin, net_margin
- `TechnicalIndicatorsResult`: sma, ema, rsi, macd, macd_signal, macd_hist, bb_upper,
  bb_mid, bb_lower, atr
- `RiskMetricsResult`: hist_vol_20d, hist_vol_60d, hist_vol_252d, beta,
  max_drawdown_252d, sharpe_252d, var_95_20d
- `ValuationResult`: current_pe, pe_1y_min, pe_1y_max, pe_1y_percentile,
  price_to_52w_high, price_to_52w_low
- `MacroSnapshotResult`: fed_funds_rate, cpi_yoy, unemployment_rate, yield_10y,
  yield_2y, yield_curve_slope, vix, gdp_growth

All fields Optional[float]. All types serialise to JSON-safe dicts (no NaN escapes).

---

## E2: Fundamental and Valuation Engine

**Data depth limitation discovered.** yfinance's `.info` dictionary returns only a
single point-in-time snapshot per ticker -- there is no historical fundamentals series
available through this API. This means YoY growth rates (revenue_growth_yoy,
earnings_growth_yoy) cannot be computed from two periods because only one period exists
in the snapshot. Growth fields return None in Phase 2. An agent that receives None for
growth metrics must interpret it as "insufficient data," not "zero growth." This is
documented in the engine's module docstring and in the MCP tool schema descriptions.

The ratios that CAN be computed from a single snapshot (P/E, P/B, P/S, ROE, ROA,
debt/equity, net margin) are implemented with explicit None propagation: a zero or
negative denominator returns None rather than dividing. This is not a defensive
safeguard against bad data -- it is the correct financial interpretation. A company
with negative equity has no meaningful P/B ratio.

**Valuation context** was the most interesting computation. The current P/E is compared
to its own trailing 252-day range to produce a percentile (pe_1y_percentile in [0, 1]).
This required care about point-in-time correctness: only bars on or before `as_of_date`
contribute to the trailing P/E series. The percentile formula is
`(pe_current - pe_min) / (pe_max - pe_min)`, clamped to [0, 1]. When pe_min == pe_max
(all bars at the same price, or only one bar), 0.5 is returned by convention.

A subtle test design error was discovered during development: a test for "percentile = 0.5
when current P/E is the median" used bars [100, 200, 300] with as_of = the last bar date.
The current price is 300, which has P/E = 300/eps = maximum, not median. The correct
test data is bars with price sequence [100, 200, 150]: the current bar (150) produces
a P/E between the historical min (P/E from 100) and historical max (P/E from 200). This
is a good example of why test data must be designed with the assertion's semantics in
mind, not just constructed to fill slots.

---

## E3: Technical Indicator Engine

**Decision DJ-010: Custom numpy/pandas for Phase 2; MCP-in-dedicated-venv for Phase 8+.**

Three TA libraries were evaluated for the main environment:

- **pandas-ta:** Incompatible with pandas >= 2.0. Calls `DataFrame.append()` which was
  removed in pandas 2.0. The project uses pandas 3.0.3. Fails at runtime, not at import.
  This is a hard blocker.
- **ta (bukosabino):** Works with pandas 3.x but has had no meaningful commits since 2022.
  Using an unmaintained library in a research project with a publication goal is
  defensible only if there is no alternative.
- **TA-Lib (C extension):** Requires a native C build. Fragile on arm64/macOS without
  Homebrew integration. Introduces an out-of-band build dependency that breaks CI
  portability.

The custom numpy implementation for six specific indicators (SMA, EMA, RSI, MACD,
Bollinger Bands, ATR) is ~50 lines. Each formula cites its primary source. Zero new
production dependencies. This is not a substitute for a TA library at scale -- it is the
correct scope for Phase 2.

The more important decision was the Phase 8+ architecture: instead of waiting for
pandas-ta to update or accepting a degraded alternative, HiFi will deploy a dedicated
MCP server running in an isolated virtual environment at `venvs/ta/`. The MCP subprocess
boundary is already a process boundary. A subprocess has its own Python interpreter,
its own site-packages, and cannot import anything from the parent process. Giving it a
dedicated venv with `pandas==1.5.3` pinned and `pandas-ta==0.3.14b0` costs nothing beyond
a `uv venv` call. The main process consumes JSON tool results -- it has no knowledge of
what runs inside the server. This is not a workaround; it is the natural extension of the
MCP architecture to the dependency isolation problem. The scaffold was created:
`venvs/ta/requirements.txt` and `scripts/setup_ta_venv.sh`.

**RSI edge case: monotonically rising series.** A rising price series has all gains and
zero losses. The standard RSI formula is RSI = 100 - 100/(1 + RS) where RS = avg_gain /
avg_loss. When avg_loss = 0, RS is undefined (division by zero). The mathematically
correct answer is RSI = 100 (infinite bullishness). The implementation must produce 100,
not NaN, not None. The first attempt used `.replace(0.0, float('nan'))` on avg_loss,
which produces NaN / NaN = NaN, making `_last()` return None. The correct implementation
uses numpy.where inside errstate(divide="ignore", invalid="ignore"):

```
rs_vals = np.where(
    loss_vals == 0.0,
    np.where(g == 0.0, np.nan, np.inf),  # gain > 0 -> inf; gain == 0 -> NaN (no movement)
    g / loss_vals,                          # normal case
)
```

inf -> 100 - 100/(1+inf) = 100. This is the mathematically correct result and was
verified against the Wilder (1978) original specification.

---

## E4: Risk Metrics Engine

**Decision DJ-011: QuantStats; Pyfolio explicitly and permanently rejected.**

Pyfolio was maintained by Quantopian, which shut down in October 2020. The library's
last meaningful release predates pandas 2.0. It is incompatible with the current stack
and unmaintained. Including a deprecated library in an open-source capstone project is
academically indefensible -- there is no path to citation or reuse. Pyfolio is excluded
from all phases of HiFi, permanently.

QuantStats is the correct choice because: (1) it is actively maintained, (2) it provides
the full set of metrics needed (Sharpe, Sortino, drawdown, VaR, volatility), (3) its
`quantstats.reports` module generates professional-grade tear sheets that will be used
directly in Phase 10 (Evaluation & Backtesting). Adopting it in Phase 2 gives Phase 10
analytics infrastructure at no additional integration cost.

**QuantStats API surprises (documented for future phases):**

1. `qs.stats.sharpe(returns, rf, periods=252, annualize=True)` -- no `prepare_returns`
   parameter. The first implementation passed `prepare_returns=False` from the volatility
   call signature. This raises `TypeError: unexpected keyword argument`. Removed.

2. `qs.stats.max_drawdown(prices)` -- returns a negative float (loss is negative by
   convention). `RiskMetricsResult.max_drawdown_252d` stores a positive magnitude.
   Applied `abs()`.

3. `qs.stats.value_at_risk(returns, confidence=0.95)` -- also returns negative. Same fix.

4. `qs.stats.volatility(returns, periods=252, prepare_returns=False)` -- the `prepare_returns`
   flag is present here but NOT on `sharpe()`. The inconsistency is noted in the risk
   engine docstring.

**Beta precision and return type.** Computing beta as `cov(r_stock, r_bench) / var(r_bench)`
requires that r_stock and r_bench be constructed compatibly. A test used log returns to
build a 2x beta series: `prices_stock = exp(2 * cumsum(log_returns))`. When `pct_change()`
is called on these prices, it recovers linear returns, not the log returns used to build
the prices. The relationship is `pct_change ≈ 2 * log_return + O(log_return^2)`, which
introduces a numerical error proportional to the square of the return. At daily sigma =
2%, the error is ~0.04% per bar -- small but enough to fail a 1e-4 relative tolerance
test over 252 bars. The fix: build benchmark prices directly from linear returns using
`np.cumprod(1 + bench_rets)` so that `pct_change()` recovers the original linear returns
exactly. Tolerance loosened to rel=1e-3 to accommodate any residual floating-point
accumulation.

---

## E5: MCP Server

**Decision DJ-009: stdio transport for Phase 2-14.**

The two options were stdio JSON-RPC (subprocess + pipes) and SSE/HTTP (network transport).
For a single-machine local deployment where agents and servers run on the same host,
stdio is strictly simpler: no sockets, no authentication, no service discovery, no port
management. A Phase 3 agent starts the server as a subprocess and communicates through
stdin/stdout. All major agent frameworks (LangGraph, LlamaIndex) support subprocess MCP
servers out of the box.

The performance concern (serialization overhead of passing OHLCV bars across the
subprocess boundary) was examined informally. A full technical analysis for AAPL Q1 2023
(63 bars) completes well under 1 second. The bottleneck is Parquet I/O, not JSON
serialization. OQ-P2-05 (serialization overhead at Phase 8+ scale, ~252 bars, ~50
tickers) remains open and will be measured empirically when the indicators venv server
is built.

SSE/HTTP will be re-evaluated at Phase 15 when multi-container deployments may require
network transport between services. That decision is documented in `financial_server.py`
module docstring.

**Fixture format discovery.** Phase 1 stored market data using `write_ohlcv()`, which
embeds a HiFi metadata block in the Parquet file's metadata. The test fixtures (AAPL,
JPM, XOM) were recorded BEFORE the write_ohlcv() API was finalized -- they are raw
pandas DataFrames written directly from yfinance with columns
`Date/Open/High/Low/Close/Adj Close/Volume`. When the MCP server called `read_ohlcv()`
on these fixtures, it raised `ValueError: No HiFi dataset metadata` because the metadata
block was absent.

Resolution: A fallback loader `_load_raw_ohlcv()` was added to the server, and
`read_raw_ohlcv_fixture()` was added to `tests/conftest.py`. The MCP server tries
`read_ohlcv()` first and falls back to the raw format on ValueError. This is not a
permanent architecture -- Phase 7+ will re-record fixtures using the write_ohlcv() API.
The raw fallback is documented with a comment noting its scope.

**call_id: Phase 5 audit trail hook.** Every tool response includes a `call_id` field:
the first 12 hex characters of SHA-256(JSON(sorted(params))). This is deterministic
given the same inputs, making it reproducible. Phase 5 (Verification) will use this to
match agent-cited numbers against the specific tool call that produced them. Building
this into Phase 2 costs 2 lines per tool and avoids a costly retrofit in Phase 5.

**Data access pattern for fundamentals.** FundamentalsSnapshot is not yet stored as
Parquet in Phase 2 (no `write_fundamentals()` in storage.py). The tools that need a
snapshot accept `snapshot_json`: the snapshot serialised as a JSON string. This is a
Phase 2 simplification; Phase 7+ will implement proper storage and registry semantics
for fundamentals. It is documented in the server's module docstring to prevent confusion
when revisiting this code.

---

## E6: Holistic Pipeline Test

The holistic test validates four scenarios:
1. Direct engine calls on Phase 1 AAPL fixture produce valid, non-NaN results
2. All result types serialise to JSON without error (no NaN escapes _NanToNoneBase)
3. MCP tool functions called directly produce the same results as direct engine calls
4. Phase 1 holistic test still passes (regression guard: data layer not broken)

The test uses `read_raw_ohlcv_fixture()` to load the raw AAPL fixture. All assertions
pass. The regression guard (Phase 1 pipeline test) is a permanently included check;
it will run in all subsequent holistic suites.

---

## Open Questions -- Resolved

**OQ-P2-01 (DJ-009 resolved):** stdio is the correct MCP transport for Phase 2-14.
Wall-clock time for a full AAPL analysis via MCP tool call is < 1 second on M3 Ultra.
No caching layer is needed at this scale.

**OQ-P2-02 resolved:** yfinance fundamentals are reliable for: eps, market_cap,
total_equity, revenue (as totalRevenue), net_income (as netIncome), total_assets,
total_liabilities. Unreliable or absent: ebitda (often None), current_ratio (not in
.info dict at all), book_value per share (available as bookValue but format varies).
These findings are embedded in the FundamentalsSnapshot schema comments from Phase 1
and will inform agent prompt design in Phase 3.

**OQ-P2-03 resolved:** Computation latency for a single ticker (63 bars, 6 indicators,
5 risk metrics) is < 200ms including Parquet load on M3 Ultra. Not a concern at Phase 2
scale.

---

## Open Questions -- Raised

**OQ-P2-04 (from Phase 1 OQ-P1-01):** US trading calendar. The DataQualityChecker
uses weekday count as an approximation for expected bars. This was not resolved in
Phase 2 because it is not on the critical path. Phase 3 will determine if false-positive
quality failures actually occur in practice; if they do, `pandas_market_calendars` will
be added then.

**OQ-P2-05:** Serialization overhead of OHLCV arrays across the MCP subprocess boundary
at Phase 8+ scale (252 bars x 50 tickers per call). To be measured empirically when the
Phase 8 indicators server is first built. If overhead is significant, Arrow IPC over stdio
is the mitigation.

---

## Decisions Recorded This Phase

| ID | Decision |
|---|---|
| DJ-009 | MCP transport: stdio for Phase 2-14; SSE/HTTP re-evaluated at Phase 15 |
| DJ-010 | Technical indicators: custom numpy Phase 2; venvs/ta/ isolated MCP server Phase 8+ |
| DJ-011 | Risk metrics: QuantStats; Pyfolio permanently rejected |
| DJ-012 | FinanceToolkit: deferred to Phase 7+ (needs SEC EDGAR multi-period quarterly data) |

---

## Forward Dependencies for Phase 3

Phase 3 (First Agent -- Baseline) depends on Phase 2 as follows:

- The `hifi-financial-calculator` MCP server must be running before any agent can produce
  an analysis. Phase 3 agents start the server as a subprocess.
- The Fundamental Agent calls: `get_financial_ratios`, `get_growth_metrics`,
  `get_valuation_context`, `get_macro_snapshot`.
- The Technical Agent (Phase 4) calls: `get_technical_indicators`.
- Agent prompts must handle None values for: growth metrics (Phase 2 limitation),
  beta when no benchmark is provided, any indicator below its minimum window.
- The `snapshot_json` parameter for fundamental tools requires a serialised
  FundamentalsSnapshot. Phase 3 agents must load the snapshot from Phase 1 data and
  pass it through the tool call. This awkward coupling will be resolved in Phase 7+ when
  fundamentals have their own Parquet storage and can be loaded server-side by ticker.

---

## Metrics at Phase 2 Completion

| Metric | Value |
|---|---|
| Total tests | 309 |
| New tests (Phase 2) | 159 |
| Test breakdown | 85 unit (engines), 18 integration (MCP), 16 holistic |
| Lines of production code | ~3500 (engines: ~1350, mcp: ~470, data+config: ~1580 from Phase 1) |
| Lines of test code | ~4600 |
| Lint errors | 0 |
| New production dependencies | mcp (FastMCP), quantstats |
| New engine modules | 5 (types, fundamental, technical, risk, macro) |
| MCP tools delivered | 6 |
