# Phase 1: Data Acquisition -- Epic Plan

**Status:** COMPLETE
**David Sections:** 7.1 (Data Acquisition Layer), 8.2 (Dataset Family A -- Market Observations partial)
**Learning Guide Topics:** 10.2 (Data Engineering), 7.1 (Quantitative Analysis foundations)
**Protocol Reference:** HIFI_PROTOCOL_V1.md Phase 1

---

## Governing Philosophy for This Phase

This phase follows all HiFi development principles:

- **Interface-first:** Data schemas are defined and tested before any acquisition code
  is written. Every subsequent phase will depend on these schemas. Getting them right
  early prevents cascading rework.
- **No mocks:** External API calls are recorded as fixture files on first run.
  Tests replay those fixtures. Live calls never happen in the test suite.
- **Deterministic synthetic data:** Unit tests use seeded generators, not live data.
- **Provenance always:** Every dataset records where it came from, when, and with
  what parameters. This is the foundation of reproducibility (David 4.5).
- **Data contracts over convenience:** yfinance returns inconsistent column names,
  handles corporate actions differently depending on version, and occasionally
  returns malformed data. We normalize to our schema at ingestion, not later.
- **Measure quality before using data:** A data quality report is a deliverable,
  not an afterthought.

---

## Epic Dependency Graph

```
P1-E1 (Schemas and Contracts)
    |
    +--------+--------+
    |                 |
P1-E2 (Market Data)  P1-E3 (Macro Data)
    |                 |
    +--------+--------+
             |
        P1-E4 (Quality Validation)
             |
        P1-E5 (Versioning and Provenance)
```

E2 and E3 are independent of each other -- they can be developed in parallel.
E4 requires both E2 and E3 to have data to validate.
E5 can begin alongside E2/E3 since it defines the metadata structure,
but is only fully testable once data exists.

---

## Epic P1-E1: Data Schemas and Contracts

**Objective:** Define the canonical data structures for all data types this
phase will produce. These are interfaces, not implementations. They are the
contract between the data layer and every layer above it (engines, agents,
evaluation). Define them wrong here and the damage propagates everywhere.

**David rationale:** Section 4.3 (Verifiability) requires that every data
point carry enough metadata to trace back to its source. Section 4.5
(Reproducibility) requires that every dataset be identifiable by version.
These requirements are encoded in the schema, not left to convention.

| Ticket | Description | Status |
|---|---|---|
| P1-E1-T1 | Define OHLCVBar: single price bar with provenance fields | DONE |
| P1-E1-T2 | Define OHLCVDataset: collection of bars for one ticker + metadata | DONE |
| P1-E1-T3 | Define FundamentalsSnapshot: quarterly financial statement data | DONE |
| P1-E1-T4 | Define MacroIndicator and MacroDataset: macro time-series | DONE |
| P1-E1-T5 | Define ProvenanceRecord: source, timestamp, parameters, hash | DONE |
| P1-E1-T6 | Unit tests: schema validation accepts valid data | DONE |
| P1-E1-T7 | Unit tests: schema validation rejects invalid data (negative price, missing field) | DONE |
| P1-E1-T8 | Unit tests: provenance record produces stable content hash | DONE |

**Files to create:**
- `src/hifi/data/schemas.py` -- all Pydantic schemas
- `tests/unit/test_schemas.py` -- validation tests

**Acceptance test:** All schema tests pass. A valid OHLCV bar is accepted.
A bar with negative price is rejected. A provenance record hashes deterministically.

---

## Epic P1-E2: Market Data Acquisition (Yahoo Finance)

**Objective:** Download OHLCV and basic fundamental data for the 10-stock
universe, normalise to HiFi schemas, store as Parquet with provenance metadata.

**Why Yahoo Finance for Phase 1:** Free, no API key required, covers the full
historical range we need (2015--present), handles US equities well. Its
limitations (occasional gaps, inconsistent corporate action handling across
library versions) are documented and mitigated in this phase. SEC EDGAR and
Alpaca come in later phases when their specific capabilities are needed.

**Why not fetch live during tests:** Network calls in tests create non-determinism
(API changes, rate limits, outages) and make CI/CD impossible without credentials.
We record real responses once using a capture script, store them as fixtures, and
replay them in all tests. This is the "recorded fixture" pattern -- standard in
serious API testing.

| Ticket | Description | Status |
|---|---|---|
| P1-E2-T1 | Implement `MarketDataFetcher`: wraps yfinance, returns OHLCVDataset | DONE |
| P1-E2-T2 | Implement `FundamentalsFetcher`: wraps yfinance info/financials, returns FundamentalsSnapshot | DONE |
| P1-E2-T3 | Write `scripts/record_fixtures.py`: fetches real data and saves as fixtures for testing | DONE |
| P1-E2-T4 | Record fixtures for 3 representative tickers (AAPL, JPM, XOM) across 2 date ranges | DONE |
| P1-E2-T5 | Unit tests: fetcher normalises yfinance output to OHLCVDataset schema | DONE |
| P1-E2-T6 | Unit tests: fetcher handles missing data gracefully (gaps returned as NaN, logged) | DONE |
| P1-E2-T7 | Unit tests: fetcher attaches correct provenance metadata | DONE |
| P1-E2-T8 | Integration tests: full fetch for one ticker using recorded fixture | DONE |
| P1-E2-T9 | Integration tests: Parquet write/read round-trip preserves all values exactly | DONE |

**Files to create:**
- `src/hifi/data/market.py` -- MarketDataFetcher, FundamentalsFetcher
- `src/hifi/data/storage.py` -- Parquet read/write utilities
- `scripts/record_fixtures.py` -- one-time fixture recorder (not a test)
- `tests/fixtures/market/` -- recorded yfinance responses
- `tests/integration/test_market_acquisition.py`
- `tests/unit/test_market_fetcher.py`

**Acceptance test:** Integration test fetches AAPL OHLCV from fixture, writes
to Parquet, reads back, and every value matches the original to float precision.

---

## Epic P1-E3: Macro Data Acquisition (FRED)

**Objective:** Download key macroeconomic indicators from FRED, normalise to
HiFi schemas, store as Parquet with provenance metadata.

**Why these indicators:** The Macro Agent (David Section 10.2) needs interest
rates, inflation, unemployment, and market-level fear indicators. These are the
minimal set that drives macro-regime classification in Phase 10.

| Indicator | FRED Series ID | Rationale |
|---|---|---|
| Federal Funds Rate | FEDFUNDS | Primary monetary policy instrument; drives valuation |
| CPI (YoY inflation) | CPIAUCSL | Inflation regime detection |
| Unemployment rate | UNRATE | Economic cycle positioning |
| 10Y Treasury yield | GS10 | Risk-free rate baseline; yield curve |
| 2Y Treasury yield | GS2 | Yield curve slope (10Y-2Y = recession signal) |
| VIX | VIXCLS | Market fear index; volatility regime |
| US GDP growth | A191RL1Q225SBEA | Quarterly economic growth |

| Ticket | Description | Status |
|---|---|---|
| P1-E3-T1 | Implement `MacroDataFetcher`: wraps fredapi, returns MacroDataset | DONE |
| P1-E3-T2 | Handle FRED data alignment: convert to daily frequency by forward-filling (documented assumption) | DONE |
| P1-E3-T3 | Write FRED fixture recorder in `scripts/record_fixtures.py` (extend existing) | DONE |
| P1-E3-T4 | Record fixtures for all 7 indicators | DONE |
| P1-E3-T5 | Unit tests: fetcher normalises FRED output to MacroDataset schema | DONE |
| P1-E3-T6 | Unit tests: forward-fill logic is deterministic and does not bleed future values | DONE |
| P1-E3-T7 | Unit tests: provenance metadata is attached correctly | DONE |
| P1-E3-T8 | Integration tests: full fetch for FEDFUNDS using recorded fixture | DONE |
| P1-E3-T9 | Integration tests: Parquet write/read round-trip preserves all values | DONE |

**Files to create:**
- `src/hifi/data/macro.py` -- MacroDataFetcher
- `tests/fixtures/macro/` -- recorded FRED responses
- `tests/integration/test_macro_acquisition.py`
- `tests/unit/test_macro_fetcher.py`

**Note on forward-filling:** FRED series have different publication frequencies
(monthly for CPI, quarterly for GDP). To align with daily OHLCV data, we
forward-fill -- carry the last known value forward until the next publication.
This is a documented assumption, not a hidden decision. It means agents see the
most recently published value, not the current unrevised value. This is
point-in-time safe by construction.

**Acceptance test:** Integration test fetches FEDFUNDS from fixture, writes to
Parquet, reads back, and alignment to daily frequency is verified.

---

## Epic P1-E4: Data Quality Validation

**Objective:** Measure the completeness, consistency, and fitness-for-use of
the acquired data. Produce a structured quality report. Failures do not crash
the system -- they are reported so they can be investigated.

**Why this is a deliverable, not an afterthought:** The evaluation framework
in Phase 10 will use this data. If the data has gaps in critical periods (e.g.,
the 2020 COVID crash), our evaluation results will be wrong. We need to know now,
not when we are debugging Phase 10 results.

| Ticket | Description | Status |
|---|---|---|
| P1-E4-T1 | Implement `DataQualityChecker`: takes OHLCVDataset, returns QualityReport | DONE |
| P1-E4-T2 | Check: completeness -- % of expected trading days with data | DONE |
| P1-E4-T3 | Check: gap detection -- identify contiguous missing periods > N days | DONE |
| P1-E4-T4 | Check: price sanity -- no negative prices, no zero volume, no single-day moves > 50% | DONE |
| P1-E4-T5 | Check: OHLCV relationships -- high >= max(open,close), low <= min(open,close) for every bar | DONE |
| P1-E4-T6 | Check: corporate action consistency -- detect suspicious price discontinuities (likely unadjusted) | DONE |
| P1-E4-T7 | Implement `QualityReport`: structured summary with per-ticker metrics | DONE |
| P1-E4-T8 | Unit tests: quality checker detects known defects in synthetic bad data | DONE |
| P1-E4-T9 | Unit tests: quality checker passes clean synthetic data | DONE |
| P1-E4-T10 | Integration test (holistic): run quality check on all 10 tickers, report passes threshold | DONE |

**Files to create:**
- `src/hifi/data/quality.py` -- DataQualityChecker, QualityReport
- `tests/unit/test_quality.py`
- `tests/holistic/test_phase1_pipeline.py` -- first holistic test

**Completeness threshold for Phase 1:** >= 98% of expected US trading days
with non-null OHLCV data. Stocks below this threshold are flagged, not removed.

**Acceptance test:** Quality check on synthetic clean data passes.
Quality check on synthetic data with injected defects catches every defect.

---

## Epic P1-E5: Data Versioning and Provenance

**Objective:** Every dataset written to disk is stamped with a version identifier
derived from its content. The dataset registry records what was downloaded, when,
and with what parameters. This makes every experiment reproducible.

**Why content-based hashing over timestamps:** A timestamp tells you when the
file was written, not what is in it. If we re-download data and the API returns
different values (revised data, different adjusted prices), a new hash catches
this automatically. A new timestamp would too, but the hash also serves as an
integrity check on the file itself.

| Ticket | Description | Status |
|---|---|---|
| P1-E5-T1 | Implement `content_hash(path)`: SHA-256 hash of a Parquet file | DONE |
| P1-E5-T2 | Implement `DatasetRegistry`: JSON-backed registry of all downloaded datasets | DONE |
| P1-E5-T3 | Registry records: ticker, source, date range, download timestamp, file path, content hash | DONE |
| P1-E5-T4 | Registry supports: register, lookup by ticker+source, verify integrity (re-hash and compare) | DONE |
| P1-E5-T5 | Integrate registry into MarketDataFetcher and MacroDataFetcher | DONE |
| P1-E5-T6 | Unit tests: hash is stable for identical content | DONE |
| P1-E5-T7 | Unit tests: hash changes when content changes | DONE |
| P1-E5-T8 | Unit tests: registry stores and retrieves entries correctly | DONE |
| P1-E5-T9 | Unit tests: integrity check detects file tampering | DONE |

**Files to create:**
- `src/hifi/data/versioning.py` -- content_hash, DatasetRegistry
- `tests/unit/test_versioning.py`

**Acceptance test:** Register a dataset, verify integrity passes.
Modify the file, verify integrity check detects the change.

---

## Holistic Test: Phase 1 Pipeline

**File:** `tests/holistic/test_phase1_pipeline.py`

This test runs after all epics are complete. It validates the full Phase 1
pipeline end-to-end using recorded fixtures only (no live API calls).

**Scenario:** Acquire data for 3 tickers (AAPL, JPM, XOM) + 2 macro indicators
(FEDFUNDS, CPIAUCSL) using recorded fixtures. Run quality checks. Register in
dataset registry. Verify that a fresh load from Parquet matches the original.

**Assertions:**
- All schemas validate
- Quality report shows no critical defects on the fixture data
- Registry entries created with correct metadata
- Parquet round-trip is exact (no float precision loss)
- Content hashes are stable across two calls on the same file

**This test becomes a regression guard:** Once it passes, it must keep passing
in all subsequent phases. If Phase 2 engineering breaks data loading, this test
catches it.

---

## New Dependencies to Add

```
# Production
yfinance>=0.2
fredapi>=0.5
pandas>=2.0
pyarrow>=15.0     # Parquet read/write

# Dev / test only
responses>=0.25   # HTTP response recording for fixture replay
```

**Decision to record (DJ-008):**

- **pandas vs. polars:** pandas is chosen for Phase 1 for one reason: yfinance
  returns pandas DataFrames natively. Converting to polars at ingestion adds a
  transformation layer before the data is even validated. We normalise to our
  own schema (Pydantic + Parquet) which is library-neutral, so switching to
  polars for downstream processing is still possible in Phase 2+. This is not
  a permanent commitment to pandas.

- **pyarrow for Parquet:** pyarrow is the standard Parquet engine for Python.
  It preserves column types exactly (including datetime64 with timezone), is
  well-maintained, and is already a pandas dependency. No alternative considered.

- **responses for HTTP fixtures:** The `responses` library intercepts HTTP calls
  made by requests/urllib. yfinance and fredapi both use HTTP under the hood.
  Recording responses lets us capture real API responses once and replay them
  in tests without modifying production code. Alternative considered: `vcrpy`
  (heavier, cassette format is less readable). `responses` is simpler for our use.

---

## Phase 1 Quality Gates

| Gate | Criterion | Measured By |
|---|---|---|
| All unit tests pass | pytest -m unit, 0 failures | CI / manual run |
| All integration tests pass | pytest -m integration, 0 failures | CI / manual run |
| Holistic test passes | pytest tests/holistic/test_phase1_pipeline.py | CI / manual run |
| Linting clean | ruff check src/ tests/, 0 errors | CI / manual run |
| No live API calls in tests | grep -r "yfinance.download" tests/ returns nothing | Code review |
| Quality report threshold | >=98% completeness on fixture data | Holistic test |
| Schema coverage | Every field in OHLCVBar has at least one rejection test | Test review |
| David proxy | E1 completes: data/schemas.py maps to David 7.1 and 8.2 | Conformance matrix |

---

## Commit Strategy

One commit per epic, in dependency order:

| Commit | Epic | Files |
|---|---|---|
| Phase 1 / E1: Data schemas and contracts | P1-E1 | src/hifi/data/schemas.py, tests/unit/test_schemas.py |
| Phase 1 / E2: Market data acquisition | P1-E2 | src/hifi/data/market.py, storage.py, fixtures, tests |
| Phase 1 / E3: Macro data acquisition | P1-E3 | src/hifi/data/macro.py, fixtures, tests |
| Phase 1 / E4: Data quality validation | P1-E4 | src/hifi/data/quality.py, holistic test |
| Phase 1 / E5: Data versioning and provenance | P1-E5 | src/hifi/data/versioning.py, tests, updated registry integration |

E2 and E3 commits can be made in either order since they are independent.

---

## Open Questions This Phase Will Answer

- **OQ-D01 (partial):** What is the actual data quality from yfinance for our
  10 stocks? The quality report will give us completeness percentages,
  gap locations, and any price anomalies. We will know before Phase 2.

- **OQ-D03 (partial):** The fixture recorder will reveal what free data sources
  actually return. If yfinance has major gaps for certain tickers, we document it.
