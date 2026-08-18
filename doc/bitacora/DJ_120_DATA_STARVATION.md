# DJ-120 — Agent data starvation: three independent path defects

**Date found:** 2026-08-14 · **Branch:** `phase14/heterogeneous-ensemble`
**Severity:** invalidates the Phase 15 headline result and the Phase 16 A/B live
record from 2026-07-16 to 2026-08-13.

## Summary

Three unrelated bugs each pointed a consumer at an empty or legacy store while
the real, rich data sat elsewhere. Together they starved three of the five
ensemble agents, so the "ensemble" under study was effectively two agents on 15
tickers and a blanket no-data `Sell` on the other 83.

The defects were invisible for a month because **the MCP tools return error
payloads rather than raising**, and the agents faithfully reasoned over those
payloads. A dead tool call and a considered bearish opinion produced the same
downstream artefact: a `Sell` with a rationale. The failure therefore presented
as a plausible trading signal, not as an outage.

## The three defects

### 1. OHLCV — legacy flat glob (critical)

Five call sites globbed `data/market/{TICKER}_*.parquet`:

| file | symbol |
|---|---|
| `src/hifi/mcp/financial_server.py` | `_load_ohlcv` |
| `src/hifi/mcp/indicators_server.py` | `_load_ohlcv_df` |
| `src/hifi/collective/labeler.py` | `_load_prices` |
| `src/hifi/models/training_data.py` | `_load_close_series`, `_load_ohlcv_df` |

That pattern matches only 16 leftover fixtures, all ending **2023-06-30**. The
canonical store is nested — `data/market/{TICKER}/ohlcv.parquet`, 98 tickers,
current to the previous close. Result: 83/98 tickers returned
`TICKER_NOT_FOUND` on every pass; the remaining 15 were analysed on bars three
years stale.

Arm D was unaffected because `riskbudget_strategy.py` reads the nested store
directly — the same layout mismatch noted in DJ-113, in the opposite direction.

### 2. Macro — format mismatch (high)

`data/macro/macro.parquet` was a *wide* frame (`treasury_10y`, `treasury_2y`,
`spread_10y2y`) from an earlier regime-detection phase. `read_macro` expects
long single-series files carrying a `hifi_dataset_metadata` schema block, so it
raised, `_load_all_macro` returned `{}`, and every `get_macro_snapshot` answered
`NO_MACRO_DATA`. The macro agent voted Hold on 99.8% of passes.

### 3. Sentiment — wrong knowledge table (high)

`knowledge_server` reads `HIFI_KNOWLEDGE_DATA_DIR`, default `data/knowledge`,
whose `chunks_a` table holds **169 rows for 3 tickers** (AAPL, JPM, XOM). The
populated corpus is `data/knowledge.lance` table `hifi-dev-sec-sec-mda` —
**209,722 MD&A chunks covering all 98 tickers**, already used by the fundamental
agent via `retrieve_mda_context`. Sentiment never reached it and took its
"Insufficient Data" path on 97% of passes, returning Hold at confidence 0.0.

Arithmetic confirmation: 57 live non-default sentiment passes = 3 tickers × 19
dates exactly.

## Measured impact

Across 1,862 sidecars per LLM arm:

| agent | tools ok | tools failed |
|---|---|---|
| technical | **Buy 285/285 (100%)** | Sell 1338, Hold 239 |
| risk | Hold 201, Sell 84 | **Sell 1577/1577 (100%)** |
| fundamental | — (0 clean) | Hold 1841, Buy 21 |
| macro | — (0 clean) | Hold 1858, Sell 3 |
| sentiment | Hold 1862/1862 (conf 0.0) | — |

Three of five agents were constant. Because unanimity requires *all* members to
agree, constant members also **suppress** measured herding — so the diversity
metrics were biased, not merely noisy.

### Phase 15 headline result does not survive

`compute_phase15_ic.py` reads forward returns from the nested store, so the
*labels* were always correct; only the *signals* were starved. Decomposing the
published IC by data availability:

| condition | all | 16 data-bearing tickers | 83 dark tickers |
|---|---|---|---|
| **parallel** | **+0.0642** (p=0.0019) | **−0.1377** (p=0.0089) | **+0.0669** (p=0.0028) |
| full | +0.0232 | −0.0147 | +0.0207 |
| homogeneous | −0.0428 | −0.0481 | −0.0422 |
| no-memory | +0.0251 | −0.0427 | +0.0162 |

The positive headline IC lives **entirely in the subset where the agents had no
data**. Where they could see, the champion condition is significantly negative.

A second, independent statistical problem: p-values treat 2,352 (ticker, date)
pairs as independent when they are 25 dates × 99 tickers with overlapping
60-trading-day forward windows. Effective n is nearer the number of
non-overlapping dates, so `p=0.0019` is substantially overstated regardless of
the data defect.

**Conclusion:** Phase 15 cannot be read as evidence for or against the Page
theorem. The walk-forward harness is sound; it needs re-running on repaired
data.

## Fixes

- **New** `src/hifi/data/market_store.py` — `resolve_ohlcv_path` (nested-first,
  flat fallback) and `coverage_report`. All five call sites now use it;
  `indicators_server` duplicates the logic by design (pandas 1.5.3 env).
- `financial_server._load_raw_ohlcv` handles the nested shape (DatetimeIndex, no
  `Adj Close`) and labels provenance `market_store` vs `fixture` honestly.
- **New** `scripts/refresh_macro_store.py` — writes the 7 series
  `compute_macro_snapshot` actually reads, in `write_macro` format; retires the
  wide frame to `macro_wide_legacy.parquet.bak`.
- `edgar_retriever.retrieve_mda_context` gained an optional `query` for
  lexical chunk selection; `sentiment_agent._retrieve_context` falls back to it
  when the vector store is empty. The query differs from the fundamental
  agent's head-of-document slice **deliberately**: identical evidence would
  correlate two nominally independent members in an experiment whose dependent
  variable is agreement.
- **New** `src/hifi/analytics/decision_audit.py` — per-stock traceability with
  tool-call health as a first-class column, plus `degenerate_agents` to catch
  constant members before they are read as consensus.

## Verification

- 98/98 tickers resolve to the nested store, all `last_date = 2026-08-13`; zero
  flat-legacy fallbacks. 98/98 covered in the EDGAR corpus.
- `get_macro_snapshot("2026-08-13")` returns a full cross-section.
- End-to-end agent re-run, ACN on 2026-08-13, same ticker and date as the
  starved live pass:

| | before | after |
|---|---|---|
| tool errors | 7 | **0** |
| technical | Sell @ 1.0 ("no data available") | Buy @ 0.7 (real indicators) |
| risk | Sell @ 1.0 | Sell @ 0.95 |
| sentiment | Hold @ 0.0 (insufficient data) | Hold @ 0.7 (real MD&A) |
| macro | Hold @ 0.1 | Hold @ 0.6 |
| **ensemble** | **Sell** @ 0.769 | **Hold** @ 0.548 |
| entropy | 0.971 | 1.371 |

- 1,690 tests pass (1,660 pre-existing + 30 new), lint clean.

## Known gaps, deliberately not fixed here

- **SPY benchmark** exists only as a flat 2023 fixture, so `beta` would be stale
  if requested. No agent currently passes `benchmark_ticker`, so beta is always
  `None` — a pre-existing gap, unchanged by this work.
- **`indicators_server`** is dormant: nothing in the agent pipeline calls it. Its
  `venvs/ta` env also lacks `pyarrow`. Patched for consistency, not exercised.
- **`google/gemma-2-2b-it`** shows 139 LangFuse calls at a 100% error rate.
  Unrelated to this defect; worth a separate look.

## Consequences for the experiment

Arms A and B are void for 2026-07-16 → 2026-08-13. Arms C and D are unaffected
*by this defect*. Per the decision on 2026-08-14, A and B restart clean after
the fix rather than continuing across a regime break. This warrants **OSF
amendment 002**, and it is the more consequential of the two now pending.

> **Superseded in part by DJ-121 below.** D was *not* in fact unaffected: it
> shares the allocator with A and B and was crippled by a separate defect there,
> so D's history is void for performance purposes too. Only C — which bypasses
> the pipeline entirely — remains continuous and valid.

---

# DJ-121 — Allocator froze every arm at one position

**Found:** 2026-08-17, immediately after DJ-120 went live.

## How it surfaced

The first post-fix run (Friday 2026-08-14, executed 04:41 Monday) confirmed
DJ-120 was cured — tool-failure rate 0.000 across all agents, and A's signal
mix moved from `Sell 86 / Buy 1` to `Sell 0 / Buy 30`. But A placed **zero
orders**, and had ~95% cash against a single NVDA position.

DJ-120 had been *masking* this. While 86 of 98 names were blind-Sells there was
only ever one Buy per day, so nobody could see that the allocator was discarding
Buys. Restoring sight produced 30, and all 30 died.

## Defects in `mcp/capital_allocator.generate_orders`

**1. Rebalance band wider than the maximum position (primary).**
`_REBAL_THRESHOLD` was 0.05 absolute, and `max_single_stock` is also 0.05. A
fresh position has `current_weight = 0`, so its drift is `|target − 0| = target
≤ 0.05` — it can never exceed a 0.05 band. The guard only engages once
`current_capital > 0`, so each arm opened exactly one position on day one while
100% cash and was then structurally frozen. Sweep:

```
target=0.033 -> 0 orders      target=0.050 -> 0 orders
target=0.049 -> 0 orders      target=0.051 -> 1 order
```

**2. Mismatched denominators.** `target_value = weight × capital` (equity) but
`current_weight = current_value / current_capital` (invested value). At 61%
cash this inflated held weights ~2.6×:

| | as computed | true |
|---|---|---|
| BAC | 0.1284 | 0.0499 |
| JPM | 0.1223 | 0.0475 |

That inflation was the *only* reason D traded: BAC's fake 12.84% cleared the
band against a 5% target and `floor()` yielded a ±1-share delta — buy 1, sell 1,
buy 1 on consecutive days. Rounding churn on one name, not a strategy.

**3. Integer share truncation.** `int(floor(...))` on quantities and
`int(p.qty)` on holdings; a 3.9-share position read as 3 and manufactured
phantom rebalances. Alpaca supports fractional shares and arm C already used them.

**4. Over-allocation (found while fixing 1–3).** `compose_portfolio` spreads
~100% of capital across the Buy names alone while Hold positions stay on the
book, so targets can sum past 100%. A's repaired run demanded $100,196 of buys
against $95,464 cash.

## Fixes

- Band is now **relative to the target** (`_REBAL_DRIFT_FRAC = 0.20`): a fresh
  position is 100% adrift and always trades; a held position near target does not.
- Both weights measured against `capital`. `current_capital` is now only an
  "is there a book yet?" flag.
- Fractional quantities by default (`fractional=False` retains whole shares for
  IBKR), plus a `$50` notional deadband that makes rounding noise untradeable.
- `_fit_to_cash` scales BUYs **proportionally** to available cash less a 1%
  buffer. Proportional, not truncating the tail — the experiment measures how
  signals map to a portfolio, so dropping the end of the buy list would bias the
  book toward whichever names were enumerated first. SELLs are never scaled.
- Long-only guard: a SELL can never exceed shares held.

## Verification (against live broker state, 2026-08-17)

| arm | before | after | buy notional vs cash |
|---|---|---|---|
| A | 30 Buys → **0 orders** | 30 orders | $94,511 / $95,464 ✓ |
| B | 18 Buys → **0 orders** | 17 orders | $85,352 / $95,508 ✓ |
| D | 12 Buys → 1 churn order | 10 orders | $49,432 / $60,611 ✓ |

1,971 tests pass. Three tests encoded the old contract and were rewritten
deliberately: absolute-band, integer-quantity and `isinstance(int)` assertions.

## Consequence for the experiment

A, B and D were all crippled by one allocator, from two different signal
sources — A/B from the LLM ensemble, D from riskbudget. C is unaffected (it
bypasses the pipeline entirely) and is the only arm that ever held 98 names.

Per the decision on 2026-08-17, **A, B and D restart from the first clean run**;
their prior history is void for performance purposes. C remains continuous and
valid throughout. This compounds OSF amendment 002: the A/B restart announced
for DJ-120 had not in fact begun, because the arms could not deploy capital.

---

# DJ-122 — Position limits were absolutes, not functions of the universe

**Found:** 2026-08-17, from Alberto's observation that the arms looked "wired
to a limited number of stocks — we tried 8, 16, 30 early on".

## The finding

There is **no** hardcoded position cap. Fed 98 Buy signals, `compose_portfolio`
returns 98 weights and drops nothing. The intuition was right about the
symptom and wrong about the mechanism.

What existed instead was a set of absolute constants —
`max_single_stock=0.05`, `max_sector=0.20`, `min_position=0.01` — restated
independently in `run_mcp_pipeline` and as defaults on `compose_portfolio`,
bearing no relation to how many names were being chosen from. Measured
deployment against an equal-confidence buy list:

| Buys | positions | capital deployed |
|---|---|---|
| 8 | 8 | **20%** |
| 16 | 16 | **40%** |
| 30 | 30 | **70%** |
| 50+ | 50–98 | 100% |

A narrow or sector-concentrated buy list silently stranded most of the
account, with no error anywhere. At the other end the same 5% cap is 5x the
equal weight across 98 names and never binds at all. One number was
simultaneously far too tight and completely inert.

**Why this matters beyond capital efficiency:** the binding constraint was a
function of *how many names an arm selected*, and that is downstream of the
treatment under test. A fixed absolute cap taxes a diversifying arm and leaves
a concentrating one untouched — precisely the invariance failure recorded for
the circuit breaker in DJ-119, in a different guise.

A second defect surfaced in the same pass: the sector cap ignored positions
already held. Arm A's held NVDA (a Hold, so invisible to the risk layer) would
have pushed Information Technology to **21.86% against a 20% cap** while every
individual check passed.

## Fix

New `src/hifi/portfolio/policy.py` — `PortfolioPolicy`, the single control
point. Limits are expressed as multiples of the equal weight `1/n`, so one
knob carries the same meaning at any book width:

- `max_single_stock` = `concentration` (3.0) x equal weight, bounded to
  [2%, **10%**]. The 10% ceiling is deliberate: this is a diversified
  multi-name study, and a position above a tenth of the book would make
  arm-level returns a story about one company rather than about ensemble
  architecture.
- `max_sector` = `sector_slack` (1.5) x the largest sector's actual share of
  the buy list, floored at 25% — a genuinely sector-heavy signal set is
  allowed to be sector-heavy instead of being forced into cash.
- `min_position` = 0.25 x equal weight, held strictly below
  `max_single_stock` so the two can never invert and empty the book.

`policy.as_constraints()` is now the only place the pipeline's limit
vocabulary is assembled, so call sites cannot drift apart again.

`_apply_sector_cap` gained `existing_weights`: held names consume sector
budget, and a sector already full from holdings receives zero new allocation
rather than a negative one. `pipeline.run_pipeline` passes the current book
through automatically. Also fixed a third `int()` truncation of holdings at
`pipeline.py:169` that would have undone the DJ-121 fractional-share work.

Deliberately left alone: `_KELLY_CAP` (0.25) is a strictly looser backstop
than the 10% policy ceiling and never binds; `config/loader.py`'s
`max_position_pct` / `max_sector_exposure_pct` are dead — consumed nowhere.

## Verification (sandbox against live account state)

`scripts/simulate_next_run.py` now asserts against the policy in force rather
than copied percentages, and adds a deployment check that would have caught
the original defect.

| arm | Buys | policy | orders | deployed | largest position | largest sector |
|---|---|---|---|---|---|---|
| A | 30 | max_single 10.0%, max_sector 30.0% | 30 | **99.0%** | 4.88% | 21.83% |
| B | 18 | max_single 10.0%, max_sector 33.3% | 18 | **99.0%** | 9.54% | 19.02% |
| D | 12 | max_single 10.0%, max_sector 37.5% | 12 | **99.4%** | 7.74% | 20.55% |

All checks pass on all three arms. Before this change A deployed 4.9% and
would have breached its sector cap. 2,008 tests pass.

---

# DJ-123 — One delisted ticker took down a night

**Found:** 2026-08-18, from the first live run after DJ-120/121/122.

## What happened

**EQR (Equity Residential) was removed from Alpaca's asset universe** — not
marked untradable, deleted: `get_asset("EQR")` returns 404. Our parquet store
still held bars through 2026-08-13, so the DJ-120 coverage gate passed 98/98
and the problem only surfaced four hours later, mid-execution.

One dead ticker produced four distinct failures:

| arm | effect |
|---|---|
| **A** | 404 mid-loop aborted the cycle after **37 of 39 orders had already filled**. EXC, next in the list, was never submitted. `log_episode` never ran, so a day the account actually traded left **no decision record**. |
| **B** | unaffected — its 15 Buys did not include EQR. 15 orders, 99.4% exposure. |
| **C** | EQR position vanished: 98 -> 97 holdings. |
| **D** | EQR vanished from equity **without a matching cash credit** (cash moved +$64 while equity fell $3,862), reading as a -3.72% daily loss. The circuit breaker halted a perfectly healthy arm. |

The record/broker divergence on A is the worst of these: the equity curve moved
with no decision to attribute it to. That is strictly worse than a clean
failure.

## Fixes

- **Per-order isolation** in `execute_orders` and `run_control_strategy`. A
  rejected order is recorded with its error rather than raising, so it becomes
  visible in the report funnel as conviction that never reached the broker.
- **Episode logging is best-effort and independent** of later failures; a
  cycle that reached the broker always leaves a record.
- **Broker-side tradability pre-flight** (`check_tradability`). The data gate
  checks our store; this checks the broker, because the two can disagree.
  Reports rather than blocks — a delisting is a fact about the world, not a
  fault in the run — but it must be visible before agents spend a night
  analysing a security that cannot be bought.
- **Circuit breaker distinguishes corporate actions from losses**
  (`_vanished_position_value`). A symbol we recorded holding, now absent from
  the account *and* no longer recognised as an asset by the broker, is
  unambiguously a delisting. Its value is excluded from the daily-change
  calculation. Detection is by asset lookup rather than snapshot-date
  alignment, which is unreliable: Alpaca's `last_equity` is struck at a close
  we cannot observe, and our own snapshot may already have been rewritten
  post-removal — the state D was actually left in.

Verified against live accounts: D's raw -3.57% becomes **+0.29% -> trades**;
C's -0.36% becomes +0.57%; A and B unaffected ($0 vanished). 2,015 tests pass.

## Recovery

A's 2026-08-17 episode was reconstructed and appended, flagged
`reconstructed: true` with a note: 98 signals recovered from the ensemble
sidecars, 37 orders recovered from the broker.

## The agent layer was healthy

Worth recording separately, because it is the first clean read since DJ-120:
tool-failure rate **0.000 across all six agents in both arms**, 98/98 sidecars
carrying a collective decision, and all 9 contrarian parse failures recovered
on retry. The data-starvation fix is holding.

## New finding: the technical agent is a constant

With full data for the first time, the technical agent returned **Buy on 98/98
tickers at confidence exactly 0.70** in both arms. This is not starvation — it
reads the indicators correctly, and its rationales describe them accurately:

> AAPL: "RSI of 43.4 indicates neutral conditions. EMA 312.17 vs SMA 317.38
> shows **bearish** short-term momentum. MACD histogram of -2.290 confirms
> **negative** momentum." -> **Buy @ 0.70**

The narrative contradicts the decision. The signal field appears pinned
regardless of the evidence the model just summarised.

This was always true and was merely hidden: in the starved data the technical
agent was `Buy 285/285` whenever its tools worked. It is now fully visible
because the tools always work.

**Consequence:** a constant member contributes no information to the ensemble
and mechanically deflates disagreement. The technical agent is also what
drives arm A's 39 Buys, so the current buy list is close to a single degenerate
model's output rather than an ensemble decision. This is the next thing to
investigate — likely the Phase 11 LoRA fine-tune (max-return labels bias
toward Buy) or the prompt's output contract. Not fixed here: it needs a
deliberate look, not a quick patch.
