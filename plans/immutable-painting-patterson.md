# Freeze the signal path, then start Genesis III

## Context

The previous plan in this file — the adversarial verification of the DJ-135
cleanup — is **complete**. All eight steps ran, plus four remediations beyond
it: DJ-136 (dry runs contaminated `decisions.jsonl`), DJ-137 (arm D's provider
was in no manifest), DJ-138 (the generation reset became a tested instrument),
DJ-139 (the retired LoRA was still listening on :1235). Rung 4, the full
97-ticker DRY nightly, passed on 2026-09-02 with `live-verify` PASS. The
apparatus is verified. `doc/VERIFICATION_DJ136.md` records what was and was not
checked.

**What this plan is for.** Genesis III has not started. Once it does, every edit
to the code the nightly cycle executes is a protocol deviation *inside a live
generation* — the failure mode that voided the Phase 15 result and the A/B/D
record, and the reason `doc/bitacora/live-run-protocol-deviations` exists. So
the remaining cleanup splits on one question: **does it change what the nightly
cycle executes?** If yes, it lands before the reset. If it only adds tests or
config, it can land after.

Two bodies of work are in the "before" half.

### 1. Six readers of the same file, three incompatible contracts

`data/market/<TICKER>/ohlcv.parquet` is read by six call sites that each
re-implement the normalisation, and they do not agree about where the date is:

| reader | feeds | assumes the date is |
|---|---|---|
| `live/cycle.py:105` | the MCP pipeline, all four arms | in the **index** — resets it to a column |
| `execution/riskbudget_strategy.py:47` | arm D's price history | same, spelled with a list comprehension instead of `.str.lower()` |
| `execution/market_data.py:104` | the nightly refresh that **writes** the file | same |
| `live/walkforward.py:35` | the Phase 15 sweep | **stays** in the index — the opposite assumption |
| `live/market.py:82` | `_latest_prices` → **order sizing** | neither; takes `.iloc[-1]` with **no sort** |
| `mcp/financial_server.py:_load_raw_ohlcv` | the MCP tools and `refresh.check_ohlcv_quality` | handles both shapes explicitly — the only one that does |

Measured on disk today: a `DatetimeIndex` named `Date`, capitalised
`Open/High/Low/Close/Volume`, sorted ascending, and **no HiFi metadata block**
(so `hifi.data.storage.read_ohlcv` raises `ValueError` on these files — which is
why `financial_server` has a fallback path at all).

All six therefore work today, each for a different reason. The exposure:

- If `data/refresh.py` or `market_data.py` ever wrote the date as a column,
  `live/walkforward.py` would read the **integer** index as epoch dates and its
  `df.index <= as_of_date` filter would silently return everything or nothing.
  No error, wrong data.
- `_latest_prices` never sorts, so an unsorted write hands the allocator a stale
  close — and that number sizes orders.
- Four of the six bypass `market_store.resolve_ohlcv_path` and hard-code the
  nested path, so they see "no data" exactly where `financial_server` sees a
  stale-but-present legacy fixture. **Different arms would disagree about
  whether a ticker has data.**

That last one is DJ-120's shape. DJ-120 was "five call sites independently
globbed the flat pattern"; the fix centralised *path resolution* in
`hifi.data.market_store` — whose own docstring says "This module is the single
place that knows how to find a ticker's bars" — but left *frame normalisation*
scattered. The same defect, one layer up. Nothing pins the on-disk layout.

### 2. 44 mypy errors on the signal path

`make typecheck` reports 105 errors (down from 165 at DJ-140). 44 are on modules
the nightly cycle executes. The clusters, verified individually:

- **17 × `"object" has no attribute "invoke"/"model_name"`** across all six
  agents. Identical shape: `llm` is assigned in an `if/elif/else` whose
  `_test_llm` branch is untyped, so mypy joins the branches to `object`.
  `make_llm` (`agents/lm_client.py:45`) returns `ChatOpenAI`. Annotation only.
- **5 × `simulation/snapshot.py`** — four `float(...)` on a pandas cell typed as
  a union including `date`/`bytes`, and `period_end=as_of_date` passing a `str`
  where `FundamentalsSnapshot.period_end` is `date` (`data/schemas.py:170`).
  Pydantic coerces the ISO string, so it works — on the field that decides
  point-in-time discipline.
- **5 × the `Hashable`/`Index` cluster** (`live/cycle.py:105`,
  `execution/market_data.py:104`, `execution/riskbudget_strategy.py:47`,
  `live/walkforward.py:37,41`) — these are the *same lines* body 1 consolidates,
  so one fix closes both.
- **The rest**: `data/market.py`, `macro.py`, `regime.py`, `refresh.py`,
  `market_store.py`, `execution/market_data.py`, `data/news.py`,
  `simulation/agent_executor.py`, `model_manager.py`.

`execution/market_data.py:65` and `data/news.py:229` are the same
`Model | dict[str, Any]` alpaca-py pattern already solved at DJ-140 by
`_model()` in `execution/alpaca_executor.py` — that helper should move somewhere
shared and be reused rather than reinvented.

---

## Step 1 — Pin the OHLCV layout, then consolidate the readers

**Pin first.** `tests/unit/data/test_ohlcv_layout.py`, written against the real
store and passing before any source change, so it documents current behaviour
rather than the behaviour being introduced:

- The canonical file has a `DatetimeIndex` named `Date`, capitalised OHLCV
  columns, is monotonically increasing, and carries no HiFi metadata block.
- All six readers, given the same ticker, agree on the last bar's date and close.
- `_latest_prices(["AAPL"])` equals the close of the maximum date, not
  `.iloc[-1]` of whatever order the file happens to be in — the property that is
  true today only by accident.
- A fixture written with the date as a *column* is read identically by every
  reader (this fails before the consolidation; it is the regression test).

**Then consolidate.** Add to `hifi/data/market_store.py` — the module that
already owns "where are this ticker's bars", reusing its `resolve_ohlcv_path`:

```python
def load_ohlcv_frame(ticker, data_dir=None) -> pd.DataFrame:
    """The one normalised frame: lowercase columns, `date` as a sorted column."""
```

Its body is the shape-handling already proven in
`mcp/financial_server.py:_load_raw_ohlcv`, lifted rather than rewritten. Then
repoint all six call sites at it, deleting their local normalisation. Two
behaviour changes fall out, both intended and both to be stated in the commit:

1. The four hard-coded readers now go through `resolve_ohlcv_path`, so they gain
   the legacy-fixture fallback the other two already had. Arms stop disagreeing
   about whether a ticker has data.
2. `_latest_prices` sorts before taking the last close.

## Step 2 — Clear the 44 signal-path mypy errors

Ordered so the shared fixes come first:

1. Move `_model()` and `_num()` out of `execution/alpaca_executor.py` into
   `hifi/execution/alpaca_types.py`; reuse at `execution/market_data.py:65` and
   `data/news.py:229`.
2. Annotate `llm: ChatOpenAI` in the six agents (`fundamental`, `technical`,
   `risk`, `macro`, `sentiment`, `contrarian`) — one pattern, six files.
3. `simulation/snapshot.py`: make the `period_end` conversion explicit with
   `date.fromisoformat(as_of_date)` so a malformed date fails at the boundary
   instead of inside Pydantic, and narrow the four `float(...)` cell reads.
4. The remainder, module by module. Anything that turns out to be a real defect
   rather than an annotation gets its own regression test, as `_num()` and
   `QueryOrderStatus` did at DJ-140.

`make typecheck` must report **zero** errors for `hifi/live/**`, `hifi/mcp/**`,
`hifi/portfolio/**`, `hifi/execution/**`, the six agent modules, and
`simulation/{agent_executor,snapshot,model_manager}.py`. The remaining ~61 in
offline modules stay reported and ungated — `agents/ensemble_runner.py` (14) is
**not** on the nightly path; `live/ensemble.py` reaches the agents through
`simulation/agent_executor`, and `voting.py`'s only mention of
`ensemble_runner` is a docstring.

## Step 3 — Freeze and tag

- Full suite green, `make lint` clean, `make typecheck` clean on the signal path.
- `make live-plan` runs and writes only to `dry_runs.jsonl` (the DJ-136
  property), verified before and after.
- Annotated tag **`genesis-3`** on the frozen SHA, its message carrying the
  verification numbers: the 2026-09-02 DRY nightly result, coverage, the mypy
  signal-path count. Generation 2 has no tag, which is why "which code produced
  this night" is currently answered by reading git log against timestamps.
- Update `doc/VERIFICATION_DJ136.md` with Steps 1–2 and the tag.

## Step 4 — The handoff

Nothing here is optional or reorderable.

1. **Alberto** resets all four Alpaca paper accounts to $100,000, together.
2. `scripts/genesis_reset.sh --clear --generation 2 --genesis-date <first
   decision date>` — refuses without the complete archive (already made at
   `data/live/_genesis2_archive/`), refuses a marker that moves backwards, and
   advances `genesis_date.txt`, which nothing else writes.
3. **Alberto** runs `make live-nightly` after 16:00 ET.
4. Next morning: confirm fills, positions >> 1 per arm, and that arm B — at
   $9.94 of buying power today, able only to sell — can buy again.

## Step 5 — After Genesis III has started (adds no runtime code)

1. `scripts/check_coverage_floors.py` over `coverage json`, wired into
   `make coverage`. `fail_under` in coverage.py is global only, so per-module
   floors need this; a global floor is satisfied by testing whichever module is
   cheapest, which is never the one that matters. Floors from measured values
   on `hifi/live/**`, `mcp/portfolio_composer.py`, `portfolio/policy.py`,
   `collective/voting.py`, `data/filing_calendar.py`, `simulation/snapshot.py`.
2. Raise coverage on the worst modules, in cost-of-defect order:
   `execution/market_data.py` 17%, `live/ensemble.py` 20%,
   `observability/langchain_callback.py` 32%, `simulation/agent_executor.py`
   37%, `data/edgar.py` 54%, `data/news.py` 54%.
3. The ~61 remaining offline mypy errors.

---

## Verification

- `make test` — full suite green. New skips must name an absent external
  dependency, never a data condition that makes the test vacuous.
- `make lint` — clean. It passed for the first time at DJ-139; keep it that way.
- `make typecheck` — zero on the signal-path modules listed in Step 2.
- `tests/unit/data/test_ohlcv_layout.py` — passes before Step 1's source change
  (documenting today's behaviour) and after (proving the consolidation preserved
  it), with the date-as-column case failing before and passing after.
- `make live-plan` — `decisions.jsonl` unchanged, `dry_runs.jsonl` +1 per arm.
- `make live-status` and `make live-snapshot` — still run against the live
  accounts.
- `git tag -v genesis-3` resolves, and its SHA is the one `make live-nightly`
  runs.

## Scope boundaries

- **The untracked paper work is not touched**: `simulation/diversity.py`,
  `simulation/synthetic_collective.py`, `scripts/analyze_paper1_diversity.py`,
  `tests/unit/simulation/test_diversity.py`, `doc/PAPPERS/**`,
  `plans/PHASE_22_PLAN_V2.md`, and the modification to
  `simulation/metrics.py`. A concurrent edit is how work gets lost. One was
  swept into a commit at DJ-139 and backed out; do not repeat that.
- No changes to arm definitions, model configuration, risk limits, universe or
  `data/` layout. Step 1 changes how the layout is *read*, never what is
  written.
- Genesis III itself and OSF amendment 002 remain Alberto's to trigger.
