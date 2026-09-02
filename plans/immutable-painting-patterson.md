# Adversarial verification of the DJ-135 cleanup

## Context

The DJ-135 cleanup (`02dccd9`, `45c5437`, `b1d3413`) moved 2,340 lines of the
running experiment out of `scripts/` into `hifi.live`, deleted two orphan
modules, archived 30 one-shot scripts, and removed two silent-substitution
hazards. It was verified by 2,622 passing tests, clean lint, and a
byte-identical `make live-plan`.

That is weaker evidence than it looks, and reconnaissance for this plan found
why. **Coverage has never been measured on this project** — `pytest-cov` is
declared in `[project.optional-dependencies].dev` but the environment resolves
`[dependency-groups].dev`, which omits it, so "2,622 tests pass" says nothing
about what fraction of the code they execute. `mypy` is configured in
`[tool.mypy]` and is likewise uninstalled and invoked by nothing. Seven modules
have **zero tests naming any of their public symbols**. No test anywhere
exercises `run_account_cycle` or `run_batch` — the 379-line spine of the
nightly cycle. The shell wrapper that guards the experiment's timing protocol is
untested.

And probing turned up a live defect that pre-dates the cleanup and that my own
verification runs made worse (below). The purpose of this pass is to replace
"the tests pass" with "we know what is and is not exercised, and the gaps are
either closed or named."

**Outcome:** measured coverage with a floor on the live path, the seven blind
modules tested, an integration test that runs a whole cycle against a mock
broker, the nightly wrapper's guard tested, and a graduated end-to-end run
ending in the full DRY nightly that gates Genesis III.

---

## Step 0 — DJ-136: dry runs contaminate the experimental record

**Found during reconnaissance. Fix before anything else; it currently blocks a
correct Genesis III start.**

`run_account_cycle` (`src/hifi/live/cycle.py:349-360`) calls `log_episode` and
`record_account` unconditionally, after the `is_dry` branches. So a run that
places no orders still appends a full decision row and an equity snapshot,
structurally indistinguishable from a day the arm actually traded.

Measured now:

| Arm | `decisions.jsonl` rows | contaminated | dates |
|---|---|---|---|
| A | 5 | 1 | 2026-08-31 |
| B | 5 | 1 | 2026-08-31 |
| C | 8 | **3** | 2026-09-01 ×3 |
| D | 5 | 1 | 2026-08-31 |

Two distinct leaks. `--no-execute` (the `DRY=1` verification path) logs all four
arms. `--dry-run` (`make live-plan`) logs **arm C only** — A/B/D return early at
`cycle.py:333`, but the control arm falls through — and because
`already_decided` is skipped when `is_dry`, every invocation appends another
row. Arm C's three rows are the three times I ran `make live-plan` today.

Consequences, in order of severity:

1. `already_decided("C", "2026-09-01")` is now **True**. A real cycle tonight
   would silently skip arm C — the null model, the arm every other arm is
   measured against.
2. `already_decided(*, "2026-08-31")` is True for all four arms.
3. `live_report.signal_distribution` counts phantom decision days.
   `equity_curves` happens to survive this because it de-duplicates on index
   (`live_report.py:71`), but the decision log has no such guard.

**Fix:** a dry run must not write to the experimental record. Route dry episodes
to `data/live/<ARM>/dry_runs.jsonl` instead — the audit value of "a verification
run happened tonight" is real, so the answer is a separate file rather than
silence. Skip `record_account` entirely when `is_dry`: the broker state it
snapshots is unchanged by a run that placed no orders.

Then remove the contaminated rows from `decisions.jsonl` and `equity.jsonl` for
the four arms, so tonight's cycle is not blocked. Genesis III archives these
files anyway; this is about not starting from a false state.

Regression tests: a dry cycle writes zero rows to `decisions.jsonl` for every
one of the four conditions; a non-dry cycle writes exactly one.

## Step 1 — Make coverage measurable

`pyproject.toml` has two `dev` groups. `[project.optional-dependencies].dev`
(line 35) lists pytest-cov, ruff and mypy; `[dependency-groups].dev` (line 83)
lists pytest, responses and vcrpy, and is what the environment actually has.
Consolidate into `[dependency-groups].dev` so `uv sync` installs everything, and
point `make install` at it.

Add:

```
make coverage        # term-missing report, per-module, branch coverage on
make typecheck       # mypy; reports, does not gate (Step 6)
```

Configure `[tool.coverage.run] branch = true, source = ["src/hifi"]`. Branch
coverage matters more than line coverage here: the code is dense with
`if not is_dry`, `if dry_run`, and fail-open `except` paths, and line coverage
scores those as covered while never taking the branch that matters.

## Step 2 — Baseline the coverage, then set a floor where it counts

Run the suite under coverage and record per-module numbers as the starting
point. Then set `fail_under` **only** on the modules whose failure corrupts the
experiment rather than a report:

- `hifi/live/**` — the nightly cycle
- `hifi/mcp/portfolio_composer.py`, `hifi/portfolio/policy.py` — capital
  allocation (DJ-122, DJ-132)
- `hifi/collective/voting.py` — the collective decision itself
- `hifi/data/filing_calendar.py`, `hifi/simulation/snapshot.py` —
  point-in-time discipline

A global floor is the wrong instrument: it would be satisfied by testing
whichever module is cheapest, which is never the one that matters. The number
itself should be set from the measured baseline, not picked in advance — pick it
after Step 2 reports, and state it in the commit.

## Step 3 — Close the seven blind modules

Zero tests name any public symbol of these. Ordered by what a defect costs:

| Module | loc | Untested surface | Why it matters |
|---|---|---|---|
| `data/filing_calendar.py` | 281 | `ticker_to_cik`, `build_filing_calendar`, `load_filing_calendar`, `latest_filed_period` | **Highest.** `latest_filed_period` decides which fundamentals an agent may see on a given date. A lookahead bug here silently invalidates every result, exactly as DJ-120 did |
| `engines/macro.py` | 117 | `compute_macro_snapshot` | The macro agent's entire input. DJ-133c blinded this agent for a night and nothing failed |
| `verification/metrics.py` | 96 | `compute_verification_metrics` | Grounding/hallucination rates — numbers that go in the paper |
| `data/refresh.py` | 344 | `refresh_ticker`, `refresh_series`, `check_ohlcv_quality` | New in `b1d3413`; the merge semantics that prevent history loss are asserted nowhere |
| `data/regime.py` | 188 | `classify_regime` | DJ-130 regime context fed to agents |
| `live/walkforward.py` | 191 | `run_pipeline_mode`, `run_status_mode` | The Phase 15 re-run path, which is pending work |
| `observability/langchain_callback.py` | 91 | `HiFiLangfuseCallbackHandler` | Telemetry; fail-open, so a break is invisible |

For `filing_calendar` the tests that matter are adversarial, not smoke tests:
`latest_filed_period` must never return a period whose `filingDate` is after the
as-of date; the `_CIK_SUCCESSIONS` path (XOM) must resolve; a ticker with fewer
than `_MIN_PERIODIC_FILINGS` must fail loudly rather than return a partial
calendar.

For `data/refresh`, pin the two rules the module exists to enforce: the merge
never drops a period present only locally, and a macro write round-trips through
`read_macro` (the DJ-133c guard) — with a test that plants a bad write and
confirms it is caught.

## Step 4 — An integration test that runs a whole cycle

No test exercises `run_account_cycle` or `run_batch`. Add
`tests/integration/test_live_cycle.py` using the mock-executor pattern already
in `tests/unit/execution/test_phase16_live.py:18` (`_mock_executor`), with
`tmp_path` redirection through `hifi.live.paths` (the module-qualified
references from `45c5437` make one patch point sufficient).

Cover, per condition:

- `control` — buys the universe once, holds thereafter
- `riskbudget` — signals from the provider, through the pipeline, to orders
- `parallel` / `full` — ensemble signals loaded from stored sidecars (fixtures,
  no LLM), through the pipeline

And the interactions that only appear in the whole cycle:

- a tripped breaker suppresses orders but still records equity (DJ-119)
- `_halt_before_submit` firing between signal generation and submission (DJ-129c)
- HWM ratchets before anything trades, and never falls (DJ-129b)
- one arm raising does not abort the others (DJ-117)
- a rejected order is recorded, not raised (DJ-123)
- **Step 0's property:** dry cycles leave `decisions.jsonl` untouched

## Step 5 — Test the nightly wrapper's guard

`scripts/nightly_live_execute.sh` enforces the experiment's timing protocol —
decisions on completed closes, fills at the next open — and nothing tests it.

`--check-window` is pure and exits 0/1, so it is testable by shimming `date` on
`PATH` in a `tmp_path` bin directory. Assert: refuses inside 09:30–16:00 ET;
allows evenings; allows weekends (DJ-121) with the "last completed session"
message; warns but proceeds pre-market; `ALLOW_MARKET_HOURS=1` overrides.

Add a static check that the wrapper's `--no-execute` path and its `--execute`
path differ **only** by that flag — the invariant `make live-nightly DRY=1`
depends on.

## Step 6 — mypy: measure, report, do not gate

Install mypy, run it across `src/`, and report the error count and where errors
cluster. Do not fix and do not add to `make lint` in this pass. On 27,336 lines
that have never been type-checked the count could be in the hundreds, and that
is a separate decision to take with the number in hand.

## Step 7 — Graduated end-to-end

Each rung is only attempted if the previous one passed.

1. **`make test` + `make coverage`** — full suite, coverage at or above the
   Step 2 floor.
2. **Every read-only Makefile target executes**: `live-plan`, `live-status`,
   `live-snapshot`, `refresh-data --check`, `archive-help`, `walkforward-status`.
   Plus an import-integrity check over every module in `src/` and `scripts/`
   (including `scripts/archive/`, whose repo-root resolution changed in
   `b1d3413`).
3. **Smoke DRY nightly** — `--smoke` (22 tickers), all four arms, real agents,
   no orders (~1.5 h). Then `verify_agent_repair.py` on that date. This is the
   first rung where model routing, LangFuse binding and the data gate are
   actually exercised.
4. **Full DRY nightly** — 97 tickers, four arms, `make live-nightly DRY=1`
   (~5–6.5 h), followed by `make live-verify DATE=<that date>` against the
   pre-declared thresholds. This is the gate for Genesis III.

Rungs 3 and 4 must run **after** Step 0, or they will contaminate the record
again — which is itself the cleanest demonstration that Step 0's fix works.

## Step 8 — Report

`doc/VERIFICATION_DJ136.md`: coverage per module before and after, what each new
test pins and which defect it descends from, the mypy count, the end-to-end
results, and — explicitly — **what remains unverified and why**. A verification
report that claims completeness is the same failure as a test suite that passes
without executing anything.

---

## Verification of this work

- `make test` — full suite green, no new skips. Any test that skips must state a
  reason naming an absent external dependency, never a data condition that
  silently makes it vacuous.
- `make coverage` — meets the Step 2 floor; the seven modules in Step 3 move
  from 0% to covered.
- `make typecheck` — runs and reports (non-gating).
- `make live-plan` — still byte-identical to the pre-cleanup capture.
- **`decisions.jsonl` gains no row from any dry run**, checked before and after
  rungs 3 and 4.
- `make live-verify DATE=<full DRY date>` — PASS on the pre-declared thresholds
  (`MAX_MODAL_SHARE 0.95`, `MIN_UNIQUE_CONFIDENCES 3`, `MIN_RATIO_COVERAGE 0.90`)
  for both `parallel` and `full`.

## Scope boundaries

- **The four untracked paper modules** (`simulation/diversity.py`,
  `simulation/synthetic_collective.py`, `scripts/analyze_paper1_diversity.py`,
  `tests/unit/simulation/test_diversity.py`) are **measured and reported, never
  modified.** Another process was writing them during the last session; a
  concurrent edit is how work gets lost.
- No changes to arm definitions, model configuration, risk limits or any
  `data/` layout. This pass verifies the apparatus; it does not redesign it.
- Genesis III itself, and OSF amendment 002, remain yours to trigger.
