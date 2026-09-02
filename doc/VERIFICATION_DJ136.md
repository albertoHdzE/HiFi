# Adversarial verification of the DJ-135 cleanup

**Date:** 2026-09-01 / 09-02
**Commits:** `8a89672`, `d4ed914`, `b2aa316`, `ab81571`
**Verdict:** the apparatus is sound and two live defects were found and fixed.
Genesis III is not yet cleared — see *Remaining before Genesis III*.

---

## Why this pass existed

The DJ-135 cleanup was reported as verified by "2,622 tests pass, lint clean,
`make live-plan` byte-identical". That is weaker evidence than it sounds, and
the first thing this pass established is *why*: **coverage had never been
measured on this project.** `pytest-cov` was declared in
`[project.optional-dependencies].dev` while the environment resolves
`[dependency-groups].dev`, which omitted it. So "2,622 tests pass" said nothing
about what fraction of the code those tests execute. `mypy` was configured in
`[tool.mypy]` and had, likewise, never run.

Measured cold, the answer was 79% — and the holes were where the risk is, not
where the code is dull.

---

## Defects found

### DJ-136 — a dry run entered the experimental record

`run_account_cycle` called `log_episode` and `record_account` after the `is_dry`
branches, unconditionally. A run that placed no orders still appended a row to
`decisions.jsonl` with the same shape as a day the arm had traded.

Measured at the time of discovery:

| Arm | rows | contaminated | dates |
|---|---|---|---|
| A | 5 | 1 | 2026-08-31 |
| B | 5 | 1 | 2026-08-31 |
| C | 8 | **3** | 2026-09-01 ×3 |
| D | 5 | 1 | 2026-08-31 |

Two leaks. `--no-execute` logged all four arms; `--dry-run` logged **arm C
alone**, because A/B/D return early before the pipeline while the control arm
falls through — and since `already_decided` is skipped when dry, each
invocation appended another row. Arm C's three rows were three `make live-plan`
runs.

Consequence, in order of severity:

1. `already_decided("C", "2026-09-01")` had become `True`. **A real cycle would
   have silently skipped the control arm** — the null model every other arm is
   measured against.
2. All four arms were blocked for 2026-08-31.
3. `signal_distribution` counted days on which nothing traded.

**Fixed.** Dry episodes go to `data/live/<ARM>/dry_runs.jsonl` with
`"dry_run": true` — a separate file rather than silence, because *that a
verification run happened* is worth knowing; it is simply not the experiment.
`record_account` is skipped when dry. The record was repaired: the contaminated
rows were migrated with a note, originals backed up to
`data/live/_dj136_backup/`. All four arms now show the same four real decision
days, 2026-08-24 to 08-27, and no date is blocked.

### DJ-137 — arm D depended on a package no manifest mentioned

The smoke verification cycle ran all four arms. A, B and C completed; **D failed
with `ModuleNotFoundError: No module named 'riskbudget'`.**

`riskbudget` is arm D's entire signal source. It was never in `pyproject.toml`
and never in `uv.lock` — it was in the venv because it had once been installed
by hand from `../decon-fin/riskbudget`. The `uv sync` that added `pytest-cov`
removed it.

That is a regression I introduced, and it exposed the larger problem: **any
fresh checkout or routine `uv sync` would have silently disabled a live
experimental arm.** The failure is loud in the log and silent in the result —
nothing compares the arms that ran against the arms that were supposed to, so a
four-arm ablation quietly becomes a three-arm one and the record still looks
well-formed.

**Fixed.** Declared as a `[tool.uv.sources]` path dependency, restored, verified
(22 signals, 2 orders, `calm_exposure`, call_id recorded). A test now asserts
every arm's signal source is importable, driven off the `_ACCOUNTS` table so a
fifth arm cannot be added without one.

---

## Coverage

**79.0% → 84.6%** branch coverage. Branch, not line: this code is dense with
`if not is_dry` and fail-open `except` handlers that line coverage scores as
executed while never taking the branch that matters. DJ-136 lived on exactly
such a branch.

| module | before | after | stmts |
|---|---|---|---|
| `verification/metrics.py` | 0% | 100% | 15 |
| `live/walkforward.py` | 0% | 98% | 89 |
| `data/refresh.py` | 0% | 96% | 134 |
| `data/filing_calendar.py` | 19% | 97% | 119 |
| `engines/macro.py` | 27% | 100% | 27 |
| `live/accounts.py` | 66% | 100% | 90 |
| `live/market.py` | 62% | 100% | 41 |
| `live/guards.py` | 62% | 98% | 177 |
| `data/regime.py` | 66% | 97% | 83 |
| `live/cycle.py` | 0%* | 85% | 195 |

\* `cycle.py` had no direct tests at all before `test_live_cycle.py`.

Priority was by what a defect costs, not by what was cheapest:

- **`filing_calendar`** first. It decides which fiscal period an agent may see
  on a given date. If it ever gated on `period_end` instead of `filing_date`,
  every result in the paper would be lookahead-contaminated and nothing would
  fail — the agents would simply be a little too good. Three of its tests run
  against the built calendar and assert no filing precedes its own period end,
  every universe ticker has four filed quarters, and no quarter is double-counted.
- **`engines/macro`** is the macro agent's entire input; DJ-133c blinded it for
  a night with no error anywhere. Two tests run against the live store and
  confirm all seven series still parse.
- **`data/refresh`** plants both failures it exists to prevent — a short
  yfinance response that would truncate eight quarters to three, and a
  `df.to_parquet()` that strips the schema metadata `read_macro` requires.

Every DJ-136 regression test was **mutation-checked**: reverted against the
pre-fix code, each fails with its own message rather than a collateral error,
and passes after. A regression test that cannot fail is not a guard.

---

## What the end-to-end run showed

`bash scripts/nightly_live_execute.sh --no-execute --smoke` — 22 tickers, four
arms, real agents, no orders. 1h38m.

- **264 agent passes** completed: 44 per agent (22 tickers × 2 LLM arms) × 6 agents.
- **The pre-flight started Docker and LangFuse itself** — "LangFuse down →
  starting stack → up, tracing enabled". This is precisely what was skipped on
  2026-08-31 when a cycle ran with no telemetry at all.
- Data coverage 22/22, tradability 22/22, arm-invariance probe ran.
- **DJ-136 confirmed in production:** `decisions.jsonl` stayed at 4 rows for
  every arm; `dry_runs.jsonl` grew.
- Arm D failed → DJ-137 above, now fixed and re-verified.

`verify_agent_repair.py --date 2026-09-01`: **PASS**

| agent | n | modal share | unique conf |
|---|---|---|---|
| fundamental | 44 | 91% | 3 |
| technical | 44 | 66% | 4 |
| risk | 44 | 64% | 6 |
| macro | 44 | 64% | 4 |
| sentiment | 44 | 86% | 5 |

Ratio coverage 44/44 (100%). Ensemble unanimity 3/22 = **14%**, against 59% on
2026-08-27.

---

## Repo integrity

202 checks, all passing: every one of the 100 `src` modules imports; every
script parses; every archived script resolves the repo root through
`parents[2]` rather than the `parent.parent` that would now point at `scripts/`;
every Makefile target resolves, is declared `.PHONY`, and appears in
`make help`; nothing in `src/` reaches into `scripts/archive/`; every archived
script has an index entry; both CLIs define no functions beyond argument
parsing; `RUNBOOK.md` names no target or module path that does not exist.

The nightly wrapper's market-hours guard is tested at shell level for the first
time (30 tests, `date` shimmed onto `PATH`). The load-bearing one compares the
two invocation lines: **the dry path and the real path must differ only by
`--execute`**. `make live-nightly DRY=1` is evidence about the real run only
while that holds.

---

## mypy: measured, reported, not gated

**165 errors in 55 files.** Honestly categorised:

| count | category | assessment |
|---|---|---|
| 36 | `import-untyped` | missing library stubs (pandas, networkx, PyYAML) — not defects |
| 32 | `union-attr` in `alpaca_executor` | one false positive: alpaca-py declares `get_account() -> TradeAccount \| dict`; the code uses the correct branch |
| 36 | `no-any-return` | returning `Any` from an untyped library call — cosmetic |
| ~61 | `arg-type`, `attr-defined`, other | worth a look, none urgent |

**Zero errors of substance on `hifi/live`, `hifi/mcp` or `hifi/portfolio`.** The
only `union-attr` outside the alpaca cluster on a live-ish path is two in
`collective/debate_nodes.py`, where LangChain message content can be `str` or
`list` — a real but narrow risk in the debate path, which is not on the nightly
cycle.

`make typecheck` runs it and does not gate.

---

## What remains unverified, and why

A verification report that claims completeness is the same failure as a test
suite that passes without executing anything.

| area | coverage | why it is not closed |
|---|---|---|
| `live/ensemble.py` | 20% | The agent-execution loop. Its real test is a live run; unit tests here would mock the LLM and assert on the mock. The 264-pass smoke run **is** its integration evidence, but that is not a repeatable check. |
| `simulation/agent_executor.py` | 37% | Same reason — it is the LLM call site. |
| `data/edgar.py` | 54% | Network-bound. Partially covered by recorded fixtures. |
| `data/news.py` | 54% | Network-bound; the offline-cache path is covered, the fetch path is not. |
| `execution/market_data.py` | 17% | Alpaca bars API. |
| `observability/langchain_callback.py` | 32% | Fail-open telemetry; a break here is invisible by design, which is itself worth revisiting. |

**Not done at all:**

- **The full 97-ticker DRY nightly** (rung 4 of the plan). The smoke run covers
  the same code path at 22 tickers; what it does *not* exercise is the
  97-ticker data gate, the full sector-cap arithmetic in the allocator, and the
  5–6.5h runtime. This is the remaining gate before Genesis III.

  It was **not run** because the market-hours guard correctly refused: the
  attempt was made at 09:49 ET on 2026-09-02, inside the cash session, when the
  last OHLCV bar is a live partial. That refusal is itself a verification —
  the guard tested in `test_nightly_wrapper.py` behaving as specified against
  the real clock — but it means rung 4 must run after 16:00 ET.
- **The four untracked paper modules** (`simulation/diversity.py`,
  `simulation/synthetic_collective.py`, `scripts/analyze_paper1_diversity.py`,
  `tests/unit/simulation/test_diversity.py`) were measured and left untouched,
  as agreed. `diversity.py` reports 89%. Another process was writing them during
  this session.
- **`riskbudget` version ambiguity.** Its `pyproject.toml` says 1.1.0;
  `riskbudget.__version__` says 1.0.0, and the `strategy_meta` recorded against
  arm D's decisions carries the latter. The attribution in the experimental
  record is therefore ambiguous about which build produced which orders. Not
  fixed: HiFi does not modify that project's internals.

---

## Test skips

Four, and each names an absent external dependency rather than a data condition
that would make the test vacuous — the distinction that matters, since a test
which skips itself on missing data passes for the wrong reason:

- 1 × a script that resolves no repo root (structural, `test_repo_integrity`)
- 3 × no EDGAR MD&A in the dev LanceDB table for specific ticker/date pairs

## Remaining before Genesis III

1. **Full 97-ticker DRY nightly, after 16:00 ET**: `make live-nightly DRY=1`,
   then `make live-verify DATE=<date>`. ~5–6.5h. The guard will refuse before
   16:00 ET, correctly.
2. Confirm afterwards that `decisions.jsonl` gained no row and `dry_runs.jsonl`
   gained four.
3. Reset the four Alpaca accounts to $100,000 **together**.
4. Archive `data/live/<ARM>/` and `data/live/_dj136_backup/`.
5. `make live-nightly`.
6. OSF amendment 002 — only Alberto can file it.
