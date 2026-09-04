# Adversarial verification of the DJ-135 cleanup

**Date:** 2026-09-01 / 09-02 / 09-03
**Commits:** `8a89672`, `d4ed914`, `b2aa316`, `ab81571`, `379ace8`, `ec0b999`
**Verdict:** the apparatus is sound and two live defects were found and fixed.
**Rung 4 — the full 97-ticker DRY nightly — ran on 2026-09-02 and passed.**
Genesis III is cleared; see *Remaining before Genesis III*.

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

## Rung 4 — the full 97-ticker DRY nightly

`make live-nightly DRY=1`, started 2026-09-02 21:28:38 ET — after the close, as
the market-hours guard requires — and finished **rc=0 at 04:41:12** on 09-03.
7h12m. Log: `data/live/logs/verify_20260902.log`.

What the smoke run could not exercise, and this one did:

- **97/97 data coverage** and **97/97 tradability**. The DJ-120 gate ran against
  the real universe, not a 22-ticker subset; every bar resolved through
  2026-09-02.
- **1,164 agent passes** — 97 tickers × 6 agents × 2 LLM arms — with
  `done=97 skip=0 fail=0` for every agent in both arms, and `fail=0` on the
  aggregate step. **Zero ERROR, CRITICAL or Traceback lines in the whole log.**
- **The full sector-cap arithmetic** in the allocator, at 97 names rather than
  22. It bound in all three trading arms and reported doing so:

  | arm | BUY demand | buying power | scale | orders |
  |---|---|---|---|---|
  | A | $89,017.69 | $9,730.56 | 0.109 | 7 |
  | B | $65,480.49 | **$9.94** | 0.000 | 1 (a sell) |
  | D | $43,172.53 | $9,572.52 | 0.222 | 11 |

  Arm B is fully invested and can currently only sell. That is the allocator
  behaving as specified against a depleted cash balance, not a defect — and it
  is exactly the state the Genesis III reset exists to clear. It is recorded
  here because an arm that can only sell is not producing the same experiment as
  one that can buy, and nothing in the log says so on its own.
- **Arm invariance probe (DJ-119)** across the real universe: A 9 positions /
  90.2% exposure / halt at 20%; B 12 / 100.0% / 24%; C 97 / 99.0% / 196%;
  D 12 / 90.2% / 27%. The thresholds move with exposure, as DJ-122 requires.
- **Shadow replay**: `[A]` and `[B]`, 97 tickers, **0 baseline mismatches**.
- **DJ-136 confirmed at full scale.** `decisions.jsonl` stayed at exactly four
  rows per arm (2026-08-24 … 08-27), `equity.jsonl` at four; `dry_runs.jsonl`
  gained **exactly one row per arm** for `decision_date` 2026-09-02, every one
  flagged `"dry_run": true`. Nothing about the seven-hour run entered the
  experimental record.

`make live-verify DATE=2026-09-02`: **PASS**

| agent | n | modal share | was | unique conf | was |
|---|---|---|---|---|---|
| fundamental | 194 | 88% | 100% | 4 | 2 |
| technical | 194 | 60% | 75% | 4 | 5 |
| risk | 194 | 63% | 80% | 6 | 6 |
| macro | 194 | 53% | 92% | 7 | 6 |
| sentiment | 194 | 87% | 100% | 5 | 2 |

Ratio coverage **192/194 (99%)**, against 0/97 on 2026-08-27; threshold 90%.
Ensemble unanimity **8/97 (8%)**, against 59%.

The two uncovered passes are both **APD**, one per arm, and both are correct.
APD's TTM diluted EPS is 0.02 + 3.04 + 3.19 − 6.47 = **−0.22**: earnings are
negative, so P/E is undefined and the agent is told so rather than shown a
fabricated number. `pb` (4.96), `ps` (5.47) and `ev_ebitda` (51.2) were all
delivered, and `roe` is −0.003, consistent with the same loss.

This does expose a small imprecision in the *metric*, not the pipeline:
`_ratio_coverage` (`scripts/verify_agent_repair.py:102`) counts a pass as blind
whenever `pe`, `pb` or `ps` appears in `data_gaps`, which conflates *a ratio that
was not delivered* with *a ratio that does not exist for this company*. At 99%
it does not change the verdict. It would matter if the universe ever held enough
loss-making names to push the measured coverage under the 90% floor, and the
gate would then fail for the wrong reason. Left as-is: the plan's scope was to
verify the apparatus, not to redesign the verifier.

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

## Freezing the signal path before Genesis III (DJ-141, DJ-142)

Once a generation starts, an edit to the code the nightly cycle runs is a
protocol deviation *inside* it — the failure that voided Phase 15 and the A/B/D
record. So the remaining cleanup was split on one question: does it change what
the cycle executes? Two bodies of work did, and both landed before the reset.

### DJ-141 — seven readers of one file, and one of them dated the cycle 1970

`data/market/<TICKER>/ohlcv.parquet` was read from seven places, each
normalising it itself, disagreeing about where the date lives:

| contract | readers |
|---|---|
| date in the index, reset to a column | `live/cycle`, `execution/riskbudget_strategy`, `execution/market_data` |
| date in the index, left there | `live/walkforward`, `live/market._last_completed_session` |
| neither; `.iloc[-1]`, unsorted | `live/market._latest_prices` |
| both shapes, explicitly | `mcp/financial_server` |

All seven worked against the file as written — so six worked by accident. The
pinning tests were written **first** and run against the real store before any
source change. Two failed, both with the same value:

- **`_last_completed_session`** took `read_parquet(p).index.max()` and stamped
  every arm's decision with it. Against a store whose date sits in a column that
  index is a RangeIndex, its max is an integer, and `pd.Timestamp(5712)` is
  1970-01-01. **Measured: `'1970-01-01'` where `'2026-09-02'` was expected** —
  for the whole cycle, all four arms, nothing raised.
- **`live/walkforward`** filtered `df.index <= as_of_date` on that RangeIndex
  and returned everything or nothing, also silently.

`_latest_prices` never sorted, and its number sizes orders (DJ-126). Four of the
seven bypassed `resolve_ohlcv_path` and hard-coded the nested path, so they
reported "no data" exactly where the other two found a legacy fixture — **arms
disagreeing about whether a ticker exists.**

That is DJ-120's shape one layer up: its fix centralised where a ticker's bars
are *found*; how they are *read* stayed scattered.
`hifi.data.market_store.load_ohlcv_frame` is now the one normaliser, and it
refuses an integer date column rather than letting `pd.to_datetime` turn 0, 1, 2
into 1970-01-01.

One regression the existing suite caught: moving the handler from around the
whole loop to per-ticker made corrupt stores skip **silently**, because
`pyarrow.ArrowInvalid` subclasses `ValueError`. Per-ticker isolation is right —
one bad file used to blind a whole sweep — but silence is exactly DJ-120. Every
skip now logs.

### DJ-142 — 99 type errors to 32, none on the signal path

`hifi/live`, `hifi/mcp`, `hifi/portfolio`, `hifi/execution`, the six agents,
`hifi/engines`, `simulation/{agent_executor,snapshot,model_manager}` and
`collective/voting` are clean. The 32 remaining are all offline: reporting,
knowledge graph, training data, EDGAR's network fetch, and two in the untracked
paper modules.

Two were defects, not annotations:

- **`BaseMessage.content` is `str | list[str | dict]`.** All seven call sites
  passed it into `extract_json`, which begins `text.strip()`. A provider
  answering with content *blocks* would have raised AttributeError inside the
  agent's own try/except and been recorded as a parse failure naming neither the
  cause nor the text — the identical shape to the list-from-`json.loads` case
  fixed at DJ-140, one layer earlier. Local LM Studio models return strings;
  that is a property of the serving stack, not of the interface.
- **MCP tool results, LM Studio responses and `book_state.json`** were read with
  `json.loads` and handed to callers that immediately `.get()`. Each now refuses
  a non-object at the boundary.

`_test_llm` was `object | None` in all six agents and in `agent_executor`, which
joined with the `make_llm()` branch to `object` — three errors per agent, and it
was **masking the content-union finding underneath.** `ChatModel` is now a
Protocol for what the nodes actually call.

`simulation/snapshot` passed a `str` to `period_end`, declared `date` — the
field that decides which fundamentals an agent may see. Pydantic coerced it;
it is now explicit.

### Freeze checks

- `decisions.jsonl` and `equity.jsonl` unchanged at 4 rows per arm across a
  `make live-plan`; only arm C's `dry_runs.jsonl` grew, exactly as DJ-136
  documents.
- `make lint` clean, `make typecheck` runs, `make live-status` and
  `make live-plan` run against the live accounts.
- 3,237 tests pass, 6 skipped.

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

1. ~~Full 97-ticker DRY nightly~~ — **done 2026-09-02, PASS.** It ran at 21:28
   ET; an earlier attempt at 09:49 ET was correctly refused by the market-hours
   guard, which is itself a verification of `test_nightly_wrapper.py` against
   the real clock.
2. ~~Confirm `decisions.jsonl` gained no row and `dry_runs.jsonl` gained four~~
   — **confirmed.** Four decision rows per arm before and after; one dry row per
   arm added.

3. ~~Archive `data/live/<ARM>/` and `data/live/_dj136_backup/`~~ — **done
   2026-09-03**, `data/live/_genesis2_archive/`, 81 MB, all four arms verified
   byte-identical with `cmp`, originals untouched.
4. ~~Freeze the signal path~~ — **done**: DJ-141 and DJ-142 above. The frozen
   commit is tagged **`genesis-3`**.

Nothing technical now gates Genesis III. What remains is Alberto's to trigger,
in this order:

5. Reset the four Alpaca accounts to $100,000 **together**. Arm B in particular
   has $9.94 of buying power and can only sell until this happens.
6. `scripts/genesis_reset.sh --clear --generation 2 --genesis-date <first
   decision date>` — refuses without the archive, refuses a marker that moves
   backwards, and advances `genesis_date.txt`, which nothing else writes.
7. `make live-nightly`, after 16:00 ET.
8. OSF amendment 002.
