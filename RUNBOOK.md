# HiFi runbook

Operating the live experiment. For what the project *is*, see `README.md`; for
why any given decision was made, `doc/bitacora/`.

---

## The nightly cycle

One command:

```bash
make live-nightly            # places paper orders
make live-nightly DRY=1      # identical path, agents run, no orders placed
```

`DRY=1` is the only difference between a verification run and a production run.
Do not assemble a "safe" invocation by hand — calling `scripts/hifi_live.py`
directly skips the wrapper's pre-flight, and on 2026-08-31 that produced a full
cycle with **no telemetry at all**: the tracer bound to a dead LangFuse endpoint
at import and never recovered, even after the stack came up.

The run detaches, so closing the terminal is safe. Follow it with:

```bash
tail -f data/live/logs/nightly_$(date +%Y%m%d).log   # or verify_… under DRY=1
```

### Before it starts

The wrapper checks these itself; you only need to intervene if it says so.

| Dependency | Where | If missing |
|---|---|---|
| LM Studio | `localhost:1234` | The wrapper waits 3 minutes, then proceeds and every agent pass fails |
| LangFuse | `localhost:3000` | The wrapper starts Docker and the stack; fail-open, sidecars still record |
| Alpaca keys | `.env` | The arm is skipped with a warning |

### When it refuses

Launching inside the US cash session (09:30–16:00 ET) is **blocked**: the last
OHLCV bar would be a live partial and orders would fill intraday rather than at
an open. Run after 16:00 ET, or override with
`ALLOW_MARKET_HOURS=1 make live-nightly` and annotate the date as off-protocol.

Weekends are allowed. The decision date resolves to the last *completed session*
read from the OHLCV store, so a Friday-night run and a Sunday run both date to
Friday and the second is collapsed by `already_decided`.

### Timing

A cycle takes ~5–6.5 hours (six agents × 97 tickers × two LLM arms). Start it in
the evening. A pre-market start is warned about, not blocked: the inputs are
still complete closes, which is the part that matters scientifically.

---

## The four arms

| Arm | Condition | What it tests | Signals from |
|---|---|---|---|
| A | `parallel` | Independent agents, no inter-agent context | 6 LLM agents |
| B | `full` | Sequential — each agent sees its predecessors | 6 LLM agents |
| C | `control` | Null model: equal-weight buy-and-hold | none, by design |
| D | `riskbudget` | Deterministic quant baseline | external provider |

A and B differ **only** in whether agents see each other. That is the treatment;
everything else — universe, capital, limits, breakers, execution — is held
identical, and `log_arm_invariance` prints each arm's book width, exposure and
effective halt threshold on every multi-arm run so divergence is visible rather
than inferred.

Credentials come from `.env` by suffix: `_FIRST`/`_A` (A, also unsuffixed),
`_SECOND`/`_B`, `_THIRD`/`_C`, `_FOURTH`/`_D`.

---

## Where things land

```
data/live/<ARM>/
  decisions.jsonl          one row per decision date: signals, orders, equity
  circuit_breakers.jsonl   every halt and every flag
  hwm.json                 persisted high-water mark (the drawdown breaker's baseline)
  equity.jsonl             equity + positions snapshots
  walkforward/<DATE>/<condition>/<YYYY>/<MM>/<TICKER>.json   ensemble decisions
data/runs/<condition>-<DATE>-<TICKER>/<TICKER>_<agent>.json  per-agent sidecars
data/live/logs/nightly_<YYYYMMDD>.log
```

The **sidecars are the durable record**. They hold each agent's tool calls,
inputs, reasoning and decision, and they survive LangFuse being down. If you
want to know why an arm did something, start there — not with the equity curve.

---

## Verifying a run

```bash
make live-verify DATE=2026-08-31
```

Scores whether the agents are actually responding to their inputs, against
thresholds declared *before* the run (`scripts/verify_agent_repair.py`):

- **modal share ≤ 95%** — an agent emitting one answer for the whole
  cross-section is stuck, not decisive
- **≥ 3 distinct confidence values** — two means the model is pattern-matching
  its own template
- **ratio coverage ≥ 90%** — the fundamental agent must actually receive
  fundamentals

These exist because of DJ-120: when the MCP tools answered `TICKER_NOT_FOUND`,
the agents did not error. They reasoned over the absence and returned
`"no data available -- Sell"` at confidence 1.0, on 83 of 98 tickers, for a
month. **A blinded agent looks like a confident one.** Nothing downstream can
tell the difference, which is why the check is on inputs and spread rather than
on outputs.

---

## Data

```bash
make refresh-data          # fundamentals + FRED macro, merge-not-overwrite, + quality scoring
make live-update-data      # OHLCV bars through today via Alpaca
```

Everything merges: union the periods, fresh values win on overlap, existing
history is never dropped. The acquisition scripts in `scripts/archive/` do the
opposite — yfinance serves only 5–7 quarters, so a plain re-run buys the newest
quarter at the cost of the oldest.

Two rules that are load-bearing rather than stylistic:

- **Never write a macro parquet with `df.to_parquet()`.** `write_macro` embeds
  series metadata in the Parquet *schema*, and `read_macro` raises without it.
  Bypassing it once made all seven series unreadable and the macro agent voted
  Hold on 193 of 194 passes (DJ-133c). Every macro write is now round-tripped
  before it is committed.
- **OHLCV completeness reads ~96.4%, and that is correct.** The checker counts
  weekdays without subtracting market holidays. Measured on AAPL over 22.7
  years: 211 missing days, 9.31/year, against 9–10 US market holidays/year, with
  zero gaps detected. The threshold is 0.95 for that reason; `gap_count` is the
  sharper instrument.

---

## Starting a Genesis (resetting the arms)

A "Genesis" is a clean restart of all four arms from the same capital on the
same date. Reset them **together** — an arm restarted alone is not comparable to
the others, and comparability is the whole design.

`scripts/genesis_reset.sh` does steps 1 and 3. `N` is the generation being
**retired**, not the one being opened.

1. **Archive**, before anything moves:

       scripts/genesis_reset.sh --archive --generation N

   Copies every arm, the genesis marker, the DJ-136 repair backup and this
   generation's nightly/verify logs into `data/live/_genesisN_archive`. Refuses
   to overwrite an existing archive, and refuses a partial one.

2. **Reset all four Alpaca paper accounts to $100,000 — together.**

3. **Clear**, only after the accounts are actually reset:

       scripts/genesis_reset.sh --clear --generation N --genesis-date YYYY-MM-DD

   `--genesis-date` is the first decision date of the *new* generation. Refuses
   without a complete archive, and refuses a date that is not after the current
   marker.

   Removes the state tied to the old capital — `hwm.json`, `decisions.jsonl`,
   `equity.jsonl`, `portfolio_history.json`, `circuit_breakers.jsonl`,
   `book_state.json`, `dry_runs.jsonl`. **Keeps** `walkforward/` and
   `shadow_personality.jsonl`: those record what the ensemble said about a
   security on a date, which no capital reset invalidates.

   It also advances `data/live/genesis_date.txt`. Nothing else in the codebase
   writes that file — `hifi.agents.context` only reads it, to tell each agent
   how many sessions old the deployment is and whether it is in DEPLOYMENT or
   STEADY phase. Left stale it does not error; it tells the agents they are
   managing an established book on night one.

4. `make live-nightly`.
5. Record the amendment on OSF. Only Alberto can file this.

---

## Where the code is

```
src/hifi/live/          the running experiment
  paths.py              where everything lives
  accounts.py           the four arms, credentials, per-arm state
  market.py             OHLCV refresh, session date, last closes
  models.py             which model each agent runs on
  ensemble.py           the six agent passes and their aggregation
  guards.py             everything that can stop or qualify a cycle
  strategies.py         arm C, the null model
  cycle.py              one arm, one evening, end to end
  walkforward.py        the offline multi-date sweep

scripts/hifi_live.py        CLI for the nightly cycle — argparse only
scripts/hifi_walkforward.py CLI for the offline sweep — argparse only
scripts/archive/            one-shot phase scripts, kept for provenance
```

Agents, engines and MCP servers live under `src/hifi/{agents,engines,mcp}`. The
ensemble roster has exactly one definition, `hifi.agents.roster`; a test fails if
a second one appears.

---

## Things that will bite

- **A model that will not load aborts the pass.** There is no fallback and that
  is deliberate: substituting a different model succeeds silently and writes
  sidecars that look valid (DJ-124, DJ-135).
- **Position limits are derived from universe size**, never restated as
  constants. `hifi.portfolio.PortfolioPolicy` is the single control point; a
  hardcoded 5%/20%/1% stranded capital on narrow books and taxed diversified
  arms harder than concentrated ones (DJ-122).
- **A delisted ticker is data, not a crash.** Orders are submitted per-symbol so
  one dead name cannot take down an arm mid-execution (DJ-123).
- **Re-running the same date is safe.** Orders carry deterministic
  `client_order_id`s and the broker refuses duplicates (DJ-129a). `--force`
  overrides the once-per-day guard and makes the date a protocol deviation.
