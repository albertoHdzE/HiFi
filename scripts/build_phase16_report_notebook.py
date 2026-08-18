"""Generate notebooks/phase16_live_report.ipynb (DJ-120).

The report is generated rather than hand-edited so its structure stays in step
with src/hifi/analytics/{live_report,decision_audit,langfuse_report}.py. Run
after changing those modules:

    uv run python scripts/build_phase16_report_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------

md("""
# HiFi Phase 16 — Live 4-Arm Paper-Trading Report

**Status: EXPERIMENT IN PROGRESS — not for performance conclusions yet.**
Paper trading only. Not investment advice. Arms have different inception dates.

| Arm | Strategy | Type |
|---|---|---|
| **A** | parallel ensemble (champion) | LLM |
| **B** | full sequential (herding contrast) | LLM |
| **C** | equal-weight buy-and-hold | null control |
| **D** | riskbudget calm_exposure | deterministic quant |

### Read section 0 first

On 2026-08-14 a data-resolution defect (DJ-120) was found: the MCP tools
globbed a legacy flat parquet layout holding 16 tickers stale since 2023-06-30,
while the canonical nested store held all 98 current to the previous close. For
the first month of live trading **83 of 98 tickers returned `TICKER_NOT_FOUND`
on every pass**, and the agents rendered that absence as conviction — the risk
agent answered `Sell` at confidence 1.0 with the rationale *"decision based
solely on absence of information"*. Macro and sentiment were starved by two
further, independent path bugs.

The consequence is that **every decision-layer statistic before the fix
measures the data layer, not the models**. Section 0 is the gate: if tool
failure rates are non-zero, nothing below it can be read as an AI result.

Layers: **0 Provenance** · **1 Financial** · **2 Tearsheets** ·
**3 Decisions** · **4 Per-stock traceability** · **5 AI-operations**.
""")

code("""
import sys
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
DATA = str(ROOT / "data")
try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")  # LangFuse creds for section 5
except Exception:
    pass

import pandas as pd, matplotlib.pyplot as plt
from hifi.analytics import live_report as lr
from hifi.analytics import decision_audit as da
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 90)
plt.rcParams["figure.figsize"] = (11, 5)

status = lr.experiment_status(DATA)
print("Repo:", ROOT)
print("Days of data:", status["days_of_data"],
      "| window:", status["first_date"], "->", status["last_date"])
print("Arms live:", status["arms_live"])
print("Live IC:", status["ic_status"])
""")

# --- Section 0 ------------------------------------------------------------
md("""
## 0 — Data provenance and agent health

**The gate.** Two questions before any interpretation: did the agents' tools
return data, and did any agent collapse to a constant?
""")

md("""
### 0a — Store coverage

`found` must be 98/98, `layout` must be `nested` for all, and `last_date`
should be the previous close. Any `flat-legacy` row is a stale 2023 fixture
shadowing the real store — the DJ-120 defect.
""")

code("""
from hifi.data.universe import PHASE14_UNIVERSE
from hifi.data.market_store import coverage_report

tickers = [e["ticker"] for e in PHASE14_UNIVERSE]
cov = pd.DataFrame(coverage_report(tickers, DATA)).T
print(f"tickers found : {int(cov['found'].sum())}/{len(cov)}")
print(f"layouts       : {cov['layout'].value_counts().to_dict()}")
print(f"last_date     : {sorted(cov['last_date'].dropna().unique())[-3:]}")
stale = cov[cov["layout"] == "flat-legacy"]
if len(stale):
    print(f"\\n!! {len(stale)} ticker(s) on the stale flat layout: {list(stale.index)}")
else:
    print("\\nAll tickers resolve to the canonical nested store.")
""")

md("""
### 0b — MCP tool-call failure rate per agent, over time

The headline diagnostic. Flat high lines are a broken data path, not a market
view. The DJ-120 fix lands as a cliff to zero; dates left of it are void for
decision-science purposes.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), sharey=True)
for ax, arm in zip(axes, ["A", "B"]):
    pm = da.provenance_matrix(arm, DATA)
    if pm.empty:
        ax.set_title(f"{arm}: no sidecars"); continue
    piv = pm.pivot(index="decision_date", columns="agent", values="failure_rate")
    piv = piv.drop(columns=[c for c in piv.columns if piv[c].isna().all()], errors="ignore")
    piv.plot(ax=ax, marker="o", ms=3)
    ax.set_title(f"Arm {arm} — tool-call failure rate"); ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=90)
plt.suptitle("Section 0b — fraction of MCP tool calls returning an error")
plt.tight_layout(); plt.show()
""")

md("""
### 0c — Did any agent collapse to a constant?

`when_tools_ok` vs `when_tools_failed` is the decisive split. An agent whose
decision changes completely between those columns is reporting on data
availability, not on the security. A constant member contributes no
information *and* mechanically suppresses measured unanimity, so this must be
clean before any Page-theorem claim is read off section 3.
""")

code("""
for arm in ["A", "B"]:
    beh = da.agent_behaviour(arm, DATA)
    if beh.empty:
        print(f"Arm {arm}: no sidecars yet."); continue
    print(f"\\n{'='*78}\\nArm {arm} — agent behaviour\\n{'='*78}")
    print(beh[["agent", "n", "mean_confidence", "tool_failure_rate", "modal_share"]]
          .to_string(index=False))
    print("\\n  decision mix, split by whether the agent's tools worked:")
    for _, r in beh.iterrows():
        print(f"    {r['agent']:<12} ok={r['when_tools_ok'] or '{}'}"
              f"   failed={r['when_tools_failed'] or '{}'}")
    deg = da.degenerate_agents(arm, DATA)
    if not deg.empty:
        print(f"\\n  !! CONSTANT AGENTS: {list(deg['agent'])}"
              f" — these contribute no information to the ensemble")
""")

# --- Section 1 ------------------------------------------------------------
md("""
## 1 — Financial layer

Alpaca's authoritative close-marked equity curve, rebased to 100.

**Exposure caveat (DJ-119).** The arms are not equally invested, so raw return,
Sharpe and drawdown partly measure capital deployment rather than signal
quality. The exposure column below must be read beside the metrics table.
""")

code("""
curves = lr.equity_curves(DATA, rebased=True)
if curves.empty:
    print("No equity data yet.")
else:
    ax = curves.plot(title="Equity curves, rebased to 100")
    ax.set_ylabel("index (start = 100)"); ax.axhline(100, color="k", lw=0.8, ls=":")
    plt.tight_layout(); plt.show()
    display(curves.tail(5))
""")

code("""
m = lr.metrics_table(DATA)
exp = {a: lr.exposure_series(a, DATA) for a in lr.ARMS}
m["exposure_pct"] = [
    round(float(e["exposure"].iloc[-1]) * 100, 1)
    if not e.empty and e["exposure"].notna().any() else None
    for e in exp.values()
]
m["n_positions"] = [
    int(e["n_positions"].iloc[-1]) if not e.empty else None for e in exp.values()
]
display(m)
if not m["enough_data"].all():
    print("Arms marked enough_data=False have <20 daily returns; "
          "history-dependent metrics are withheld rather than reported as noise.")
""")

md("""
### 1b — Halted days

Days the circuit breaker stopped an arm from trading. Flags (a position breach
too small to matter at portfolio level) are excluded — they are observations,
not interventions.
""")

code("""
for arm in lr.ARMS:
    h = lr.halted_days(arm, DATA)
    print(f"Arm {arm}: {len(h)} halted day(s)" + (f" — {list(h['date'])}" if len(h) else ""))
""")

# --- Section 2 ------------------------------------------------------------
md("""
## 2 — QuantStats tearsheet per arm

The full metric block renders from ~20 daily returns. The *monthly* and
*EOY* panels of `qs.reports.full` need at least two calendar months and raise
until then (`index 0 is out of bounds`), which would also abort the HTML —
so those are attempted separately and skipped cleanly while the record is short.

Saved to `data/live/reports/{arm}.html` once the full report succeeds.

**Interpretation warning.** At n ≈ 20 daily observations an annualised Sharpe
is descriptive, not inferential: its standard error is roughly √(252/n) ≈ 3.5,
so a printed Sharpe of 2.7 is not distinguishable from 0. The same applies to
CAGR extrapolated from three weeks. Read section 1's exposure column too — the
arms are not equally invested, so these numbers partly rank deployment.
""")

code("""
import quantstats as qs
MIN = 20
reports_dir = Path(DATA) / "live" / "reports"; reports_dir.mkdir(parents=True, exist_ok=True)

for arm in lr.ARMS:
    r = lr.load_returns(arm, DATA)
    label = lr.ARMS[arm]["label"]
    print(f"\\n{'='*70}\\nArm {arm}: {label}  ({len(r)} return days)\\n{'='*70}")
    if len(r) == 0:
        print("no returns captured yet."); continue

    # 1. Metric block — works on daily data alone.
    try:
        qs.reports.metrics(r, mode="full" if len(r) >= MIN else "basic", display=True)
    except Exception as e:
        print("metrics unavailable:", e)

    # 2. Daily-only plots — the parts of the tearsheet meaningful at this length.
    # Each is attempted separately: the rolling views use a 6-month window by
    # default and raise on a record this short, which must not suppress the
    # cumulative-return and drawdown plots that do work.
    for name, fn in [("returns", qs.plots.returns),
                     ("drawdown", qs.plots.drawdown),
                     ("rolling_volatility", qs.plots.rolling_volatility)]:
        try:
            fn(r, show=True)
        except Exception as e:
            print(f"  plot '{name}' skipped ({e}) — needs a longer record")

    # 3. Full HTML tearsheet — needs >= 2 calendar months for the monthly/EOY panels.
    months = r.index.to_period("M").nunique() if len(r) else 0
    if len(r) >= MIN and months >= 2:
        try:
            out = reports_dir / f"{arm}.html"
            qs.reports.html(r, output=str(out), title=f"HiFi Arm {arm} — {label}")
            print(f"saved full tearsheet -> {out}")
        except Exception as e:
            print("html tearsheet error:", e)
    else:
        print(f"full HTML tearsheet pending: {len(r)} return days across {months} "
              f"calendar month(s) (needs >={MIN} days and >=2 months for monthly/EOY panels)")
""")

# --- Section 3 ------------------------------------------------------------
md("""
## 3 — Decision layer

### 3a — Signal distribution over time

Each arm produces signals differently, so a blank panel is annotated rather
than left ambiguous:

- **A, B** — LLM ensemble, Buy/Hold/Sell per ticker.
- **C** — *no signal layer by design*: a buy-once-hold null model that bypasses
  the pipeline and writes `signals: []` forever. Blank here is correct.
- **D** — deterministic reason codes (section 3c).
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(19, 3.6), sharey=True)
for ax, arm in zip(axes, lr.ARMS):
    kind = lr.signal_layer_kind(arm)
    sd = lr.signal_distribution(arm, DATA)
    has = not sd.empty and sd[["Buy", "Hold", "Sell"]].to_numpy().sum() > 0
    if has:
        sd.set_index("decision_date")[["Buy", "Hold", "Sell"]].plot(
            kind="bar", stacked=True, ax=ax, legend=(arm == "A"),
            color={"Buy": "#2ca02c", "Hold": "#888888", "Sell": "#d62728"})
    else:
        msg = ("no signal layer\\n(buy-once-hold null model,\\nby design)"
               if kind == "none-by-design" else "no signals recorded")
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#666")
        ax.set_xticks([])
    ax.set_title(f"{arm}: {lr.ARMS[arm]['label'][:24]}", fontsize=9); ax.set_xlabel("")
plt.suptitle("Section 3a — signal distribution over time (Buy/Hold/Sell)")
plt.tight_layout(); plt.show()
""")

md("""
### 3b — Herding vs diversity (LLM arms)

`unanimity` = fraction of tickers where all agents agreed (Page-theorem
herding). `mean_entropy` = mean cross-agent disagreement entropy.

**Read against section 0c.** Unanimity requires *all* members to agree, so a
constant agent mechanically drives unanimity down — a diverse-looking arm may
simply contain a dead member. These numbers are only interpretable when no
agent is flagged constant and tool failure rates are zero.
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for arm in ["A", "B"]:
    h = lr.herding_series(arm, DATA)
    if h.empty:
        print(f"Arm {arm}: no ensemble sidecars yet."); continue
    print(f"\\n=== Arm {arm} ({lr.ARMS[arm]['label']}) ===")
    print(h.to_string(index=False))
    h.plot(x="decision_date", y="unanimity", ax=ax[0], marker="o", ms=3, label=arm)
    h.plot(x="decision_date", y="mean_entropy", ax=ax[1], marker="o", ms=3, label=arm)
ax[0].set_title("Unanimity (herding)"); ax[1].set_title("Mean disagreement entropy")
for a in ax:
    a.set_xlabel(""); a.tick_params(axis="x", rotation=90)
plt.tight_layout(); plt.show()
""")

md("""
### 3c — Arm D: deterministic reason codes

D is rule-based and has no ensemble, so it has no herding metric. Its `reason`
field is the deterministic analogue of the LLM rationale and belongs in the
decision layer rather than being omitted.
""")

code("""
dr = lr.deterministic_reasons("D", DATA)
if dr.empty:
    print("No reason codes recorded for arm D.")
else:
    dr.plot(kind="area", stacked=True, alpha=0.85,
            title="Arm D — riskbudget reason codes over time")
    plt.ylabel("tickers"); plt.xlabel(""); plt.xticks(rotation=90)
    plt.tight_layout(); plt.show()
    display(dr.tail(8))
""")

md("""
### 3d — Live Information Coefficient

Deferred by design: IC needs 60-trading-day forward labels (Phase 16 E3), which
do not exist for recent decisions. The report says "pending" rather than
computing a noisy proxy.

Two cautions for when it does land:

1. **Provenance.** Restrict IC to dates after the DJ-120 fix. In the Phase 15
   offline data the pre-fix IC decomposed to **−0.1377 on the 16 tickers that
   had data** and **+0.0669 on the 83 that had none** — the positive headline
   lived entirely where the agents were blind.
2. **Effective sample size.** Pairs are not independent: ~98 tickers share each
   date and 60-day forward windows overlap heavily across dates. A p-value
   computed on `n_pairs` is badly overstated; the effective n is closer to the
   number of non-overlapping dates.
""")

code("""
print("Live IC:", lr.experiment_status(DATA)["ic_status"])
""")

# --- Section 4 ------------------------------------------------------------
md("""
## 4 — Per-stock decision traceability: why, how, when

The audit trail for a single decision. For any (arm, ticker, date):

- **why** — each agent's decision, confidence, rationale, key concern, declared data gaps
- **how** — which model, which MCP tool calls, whether they succeeded, latency
- **when** — the decision date, and whether it reached the broker as an order

Every view carries tool-call health, so a failed call can never again be
mistaken for a considered bearish opinion.
""")

code("""
# Edit these three and re-run.
ARM, TICKER, DATE = "A", "ACN", None   # DATE=None -> most recent available

hist = da.ticker_history(ARM, TICKER, DATA)
if hist.empty:
    print(f"No sidecars for {ARM}/{TICKER}.")
else:
    DATE = DATE or hist["decision_date"].iloc[-1]
    print(da.format_trace(da.trace(ARM, TICKER, DATE, DATA)))
""")

md("""
### 4b — One ticker through time

`n_tool_failures` sits beside every decision on purpose: a long run of
identical Sells with a constant non-zero failure count is a broken feed, not a
sustained view.
""")

code("""
if not hist.empty:
    display(hist)
    changed = hist["ensemble"].nunique()
    print(f"{TICKER} in arm {ARM}: {len(hist)} decision days, "
          f"{changed} distinct ensemble decision(s), "
          f"{int(hist['ordered'].sum())} order(s) placed.")
    if hist["n_tool_failures"].gt(0).all():
        print("!! every decision for this ticker was made with failed tool calls "
              "— treat the whole series as uninformed")
""")

md("""
### 4c — Why do the AI arms trade so few names?

The question this report was built to answer. Three distinct filters sit
between a universe of 98 and the orders actually placed, and they must not be
conflated:

1. **Evidence** — did the agents get data at all? (section 0)
2. **Conviction** — how many tickers did the ensemble mark Buy?
3. **Allocation** — how many Buys survived the allocator's rebalance band,
   position caps and the circuit breaker?

A small order count can come from any of the three. The table separates them.
""")

code("""
rows = []
for arm in ["A", "B"]:
    sd = lr.signal_distribution(arm, DATA)
    pm = da.provenance_matrix(arm, DATA)
    if sd.empty:
        continue
    for _, r in sd.iterrows():
        d = r["decision_date"]
        day = pm[pm.decision_date == d]
        calls, failed = int(day["calls"].sum()), int(day["failed"].sum())
        rows.append({
            "arm": arm, "date": d,
            "tool_failure_rate": round(failed / calls, 3) if calls else None,
            "Buy": int(r["Buy"]), "Hold": int(r["Hold"]), "Sell": int(r["Sell"]),
            "orders": int(r["n_orders"]),
            "buys_not_ordered": int(r["Buy"]) - int(r["n_orders"]),
        })
funnel = pd.DataFrame(rows)
if funnel.empty:
    print("No signal data yet.")
else:
    display(funnel.tail(20))
    print("\\nInterpretation:")
    print("  tool_failure_rate > 0  -> stage 1: the agents were blind; Buy counts are meaningless")
    print("  Buy small, failures 0  -> stage 2: a genuine AI decision to stay out")
    print("  buys_not_ordered > 0   -> stage 3: the allocator, not the LLM, suppressed the trade")
""")

# --- Section 5 ------------------------------------------------------------
md("""
## 5 — AI-operations layer (LangFuse)

Token, latency and error telemetry per model. Requires LangFuse credentials in
`.env`; an empty table here means credentials, not an absence of activity.
""")

code("""
from hifi.analytics import langfuse_report as lf
gens = lf.fetch_generations(max_items=20000)
if gens is None or gens.empty:
    print("No LangFuse telemetry (check LANGFUSE_* credentials in .env).")
else:
    print("LLM-ops summary:", lf.llm_ops_summary(gens))
    usage = lf.model_usage_table(gens)
    # Shorten filesystem-path model ids and surface unlabelled rows explicitly.
    usage.index = [str(i).split("/")[-1] if pd.notna(i) else "(unlabelled)" for i in usage.index]
    display(usage)
    broken = usage[usage["error_rate"] > 0.5]
    if len(broken):
        print("\\n!! models failing >50% of calls:")
        for name, r in broken.iterrows():
            print(f"   {name}: {int(r['n_calls'])} calls, error_rate={r['error_rate']:.2f}")
""")

code("""
if gens is not None and not gens.empty:
    u = lf.model_usage_table(gens)
    u.index = [str(i).split("/")[-1] if pd.notna(i) else "(unlabelled)" for i in u.index]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    u["n_calls"].plot(kind="barh", ax=ax[0], title="LLM calls per model")
    ax[0].set_xlabel("calls")
    u["mean_latency_s"].plot(kind="barh", ax=ax[1], color="#d97706",
                             title="Mean latency per model (s)")
    ax[1].set_xlabel("seconds")
    plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------


def build() -> dict:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        # Stable ids keep diffs readable and satisfy nbformat >= 4.5.
        cell_id = f"hifi-{i:02d}"
        if kind == "markdown":
            cells.append({
                "cell_type": "markdown", "id": cell_id,
                "metadata": {}, "source": source,
            })
        else:
            cells.append({
                "cell_type": "code", "id": cell_id, "execution_count": None,
                "metadata": {}, "outputs": [], "source": source,
            })
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "hifi-venv", "language": "python", "name": "hifi-venv",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks" / "phase16_live_report.ipynb"
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")
