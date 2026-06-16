"""
E4-T4: Agent memory influence evaluation → OQ-M03 (P13-E4, DJ-076).

Tests whether in-context memory prefixes change agent decisions vs. no-memory baseline.

Design
------
- 10 dates × 3 tickers = 30 (date, ticker) pairs.
- agents=["fundamental", "technical"] — matches Phase 12 factorial setup for consistency.
- Each pair run twice:
    (a) memory_prefixes=None   (no history)
    (b) memory_prefixes built from AgentMemoryStore populated with 3 synthetic prior
        records per (ticker, agent_type) — alternating Buy/Hold/Sell with known
        outcome metadata.
- Synthetic priors use dates before the earliest eval date (2020-Q1 window)
  so they are always "prior" decisions, never future leakage.
- Metric (OQ-M03): fraction of (date, ticker) pairs where ≥1 agent changed decision.

OQ-M03 scientific criteria
---------------------------
  If changed_fraction >= 0.10: YES — memory has measurable influence.
  If changed_fraction < 0.10:  NEGLIGIBLE — memory prefix does not alter decisions.

Hypothesis (pre-registered): YES (weakly) — structured memory prefix creates
anchoring bias that shifts ≥10% of decisions on at least a subset of dates.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hifi.agents.ensemble_runner import run_ensemble  # noqa: E402
from hifi.collective.memory import AgentMemoryRecord, AgentMemoryStore  # noqa: E402
from hifi.data.schemas import FundamentalsSnapshot  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACTORIAL_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "baseline" / "phase12_factorial_results.json"
)
OUTPUT_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "baseline" / "phase13_memory_eval.json"
)

TICKERS = ["AAPL", "JPM", "XOM"]
AGENTS = ["fundamental", "technical"]
FETCHED_AT = "2026-06-15T00:00:00Z"

_SNAPSHOTS = {
    "AAPL": {
        "ticker": "AAPL", "period_end": "2022-12-31",
        "revenue": 117_154_000_000, "net_income": 29_998_000_000,
        "total_assets": 346_747_000_000, "total_liabilities": 290_437_000_000,
        "total_equity": 50_672_000_000, "eps": 1.88,
        "market_cap": 2_350_000_000_000, "source": "reference",
        "fetched_at": FETCHED_AT,
        "provenance": {"source": "10-Q reference", "fetched_at": FETCHED_AT},
    },
    "JPM": {
        "ticker": "JPM", "period_end": "2022-12-31",
        "revenue": 128_695_000_000, "net_income": 37_676_000_000,
        "total_assets": 3_665_743_000_000, "total_liabilities": 3_373_000_000_000,
        "total_equity": 292_000_000_000, "eps": 12.09,
        "market_cap": 390_000_000_000, "source": "reference",
        "fetched_at": FETCHED_AT,
        "provenance": {"source": "10-K reference", "fetched_at": FETCHED_AT},
    },
    "XOM": {
        "ticker": "XOM", "period_end": "2022-12-31",
        "revenue": 398_675_000_000, "net_income": 55_740_000_000,
        "total_assets": 369_067_000_000, "total_liabilities": 167_961_000_000,
        "total_equity": 168_577_000_000, "eps": 14.18,
        "market_cap": 440_000_000_000, "source": "reference",
        "fetched_at": FETCHED_AT,
        "provenance": {"source": "10-K reference", "fetched_at": FETCHED_AT},
    },
}

# Synthetic prior records injected per (ticker, agent_type).
# Dates chosen to precede the earliest eval date (2020-03-31).
# Alternating Buy/Hold/Sell with outcome metadata creates maximal conflict signal.
_SYNTHETIC_PRIORS: list[tuple[str, str, float, float, bool]] = [
    # (as_of_date,    decision, confidence, actual_60d_return, outcome_correct)
    ("2019-09-30", "Buy",  0.75,  0.05,  True),
    ("2019-12-31", "Hold", 0.60, -0.02,  False),
    ("2020-01-31", "Sell", 0.70, -0.08,  True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_store(tmp_dir: str) -> AgentMemoryStore:
    """Populate a temporary store with 3 synthetic priors per (ticker, agent_type)."""
    store = AgentMemoryStore(tmp_dir)
    for ticker in TICKERS:
        for agent_type in AGENTS:
            for date, decision, conf, ret, correct in _SYNTHETIC_PRIORS:
                rec = AgentMemoryRecord(
                    ticker=ticker,
                    as_of_date=date,
                    agent_type=agent_type,
                    decision=decision,  # type: ignore[arg-type]
                    confidence=conf,
                    actual_60d_return=ret,
                    outcome_correct=correct,
                )
                store.record(rec)
    return store


def _build_prefixes(store: AgentMemoryStore, ticker: str) -> dict[str, str]:
    """Build memory_prefixes dict for run_ensemble()."""
    return {
        agent_type: store.format_for_prompt(store.recall(ticker, agent_type, n=3))
        for agent_type in AGENTS
    }


def _agent_decisions(output) -> dict[str, str]:
    """Extract {agent_type: decision} from EnsembleOutput via signal.decision."""
    decisions: dict[str, str] = {}
    fa = output.fundamental_analysis
    if fa and fa.signal and fa.signal.decision:
        decisions["fundamental"] = fa.signal.decision
    ta = output.technical_analysis
    if ta and ta.signal and ta.signal.decision:
        decisions["technical"] = ta.signal.decision
    return decisions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("E4-T4: Agent Memory Influence Evaluation (OQ-M03)")
    print("=" * 60)

    factorial = json.loads(FACTORIAL_FIXTURE.read_text())
    dates: list[str] = factorial["metadata"]["dates"]
    print(f"Dates: {dates}")
    print(
        f"Setup: {len(dates)} dates × {len(TICKERS)} tickers × 2 runs"
        f" × {len(AGENTS)} agents = {len(dates)*len(TICKERS)*2*len(AGENTS)} LLM calls"
    )
    print()

    snapshots = {
        t: FundamentalsSnapshot.model_validate(s).model_dump_json()
        for t, s in _SNAPSHOTS.items()
    }

    results: list[dict] = []
    n_changed = 0
    n_pairs = 0
    n_fail = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = _build_store(tmp_dir)

        for date in dates:
            for ticker in TICKERS:
                n_pairs += 1
                pair_label = f"{ticker} {date}"
                print(f"\n[{n_pairs}/{len(dates)*len(TICKERS)}] {pair_label}")

                # --- (a) No memory ---
                try:
                    print("  (a) no memory ...", end="", flush=True)
                    out_a = run_ensemble(
                        ticker=ticker,
                        as_of_date=date,
                        snapshot_json=snapshots[ticker],
                        agents=AGENTS,
                    )
                    dec_a = _agent_decisions(out_a)
                    print(f" {dec_a}")
                except Exception as exc:
                    print(f" FAILED: {exc}")
                    n_fail += 1
                    results.append({"ticker": ticker, "as_of_date": date, "error_a": str(exc)})
                    continue

                # --- (b) With memory ---
                try:
                    print("  (b) with memory ...", end="", flush=True)
                    prefixes = _build_prefixes(store, ticker)
                    out_b = run_ensemble(
                        ticker=ticker,
                        as_of_date=date,
                        snapshot_json=snapshots[ticker],
                        agents=AGENTS,
                        memory_prefixes=prefixes,
                    )
                    dec_b = _agent_decisions(out_b)
                    print(f" {dec_b}")
                except Exception as exc:
                    print(f" FAILED: {exc}")
                    n_fail += 1
                    results.append({
                        "ticker": ticker, "as_of_date": date,
                        "decisions_no_memory": dec_a, "error_b": str(exc),
                    })
                    continue

                # --- Compare ---
                changed_agents = [
                    agent for agent in dec_a
                    if dec_a.get(agent) != dec_b.get(agent)
                ]
                any_changed = len(changed_agents) > 0
                if any_changed:
                    n_changed += 1

                print(
                    f"  changed={any_changed} "
                    f"({'|'.join(changed_agents) if changed_agents else 'none'})"
                )

                results.append({
                    "ticker": ticker,
                    "as_of_date": date,
                    "decisions_no_memory": dec_a,
                    "decisions_with_memory": dec_b,
                    "collective_no_memory": (
                        out_a.ensemble_decision.collective_decision
                        if out_a.ensemble_decision else None
                    ),
                    "collective_with_memory": (
                        out_b.ensemble_decision.collective_decision
                        if out_b.ensemble_decision else None
                    ),
                    "changed_agents": changed_agents,
                    "any_changed": any_changed,
                    "memory_prefixes_used": {
                        k: v[:80] + "..." if len(v) > 80 else v
                        for k, v in prefixes.items()
                    },
                })

    # Aggregate
    valid = [r for r in results if "any_changed" in r]
    changed_fraction = n_changed / len(valid) if valid else None

    print(f"\n{'='*60}")
    print("Aggregate:")
    print(f"  Pairs completed : {len(valid)}/{n_pairs} (failed: {n_fail})")
    if changed_fraction is not None:
        print(f"  Pairs where memory changed ≥1 decision: {n_changed}/{len(valid)}")
        print(f"  Changed fraction: {changed_fraction:.3f}")
        if changed_fraction >= 0.10:
            oq_m03 = "YES — memory prefix has measurable influence on agent decisions"
        else:
            oq_m03 = "NEGLIGIBLE — memory prefix does not alter decisions (< 10% change rate)"
        print(f"\nOQ-M03: {oq_m03}")
    else:
        oq_m03 = "INCONCLUSIVE (insufficient completed runs)"
        print(f"\nOQ-M03: {oq_m03}")

    output = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "phase": 13, "ticket": "E4-T4",
            "description": "Agent memory influence evaluation: no-memory vs. with-memory",
            "dates_evaluated": dates,
            "tickers": TICKERS,
            "agents": AGENTS,
            "n_synthetic_priors_per_agent_per_ticker": len(_SYNTHETIC_PRIORS),
            "synthetic_priors": [
                {"as_of_date": d, "decision": dec, "confidence": c,
                 "actual_60d_return": r, "outcome_correct": ok}
                for d, dec, c, r, ok in _SYNTHETIC_PRIORS
            ],
            "n_pairs_planned": len(dates) * len(TICKERS),
            "n_pairs_completed": len(valid),
            "n_pairs_failed": n_fail,
        },
        "n_pairs_changed": n_changed,
        "changed_fraction": changed_fraction,
        "oq_m03": oq_m03,
        "per_pair_results": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nOutput written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
