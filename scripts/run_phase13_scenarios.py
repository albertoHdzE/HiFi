"""
E6-T2: Scenario evaluation → Dataset Family F (P13-E6, DJ-078).

Runs all 7 Phase 13 scenarios through the ensemble and records alignment with
expected_direction. Results written to data/scenarios/{scenario_id}.json.

Scenarios (from PHASE13_SCENARIOS in src/hifi/collective/scenarios.py):
  F-001   AAPL  2020-03-16  Black Monday II            Risk-Off
  F-001b  JPM   2020-03-16  Black Monday II (banking)  Risk-Off
  F-001c  XOM   2020-03-16  COVID + oil war            Sell
  F-002   AAPL  2022-03-31  Rate shock                 Risk-Off
  F-002b  JPM   2022-03-31  Rate shock (financials)    Hold
  F-002c  XOM   2022-03-31  Russia/Ukraine energy      Buy
  F-003   AAPL  2023-02-02  Earnings beat              Buy

Uses reference fundamentals snapshots (period_end 2022-12-31) for all scenarios.
This is a methodological simplification (noted in DJ-078 limitation):
snapshot date does not match scenario date, but the Fundamental Agent's
price-based signals still respond to market data at the scenario as_of_date.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hifi.agents.roster import VOTING_AGENTS  # noqa: E402
from hifi.collective.scenarios import PHASE13_SCENARIOS, ScenarioEvaluator  # noqa: E402
from hifi.data.schemas import FundamentalsSnapshot  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIOS_DIR = REPO_ROOT / "data" / "scenarios"
OUTPUT_SUMMARY = SCENARIOS_DIR / "README.md"

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("E6-T2: Phase 13 Scenario Evaluation (Dataset Family F)")
    print("=" * 60)
    print(f"Scenarios: {len(PHASE13_SCENARIOS)}")
    for s in PHASE13_SCENARIOS:
        print(f"  {s.scenario_id:6s}  {s.ticker}  {s.as_of_date}  "
              f"expected={s.expected_direction}  regime={s.regime}")
    print()

    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    # Contrarian (qwen3.5-35b) excluded — model fails to load in LM Studio (DJ-087).
    # 5-agent subset: fundamental, technical, risk, macro, sentiment.
    agents = list(VOTING_AGENTS)
    print(f"Agents: {agents} (contrarian excluded — LM Studio load failure)")
    print()

    # Build one evaluator per ticker (snapshot_json is ticker-specific)
    evaluators: dict[str, ScenarioEvaluator] = {
        ticker: ScenarioEvaluator(
            snapshot_json=FundamentalsSnapshot.model_validate(snap).model_dump_json(),
            agents=agents,
        )
        for ticker, snap in _SNAPSHOTS.items()
    }

    scenario_results = []
    n_aligned = 0
    n_fail = 0

    for i, scenario in enumerate(PHASE13_SCENARIOS, 1):
        print(
            f"[{i}/{len(PHASE13_SCENARIOS)}] {scenario.scenario_id} "
            f"{scenario.ticker} {scenario.as_of_date} "
            f"(expected={scenario.expected_direction}) ...",
            end="",
            flush=True,
        )
        evaluator = evaluators[scenario.ticker]
        try:
            result = evaluator.run(scenario)
            aligned_str = "ALIGNED" if result.aligned else "MISALIGNED"
            print(
                f" decision={result.collective_decision} [{aligned_str}]"
            )
            if result.aligned:
                n_aligned += 1

            # Write per-scenario JSON to data/scenarios/
            scenario_path = SCENARIOS_DIR / f"{scenario.scenario_id}.json"
            scenario_path.write_text(
                json.dumps(result.model_dump(), indent=2)
            )

            scenario_results.append({
                "scenario_id": scenario.scenario_id,
                "ticker": scenario.ticker,
                "as_of_date": scenario.as_of_date,
                "regime": scenario.regime,
                "event_description": scenario.event_description,
                "expected_direction": scenario.expected_direction,
                "collective_decision": result.collective_decision,
                "aligned": result.aligned,
            })
        except Exception as exc:
            print(f" FAILED: {exc}")
            n_fail += 1
            scenario_results.append({
                "scenario_id": scenario.scenario_id,
                "ticker": scenario.ticker,
                "as_of_date": scenario.as_of_date,
                "regime": scenario.regime,
                "expected_direction": scenario.expected_direction,
                "error": str(exc),
            })

    # Summary
    n_completed = len(PHASE13_SCENARIOS) - n_fail
    alignment_rate = n_aligned / n_completed if n_completed > 0 else None

    print(f"\n{'='*60}")
    print(f"Scenarios completed : {n_completed}/{len(PHASE13_SCENARIOS)} (failed: {n_fail})")
    if alignment_rate is not None:
        print(f"Alignment rate      : {n_aligned}/{n_completed} = {alignment_rate:.3f}")

    # Per-regime alignment
    regimes: dict[str, dict[str, int]] = {}
    for r in scenario_results:
        if "error" in r:
            continue
        reg = r["regime"]
        if reg not in regimes:
            regimes[reg] = {"aligned": 0, "total": 0}
        regimes[reg]["total"] += 1
        if r["aligned"]:
            regimes[reg]["aligned"] += 1

    print("\nPer-regime alignment:")
    for reg, counts in regimes.items():
        rate = counts["aligned"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {reg:<16s}: {counts['aligned']}/{counts['total']} = {rate:.2f}")

    # Write summary JSON
    summary = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "phase": 13, "ticket": "E6-T2",
            "description": "Phase 13 scenario evaluation — Dataset Family F",
            "n_scenarios": len(PHASE13_SCENARIOS),
            "n_completed": n_completed,
            "n_failed": n_fail,
        },
        "alignment_rate": alignment_rate,
        "n_aligned": n_aligned,
        "per_regime_alignment": regimes,
        "scenarios": scenario_results,
    }
    summary_json = SCENARIOS_DIR / "scenario_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {summary_json}")
    print(f"Per-scenario JSONs: {SCENARIOS_DIR}/{{scenario_id}}.json")


if __name__ == "__main__":
    main()
