"""
E2-T4: Multi-round debate evaluation → OQ-D04 (P13-E2, DJ-074).

Compares herding_coefficient between 1-round (Phase 12 condition C baseline)
and 2-round debate on a subset of Phase 12 evaluation dates.

Design
------
- 5 dates × 3 tickers = 15 runs at max_rounds=2
- Baseline herding (1-round, condition C): from phase12_factorial_results.json
  mean_herding_coefficient = 0.950 (condition C: debate=True, FT=False)
- New measurement: same setup, max_rounds=2

OQ-D04: Does a second debate round reduce herding vs. one round?
Hypothesis (pre-registered): NO — herding is determined by architecture diversity,
not round count. A second round may reinforce the majority view rather than
reduce it (anchoring effect).

Scientific criteria for answer
-------------------------------
  If |mean_herding_2round - mean_herding_1round| < 0.05: NEGLIGIBLE difference.
  If mean_herding_2round < mean_herding_1round - 0.05: POSITIVE (2nd round reduces herding).
  If mean_herding_2round > mean_herding_1round + 0.05: NEGATIVE (2nd round increases herding).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hifi.agents.ensemble_runner import run_debate_ensemble  # noqa: E402
from hifi.data.schemas import FundamentalsSnapshot  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACTORIAL_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "baseline" / "phase12_factorial_results.json"
OUTPUT_PATH = REPO_ROOT / "tests" / "fixtures" / "baseline" / "phase13_debate_multiround.json"

TICKERS = ["AAPL", "JPM", "XOM"]
FETCHED_AT = "2026-06-15T00:00:00Z"

_SNAPSHOTS = {
    "AAPL": {"ticker": "AAPL", "period_end": "2022-12-31",
              "revenue": 117_154_000_000, "net_income": 29_998_000_000,
              "total_assets": 346_747_000_000, "total_liabilities": 290_437_000_000,
              "total_equity": 50_672_000_000, "eps": 1.88,
              "market_cap": 2_350_000_000_000, "source": "reference",
              "fetched_at": FETCHED_AT, "provenance": {"source": "10-Q reference", "fetched_at": FETCHED_AT}},
    "JPM":  {"ticker": "JPM", "period_end": "2022-12-31",
              "revenue": 128_695_000_000, "net_income": 37_676_000_000,
              "total_assets": 3_665_743_000_000, "total_liabilities": 3_373_000_000_000,
              "total_equity": 292_000_000_000, "eps": 12.09,
              "market_cap": 390_000_000_000, "source": "reference",
              "fetched_at": FETCHED_AT, "provenance": {"source": "10-K reference", "fetched_at": FETCHED_AT}},
    "XOM":  {"ticker": "XOM", "period_end": "2022-12-31",
              "revenue": 398_675_000_000, "net_income": 55_740_000_000,
              "total_assets": 369_067_000_000, "total_liabilities": 167_961_000_000,
              "total_equity": 168_577_000_000, "eps": 14.18,
              "market_cap": 440_000_000_000, "source": "reference",
              "fetched_at": FETCHED_AT, "provenance": {"source": "10-K reference", "fetched_at": FETCHED_AT}},
}


def _herding(decisions: list[str]) -> float:
    if not decisions:
        return 0.0
    counts = Counter(decisions)
    return counts.most_common(1)[0][1] / len(decisions)


def main() -> None:
    print("=" * 60)
    print("E2-T4: Multi-Round Debate Evaluation (OQ-D04)")
    print("=" * 60)

    # Load Phase 12 dates and baseline herding
    factorial = json.loads(FACTORIAL_FIXTURE.read_text())
    all_dates: list[str] = factorial["metadata"]["dates"]
    dates = all_dates[:5]  # First 5 of 10 quarterly dates
    baseline_herding_1round: float = factorial["conditions"]["C"]["mean_herding_coefficient"]
    print(f"Phase 12 condition C herding (1-round): {baseline_herding_1round:.4f}")
    print(f"Evaluation: {len(dates)} dates × {len(TICKERS)} tickers = {len(dates)*len(TICKERS)} runs")
    print(f"Dates: {dates}\n")

    snapshots = {t: FundamentalsSnapshot.model_validate(s).model_dump_json()
                 for t, s in _SNAPSHOTS.items()}

    results: list[dict] = []
    n_run = 0
    n_fail = 0

    for date in dates:
        for ticker in TICKERS:
            n_run += 1
            print(f"  [{n_run}/{len(dates)*len(TICKERS)}] {ticker} {date} max_rounds=2 ...", end="", flush=True)
            try:
                out = run_debate_ensemble(
                    ticker=ticker,
                    as_of_date=date,
                    snapshot_json=snapshots[ticker],
                    max_rounds=2,
                    use_rag=True,
                )
                decisions = [s.decision for s in (out.signals or []) if s.decision]
                h = _herding(decisions)
                n_rounds = len(out.debate_transcripts) if out.debate_transcripts else 1
                converged = (out.debate_transcripts[-1].converged
                             if out.debate_transcripts else True)
                print(f" decision={out.collective_decision} herding={h:.3f} rounds={n_rounds} converged={converged}")
                results.append({
                    "ticker": ticker, "as_of_date": date,
                    "collective_decision": out.collective_decision,
                    "herding_coefficient_2round": h,
                    "n_rounds_run": n_rounds,
                    "converged": converged,
                    "agent_decisions": decisions,
                })
            except Exception as exc:
                print(f" FAILED: {exc}")
                n_fail += 1
                results.append({"ticker": ticker, "as_of_date": date, "error": str(exc)})

    # Aggregate
    valid = [r for r in results if "herding_coefficient_2round" in r]
    mean_herding_2round = sum(r["herding_coefficient_2round"] for r in valid) / len(valid) if valid else None
    delta = (mean_herding_2round - baseline_herding_1round) if mean_herding_2round is not None else None

    print(f"\nAggregate:")
    print(f"  Runs completed : {len(valid)}/{n_run} (failed: {n_fail})")
    if mean_herding_2round is not None:
        print(f"  Mean herding 1-round (Phase 12 condition C): {baseline_herding_1round:.4f}")
        print(f"  Mean herding 2-round (this eval):            {mean_herding_2round:.4f}")
        print(f"  Delta (2-round minus 1-round):               {delta:+.4f}")
        if abs(delta) < 0.05:
            oq_d04 = "NEGLIGIBLE — second round does not meaningfully change herding"
        elif delta < -0.05:
            oq_d04 = "POSITIVE — second round reduces herding"
        else:
            oq_d04 = "NEGATIVE — second round increases herding (anchoring)"
        print(f"\nOQ-D04: {oq_d04}")

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phase": 13, "ticket": "E2-T4",
            "description": "Multi-round debate evaluation: herding 1-round vs 2-round",
            "dates_evaluated": dates, "tickers": TICKERS,
            "n_runs_planned": len(dates) * len(TICKERS),
            "n_runs_completed": len(valid), "n_runs_failed": n_fail,
        },
        "baseline_1round_herding_condition_c": baseline_herding_1round,
        "mean_herding_2round": mean_herding_2round,
        "delta_2round_minus_1round": delta,
        "oq_d04": oq_d04 if mean_herding_2round is not None else "INCONCLUSIVE (insufficient runs)",
        "per_run_results": results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
