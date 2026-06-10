"""
Phase 5 verification baseline runner (P5-E6-T2, P5-E6-T3).

Loads the Phase 4 ensemble fixture (tests/fixtures/baseline/phase4_ensemble.json),
runs verify_ensemble on each ticker, and saves the results to
tests/fixtures/baseline/phase5_verification.json.

IMPORTANT: No live LM Studio instance is required. Verification operates
entirely on already-produced agent outputs stored in the Phase 4 fixture.
This is the key design property of Phase 5: measurement infrastructure that
can be run at any time without inference cost.

Usage
-----
    uv run python scripts/run_phase5_verification.py

Output
------
tests/fixtures/baseline/phase5_verification.json with the structure:
{
  "metadata": { "phase": "5", "verified_from": "phase4_ensemble.json",
                "run_date": "...", "hifi_commit": "..." },
  "reports": {
    "AAPL": { EnsembleVerificationReport dict },
    "JPM":  { ... },
    "XOM":  { ... }
  },
  "metrics": {
    "fundamental": { mean_hr, mean_gr, mean_unresolvable_rate, ... },
    "technical":   { ... },
    "ensemble":    { mean_ehr, n_contradictions_total, ... }
  }
}

Decisions recorded by this script
----------------------------------
DJ-019: Alias table coverage measured from unresolvable_rate.
DJ-021: Hallucination rate threshold confirmed or revised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hifi.collective.schemas import EnsembleOutput  # noqa: E402
from hifi.verification.metrics import compute_verification_metrics  # noqa: E402
from hifi.verification.schemas import (  # noqa: E402
    AgentVerificationReport,
    EnsembleVerificationReport,
)
from hifi.verification.verifier import verify_ensemble  # noqa: E402

_PHASE4_PATH = _ROOT / "tests" / "fixtures" / "baseline" / "phase4_ensemble.json"
_OUTPUT_PATH = _ROOT / "tests" / "fixtures" / "baseline" / "phase5_verification.json"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _print_agent_metrics(label: str, metrics: dict) -> None:
    print(f"\n  {label}:")
    print(f"    mean_hallucination_rate : {metrics['mean_hallucination_rate']:.4f}")
    print(f"    mean_grounding_rate     : {metrics['mean_grounding_rate']:.4f}")
    print(f"    mean_unresolvable_rate  : {metrics['mean_unresolvable_rate']:.4f}")
    print(f"    alias_table_coverage    : {metrics['alias_table_coverage']:.4f}")
    print(f"    total_claims            : {metrics['total_claims']}")
    print(f"    total_verified          : {metrics['total_verified']}")
    print(f"    total_hallucinated      : {metrics['total_hallucinated']}")
    print(f"    total_unresolvable      : {metrics['total_unresolvable']}")
    print(f"    n_flagged_high_hr       : {metrics['n_flagged_high_hr']}")


def run_verification() -> None:
    if not _PHASE4_PATH.exists():
        print(f"ERROR: {_PHASE4_PATH} not found.")
        print("Run: uv run python scripts/run_phase4_ensemble.py first.")
        sys.exit(1)

    print(f"Loading Phase 4 fixture: {_PHASE4_PATH}")
    with open(_PHASE4_PATH, encoding="utf-8") as f:
        phase4 = json.load(f)

    tickers = list(phase4["outputs"].keys())
    print(f"Tickers: {tickers}")
    print()

    ensemble_reports: dict[str, EnsembleVerificationReport] = {}
    fund_reports: dict[str, AgentVerificationReport] = {}
    tech_reports: dict[str, AgentVerificationReport] = {}

    for ticker in tickers:
        output_dict = phase4["outputs"][ticker]
        output = EnsembleOutput.model_validate(output_dict)

        print(f"  Verifying {ticker} ...", flush=True)
        report = verify_ensemble(output, always_verify=True)

        ensemble_reports[ticker] = report
        fund_reports[ticker] = report.fundamental_report
        tech_reports[ticker] = report.technical_report

        fr = report.fundamental_report
        tr = report.technical_report
        print(
            f"    fundamental: claims={fr.n_claims} verified={fr.n_verified} "
            f"hallucinated={fr.n_hallucinated} unresolvable={fr.n_unresolvable} "
            f"HR={fr.hallucination_rate:.3f}"
        )
        print(
            f"    technical  : claims={tr.n_claims} verified={tr.n_verified} "
            f"hallucinated={tr.n_hallucinated} unresolvable={tr.n_unresolvable} "
            f"HR={tr.hallucination_rate:.3f}"
        )
        if report.contradictions:
            print(f"    contradictions: {len(report.contradictions)} found!")
        else:
            print("    contradictions: 0")

    # Aggregate metrics
    fund_metrics = compute_verification_metrics(fund_reports)
    tech_metrics = compute_verification_metrics(tech_reports)

    # Ensemble-level summary
    total_contradictions = sum(r.n_contradictions for r in ensemble_reports.values())
    mean_ehr = (
        sum(r.ensemble_hallucination_rate for r in ensemble_reports.values())
        / len(ensemble_reports)
    )
    n_triggered = sum(1 for r in ensemble_reports.values() if r.triggered_by_disagreement)

    ensemble_summary = {
        "mean_ensemble_hallucination_rate": round(mean_ehr, 6),
        "total_contradictions": total_contradictions,
        "n_triggered_by_disagreement": n_triggered,
        "n_tickers": len(ensemble_reports),
    }

    print("\nMetrics:")
    _print_agent_metrics("Fundamental", fund_metrics)
    _print_agent_metrics("Technical", tech_metrics)
    print("\n  Ensemble:")
    for k, v in ensemble_summary.items():
        print(f"    {k}: {v}")

    # DJ-019 coverage assessment
    fund_cov = fund_metrics["alias_table_coverage"]
    tech_cov = tech_metrics["alias_table_coverage"]
    print(f"\nDJ-019 alias table coverage: fundamental={fund_cov:.3f}, technical={tech_cov:.3f}")
    if fund_cov < 0.90 or tech_cov < 0.90:
        print("  WARNING: Coverage below 0.90. Extend FIELD_ALIAS_TABLE in extractor.py.")
    else:
        print("  Coverage meets goal (>= 0.90). DJ-019 confirmed: regex + alias table sufficient.")

    # Build output payload
    payload = {
        "metadata": {
            "phase": "5",
            "verified_from": "phase4_ensemble.json",
            "run_date": date.today().isoformat(),
            "hifi_commit": _git_sha(),
        },
        "reports": {
            ticker: r.model_dump()
            for ticker, r in ensemble_reports.items()
        },
        "metrics": {
            "fundamental": fund_metrics,
            "technical": tech_metrics,
            "ensemble": ensemble_summary,
        },
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nSaved to {_OUTPUT_PATH}")


if __name__ == "__main__":
    run_verification()
