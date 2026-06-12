"""
Phase 10 calibration: compare bootstrap vs real-LLM weights (P10-E4-T1).

Loads data/agent_performance_history.json, separates bootstrap records
(analysis_date in 2018-2022) from real LLM records (2023+), calls
labeler.build_calibration_report(), saves data/calibration_report.json,
and prints the weight comparison table.

Usage: uv run python scripts/run_phase10_calibration.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hifi.collective.labeler import build_calibration_report  # noqa: E402
from hifi.collective.performance_store import load_history  # noqa: E402
from hifi.collective.schemas import CalibrationReport  # noqa: E402

_HISTORY_FILE = "agent_performance_history.json"
_CALIBRATION_OUT = _ROOT / "data" / "calibration_report.json"

# Method records are needed for divergence rates; load from the accuracy fixture
# if available, else use an empty list (divergence rates will all be 0.0).
_ACCURACY_FIXTURE = _ROOT / "tests" / "fixtures" / "baseline" / "phase10_accuracy.json"


def _load_method_records():
    if not _ACCURACY_FIXTURE.exists():
        return []
    from hifi.collective.schemas import MethodAccuracyReport
    data = json.loads(_ACCURACY_FIXTURE.read_text())
    report = MethodAccuracyReport.model_validate(data)
    return report.records


def _print_weight_table(report: CalibrationReport) -> None:
    all_types = sorted(
        set(report.bootstrap_weights)
        | set(report.real_label_weights)
        | set(report.combined_weights)
    )

    print("\nWeight comparison (accuracy per agent type):")
    header = f"  {'agent_type':<18} {'bootstrap':>10} {'real_llm':>10} {'combined':>10}"
    print(header)
    print("  " + "-" * 52)
    for agent_type in all_types:
        bw = report.bootstrap_weights.get(agent_type, float("nan"))
        rw = report.real_label_weights.get(agent_type, float("nan"))
        cw = report.combined_weights.get(agent_type, float("nan"))
        print(f"  {agent_type:<18} {bw:>10.4f} {rw:>10.4f} {cw:>10.4f}")

    print(f"\n  n_bootstrap_labeled : {report.n_bootstrap_labeled}")
    print(f"  n_real_labeled      : {report.n_real_labeled}")

    if report.divergence_rates:
        print("\nPairwise method divergence rates:")
        for pair, rate in sorted(report.divergence_rates.items()):
            print(f"  {pair:<20} {rate:.4f}")


def run_calibration(data_dir: str) -> None:
    print("Phase 10 Calibration: Bootstrap vs Real-LLM Weight Analysis")
    print("=" * 60)

    history_path = Path(data_dir) / _HISTORY_FILE
    if not history_path.exists():
        print(f"ERROR: Performance history not found: {history_path}")
        print("Run first: make bootstrap  (or make baseline-phase10 for real records)")
        sys.exit(1)

    history = load_history(data_dir)
    all_records = history.records
    print(f"Total records in history: {len(all_records)}")

    # Separate bootstrap (2018-2022) from real LLM records (2023+)
    bootstrap_records = [r for r in all_records if r.analysis_date < "2023-01-01"]
    real_records = [r for r in all_records if r.analysis_date >= "2023-01-01"]

    print(f"  Bootstrap records (2018-2022): {len(bootstrap_records)}")
    print(f"  Real LLM records  (2023+):     {len(real_records)}")

    # Load method records for divergence rate computation
    method_records = _load_method_records()
    if method_records:
        print(f"  Method records (for divergence): {len(method_records)}")
    else:
        print("  No method records found — divergence rates will be 0.0")
        print("  (run make baseline-phase10 first to generate phase10_accuracy.json)")

    report = build_calibration_report(
        bootstrap_records=bootstrap_records,
        real_records=real_records,
        method_records=method_records,
    )

    _CALIBRATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    _CALIBRATION_OUT.write_text(report.model_dump_json(indent=2))
    print(f"\nCalibration report saved: {_CALIBRATION_OUT}")

    _print_weight_table(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(_ROOT / "data"),
        help="Root data directory (default: <repo>/data)",
    )
    args = parser.parse_args()
    run_calibration(args.data_dir)


if __name__ == "__main__":
    main()
