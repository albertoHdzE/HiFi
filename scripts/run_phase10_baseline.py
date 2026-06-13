"""
Phase 10 baseline: accuracy labeling + tear sheets (P10-E5, no LM Studio required).

Reads the Phase 9 collective fixture (tests/fixtures/baseline/phase9_collective.json),
labels each method's decision against the 60-day and 20-day forward returns, and
writes the Phase 10 accuracy fixture. Also computes QuantStats tear sheets for all
four methods and saves JSON summaries to data/tearsheets/.

Usage: uv run python scripts/run_phase10_baseline.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hifi.collective.labeler import (  # noqa: E402
    build_method_accuracy_report,
    label_agent_decisions,
    label_method_decisions,
)
from hifi.collective.performance_store import update_and_save  # noqa: E402
from hifi.collective.schemas import EnsembleOutput  # noqa: E402

_FIXTURE_IN = _ROOT / "tests" / "fixtures" / "baseline" / "phase9_collective.json"
_FIXTURE_OUT = _ROOT / "tests" / "fixtures" / "baseline" / "phase10_accuracy.json"


def _load_ohlcv_map(data_dir: str, tickers: list[str]) -> dict:
    import pandas as pd
    ohlcv_map = {}
    for ticker in tickers:
        files = sorted(_glob.glob(str(Path(data_dir) / "market" / f"{ticker}_*.parquet")))
        if not files:
            continue
        df = pd.read_parquet(files[-1])
        if "date" in df.columns:
            df = df.set_index("date")
        elif "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
        ohlcv_map[ticker] = df
    return ohlcv_map


def run_baseline(data_dir: str) -> None:
    print("Phase 10 Baseline: Accuracy Labeling + Tear Sheets")
    print("=" * 60)

    if not _FIXTURE_IN.exists():
        print(f"ERROR: Phase 9 fixture not found: {_FIXTURE_IN}")
        print("Run first: make baseline-phase9  (requires LM Studio)")
        sys.exit(1)

    raw = json.loads(_FIXTURE_IN.read_text())
    # phase9_collective.json: {"metadata":..., "outputs":{"AAPL":{...},...}, ...}
    if isinstance(raw, dict) and "outputs" in raw:
        raw_outputs = raw["outputs"]
        output_list = list(raw_outputs.values()) if isinstance(raw_outputs, dict) else raw_outputs
    elif isinstance(raw, list):
        output_list = raw
    else:
        output_list = [raw]
    outputs = [EnsembleOutput.model_validate(item) for item in output_list]

    tickers = sorted({o.ticker for o in outputs})
    print(f"Tickers:    {', '.join(tickers)}")
    print(f"Outputs:    {len(outputs)}")

    # Label at both horizons
    records_60 = label_method_decisions(outputs, data_dir, horizon_days=60)
    label_method_decisions(outputs, data_dir, horizon_days=20)  # secondary horizon, not used here

    # Label agent decisions (for performance history update)
    agent_records = label_agent_decisions(outputs, data_dir, horizon_days=60)
    if agent_records:
        update_and_save(agent_records, data_dir)
        print(f"Agent records added to performance history: {len(agent_records)}")

    # Build accuracy report (primary horizon 60d)
    report = build_method_accuracy_report(records_60)
    _FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_OUT.write_text(report.model_dump_json(indent=2))
    print(f"\nAccuracy fixture: {_FIXTURE_OUT}")

    # Print method accuracy table
    print("\nMethod accuracy (2023-03-31 baseline, horizon=60d):")
    header = f"  {'method':<25} {'correct':>7} {'total':>6} {'accuracy':>9}"
    print(header)
    print("  " + "-" * 55)
    for method_name in sorted(report.accuracy_by_method):
        acc = report.accuracy_by_method[method_name]
        method_recs = [r for r in records_60 if r.method_name == method_name
                       and r.outcome_correct is not None]
        n_correct = sum(1 for r in method_recs if r.outcome_correct)
        n_total = len(method_recs)
        print(f"  {method_name:<25} {n_correct:>7} {n_total:>6} {acc:>9.3f}")

    # Tear sheets
    ohlcv_map = _load_ohlcv_map(data_dir, tickers)
    if ohlcv_map:
        from hifi.analytics.tearsheet import compute_all_tearsheets
        tearsheets_dir = Path(data_dir) / "tearsheets"
        tearsheets_dir.mkdir(parents=True, exist_ok=True)
        summaries = compute_all_tearsheets(records_60, ohlcv_map, horizon_days=60)

        print(f"\nTear sheet summary (horizon=60d, {len(tickers)} tickers):")
        print(
            f"  {'method':<25} {'sharpe':>8} {'sortino':>8}"
            f" {'max_dd':>8} {'cagr':>8} {'win_rate':>9}"
        )
        print("  " + "-" * 75)
        for method_name, ts in sorted(summaries.items()):
            def _fmt(v): return f"{v:>8.4f}" if v is not None else "     N/A"
            print(f"  {method_name:<25} {_fmt(ts.sharpe_annual)}"
                  f" {_fmt(ts.sortino_annual)} {_fmt(ts.max_drawdown)}"
                  f" {_fmt(ts.cagr)} {_fmt(ts.win_rate)}")
            out_path = tearsheets_dir / f"{method_name}_summary.json"
            out_path.write_text(ts.model_dump_json(indent=2))

        print(f"\nTear sheet files: {tearsheets_dir}/")
    else:
        print("\nNo OHLCV data found — tear sheets skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10 baseline: accuracy + tear sheets.")
    parser.add_argument("--data-dir", default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")))
    args = parser.parse_args()
    run_baseline(args.data_dir)


if __name__ == "__main__":
    main()
