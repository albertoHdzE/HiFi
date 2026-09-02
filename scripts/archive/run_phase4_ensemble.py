"""
Phase 4 ensemble baseline runner (P4-E5-T1).

Runs the full two-agent ensemble (Fundamental + Technical) for AAPL, JPM, and
XOM at Q1 2023 and saves the output to tests/fixtures/baseline/phase4_ensemble.json.

Usage
-----
    uv run python scripts/run_phase4_ensemble.py [--data-dir DATA_DIR]

Requires
--------
- LM Studio running at HIFI_LM_STUDIO_URL (default http://localhost:1234/v1)
  with both the fundamental model (qwen2.5-coder-32b-instruct-mlx) and the
  technical model (HIFI_TECHNICAL_MODEL, default DJ-016) loaded.
- Phase 1 market/macro Parquet files in DATA_DIR (default: data/).

Output
------
tests/fixtures/baseline/phase4_ensemble.json with the format:
{
  "metadata": { phase, models, prompt_versions, data_as_of, run_date, hifi_commit },
  "outputs": { ticker: EnsembleOutput dict, ... },
  "metrics": { fundamental_compliance_rate, technical_compliance_rate, ... }
}

Notes
-----
Reference snapshots for AAPL/JPM/XOM use publicly reported Q1 2023 values from
10-Q filings (same as scripts/run_phase3_baseline.py).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from hifi.agents.ensemble_runner import run_ensemble  # noqa: E402
from hifi.collective.metrics import compute_ensemble_metrics  # noqa: E402
from hifi.data.schemas import FundamentalsSnapshot  # noqa: E402

_AS_OF = "2023-03-31"
_FETCHED_AT = datetime(2023, 4, 1)

_REFERENCE_SNAPSHOTS: dict[str, dict] = {
    "AAPL": {
        "ticker": "AAPL",
        "period_end": "2022-12-31",
        "revenue": 117_154_000_000,
        "net_income": 29_998_000_000,
        "total_assets": 346_747_000_000,
        "total_liabilities": 290_437_000_000,
        "total_equity": 50_672_000_000,
        "eps": 1.88,
        "market_cap": 2_350_000_000_000,
        "source": "reference",
        "fetched_at": _FETCHED_AT.isoformat(),
        "provenance": {
            "source": "10-Q reference",
            "fetched_at": _FETCHED_AT.isoformat(),
        },
    },
    "JPM": {
        "ticker": "JPM",
        "period_end": "2023-03-31",
        "revenue": 38_349_000_000,
        "net_income": 12_622_000_000,
        "total_assets": 3_744_305_000_000,
        "total_liabilities": 3_454_000_000_000,
        "total_equity": 290_000_000_000,
        "eps": 4.10,
        "market_cap": 400_000_000_000,
        "source": "reference",
        "fetched_at": _FETCHED_AT.isoformat(),
        "provenance": {
            "source": "10-Q reference",
            "fetched_at": _FETCHED_AT.isoformat(),
        },
    },
    "XOM": {
        "ticker": "XOM",
        "period_end": "2023-03-31",
        "revenue": 86_564_000_000,
        "net_income": 11_432_000_000,
        "total_assets": 376_317_000_000,
        "total_liabilities": 163_567_000_000,
        "total_equity": 168_577_000_000,
        "eps": 2.79,
        "market_cap": 440_000_000_000,
        "source": "reference",
        "fetched_at": _FETCHED_AT.isoformat(),
        "provenance": {
            "source": "10-Q reference",
            "fetched_at": _FETCHED_AT.isoformat(),
        },
    },
}


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _build_snapshot_json(raw: dict) -> str:
    snap = FundamentalsSnapshot.model_validate(raw)
    return snap.model_dump_json()


def run_baseline(data_dir: str) -> None:
    output_path = _ROOT / "tests" / "fixtures" / "baseline" / "phase4_ensemble.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    technical_model = os.environ.get(
        "HIFI_TECHNICAL_MODEL",
        "mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled",
    )
    fundamental_model = "qwen2.5-coder-32b-instruct-mlx"

    print(f"Running Phase 4 ensemble baseline for AAPL, JPM, XOM at {_AS_OF}")
    print(f"Fundamental model: {fundamental_model}")
    print(f"Technical model:   {technical_model}")
    print(f"Data directory:    {data_dir}")
    print(f"Output:            {output_path}")
    print()

    ensemble_outputs = {}
    for ticker, raw in _REFERENCE_SNAPSHOTS.items():
        print(f"  Running ensemble for {ticker} ...", flush=True)
        snapshot_json = _build_snapshot_json(dict(raw))
        output = run_ensemble(
            ticker=ticker,
            as_of_date=_AS_OF,
            snapshot_json=snapshot_json,
            data_dir=data_dir,
        )
        ensemble_outputs[ticker] = output

        fa_sig = output.fundamental_analysis.signal
        ta_sig = output.technical_analysis.signal
        fund_dec = fa_sig.decision if fa_sig else "FAILED"
        tech_dec = ta_sig.decision if ta_sig else "FAILED"
        collective = output.ensemble_decision.collective_decision or "NONE"
        latency = f"{output.latency_ms:.0f}ms"
        print(
            f"    {ticker}: fundamental={fund_dec}, technical={tech_dec}, "
            f"collective={collective} ({latency})"
        )

    metrics = compute_ensemble_metrics(ensemble_outputs)
    print()
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    payload = {
        "metadata": {
            "phase": "4",
            "models": {
                "fundamental": fundamental_model,
                "technical": technical_model,
            },
            "prompt_versions": {
                "fundamental": "fundamental_v1",
                "technical": "technical_v1",
            },
            "data_as_of": _AS_OF,
            "run_date": date.today().isoformat(),
            "hifi_commit": _git_sha(),
        },
        "outputs": {
            ticker: o.model_dump() for ticker, o in ensemble_outputs.items()
        },
        "metrics": metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print()
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 ensemble baseline.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")),
        help="Path to the data root directory (default: data/).",
    )
    args = parser.parse_args()
    run_baseline(args.data_dir)


if __name__ == "__main__":
    main()
