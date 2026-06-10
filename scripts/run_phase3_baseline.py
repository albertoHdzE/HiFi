"""
Phase 3 baseline runner (P3-E4-T2, P3-E5-T1).

Runs the Fundamental Analyst Agent for AAPL, JPM, and XOM at Q1 2023
and saves the full baseline output to tests/fixtures/baseline/phase3_baseline.json.

Usage
-----
    uv run python scripts/run_phase3_baseline.py [--data-dir DATA_DIR]

Requires
--------
- LM Studio running at HIFI_LM_STUDIO_URL (default http://localhost:1234/v1)
  with qwen2.5-coder-32b-instruct-mlx loaded and serving requests.
- Phase 1 market/macro Parquet files in DATA_DIR (default: data/).
  The MCP server subprocess reads these files.

Output
------
tests/fixtures/baseline/phase3_baseline.json with the format:
{
  "metadata": { phase, model, prompt_version, data_as_of, run_date, hifi_commit },
  "analyses": { ticker: FundamentalAnalysis dict, ... },
  "metrics": { compliance_rate, hallucinated_numbers, ... }
}

Notes
-----
Reference snapshots for AAPL/JPM/XOM use publicly reported Q1 2023 values from
10-Q filings (approximate; sufficient for establishing the LLM behaviour baseline).
These are NOT substitutes for the Phase 1 fetched data. When Phase 1 data is
available in data/fundamentals/, replace with loaded snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

# Resolve project root so the script can be run from any directory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hifi.agents.baseline_metrics import compute_metrics  # noqa: E402
from hifi.agents.fundamental_agent import run_analysis  # noqa: E402
from hifi.data.schemas import FundamentalsSnapshot  # noqa: E402

_AS_OF = "2023-03-31"
_FETCHED_AT = datetime(2023, 4, 1)

# Reference snapshots: approximate Q1 2023 values from public 10-Q filings.
# Monetary values in USD. Sufficient precision for LLM interpretation baseline.
_REFERENCE_SNAPSHOTS: dict[str, dict] = {
    "AAPL": {
        "ticker": "AAPL",
        "period_end": "2022-12-31",  # Q1 FY2023 (Apple fiscal Q1 ends Dec)
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
    prov = raw.pop("provenance")
    raw["provenance"] = prov
    snap = FundamentalsSnapshot.model_validate(raw)
    return snap.model_dump_json()


def run_baseline(data_dir: str) -> None:
    output_path = _ROOT / "tests" / "fixtures" / "baseline" / "phase3_baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running Phase 3 baseline for AAPL, JPM, XOM at {_AS_OF}")
    print(f"Data directory: {data_dir}")
    print(f"Output: {output_path}")
    print()

    analyses = {}
    for ticker, raw in _REFERENCE_SNAPSHOTS.items():
        print(f"  Analysing {ticker} ...", flush=True)
        snapshot_json = _build_snapshot_json(dict(raw))
        analysis = run_analysis(
            ticker=ticker,
            as_of_date=_AS_OF,
            snapshot_json=snapshot_json,
            data_dir=data_dir,
        )
        analyses[ticker] = analysis
        decision = analysis.signal.decision if analysis.signal else "FAILED"
        latency = f"{analysis.latency_ms:.0f}ms" if analysis.latency_ms else "N/A"
        print(f"    {ticker}: {decision} ({latency})")

    metrics = compute_metrics(analyses)
    print()
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Determine model_id from first valid signal
    model_id = next(
        (a.signal.model_id for a in analyses.values() if a.signal is not None),
        "unknown",
    )
    prompt_version = next(
        (a.prompt_version for a in analyses.values()),
        "fundamental_v1",
    )

    payload = {
        "metadata": {
            "phase": "3",
            "model": model_id,
            "prompt_version": prompt_version,
            "data_as_of": _AS_OF,
            "run_date": date.today().isoformat(),
            "hifi_commit": _git_sha(),
        },
        "analyses": {ticker: a.model_dump() for ticker, a in analyses.items()},
        "metrics": metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print()
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 baseline evaluation.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")),
        help="Path to the data root directory (default: data/).",
    )
    args = parser.parse_args()
    run_baseline(args.data_dir)


if __name__ == "__main__":
    main()
