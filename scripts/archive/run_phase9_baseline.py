"""
Phase 9 collective engine baseline (P9-E6).

Runs run_ensemble() for AAPL, JPM, XOM at 2023-03-31 with all four aggregation
methods active. Prints a method comparison table: decision, collective confidence,
disagreement entropy H, and opinion dispersion D per method per ticker.

If data/agent_performance_history.json exists (produced by run_phase9_bootstrap.py),
also prints rolling herding coefficient (kappa) and consensus stability (S) from
the bootstrap window for each ticker.

Saves full outputs to:
    tests/fixtures/baseline/phase9_collective.json

Requires
--------
- LM Studio running at HIFI_LM_STUDIO_URL (default http://localhost:1234/v1)
  with required models loaded (DJ-032)
- Phase 1 market/macro Parquet files in DATA_DIR (default: data/)
- knowledge store in HIFI_KNOWLEDGE_DATA_DIR (default: data/knowledge/)
- (Optional) data/agent_performance_history.json from run_phase9_bootstrap.py

Usage
-----
    uv run python scripts/run_phase9_baseline.py [--data-dir DIR]

Output
------
tests/fixtures/baseline/phase9_collective.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.agents.ensemble_runner import run_ensemble  # noqa: E402
from hifi.collective.metrics import compute_rolling_metrics  # noqa: E402
from hifi.collective.performance_store import load_history  # noqa: E402
from hifi.collective.schemas import EnsembleOutput  # noqa: E402
from hifi.data.schemas import FundamentalsSnapshot  # noqa: E402

_AS_OF = "2023-03-31"
_FETCHED_AT = datetime(2023, 4, 1)
_BASELINE_DIR = _ROOT / "tests" / "fixtures" / "baseline"
_OUTPUT_PATH = _BASELINE_DIR / "phase9_collective.json"

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

_METHODS = ("confidence_weighted", "majority", "performance_weighted", "contrarian_adjusted")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _check_lm_studio() -> bool:
    import urllib.request
    base = os.environ.get("HIFI_LM_STUDIO_URL", "http://localhost:1234/v1")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _rolling_from_bootstrap(data_dir: str) -> dict[str, dict]:
    """
    Compute rolling kappa and S per ticker from the bootstrap history.

    Groups bootstrap DecisionRecords by (ticker, analysis_date) to reconstruct
    agent_votes_per_period and the majority collective_decisions sequence.
    Returns an empty dict when the history file is absent or empty.
    """
    history = load_history(data_dir)
    if not history.records:
        return {}

    result: dict[str, dict] = {}
    tickers = sorted(set(r.ticker for r in history.records))

    for ticker in tickers:
        ticker_records = [r for r in history.records if r.ticker == ticker]
        dates = sorted(set(r.analysis_date for r in ticker_records))

        votes_per_period: list[list[str]] = []
        collective_decisions: list[str] = []

        for d in dates:
            period_votes = [r.decision for r in ticker_records if r.analysis_date == d]
            votes_per_period.append(period_votes)
            # Majority vote (plurality) as the bootstrap collective decision
            counts: dict[str, int] = {}
            for v in period_votes:
                counts[v] = counts.get(v, 0) + 1
            plurality = max(counts, key=lambda k: counts[k])
            collective_decisions.append(plurality)

        result[ticker] = compute_rolling_metrics(votes_per_period, collective_decisions)

    return result


def _print_method_table(ticker: str, output: EnsembleOutput) -> None:
    """Print the 4-method comparison table for one ticker."""
    print(f"\n  Method comparison for {ticker}:")
    print(f"  {'Method':<24} {'Decision':<8} {'CC':>6} {'H':>7} {'D':>7}")
    print(f"  {'-'*24} {'-'*8} {'-'*6} {'-'*7} {'-'*7}")

    # ensemble_decision is the canonical confidence_weighted result
    ed = output.ensemble_decision
    print(
        f"  {'confidence_weighted':<24} "
        f"{ed.collective_decision or 'NONE':<8} "
        f"{ed.collective_confidence:>6.3f} "
        f"{ed.disagreement_entropy:>7.4f} "
        f"{ed.opinion_dispersion:>7.4f}"
    )

    for method_key in ("majority", "performance_weighted", "contrarian_adjusted"):
        if method_key not in output.method_comparison:
            print(f"  {method_key:<24} (not available)")
            continue
        mc = output.method_comparison[method_key]
        flag = " [review]" if mc.review_flagged else ""
        print(
            f"  {method_key:<24} "
            f"{mc.collective_decision or 'NONE':<8} "
            f"{mc.collective_confidence:>6.3f} "
            f"{mc.disagreement_entropy:>7.4f} "
            f"{mc.opinion_dispersion:>7.4f}"
            f"{flag}"
        )


def run_baseline(data_dir: str) -> None:
    print("Phase 9 Collective Engine Baseline")
    print("=" * 60)
    print(f"Date:     {_AS_OF}")
    print("Tickers:  AAPL, JPM, XOM")
    print(f"Data dir: {data_dir}")
    print()

    outputs: dict[str, EnsembleOutput] = {}

    for ticker, raw in _REFERENCE_SNAPSHOTS.items():
        snap = FundamentalsSnapshot.model_validate(dict(raw))
        print(f"\n{ticker} ... ", end="", flush=True)

        output = run_ensemble(
            ticker=ticker,
            as_of_date=_AS_OF,
            snapshot_json=snap.model_dump_json(),
            data_dir=data_dir,
        )
        outputs[ticker] = output

        ed = output.ensemble_decision
        print(
            f"H={ed.disagreement_entropy:.4f}  "
            f"n_signals={ed.n_valid_signals}  "
            f"decision={ed.collective_decision or 'NONE'}  "
            f"({output.latency_ms:.0f}ms)"
        )

        _print_method_table(ticker, output)

        if output.contrarian_analysis is not None:
            ca = output.contrarian_analysis
            print(
                f"\n  Contrarian: confidence={ca.confidence:.2f}  "
                f"discount={ed.contrarian_confidence_discount:.3f}  "
                f"review_flagged={ed.review_flagged}"
            )
            print(f"    thesis={ca.alternative_thesis[:80]!r}")

    # Rolling metrics from bootstrap history (if available)
    rolling = _rolling_from_bootstrap(data_dir)
    if rolling:
        print("\n\nRolling metrics from bootstrap history:")
        print(f"  {'Ticker':<8} {'kappa_W5':>9} {'kappa_W10':>10} {'kappa_W20':>10}"
              f" {'S_W5':>8} {'S_W10':>9} {'S_W20':>9}")
        print(f"  {'-'*8} {'-'*9} {'-'*10} {'-'*10} {'-'*8} {'-'*9} {'-'*9}")
        for ticker in _REFERENCE_SNAPSHOTS:
            m = rolling.get(ticker, {})
            def _fmt(v):
                return f"{v:.4f}" if v is not None else "   N/A"
            print(
                f"  {ticker:<8} "
                f"{_fmt(m.get('kappa_W5')):>9} "
                f"{_fmt(m.get('kappa_W10')):>10} "
                f"{_fmt(m.get('kappa_W20')):>10} "
                f"{_fmt(m.get('stability_W5')):>8} "
                f"{_fmt(m.get('stability_W10')):>9} "
                f"{_fmt(m.get('stability_W20')):>9}"
            )
    else:
        print(
            "\nNo bootstrap history found. Run scripts/run_phase9_bootstrap.py "
            "first to seed agent_performance_history.json."
        )

    # Save fixture
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "phase": "9",
            "as_of": _AS_OF,
            "run_date": date.today().isoformat(),
            "hifi_commit": _git_sha(),
        },
        "outputs": {
            ticker: o.model_dump() for ticker, o in outputs.items()
        },
        "rolling_metrics": rolling,
    }

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nSaved to {_OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 9 collective engine baseline (requires LM Studio)."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")),
        help="Path to the market/macro data root directory (default: data/).",
    )
    args = parser.parse_args()

    if not _check_lm_studio():
        print(
            "ERROR: LM Studio not reachable. "
            "Start LM Studio and load the required models, then retry."
        )
        print(
            "Required models (DJ-032): gemma-3-4b (risk), qwen3.5-27b (macro), "
            "qwen2.5-coder-32b (sentiment), qwen3.5-35b (contrarian)."
        )
        sys.exit(1)

    run_baseline(args.data_dir)


if __name__ == "__main__":
    main()
