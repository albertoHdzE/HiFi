"""
Phase 8 agent population baseline runner (P8-E7).

Evaluates the incremental contribution of each Phase 8 agent by running
5 ensemble configurations in sequence on AAPL, JPM, and XOM at Q1 2023:

  Config 1: ["fundamental", "technical"]          (Phase 6 baseline)
  Config 2: ["fundamental", "technical", "risk"]
  Config 3: + "macro"
  Config 4: + "sentiment"
  Config 5: all 6 agents (+ "contrarian")

For each configuration and ticker, prints:
  - disagreement_entropy H
  - n_valid_signals
  - collective_decision

Saves the full output of Config 5 (all agents) to:
  tests/fixtures/baseline/phase8_agent_population.json

Marginal contribution analysis (DJ-037):
  - Compare disagreement_entropy H across configs to see if adding an agent
    increases diversity
  - Compare n_valid_signals to confirm all agents contributed
  - Contrarian analysis fields are logged but do NOT change the decision

Requires
--------
- LM Studio running at HIFI_LM_STUDIO_URL (default http://localhost:1234/v1)
  with all required models loaded (DJ-032):
    Risk:       HIFI_RISK_MODEL       (default: google/gemma-3-4b)
    Macro:      HIFI_MACRO_MODEL      (default: qwen3.5-27b-distilled)
    Sentiment:  HIFI_SENTIMENT_MODEL  (default: qwen2.5-coder-32b-instruct-mlx)
    Contrarian: HIFI_CONTRARIAN_MODEL (default: mlx-qwen3.5-35b-a3b)
- Phase 1 market/macro Parquet files in DATA_DIR (default: data/)
- knowledge store in HIFI_KNOWLEDGE_DATA_DIR (default: data/knowledge/)

Usage
-----
    uv run python scripts/run_phase8_baseline.py [--data-dir DIR]

Output
------
tests/fixtures/baseline/phase8_agent_population.json
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

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.agents.ensemble_runner import run_ensemble  # noqa: E402
from hifi.data.schemas import FundamentalsSnapshot  # noqa: E402

_AS_OF = "2023-03-31"
_FETCHED_AT = datetime(2023, 4, 1)
_BASELINE_DIR = _ROOT / "tests" / "fixtures" / "baseline"
_OUTPUT_PATH = _BASELINE_DIR / "phase8_agent_population.json"

_CONFIGS: list[tuple[str, list[str]]] = [
    ("Config 1: fundamental+technical (Phase 6 baseline)",
     ["fundamental", "technical"]),
    ("Config 2: + risk",
     ["fundamental", "technical", "risk"]),
    ("Config 3: + macro",
     ["fundamental", "technical", "risk", "macro"]),
    ("Config 4: + sentiment",
     ["fundamental", "technical", "risk", "macro", "sentiment"]),
    ("Config 5: all 6 agents (+ contrarian)",
     ["fundamental", "technical", "risk", "macro", "sentiment", "contrarian"]),
]

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


def _check_lm_studio() -> bool:
    """Return True if LM Studio is reachable at the configured URL."""
    import urllib.request
    base = os.environ.get("HIFI_LM_STUDIO_URL", "http://localhost:1234/v1")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_baseline(data_dir: str) -> None:
    print("Phase 8 Agent Population Baseline")
    print("=" * 60)
    print(f"Date:     {_AS_OF}")
    print("Tickers:  AAPL, JPM, XOM")
    print(f"Data dir: {data_dir}")
    print()

    # Incremental evaluation: one block per config
    final_outputs: dict = {}  # populated in Config 5 (all agents)

    for config_label, agent_list in _CONFIGS:
        print(f"\n{config_label}")
        print(f"  Agents: {agent_list}")
        print()

        for ticker, raw in _REFERENCE_SNAPSHOTS.items():
            snap = FundamentalsSnapshot.model_validate(dict(raw))
            print(f"  {ticker} ... ", end="", flush=True)

            output = run_ensemble(
                ticker=ticker,
                as_of_date=_AS_OF,
                snapshot_json=snap.model_dump_json(),
                data_dir=data_dir,
                agents=agent_list,
            )

            ed = output.ensemble_decision
            print(
                f"H={ed.disagreement_entropy:.3f}  "
                f"n_signals={ed.n_valid_signals}  "
                f"decision={ed.collective_decision or 'NONE'}  "
                f"({output.latency_ms:.0f}ms)"
            )

            # Log contrarian stress test if present
            if output.contrarian_analysis is not None:
                ca = output.contrarian_analysis
                print(
                    f"    Contrarian: confidence={ca.confidence:.2f}  "
                    f"thesis={ca.alternative_thesis[:60]!r}"
                )

            # Collect Config 5 outputs for the fixture
            if "contrarian" in agent_list:
                final_outputs[ticker] = output

    # Save fixture from Config 5 (all agents)
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "phase": "8",
            "configs_evaluated": [c for _, c in _CONFIGS],
            "final_config": _CONFIGS[-1][1],
            "data_as_of": _AS_OF,
            "run_date": date.today().isoformat(),
            "hifi_commit": _git_sha(),
        },
        "outputs": {
            ticker: o.model_dump() for ticker, o in final_outputs.items()
        },
    }

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nSaved to {_OUTPUT_PATH}")
    print("\nMarginal contribution table: fill in from printed H/n_signals above.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 8 incremental agent population baseline."
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
