"""
Phase 13 verification baseline runner (P13-E0-T6).

Establishes HR/GR baselines for the Risk and Macro agents, and SGR baseline
for the Sentiment agent, on AAPL, JPM, XOM at 2023-03-31. This is the same
date used for Phase 5/11 baselines, enabling direct comparability.

This script REQUIRES LM Studio running with the full agent model set (DJ-032):
  Risk:      HIFI_RISK_MODEL       (default: google/gemma-3-4b)
  Macro:     HIFI_MACRO_MODEL      (default: qwen3.5-27b-distilled)
  Sentiment: HIFI_SENTIMENT_MODEL  (default: qwen2.5-coder-32b-instruct-mlx)

Output
------
tests/fixtures/baseline/phase13_verification_baseline.json

Structure:
{
  "metadata": { "phase": "13", "run_date": "...", "hifi_commit": "..." },
  "reports": {
    "AAPL": {
      "risk":      { AgentVerificationReport },
      "macro":     { AgentVerificationReport },
      "sentiment": { SentimentGroundingReport }
    },
    ...
  },
  "metrics": {
    "risk":      { mean_hr, mean_gr, ... },
    "macro":     { mean_hr, mean_gr, ... },
    "sentiment": { mean_sgr, n_signals_total, n_grounded_total }
  }
}

Usage
-----
    uv run python scripts/run_phase13_verification_baseline.py [--data-dir DIR]

Answers P13-E0 success criterion: HR/GR baselines for Risk, Macro, Sentiment
exist before fine-tuning (gates E1 Sentiment fine-tuning).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.agents.mcp_client import call_tool  # noqa: E402
from hifi.verification.metrics import compute_verification_metrics  # noqa: E402
from hifi.verification.schemas import (  # noqa: E402
    AgentVerificationReport,
    SentimentGroundingReport,
)
from hifi.verification.verifier import verify_agent, verify_sentiment_agent  # noqa: E402

_AS_OF = "2023-03-31"
_TICKERS = ["AAPL", "JPM", "XOM"]
_OUTPUT_PATH = _ROOT / "tests" / "fixtures" / "baseline" / "phase13_verification_baseline.json"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _retrieve_sentiment_context(ticker: str, data_dir: str) -> str:
    """
    Retrieve SEC filing context for one ticker using the knowledge MCP server.

    Mirrors the retrieval logic in sentiment_agent._retrieve_context so that
    verify_sentiment_agent() receives the same context the Sentiment agent used.
    """
    query = (
        f"{ticker} management outlook guidance forward-looking statements risks "
        f"revenue growth margin services"
    )
    try:
        result = call_tool(
            "retrieve_context",
            {"query": query, "ticker": ticker, "top_k": 5},
            data_dir=data_dir,
            server_module="hifi.mcp.knowledge_server",
        )
        passages = result.get("passages", [])
        if not passages:
            return ""
        lines = []
        for p in passages:
            lines.append(
                f"[{p['rank']}] {ticker} / {p['filing_type']} / "
                f"{p['section']} / {p['period']}"
            )
            lines.append(p["text"])
            lines.append("---")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("retrieve_context failed for %s: %s", ticker, exc)
        return ""


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


def run_baseline(data_dir: str) -> None:
    # Import agents lazily (avoid import-time side effects and model loads)
    from hifi.agents.macro_agent import run_macro_analysis  # noqa: PLC0415
    from hifi.agents.risk_agent import run_risk_analysis  # noqa: PLC0415
    from hifi.agents.sentiment_agent import run_sentiment_analysis  # noqa: PLC0415

    risk_reports: dict[str, AgentVerificationReport] = {}
    macro_reports: dict[str, AgentVerificationReport] = {}
    sentiment_reports: dict[str, SentimentGroundingReport] = {}

    for ticker in _TICKERS:
        print(f"\n{ticker} ({_AS_OF})")

        # --- Risk ---
        print("  Running Risk agent ...", flush=True)
        risk_analysis = run_risk_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
        risk_report = verify_agent(risk_analysis)
        risk_reports[ticker] = risk_report
        print(
            f"    Risk:  claims={risk_report.n_claims} verified={risk_report.n_verified} "
            f"hallucinated={risk_report.n_hallucinated} unresolvable={risk_report.n_unresolvable} "
            f"HR={risk_report.hallucination_rate:.3f} GR={risk_report.grounding_rate:.3f}"
        )

        # --- Macro ---
        print("  Running Macro agent ...", flush=True)
        macro_analysis = run_macro_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
        macro_report = verify_agent(macro_analysis)
        macro_reports[ticker] = macro_report
        print(
            f"    Macro: claims={macro_report.n_claims} verified={macro_report.n_verified} "
            f"hallucinated={macro_report.n_hallucinated} "
            f"unresolvable={macro_report.n_unresolvable} "
            f"HR={macro_report.hallucination_rate:.3f} GR={macro_report.grounding_rate:.3f}"
        )

        # --- Sentiment ---
        print("  Running Sentiment agent ...", flush=True)
        sentiment_analysis = run_sentiment_analysis(
            ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir
        )
        retrieved_context = _retrieve_sentiment_context(ticker, data_dir)
        sentiment_report = verify_sentiment_agent(sentiment_analysis, retrieved_context)
        sentiment_reports[ticker] = sentiment_report
        print(
            f"    Sentiment: signals={sentiment_report.n_signals} "
            f"grounded={sentiment_report.n_grounded} "
            f"SGR={sentiment_report.grounding_rate:.3f}"
        )

    # --- Aggregate metrics ---
    risk_metrics = compute_verification_metrics(risk_reports)
    macro_metrics = compute_verification_metrics(macro_reports)

    total_signals = sum(r.n_signals for r in sentiment_reports.values())
    total_grounded = sum(r.n_grounded for r in sentiment_reports.values())
    mean_sgr = (
        sum(r.grounding_rate for r in sentiment_reports.values()) / len(sentiment_reports)
        if sentiment_reports
        else 0.0
    )
    sentiment_summary = {
        "mean_sgr": round(mean_sgr, 6),
        "n_signals_total": total_signals,
        "n_grounded_total": total_grounded,
        "n_tickers": len(sentiment_reports),
    }

    print("\n\nAggregate Metrics:")
    _print_agent_metrics("Risk", risk_metrics)
    _print_agent_metrics("Macro", macro_metrics)
    print("\n  Sentiment (SGR):")
    for k, v in sentiment_summary.items():
        print(f"    {k}: {v}")

    # --- Build and save payload ---
    payload = {
        "metadata": {
            "phase": "13",
            "epic": "E0",
            "tickers": _TICKERS,
            "as_of_date": _AS_OF,
            "run_date": date.today().isoformat(),
            "hifi_commit": _git_sha(),
        },
        "reports": {
            ticker: {
                "risk": risk_reports[ticker].model_dump(),
                "macro": macro_reports[ticker].model_dump(),
                "sentiment": sentiment_reports[ticker].model_dump(),
            }
            for ticker in _TICKERS
        },
        "metrics": {
            "risk": risk_metrics,
            "macro": macro_metrics,
            "sentiment": sentiment_summary,
        },
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nSaved to {_OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 13 E0 verification baseline: HR/GR for Risk+Macro, SGR for Sentiment."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", "data"),
        help="Path to data root directory (default: $HIFI_DATA_DIR or 'data')",
    )
    args = parser.parse_args()
    run_baseline(args.data_dir)


if __name__ == "__main__":
    main()
