"""
run_phase12_baseline.py -- Phase 12 minimal baseline (P12-E5-T1).

Runs all 4 conditions on 3 tickers at 1 date (2021-06-30) to populate
tests/fixtures/baseline/phase12_baseline.json. This is the fast structural
check analogous to Phase 9's baseline; the full 120-run evaluation is
run_phase12_evaluation.py.

Requires LM Studio at localhost:1234. Fine-tuned servers are optional —
conditions B and D are skipped if HIFI_TECHNICAL_FINETUNE_URL is not set.

Output:
  tests/fixtures/baseline/phase12_baseline.json

Usage:
    uv run python scripts/run_phase12_baseline.py [--data-dir DIR]
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FIXTURE_OUT = _ROOT / "tests" / "fixtures" / "baseline" / "phase12_baseline.json"

# Single date for the baseline (mid-range of evaluation window)
_BASELINE_DATE = "2021-06-30"
_TICKERS = ["AAPL", "JPM", "XOM"]
_AGENTS = ["fundamental", "technical"]

_FETCHED_AT = "2021-06-30T00:00:00+00:00"
_SNAPSHOTS: dict[str, dict] = {
    "AAPL": {
        "ticker": "AAPL", "period_end": "2021-03-31",
        "revenue": 89_584_000_000, "net_income": 23_630_000_000,
        "total_assets": 329_840_000_000, "total_liabilities": 287_912_000_000,
        "total_equity": 66_224_000_000, "eps": 1.40,
        "market_cap": 2_200_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT},
    },
    "JPM": {
        "ticker": "JPM", "period_end": "2021-03-31",
        "revenue": 33_073_000_000, "net_income": 14_300_000_000,
        "total_assets": 3_390_417_000_000, "total_liabilities": 3_100_000_000_000,
        "total_equity": 290_417_000_000, "eps": 4.50,
        "market_cap": 450_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT},
    },
    "XOM": {
        "ticker": "XOM", "period_end": "2021-03-31",
        "revenue": 59_150_000_000, "net_income": 2_730_000_000,
        "total_assets": 354_628_000_000, "total_liabilities": 201_000_000_000,
        "total_equity": 153_628_000_000, "eps": 0.65,
        "market_cap": 250_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT},
    },
}


def main() -> None:
    print("Phase 12 Baseline: GraphRAG + Structured Debate")
    print("=" * 60)
    print(f"  Date    : {_BASELINE_DATE}")
    print(f"  Tickers : {', '.join(_TICKERS)}")
    print(f"  Agents  : {', '.join(_AGENTS)}")

    # Determine which conditions to run based on server availability
    ft_url = os.environ.get("HIFI_TECHNICAL_FINETUNE_URL", "http://localhost:1235/v1")
    ft_available = _check_server(ft_url)
    if not ft_available:
        logger.info(
            "Fine-tuned server not available at %s; running base conditions A, C only.", ft_url
        )
        conditions = ["A", "C"]
    else:
        conditions = ["A", "B", "C", "D"]

    print(f"  Conditions: {conditions}")
    print()

    from hifi.agents.ensemble_runner import run_debate_ensemble, run_ensemble
    from hifi.data.schemas import FundamentalsSnapshot
    from hifi.observability.tracing import NoOpTracer

    snapshots = {
        ticker: FundamentalsSnapshot.model_validate(snap).model_dump_json()
        for ticker, snap in _SNAPSHOTS.items()
    }

    results: dict[str, dict] = {}
    for condition in conditions:
        use_finetune = condition in ("B", "D")
        use_debate = condition in ("C", "D")
        results[condition] = {}

        for ticker in _TICKERS:
            logger.info("Running %s | %s | %s ...", condition, ticker, _BASELINE_DATE)
            try:
                def _run(_ticker=ticker, _use_debate=use_debate):
                    tracer = NoOpTracer()
                    if _use_debate:
                        output = run_debate_ensemble(
                            ticker=_ticker,
                            as_of_date=_BASELINE_DATE,
                            snapshot_json=snapshots[_ticker],
                            agents=_AGENTS,
                            tracer=tracer,
                        )
                    else:
                        output = run_ensemble(
                            ticker=_ticker,
                            as_of_date=_BASELINE_DATE,
                            snapshot_json=snapshots[_ticker],
                            agents=_AGENTS,
                            tracer=tracer,
                        )
                    return output.model_dump()

                if use_finetune:
                    from run_phase12_evaluation import _with_finetune_env
                    fund_url = os.environ.get(
                        "HIFI_FUNDAMENTAL_FINETUNE_URL", "http://localhost:1236/v1"
                    )
                    output_dict = _with_finetune_env(_run, ft_url, fund_url)
                else:
                    output_dict = _run()

                decision = output_dict.get("ensemble_decision", {}).get("collective_decision")
                has_debate = output_dict.get("debate_transcript") is not None
                logger.info("  -> decision=%s, has_debate=%s", decision, has_debate)
                results[condition][ticker] = output_dict

            except Exception as exc:
                logger.error("FAILED %s | %s: %s", condition, ticker, exc)
                results[condition][ticker] = {"error": str(exc)}

    payload = {
        "metadata": {
            "phase": "12",
            "baseline_date": _BASELINE_DATE,
            "tickers": _TICKERS,
            "agents": _AGENTS,
            "conditions_run": conditions,
            "run_date": datetime.now(UTC).isoformat(),
        },
        "outputs": results,
    }

    _FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nBaseline fixture saved: {_FIXTURE_OUT}")

    # Summary table
    print("\n--- Baseline Results ---")
    for cond in conditions:
        for ticker in _TICKERS:
            r = results.get(cond, {}).get(ticker, {})
            if "error" in r:
                print(f"  {cond}/{ticker}: ERROR")
            else:
                d = r.get("ensemble_decision", {}).get("collective_decision", "?")
                dt = "DEBATE" if r.get("debate_transcript") else "no-debate"
                print(f"  {cond}/{ticker}: {d} ({dt})")


def _check_server(url: str) -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


if __name__ == "__main__":
    main()
