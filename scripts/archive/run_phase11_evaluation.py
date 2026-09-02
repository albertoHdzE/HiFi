"""
run_phase11_evaluation.py -- Three-tier fine-tuning evaluation (P11-E4-T1, DJ-058).

Compares base model (LM Studio port 1234) vs fine-tuned models (mlx_lm.server
ports 1235/1236) across three tiers:
  Tier 1: Individual quality -- HR and GR from Phase 5 verification layer
  Tier 2: Collective accuracy -- method-level comparison from Phase 10
  Tier 3: Diversity impact -- pairwise_diversity and disagreement_entropy delta

Output:
  data/evaluation/phase11_evaluation.json  -- FineTuneEvaluationResult per ticker
  tests/fixtures/baseline/phase11_evaluation.json  -- fixture copy

Requires:
  LM Studio running on 1234 (base model)
  mlx_lm.server running on 1235 (technical fine-tuned)
  mlx_lm.server running on 1236 (fundamental fine-tuned)

Usage:
    uv run python scripts/run_phase11_evaluation.py [--tickers AAPL,JPM,XOM]
                                                     [--analysis-date 2023-03-31]
                                                     [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Reference fundamentals snapshots (same as Phase 9 baseline, 2023-03-31 vintage)
_FETCHED_AT = datetime(2023, 3, 31, tzinfo=UTC).isoformat()
_REFERENCE_SNAPSHOTS: dict[str, dict] = {
    "AAPL": {
        "ticker": "AAPL", "period_end": "2022-12-31",
        "revenue": 117_154_000_000, "net_income": 29_998_000_000,
        "total_assets": 346_747_000_000, "total_liabilities": 290_437_000_000,
        "total_equity": 50_672_000_000, "eps": 1.88,
        "market_cap": 2_350_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT},
    },
    "JPM": {
        "ticker": "JPM", "period_end": "2023-03-31",
        "revenue": 38_349_000_000, "net_income": 12_622_000_000,
        "total_assets": 3_744_305_000_000, "total_liabilities": 3_454_000_000_000,
        "total_equity": 290_000_000_000, "eps": 4.10,
        "market_cap": 400_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT},
    },
    "XOM": {
        "ticker": "XOM", "period_end": "2023-03-31",
        "revenue": 86_564_000_000, "net_income": 11_432_000_000,
        "total_assets": 376_317_000_000, "total_liabilities": 163_567_000_000,
        "total_equity": 168_577_000_000, "eps": 2.79,
        "market_cap": 440_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT},
    },
}

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_TICKERS = ["AAPL", "JPM", "XOM"]
_DEFAULT_DATE = "2023-03-31"


def _check_server(url: str, name: str) -> bool:
    """Check if an LLM server is reachable."""
    import urllib.error
    import urllib.request

    models_url = url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        logger.warning("%s not reachable at %s", name, url)
        return False


def _get_server_model_id(url: str) -> str:
    """Return the model ID served at url by querying /v1/models."""
    import urllib.request

    models_url = url.rstrip("/") + "/models"
    with urllib.request.urlopen(models_url, timeout=5) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["id"]


def _run_ensemble_with_model(
    ticker: str,
    analysis_date: str,
    model_url_override: dict | None,
    snapshot_json: str,
) -> dict:
    """
    Run the ensemble for (ticker, analysis_date), optionally overriding model URLs.

    model_url_override: {"technical": url, "fundamental": url} or None for base.
    snapshot_json: JSON-serialized FundamentalsSnapshot for the Fundamental Agent.
    Returns the EnsembleOutput as a dict.
    """
    # Save env vars we may mutate
    saved = {
        k: os.environ.get(k)
        for k in ("HIFI_TECHNICAL_MODEL", "HIFI_TECHNICAL_FINETUNE_URL",
                  "HIFI_FUNDAMENTAL_FINETUNE_URL", "HIFI_FUNDAMENTAL_FINETUNE_MODEL")
    }

    try:
        if model_url_override:
            if "technical" in model_url_override:
                tech_url = model_url_override["technical"]
                os.environ["HIFI_TECHNICAL_FINETUNE_URL"] = tech_url
                # mlx_lm server requires the exact model ID it serves (full local path)
                os.environ["HIFI_TECHNICAL_MODEL"] = _get_server_model_id(tech_url)
            if "fundamental" in model_url_override:
                fund_url = model_url_override["fundamental"]
                os.environ["HIFI_FUNDAMENTAL_FINETUNE_URL"] = fund_url
                os.environ["HIFI_FUNDAMENTAL_FINETUNE_MODEL"] = _get_server_model_id(fund_url)

        from hifi.agents.ensemble_runner import run_ensemble
        output = run_ensemble(
            ticker=ticker,
            as_of_date=analysis_date,
            snapshot_json=snapshot_json,
            agents=["fundamental", "technical"],
            use_rag=False,
        )
        return output.model_dump()

    finally:
        # Restore all mutated env vars
        for key, original in saved.items():
            if original is not None:
                os.environ[key] = original
            elif key in os.environ:
                del os.environ[key]


def _extract_agent_gr(verification_report: dict, agent_type: str) -> float:
    """Extract GR for a specific agent from an EnsembleVerificationReport dict."""
    report = verification_report.get(f"{agent_type}_report", {})
    return float(report.get("grounding_rate", 0.0))


def _extract_diversity(ensemble_output: dict) -> tuple[float, float]:
    """Extract (pairwise_diversity, disagreement_entropy) from EnsembleOutput dict."""
    decision = ensemble_output.get("ensemble_decision", {})
    pairwise = float(decision.get("pairwise_diversity") or 0.0)
    entropy = float(decision.get("disagreement_entropy") or 0.0)
    return pairwise, entropy


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 11 three-tier evaluation.")
    parser.add_argument("--tickers", default=",".join(_DEFAULT_TICKERS))
    parser.add_argument("--analysis-date", default=_DEFAULT_DATE)
    parser.add_argument("--data-dir", default=str(_ROOT / "data"))
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    # Server reachability check
    base_url = os.environ.get("HIFI_LM_STUDIO_URL", "http://localhost:1234/v1")
    ft_tech_url = os.environ.get("HIFI_TECHNICAL_FINETUNE_URL", "http://localhost:1235/v1")
    ft_fund_url = os.environ.get("HIFI_FUNDAMENTAL_FINETUNE_URL", "http://localhost:1236/v1")

    servers_ok = True
    for url, name in [(base_url, "LM Studio"), (ft_tech_url, "Technical fine-tuned"), (ft_fund_url, "Fundamental fine-tuned")]:  # noqa: E501
        if not _check_server(url, name):
            logger.error("%s not reachable at %s", name, url)
            servers_ok = False

    if not servers_ok:
        logger.error("Start all required servers before running evaluation.")
        sys.exit(1)

    from hifi.models.training_data import FineTuneEvaluationResult
    from hifi.verification.verifier import verify_ensemble

    results = []

    from hifi.data.schemas import FundamentalsSnapshot

    for ticker in tickers:
        logger.info("Evaluating %s on %s...", ticker, args.analysis_date)

        # Build snapshot_json for the Fundamental Agent
        raw_snap = _REFERENCE_SNAPSHOTS.get(ticker)
        if raw_snap is None:
            logger.error("No reference snapshot for ticker %s; skipping.", ticker)
            continue
        snap = FundamentalsSnapshot.model_validate(raw_snap)
        snap_json = snap.model_dump_json()

        # --- Base model run ---
        logger.info("  Base model run...")
        base_output_dict = _run_ensemble_with_model(ticker, args.analysis_date, None, snap_json)

        # Reconstruct EnsembleOutput for verification
        from hifi.collective.schemas import EnsembleOutput
        base_output = EnsembleOutput.model_validate(base_output_dict)
        base_verification = verify_ensemble(base_output)
        base_tech_gr = _extract_agent_gr(base_verification.model_dump(), "technical")
        base_fund_gr = _extract_agent_gr(base_verification.model_dump(), "fundamental")
        base_pairwise, base_entropy = _extract_diversity(base_output_dict)

        # --- Fine-tuned model run ---
        logger.info("  Fine-tuned model run...")
        ft_url_override = {
            "technical": ft_tech_url,
            "fundamental": ft_fund_url,
        }
        ft_output_dict = _run_ensemble_with_model(  # noqa: E501
            ticker, args.analysis_date, ft_url_override, snap_json
        )
        ft_output = EnsembleOutput.model_validate(ft_output_dict)
        ft_verification = verify_ensemble(ft_output)
        ft_tech_gr = _extract_agent_gr(ft_verification.model_dump(), "technical")
        ft_fund_gr = _extract_agent_gr(ft_verification.model_dump(), "fundamental")
        ft_pairwise, ft_entropy = _extract_diversity(ft_output_dict)

        result = FineTuneEvaluationResult(
            ticker=ticker,
            analysis_date=args.analysis_date,
            base_technical_gr=base_tech_gr,
            base_fundamental_gr=base_fund_gr,
            finetuned_technical_gr=ft_tech_gr,
            finetuned_fundamental_gr=ft_fund_gr,
            base_pairwise_diversity=base_pairwise,
            finetuned_pairwise_diversity=ft_pairwise,
            base_disagreement_entropy=base_entropy,
            finetuned_disagreement_entropy=ft_entropy,
            generated_at=datetime.now(UTC).isoformat(),
        )
        results.append(result)

        print(f"\n{ticker}:")
        print(f"  GR technical:   base={base_tech_gr:.3f} -> ft={ft_tech_gr:.3f} (improved: {result.gr_improved_technical})")  # noqa: E501
        print(f"  GR fundamental: base={base_fund_gr:.3f} -> ft={ft_fund_gr:.3f} (improved: {result.gr_improved_fundamental})")  # noqa: E501
        print(f"  Diversity:      base={base_pairwise:.3f} -> ft={ft_pairwise:.3f} (preserved: {result.diversity_preserved})")  # noqa: E501

    # Save evaluation results
    output_dir = Path(args.data_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / "phase11_evaluation.json"
    fixture_path = _ROOT / "tests" / "fixtures" / "baseline" / "phase11_evaluation.json"

    payload = {
        "metadata": {
            "phase": "11",
            "analysis_date": args.analysis_date,
            "tickers": tickers,
            "run_date": datetime.now(UTC).isoformat(),
        },
        "results": [r.model_dump() for r in results],
    }

    for path in [eval_path, fixture_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    print(f"\nEvaluation saved to: {eval_path}")
    print(f"Fixture saved to: {fixture_path}")

    # Summary
    all_tech_improved = all(r.gr_improved_technical for r in results)
    all_diversity_preserved = all(r.diversity_preserved for r in results)
    print(f"\nOQ-M01 (GR improvement): {'YES' if all_tech_improved else 'NO'}")
    print(f"OQ-M02 (diversity preserved): {'YES' if all_diversity_preserved else 'NO'}")


if __name__ == "__main__":
    main()
