"""
run_phase12_evaluation.py -- 2x2 factorial multi-date evaluation (P12-E4-T1/T2, DJ-067).

Runs all 4 experimental conditions across 10 quarterly dates × 3 tickers = 120 total runs.
Computes diversity, herding, debate participation, and interaction effect metrics.

2x2 factorial design (DJ-067):
  Condition A: run_ensemble()        base model,       no debate
  Condition B: run_ensemble()        fine-tuned model, no debate
  Condition C: run_debate_ensemble() base model,       with debate
  Condition D: run_debate_ensemble() fine-tuned model, with debate

Interaction effect: (D-B) - (C-A) per metric (positive = fine-tuning amplifies debate benefit)

Output:
  data/evaluation/phase12/{condition}_{ticker}_{date}.json  -- per-run EnsembleOutput
  data/evaluation/phase12/checkpoint.json                   -- resumable checkpoint
  tests/fixtures/baseline/phase12_factorial_results.json    -- aggregated metrics

Usage:
    uv run python scripts/run_phase12_evaluation.py [--conditions A,B,C,D]
                                                     [--tickers AAPL,JPM,XOM]
                                                     [--dates 2020-03-31,...]
                                                     [--agents fundamental,technical]
                                                     [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experimental design constants (DJ-067)
# ---------------------------------------------------------------------------

_DEFAULT_TICKERS = ["AAPL", "JPM", "XOM"]

# 10 quarterly dates: 2020-Q1 through 2022-Q2 (well within market data window)
_DEFAULT_DATES = [
    "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
    "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30",
]

_ALL_CONDITIONS = ["A", "B", "C", "D"]

# Reference fundamentals snapshots — approximated from quarterly filings.
# For Phase 12 the specific values are illustrative; the evaluation focuses
# on decision mechanics (debate, diversity) not financial accuracy. See DJ-067.
_FETCHED_AT = "2023-03-31T00:00:00+00:00"
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
        "ticker": "JPM", "period_end": "2022-12-31",
        "revenue": 128_695_000_000, "net_income": 37_676_000_000,
        "total_assets": 3_665_743_000_000, "total_liabilities": 3_373_000_000_000,
        "total_equity": 292_000_000_000, "eps": 12.09,
        "market_cap": 390_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-K reference", "fetched_at": _FETCHED_AT},
    },
    "XOM": {
        "ticker": "XOM", "period_end": "2022-12-31",
        "revenue": 398_675_000_000, "net_income": 55_740_000_000,
        "total_assets": 369_067_000_000, "total_liabilities": 167_961_000_000,
        "total_equity": 168_577_000_000, "eps": 14.18,
        "market_cap": 440_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-K reference", "fetched_at": _FETCHED_AT},
    },
}

_FINETUNE_TECH_URL = os.environ.get("HIFI_TECHNICAL_FINETUNE_URL", "http://localhost:1235/v1")
_FINETUNE_FUND_URL = os.environ.get("HIFI_FUNDAMENTAL_FINETUNE_URL", "http://localhost:1236/v1")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _herding_coefficient(agent_decisions: list[str]) -> float:
    """
    Majority fraction: α_t = (# agents voting with plurality) / N (David SS5.6.3).

    Range [1/N, 1.0]. Unanimous vote → 1.0. Equal 3-way split → 1/3.
    """
    if not agent_decisions:
        return 0.0
    counts = Counter(agent_decisions)
    majority_count = counts.most_common(1)[0][1]
    return majority_count / len(agent_decisions)


def _extract_metrics(output_dict: dict) -> dict:
    """Extract diversity, herding, and debate metrics from a serialised EnsembleOutput."""
    decision = output_dict.get("ensemble_decision", {})
    agent_decisions = decision.get("agent_decisions", [])

    transcript = output_dict.get("debate_transcript")
    if transcript:
        vote_delta = transcript.get("vote_delta", "unchanged")
        debate_skipped = transcript.get("debate_skipped", True)
        n_changed = transcript.get("n_agents_changed_vote", 0)
    else:
        vote_delta = None
        debate_skipped = None
        n_changed = None

    return {
        "disagreement_entropy": float(decision.get("disagreement_entropy", 0.0)),
        "opinion_dispersion": float(decision.get("opinion_dispersion", 0.0)),
        "herding_coefficient": _herding_coefficient(agent_decisions),
        "collective_decision": decision.get("collective_decision"),
        "n_valid_signals": int(decision.get("n_valid_signals", 0)),
        "vote_delta": vote_delta,
        "debate_skipped": debate_skipped,
        "n_agents_changed_vote": n_changed,
    }


def _interaction_effect(
    metric_a: float, metric_b: float, metric_c: float, metric_d: float
) -> float:
    """
    2x2 factorial interaction effect: (D-B) - (C-A).

    Positive value means fine-tuning amplifies the benefit of debate.
    This is the core empirical question for publication (David SS5.3, DJ-067).
    """
    return (metric_d - metric_b) - (metric_c - metric_a)


# ---------------------------------------------------------------------------
# Fine-tune URL management (mirrors Phase 11 evaluation pattern)
# ---------------------------------------------------------------------------


def _get_server_model_id(url: str) -> str:
    import urllib.request
    models_url = url.rstrip("/") + "/models"
    with urllib.request.urlopen(models_url, timeout=5) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["id"]


def _check_server(url: str, name: str) -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _with_finetune_env(fn, tech_url: str, fund_url: str):
    """Run fn() with fine-tune env vars set, then restore. Mirrors Phase 11 pattern."""
    keys = (
        "HIFI_TECHNICAL_MODEL", "HIFI_TECHNICAL_FINETUNE_URL",
        "HIFI_FUNDAMENTAL_FINETUNE_URL", "HIFI_FUNDAMENTAL_FINETUNE_MODEL",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["HIFI_TECHNICAL_FINETUNE_URL"] = tech_url
        os.environ["HIFI_TECHNICAL_MODEL"] = _get_server_model_id(tech_url)
        os.environ["HIFI_FUNDAMENTAL_FINETUNE_URL"] = fund_url
        os.environ["HIFI_FUNDAMENTAL_FINETUNE_MODEL"] = _get_server_model_id(fund_url)
        return fn()
    finally:
        for key, original in saved.items():
            if original is not None:
                os.environ[key] = original
            elif key in os.environ:
                del os.environ[key]


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------


def _run_condition(
    condition: str,
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    agents: list[str],
    use_finetune: bool,
    tech_url: str,
    fund_url: str,
) -> dict:
    """Execute one (condition, ticker, date) cell of the factorial grid."""
    from hifi.agents.ensemble_runner import run_debate_ensemble, run_ensemble
    from hifi.observability.tracing import NoOpTracer

    use_debate = condition in ("C", "D")
    tracer = NoOpTracer()

    def _run():
        if use_debate:
            output = run_debate_ensemble(
                ticker=ticker,
                as_of_date=as_of_date,
                snapshot_json=snapshot_json,
                agents=agents,
                tracer=tracer,
                use_rag=False,
            )
        else:
            output = run_ensemble(
                ticker=ticker,
                as_of_date=as_of_date,
                snapshot_json=snapshot_json,
                agents=agents,
                tracer=tracer,
                use_rag=False,
            )
        return output.model_dump()

    if use_finetune:
        return _with_finetune_env(_run, tech_url, fund_url)
    return _run()


# ---------------------------------------------------------------------------
# Checkpoint helpers (resumable evaluation)
# ---------------------------------------------------------------------------


def _checkpoint_key(condition: str, ticker: str, date: str) -> str:
    return f"{condition}_{ticker}_{date}"


def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text())
        return set(data.get("completed", []))
    return set()


def _save_checkpoint(checkpoint_path: Path, completed: set[str]) -> None:
    checkpoint_path.write_text(json.dumps({"completed": sorted(completed)}, indent=2))


# ---------------------------------------------------------------------------
# Aggregate analysis (E4-T2)
# ---------------------------------------------------------------------------


def compute_factorial_summary(
    all_metrics: dict[str, list[dict]],
) -> dict:
    """
    Compute 2x2 factorial summary from per-run metrics.

    Parameters
    ----------
    all_metrics : dict[str, list[dict]]
        Keys "A","B","C","D" → list of per-run metric dicts.

    Returns
    -------
    dict
        Mean metrics per condition, herding assessment, vote delta distribution,
        interaction effects, and OQ-M02 assessment.
    """
    def _mean(vals: list[float | None]) -> float | None:
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else None

    condition_summary: dict[str, dict] = {}
    for cond, runs in all_metrics.items():
        entropies = [r["disagreement_entropy"] for r in runs]
        dispersions = [r["opinion_dispersion"] for r in runs]
        herd_coeffs = [r["herding_coefficient"] for r in runs]
        vote_deltas = [r["vote_delta"] for r in runs if r["vote_delta"] is not None]
        debate_skips = [r["debate_skipped"] for r in runs if r["debate_skipped"] is not None]

        condition_summary[cond] = {
            "n_runs": len(runs),
            "mean_disagreement_entropy": _mean(entropies),
            "mean_opinion_dispersion": _mean(dispersions),
            "mean_herding_coefficient": _mean(herd_coeffs),
            "vote_delta_distribution": Counter(vote_deltas),
            "debate_skipped_count": sum(1 for s in debate_skips if s),
            "debate_participated_count": sum(1 for s in debate_skips if not s),
            "debate_participation_rate": (
                sum(1 for s in debate_skips if not s) / len(debate_skips)
                if debate_skips else None
            ),
        }

    # --- Interaction effects (DJ-067) ---
    def _cond_mean(cond: str, key: str) -> float:
        vals = [r[key] for r in all_metrics.get(cond, []) if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    interaction_herding = _interaction_effect(
        _cond_mean("A", "herding_coefficient"),
        _cond_mean("B", "herding_coefficient"),
        _cond_mean("C", "herding_coefficient"),
        _cond_mean("D", "herding_coefficient"),
    )
    interaction_entropy = _interaction_effect(
        _cond_mean("A", "disagreement_entropy"),
        _cond_mean("B", "disagreement_entropy"),
        _cond_mean("C", "disagreement_entropy"),
        _cond_mean("D", "disagreement_entropy"),
    )

    # --- OQ-M02: diversity preserved? (< 10% degradation threshold) ---
    entropy_a = _cond_mean("A", "disagreement_entropy")
    entropy_b = _cond_mean("B", "disagreement_entropy")
    entropy_c = _cond_mean("C", "disagreement_entropy")

    diversity_degradation_finetune = (
        (entropy_a - entropy_b) / entropy_a if entropy_a > 0 else 0.0
    )
    diversity_degradation_debate = (
        (entropy_a - entropy_c) / entropy_a if entropy_a > 0 else 0.0
    )
    oq_m02_finetune = diversity_degradation_finetune < 0.10
    oq_m02_debate = diversity_degradation_debate < 0.10

    # --- Herding assessment ---
    herding_a = _cond_mean("A", "herding_coefficient")
    herding_c = _cond_mean("C", "herding_coefficient")
    herding_increase = herding_c - herding_a
    debate_induces_herding = herding_increase > 0.10

    return {
        "conditions": condition_summary,
        "interaction_effects": {
            "herding_coefficient": round(interaction_herding, 4),
            "disagreement_entropy": round(interaction_entropy, 4),
            "interpretation": (
                "positive = fine-tuning amplifies debate benefit; "
                "negative = debate benefits base models more"
            ),
        },
        "oq_m02": {
            "finetune_effect_entropy_degradation": round(diversity_degradation_finetune, 4),
            "debate_effect_entropy_degradation": round(diversity_degradation_debate, 4),
            "diversity_preserved_finetune": oq_m02_finetune,
            "diversity_preserved_debate": oq_m02_debate,
            "threshold": "< 10% degradation",
        },
        "herding_assessment": {
            "herding_increase_A_to_C": round(herding_increase, 4),
            "debate_induces_herding": debate_induces_herding,
            "threshold": "> 0.10 increase flags herding",
        },
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 12 2x2 factorial evaluation.")
    parser.add_argument(
        "--conditions", default="A,B,C,D",
        help="Conditions to run: A,B,C,D (default: all 4)"
    )
    parser.add_argument("--tickers", default=",".join(_DEFAULT_TICKERS))
    parser.add_argument("--dates", default=",".join(_DEFAULT_DATES))
    parser.add_argument(
        "--agents", default="fundamental,technical",
        help="Agent subset (default: fundamental,technical)"
    )
    parser.add_argument("--data-dir", default=str(_ROOT / "data"))
    parser.add_argument(
        "--skip-finetune-check", action="store_true",
        help="Skip fine-tuned server reachability check (run A,C only)"
    )
    args = parser.parse_args()

    conditions = [c.strip().upper() for c in args.conditions.split(",")]
    tickers = [t.strip() for t in args.tickers.split(",")]
    dates = [d.strip() for d in args.dates.split(",")]
    agents = [a.strip() for a in args.agents.split(",")]

    total_runs = len(conditions) * len(tickers) * len(dates)
    print("Phase 12: 2x2 Factorial Evaluation (DJ-067)")
    print("=" * 60)
    print(f"  Conditions : {conditions}")
    print(f"  Tickers    : {tickers}")
    print(f"  Dates      : {len(dates)} ({dates[0]} .. {dates[-1]})")
    print(f"  Agents     : {agents}")
    print(f"  Total runs : {total_runs}")
    print()

    # Server checks for fine-tuned conditions
    needs_finetune = any(c in conditions for c in ("B", "D"))
    if needs_finetune and not args.skip_finetune_check:
        servers = [(_FINETUNE_TECH_URL, "technical ft"), (_FINETUNE_FUND_URL, "fundamental ft")]
        for url, name in servers:
            if not _check_server(url, name):
                logger.error(
                    "%s not reachable at %s. Start fine-tuned servers: make finetune-serve",
                    name, url
                )
                sys.exit(1)

    # Setup output directories
    eval_dir = Path(args.data_dir) / "evaluation" / "phase12"
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = eval_dir / "checkpoint.json"
    fixture_out = _ROOT / "tests" / "fixtures" / "baseline" / "phase12_factorial_results.json"
    fixture_out.parent.mkdir(parents=True, exist_ok=True)

    completed = _load_checkpoint(checkpoint_path)
    logger.info("Checkpoint: %d/%d runs already completed", len(completed), total_runs)

    # Build snapshot JSON for each ticker
    from hifi.data.schemas import FundamentalsSnapshot
    snapshots: dict[str, str] = {}
    for ticker in tickers:
        raw = _REFERENCE_SNAPSHOTS.get(ticker)
        if raw is None:
            logger.error("No reference snapshot for %s", ticker)
            sys.exit(1)
        snapshots[ticker] = FundamentalsSnapshot.model_validate(raw).model_dump_json()

    # --- Run the factorial grid ---
    all_metrics: dict[str, list[dict]] = defaultdict(list)
    n_completed = 0
    n_failed = 0

    for condition in conditions:
        use_finetune = condition in ("B", "D")

        for ticker in tickers:
            for date in dates:
                key = _checkpoint_key(condition, ticker, date)
                if key in completed:
                    # Load existing result for aggregate metrics
                    run_path = eval_dir / f"{key}.json"
                    if run_path.exists():
                        run_data = json.loads(run_path.read_text())
                        all_metrics[condition].append(_extract_metrics(run_data))
                    continue

                logger.info("Running %s | %s | %s ...", condition, ticker, date)
                try:
                    output_dict = _run_condition(
                        condition=condition,
                        ticker=ticker,
                        as_of_date=date,
                        snapshot_json=snapshots[ticker],
                        agents=agents,
                        use_finetune=use_finetune,
                        tech_url=_FINETUNE_TECH_URL,
                        fund_url=_FINETUNE_FUND_URL,
                    )

                    run_path = eval_dir / f"{key}.json"
                    run_path.write_text(json.dumps(output_dict, indent=2))

                    metrics = _extract_metrics(output_dict)
                    all_metrics[condition].append(metrics)
                    completed.add(key)
                    _save_checkpoint(checkpoint_path, completed)
                    n_completed += 1

                    coll = metrics["collective_decision"]
                    herd = metrics["herding_coefficient"]
                    logger.info(
                        "  -> decision=%s, herding=%.3f, entropy=%.3f",
                        coll, herd, metrics["disagreement_entropy"]
                    )

                except Exception as exc:
                    logger.error("FAILED %s | %s | %s: %s", condition, ticker, date, exc)
                    n_failed += 1

    print(f"\nRuns completed this session: {n_completed}")
    print(f"Runs failed: {n_failed}")
    print(f"Total checkpoint: {len(completed)}/{total_runs}")

    # --- Aggregate analysis ---
    print("\nComputing factorial summary...")
    summary = compute_factorial_summary(dict(all_metrics))

    payload = {
        "metadata": {
            "phase": "12",
            "epic": "E4-T1/T2",
            "conditions": conditions,
            "tickers": tickers,
            "dates": dates,
            "agents": agents,
            "total_runs_planned": total_runs,
            "total_runs_completed": len(completed),
            "run_date": datetime.now(UTC).isoformat(),
        },
        **summary,
    }

    fixture_out.write_text(json.dumps(payload, indent=2))
    data_out = eval_dir / "factorial_summary.json"
    data_out.write_text(json.dumps(payload, indent=2))

    # Print key results
    print("\n--- Condition Summary ---")
    for cond, cdata in summary["conditions"].items():
        n = cdata["n_runs"]
        entropy = cdata["mean_disagreement_entropy"]
        herding = cdata["mean_herding_coefficient"]
        print(f"  {cond}: n={n}, entropy={entropy:.3f}, herding={herding:.3f}", end="")
        if cond in ("C", "D"):
            rate = cdata.get("debate_participation_rate")
            print(f", debate_rate={rate:.1%}" if rate is not None else "", end="")
        print()

    ie = summary["interaction_effects"]
    print("\n--- Interaction Effect ---")
    print(f"  Herding   : {ie['herding_coefficient']:+.4f}")
    print(f"  Entropy   : {ie['disagreement_entropy']:+.4f}")

    oq = summary["oq_m02"]
    print("\n--- OQ-M02 ---")
    print(f"  Fine-tuning diversity preserved: {oq['diversity_preserved_finetune']}")
    print(f"  Debate diversity preserved:      {oq['diversity_preserved_debate']}")

    herd = summary["herding_assessment"]
    print("\n--- Herding ---")
    print(f"  Debate increases herding by {herd['herding_increase_A_to_C']:+.4f}")
    print(f"  Debate induces herding (>0.10): {herd['debate_induces_herding']}")

    print(f"\nFixture saved: {fixture_out}")


if __name__ == "__main__":
    main()
