"""
Phase 7 RAG baseline runner (P7-E8).

Builds a knowledge store from SEC EDGAR fixture files, then runs
run_ensemble(use_rag=True) for AAPL, JPM, and XOM at Q1 2023.
Compares HR/GR against Phase 5 baseline to measure RAG impact.

Requires
--------
- LM Studio running at HIFI_LM_STUDIO_URL (default http://localhost:1234/v1)
  with both fundamental and technical models loaded.
- Phase 1 market/macro Parquet files in DATA_DIR (default: data/).
- SEC fixture files in tests/fixtures/sec/ (run record_sec_fixtures.py first).
- Phase 5 verification fixture in tests/fixtures/baseline/phase5_verification.json
  (run run_phase5_verification.py first if not present -- optional, used only
  for delta comparison).

Usage
-----
    uv run python scripts/run_phase7_rag_baseline.py [--data-dir DIR] [--knowledge-dir DIR]

Output
------
tests/fixtures/baseline/phase7_rag_baseline.json
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.agents.ensemble_runner import run_ensemble  # noqa: E402
from hifi.collective.metrics import compute_ensemble_metrics  # noqa: E402
from hifi.data.schemas import FundamentalsSnapshot  # noqa: E402
from hifi.knowledge.document_ingestion import DocumentIngestionPipeline  # noqa: E402
from hifi.knowledge.embeddings import EmbeddingModel  # noqa: E402
from hifi.knowledge.schemas import FilingDocument  # noqa: E402
from hifi.knowledge.vector_store import KnowledgeStore  # noqa: E402
from hifi.verification.metrics import compute_verification_metrics  # noqa: E402
from hifi.verification.schemas import (  # noqa: E402
    AgentVerificationReport,
    EnsembleVerificationReport,
)
from hifi.verification.verifier import verify_ensemble  # noqa: E402

_AS_OF = "2023-03-31"
_FETCHED_AT = datetime(2023, 4, 1)
_SEC_FIXTURES_DIR = _ROOT / "tests" / "fixtures" / "sec"
_BASELINE_DIR = _ROOT / "tests" / "fixtures" / "baseline"
_OUTPUT_PATH = _BASELINE_DIR / "phase7_rag_baseline.json"
_PHASE5_PATH = _BASELINE_DIR / "phase5_verification.json"

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
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT.isoformat()},
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
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT.isoformat()},
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
        "provenance": {"source": "10-Q reference", "fetched_at": _FETCHED_AT.isoformat()},
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


def _load_sec_fixture(ticker: str, filing_type: str) -> FilingDocument | None:
    """Load a pre-recorded SEC fixture from tests/fixtures/sec/."""
    path = _SEC_FIXTURES_DIR / f"{ticker}_{filing_type.replace('-', '_')}_sections.json"
    if not path.exists():
        logger.warning("SEC fixture not found: %s", path)
        return None
    payload = json.loads(path.read_text())
    return FilingDocument(
        ticker=payload["ticker"],
        cik=payload["cik"],
        filing_type=payload["filing_type"],
        accession_number=payload["accession_number"],
        period_of_report=date.fromisoformat(payload["period_of_report"]),
        filed_date=date.fromisoformat(payload["filed_date"]),
        sections=payload["sections"],
        source_url=payload["source_url"],
        fetched_at=datetime.fromisoformat(payload["fetched_at"]),
    )


def _build_knowledge_store(knowledge_dir: Path, chunking_config: str = "A") -> KnowledgeStore:
    """Build and populate a KnowledgeStore from SEC fixture files."""
    print(f"Building knowledge store in {knowledge_dir} (config {chunking_config}) ...")

    store = KnowledgeStore(
        data_dir=knowledge_dir,
        chunking_config=chunking_config,
    )
    model = EmbeddingModel()
    pipeline = DocumentIngestionPipeline(chunking_config)

    total_chunks = 0
    for ticker in ("AAPL", "JPM", "XOM"):
        for filing_type in ("10-K", "10-Q", "8-K"):
            doc = _load_sec_fixture(ticker, filing_type)
            if doc is None:
                logger.warning("Skipping %s %s (fixture missing)", ticker, filing_type)
                continue
            chunks = pipeline.chunk_document(doc)
            if not chunks:
                continue
            embeddings = model.embed([c.text for c in chunks])
            store.index_chunks(chunks, embeddings)
            total_chunks += len(chunks)
            logger.info("%s %s: indexed %d chunks", ticker, filing_type, len(chunks))

    stats = store.get_stats()
    print(f"Knowledge store ready: {total_chunks} chunks indexed. Stats: {stats}")
    return store


def _print_agent_metrics(label: str, metrics: dict) -> None:
    print(f"\n  {label}:")
    print(f"    mean_hr  : {metrics['mean_hallucination_rate']:.4f}")
    print(f"    mean_gr  : {metrics['mean_grounding_rate']:.4f}")
    print(f"    coverage : {metrics['alias_table_coverage']:.4f}")
    print(f"    claims   : {metrics['total_claims']}")
    print(f"    hallucin.: {metrics['total_hallucinated']}")


def _load_phase5_baseline() -> dict | None:
    if not _PHASE5_PATH.exists():
        return None
    return json.loads(_PHASE5_PATH.read_text())


def run_baseline(data_dir: str, knowledge_dir: str) -> None:
    knowledge_path = Path(knowledge_dir)
    knowledge_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Build knowledge store from SEC fixtures
    if not _SEC_FIXTURES_DIR.exists() or not any(_SEC_FIXTURES_DIR.iterdir()):
        print(f"ERROR: No SEC fixtures found in {_SEC_FIXTURES_DIR}")
        print("Run first: uv run python scripts/record_sec_fixtures.py")
        sys.exit(1)

    _build_knowledge_store(knowledge_path)

    # Step 2: Inject knowledge store into knowledge_server module so the MCP
    #         subprocess finds it via HIFI_KNOWLEDGE_DATA_DIR
    os.environ["HIFI_KNOWLEDGE_DATA_DIR"] = str(knowledge_path)
    os.environ["HIFI_KNOWLEDGE_CHUNKING_CONFIG"] = "A"

    print(f"\nRunning Phase 7 RAG ensemble for AAPL, JPM, XOM at {_AS_OF}")
    print(f"Data directory:      {data_dir}")
    print(f"Knowledge directory: {knowledge_path}")
    print()

    ensemble_outputs: dict = {}
    ensemble_reports: dict[str, EnsembleVerificationReport] = {}
    fund_reports: dict[str, AgentVerificationReport] = {}
    tech_reports: dict[str, AgentVerificationReport] = {}

    for ticker, raw in _REFERENCE_SNAPSHOTS.items():
        print(f"  Running ensemble (use_rag=True) for {ticker} ...", flush=True)
        snap = FundamentalsSnapshot.model_validate(dict(raw))

        output = run_ensemble(
            ticker=ticker,
            as_of_date=_AS_OF,
            snapshot_json=snap.model_dump_json(),
            data_dir=data_dir,
            use_rag=True,
        )
        ensemble_outputs[ticker] = output

        fa_sig = output.fundamental_analysis.signal
        ta_sig = output.technical_analysis.signal
        fund_dec = fa_sig.decision if fa_sig else "FAILED"
        tech_dec = ta_sig.decision if ta_sig else "FAILED"
        collective = output.ensemble_decision.collective_decision or "NONE"
        latency = f"{output.latency_ms:.0f}ms"
        fund_pv = output.fundamental_analysis.prompt_version
        tech_pv = output.technical_analysis.prompt_version
        print(
            f"    {ticker}: fundamental={fund_dec} [{fund_pv}], "
            f"technical={tech_dec} [{tech_pv}], "
            f"collective={collective} ({latency})"
        )

        # Verify output
        report = verify_ensemble(output, always_verify=True)
        ensemble_reports[ticker] = report
        fund_reports[ticker] = report.fundamental_report
        tech_reports[ticker] = report.technical_report

        fr = report.fundamental_report
        tr = report.technical_report
        print(
            f"    Verification: fundamental"
            f" HR={fr.hallucination_rate:.3f} GR={fr.grounding_rate:.3f},"
            f" technical HR={tr.hallucination_rate:.3f} GR={tr.grounding_rate:.3f}"
        )

    # Aggregate metrics
    fund_metrics = compute_verification_metrics(fund_reports)
    tech_metrics = compute_verification_metrics(tech_reports)
    ensemble_metrics = compute_ensemble_metrics(ensemble_outputs)

    total_contradictions = sum(r.n_contradictions for r in ensemble_reports.values())
    mean_ehr = (
        sum(r.ensemble_hallucination_rate for r in ensemble_reports.values())
        / len(ensemble_reports)
    )

    print("\nMetrics (Phase 7 RAG):")
    _print_agent_metrics("Fundamental", fund_metrics)
    _print_agent_metrics("Technical", tech_metrics)

    # Phase 5 delta comparison
    phase5 = _load_phase5_baseline()
    if phase5:
        p5_fund_hr = phase5["metrics"]["fundamental"]["mean_hallucination_rate"]
        p5_tech_hr = phase5["metrics"]["technical"]["mean_hallucination_rate"]
        p5_fund_gr = phase5["metrics"]["fundamental"]["mean_grounding_rate"]
        p5_tech_gr = phase5["metrics"]["technical"]["mean_grounding_rate"]
        fund_hr_delta = fund_metrics["mean_hallucination_rate"] - p5_fund_hr
        tech_hr_delta = tech_metrics["mean_hallucination_rate"] - p5_tech_hr
        fund_gr_delta = fund_metrics["mean_grounding_rate"] - p5_fund_gr
        tech_gr_delta = tech_metrics["mean_grounding_rate"] - p5_tech_gr
        print("\nDelta vs Phase 5 baseline:")
        p7_fund_hr = fund_metrics["mean_hallucination_rate"]
        p7_tech_hr = tech_metrics["mean_hallucination_rate"]
        p7_fund_gr = fund_metrics["mean_grounding_rate"]
        p7_tech_gr = tech_metrics["mean_grounding_rate"]
        print(f"  Fundamental HR: {fund_hr_delta:+.4f}"
              f"  (Phase 5: {p5_fund_hr:.4f} -> Phase 7: {p7_fund_hr:.4f})")
        print(f"  Technical   HR: {tech_hr_delta:+.4f}"
              f"  (Phase 5: {p5_tech_hr:.4f} -> Phase 7: {p7_tech_hr:.4f})")
        print(f"  Fundamental GR: {fund_gr_delta:+.4f}"
              f"  (Phase 5: {p5_fund_gr:.4f} -> Phase 7: {p7_fund_gr:.4f})")
        print(f"  Technical   GR: {tech_gr_delta:+.4f}"
              f"  (Phase 5: {p5_tech_gr:.4f} -> Phase 7: {p7_tech_gr:.4f})")
    else:
        fund_hr_delta = None
        tech_hr_delta = None
        fund_gr_delta = None
        tech_gr_delta = None
        print("\nPhase 5 baseline not found; skipping delta comparison.")

    # Save fixture
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "phase": "7",
            "rag_enabled": True,
            "chunking_config": "A",
            "data_as_of": _AS_OF,
            "run_date": date.today().isoformat(),
            "hifi_commit": _git_sha(),
        },
        "outputs": {
            ticker: o.model_dump() for ticker, o in ensemble_outputs.items()
        },
        "verification": {
            ticker: r.model_dump() for ticker, r in ensemble_reports.items()
        },
        "metrics": {
            "fundamental": fund_metrics,
            "technical": tech_metrics,
            "ensemble": {
                **ensemble_metrics,
                "mean_ensemble_hallucination_rate": round(mean_ehr, 6),
                "total_contradictions": total_contradictions,
            },
        },
        "delta_vs_phase5": {
            "fundamental_hr": fund_hr_delta,
            "technical_hr": tech_hr_delta,
            "fundamental_gr": fund_gr_delta,
            "technical_gr": tech_gr_delta,
        },
    }

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nSaved to {_OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 7 RAG ensemble baseline.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", str(_ROOT / "data")),
        help="Path to the market/macro data root directory (default: data/).",
    )
    parser.add_argument(
        "--knowledge-dir",
        default=os.environ.get("HIFI_KNOWLEDGE_DATA_DIR", str(_ROOT / "data" / "knowledge")),
        help="Path to the knowledge store directory (default: data/knowledge/).",
    )
    args = parser.parse_args()
    run_baseline(args.data_dir, args.knowledge_dir)


if __name__ == "__main__":
    main()
