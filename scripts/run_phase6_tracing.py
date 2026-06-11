"""
Phase 6 observability baseline runner (P6-E6-T7, P6-E6-T8).

Reads the Phase 4 ensemble fixture (tests/fixtures/baseline/phase4_ensemble.json)
and logs the Phase 5 verification metrics as LangFuse scores, creating one trace
per ticker. No LLM inference is required -- verification operates on already-produced
agent outputs.

IMPORTANT: Requires a live LangFuse instance with credentials configured:
    LANGFUSE_HOST=http://localhost:3000
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_ENABLED=true

After running, open the LangFuse dashboard (http://localhost:3000) and verify:
- Three traces visible (AAPL, JPM, XOM)
- Each trace has six scores:
    fundamental_hr, fundamental_gr, technical_hr, technical_gr,
    disagreement_entropy, n_contradictions
- Score values match the Phase 5 baseline (see PHASE_05_VERIFICATION.md)

Usage
-----
    uv run python scripts/run_phase6_tracing.py

This is the manual verification step for DJ-022, DJ-023, DJ-024, DJ-025.
It seeds the LangFuse time series with the Phase 5 baseline values so that
Phase 7 (RAG) improvements are measurable as score changes in the dashboard.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hifi.collective.schemas import EnsembleOutput  # noqa: E402
from hifi.observability.tracing import get_tracer, log_verification_scores  # noqa: E402
from hifi.verification.verifier import verify_ensemble  # noqa: E402

_PHASE4_PATH = _ROOT / "tests" / "fixtures" / "baseline" / "phase4_ensemble.json"


def _check_prerequisites() -> None:
    enabled = os.environ.get("LANGFUSE_ENABLED", "true").lower().strip()
    if enabled in ("false", "0", "no", "off"):
        print("ERROR: LANGFUSE_ENABLED is false. Set it to true before running.")
        sys.exit(1)

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key:
        print(
            "ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set.\n"
            "  See doc/setup/LANGFUSE_SETUP.md for setup instructions."
        )
        sys.exit(1)

    if not _PHASE4_PATH.exists():
        print(f"ERROR: {_PHASE4_PATH} not found.")
        print("  Run: uv run python scripts/run_phase4_ensemble.py first.")
        sys.exit(1)


def run_tracing_baseline() -> None:
    _check_prerequisites()

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    print(f"LangFuse host : {host}")
    print(f"Phase 4 fixture: {_PHASE4_PATH}")
    print()

    tracer = get_tracer()
    tracer_type = type(tracer).__name__
    if tracer_type == "NoOpTracer":
        print("WARNING: get_tracer() returned NoOpTracer. Credentials may be invalid.")
        print("  Traces will NOT appear in the LangFuse dashboard.")
        print()

    with open(_PHASE4_PATH, encoding="utf-8") as f:
        phase4 = json.load(f)

    tickers = list(phase4["outputs"].keys())
    print(f"Tickers: {tickers}")
    print()

    trace_ids: dict[str, str] = {}

    for ticker in tickers:
        output_dict = phase4["outputs"][ticker]
        output = EnsembleOutput.model_validate(output_dict)

        print(f"  Processing {ticker} ...", flush=True)

        # Verify ensemble output (same as Phase 5)
        report = verify_ensemble(output, always_verify=True)
        decision = output.ensemble_decision

        # Create a trace and log the six verification scores
        trace_id = tracer.start_trace(
            "run_ensemble",
            ticker=ticker,
            as_of_date=output.as_of_date,
            source="phase6_baseline_run",
            run_date=date.today().isoformat(),
        )
        log_verification_scores(tracer, trace_id, report, decision)
        tracer.flush()

        trace_ids[ticker] = trace_id

        fr = report.fundamental_report
        tr = report.technical_report
        print(
            f"    fund  : HR={fr.hallucination_rate:.3f}  GR={fr.grounding_rate:.3f}"
        )
        print(
            f"    tech  : HR={tr.hallucination_rate:.3f}  GR={tr.grounding_rate:.3f}"
        )
        print(
            f"    entropy={decision.disagreement_entropy:.3f}  "
            f"contradictions={report.n_contradictions}"
        )
        print(f"    trace_id={trace_id}")
        print()

    print("Done.")
    print(f"Tracer type: {tracer_type}")
    if tracer_type != "NoOpTracer":
        print(f"Open {host} to verify {len(tickers)} traces with 6 scores each.")


if __name__ == "__main__":
    run_tracing_baseline()
