"""
Ensemble runner for the Phase 4 two-agent ensemble (P4-E4).

Runs the Fundamental Agent and Technical Agent sequentially with no shared state
between them during reasoning. This independence is the condition from David §10.1
required for ensemble theory to apply.

Sequential execution is intentional for Phase 4 (2 agents). Phase 9 will
parallelize agent runs; the additional complexity is not justified at this scale.

Phase 6 addition: run_ensemble() creates a parent LangFuse trace, passes the
tracer to both agents, and logs verification scores on the trace after calling
verify_ensemble(). The full pipeline is observable end-to-end without any changes
to agent logic or LangGraph state schemas (DJ-023, DJ-024).
"""

from __future__ import annotations

import time

from hifi.agents.fundamental_agent import run_analysis
from hifi.agents.technical_agent import run_technical_analysis
from hifi.collective.schemas import EnsembleOutput
from hifi.collective.voting import confidence_weighted_vote
from hifi.observability.tracing import (
    AbstractTracer,
    get_tracer,
    log_verification_scores,
    trace_context,
)
from hifi.verification.verifier import verify_ensemble


def run_ensemble(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
) -> EnsembleOutput:
    """
    Run both agents independently, aggregate their outputs, and verify.

    Parameters
    ----------
    ticker : str
        Ticker symbol (must match Parquet files in data_dir/market/).
    as_of_date : str
        ISO 8601 analysis date (e.g. "2023-03-31").
    snapshot_json : str
        JSON-serialised FundamentalsSnapshot for the Fundamental Agent.
        The Technical Agent does not receive this.
    data_dir : str | None
        Path to the data root directory. Defaults to HIFI_DATA_DIR env var.
    tracer : AbstractTracer | None
        Observability tracer. Defaults to get_tracer(). When LANGFUSE_ENABLED=true
        and credentials are set, this creates a parent LangFuse trace that covers
        both agent runs and the verification step. When disabled, NoOpTracer is
        used and behaviour is identical to pre-Phase-6.

    Returns
    -------
    EnsembleOutput
        Full ensemble output: both agent analyses plus collective decision.
        Agents share no state during reasoning -- independence is enforced by
        the function call boundary.
    """
    _tracer = tracer if tracer is not None else get_tracer()
    trace_id = _tracer.start_trace(
        "run_ensemble", ticker=ticker, as_of_date=as_of_date
    )

    start = time.monotonic()

    with trace_context(trace_id):
        # Step 1: Fundamental Agent (uses snapshot_json + MCP fundamentals tools)
        fundamental = run_analysis(
            ticker=ticker,
            as_of_date=as_of_date,
            snapshot_json=snapshot_json,
            data_dir=data_dir,
            tracer=_tracer,
        )

        # Step 2: Technical Agent (uses only price-derived MCP tools; no snapshot)
        technical = run_technical_analysis(
            ticker=ticker,
            as_of_date=as_of_date,
            data_dir=data_dir,
            tracer=_tracer,
        )

    # Step 3: Collect valid signals and aggregate
    valid_signals = [
        s
        for s in [fundamental.signal, technical.signal]
        if s is not None
    ]
    decision = confidence_weighted_vote(valid_signals)

    latency_ms = round((time.monotonic() - start) * 1000, 1)

    output = EnsembleOutput(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamental_analysis=fundamental,
        technical_analysis=technical,
        ensemble_decision=decision,
        latency_ms=latency_ms,
    )

    # Step 4: Verify ensemble output and log scores (DJ-024)
    verification = verify_ensemble(output)
    log_verification_scores(_tracer, trace_id, verification, decision)

    _tracer.flush()

    return output
