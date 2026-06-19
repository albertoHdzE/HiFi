"""
Ensemble runner for the Phase 8/9 N-agent ensemble (P8-E6, P9-E3).

Extends the Phase 4 two-agent runner to support up to 6 agents:
  fundamental, technical, risk, macro, sentiment, contrarian

The `agents` parameter selects which agents to run (DJ-038):
- agents=None         → all 6 agents (Phase 8 default)
- agents=["fundamental","technical"] → Phase 4/6/7 behavior (backward compat)
- Contrarian always runs LAST (after voting) regardless of list order

Phase 9 additions (P9-E3):
- All four aggregation methods run on every call via run_all_methods(); results
  stored in EnsembleOutput.method_comparison (D-02)
- Valid signals captured in EnsembleOutput.signals (D-01)
- Performance weights loaded from data/agent_performance_history.json (D-04)
- Method divergence logged at INFO when methods produce different decisions (D-06)

Sequential execution: agents run one at a time.

Observability: the same parent LangFuse trace covers all agents and verification.
"""

from __future__ import annotations

import logging
import time

from hifi.agents.fundamental_agent import run_analysis
from hifi.agents.technical_agent import run_technical_analysis
from hifi.collective.performance_store import get_weights
from hifi.collective.schemas import EnsembleOutput
from hifi.collective.voting import confidence_weighted_vote, run_all_methods
from hifi.observability.tracing import (
    AbstractTracer,
    get_tracer,
    log_verification_scores,
    trace_context,
)
from hifi.verification.verifier import verify_ensemble

logger = logging.getLogger(__name__)

_ALL_AGENTS = ["fundamental", "technical", "risk", "macro", "sentiment", "contrarian"]


def _build_graphrag_context(ticker: str, data_dir: str | None) -> str:
    """
    Build graph-expanded retrieval context for use_graphrag=True (P12-E2-T1, DJ-068).

    Loads FinancialGraph from data/knowledge_graph/financial_graph.json and
    performs graph-expanded ANN search. Returns "" on any failure (fail-open).
    """
    import os
    from pathlib import Path

    try:
        from hifi.knowledge.embeddings import EmbeddingModel
        from hifi.knowledge.graph_retrieval import GraphRetriever
        from hifi.knowledge.graph_store import FinancialGraph
        from hifi.knowledge.vector_store import KnowledgeStore

        effective_dir = Path(data_dir or os.environ.get("HIFI_DATA_DIR", "data"))
        graph_path = effective_dir / "knowledge_graph" / "financial_graph.json"

        if not graph_path.exists():
            logger.warning("Knowledge graph not found at %s; GraphRAG disabled", graph_path)
            return ""

        graph = FinancialGraph.load(graph_path)
        store = KnowledgeStore(data_dir=effective_dir)
        model = EmbeddingModel()
        retriever = GraphRetriever(store=store, embedding_model=model, graph=graph)
        chunks = retriever.retrieve(f"financial analysis {ticker}", ticker=ticker, top_k=5)
        return retriever.format_context(chunks)
    except Exception as exc:
        logger.warning("GraphRAG context build failed for %s: %s", ticker, exc)
        return ""


def run_ensemble(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
    agents: list[str] | None = None,
    use_graphrag: bool = False,
    memory_prefixes: dict[str, str] | None = None,
    _test_llms: dict | None = None,
) -> EnsembleOutput:
    """
    Run the agent ensemble, aggregate outputs, and verify.

    Parameters
    ----------
    ticker : str
        Ticker symbol (must match Parquet files in data_dir/market/).
    as_of_date : str
        ISO 8601 analysis date (e.g. "2023-03-31").
    snapshot_json : str
        JSON-serialised FundamentalsSnapshot for the Fundamental Agent.
        Only used when "fundamental" is in the agents list.
    data_dir : str | None
        Path to the data root directory. Defaults to HIFI_DATA_DIR env var.
    tracer : AbstractTracer | None
        Observability tracer. Defaults to get_tracer().
    use_rag : bool
        When True, Fundamental and Technical Agents include a retrieve_context step.
    agents : list[str] | None
        Agents to run. None means all 6 agents. Use ["fundamental","technical"] for
        Phase 4/6/7 backward compat. Contrarian always runs last (DJ-038).
    use_graphrag : bool
        When True, use graph-expanded RAG instead of plain dense RAG (DJ-068).
        Mutually exclusive with use_rag.
    memory_prefixes : dict[str, str] | None
        Optional per-agent memory prefixes (P13-E4-T3, DJ-076). Keys are agent_type
        strings ("fundamental", "technical", "risk", "macro", "sentiment"). Missing
        keys → empty prefix (no memory injection for that agent).

    Returns
    -------
    EnsembleOutput
        Full ensemble output. Phase 8 fields (risk, macro, sentiment, contrarian)
        are None when the corresponding agent was not in the `agents` list.
    """
    if use_graphrag:
        assert not use_rag, "use_rag and use_graphrag are mutually exclusive (DJ-068)"

    _tracer = tracer if tracer is not None else get_tracer()
    trace_id = _tracer.start_trace(
        "run_ensemble", ticker=ticker, as_of_date=as_of_date
    )

    active = _ALL_AGENTS if agents is None else list(agents)
    # Contrarian always runs last — remove and re-append after other agents
    run_contrarian = "contrarian" in active
    voting_agents = [a for a in active if a != "contrarian"]

    start = time.monotonic()
    _mem = memory_prefixes or {}
    _llms = _test_llms or {}

    graphrag_ctx = _build_graphrag_context(ticker, data_dir) if use_graphrag else ""

    risk_analysis = None
    macro_analysis = None
    sentiment_analysis = None
    contrarian_analysis = None

    with trace_context(trace_id):
        # --- Phase 4 agents ---
        fundamental = run_analysis(
            ticker=ticker,
            as_of_date=as_of_date,
            snapshot_json=snapshot_json,
            data_dir=data_dir,
            tracer=_tracer,
            use_rag=use_rag,
            retrieved_context=graphrag_ctx,
            memory_prefix=_mem.get("fundamental", ""),
            _test_llm=_llms.get("fundamental"),
        )

        technical = run_technical_analysis(
            ticker=ticker,
            as_of_date=as_of_date,
            data_dir=data_dir,
            tracer=_tracer,
            use_rag=use_rag,
            retrieved_context=graphrag_ctx,
            memory_prefix=_mem.get("technical", ""),
            _test_llm=_llms.get("technical"),
        )

        # --- Phase 8 agents (imported lazily to avoid import-time side effects) ---
        if "risk" in voting_agents:
            from hifi.agents.risk_agent import run_risk_analysis
            risk_analysis = run_risk_analysis(
                ticker=ticker,
                as_of_date=as_of_date,
                data_dir=data_dir,
                tracer=_tracer,
                memory_prefix=_mem.get("risk", ""),
                _test_llm=_llms.get("risk"),
            )

        if "macro" in voting_agents:
            from hifi.agents.macro_agent import run_macro_analysis
            macro_analysis = run_macro_analysis(
                ticker=ticker,
                as_of_date=as_of_date,
                data_dir=data_dir,
                tracer=_tracer,
                memory_prefix=_mem.get("macro", ""),
                _test_llm=_llms.get("macro"),
            )

        if "sentiment" in voting_agents:
            from hifi.agents.sentiment_agent import run_sentiment_analysis
            sentiment_analysis = run_sentiment_analysis(
                ticker=ticker,
                as_of_date=as_of_date,
                data_dir=data_dir,
                tracer=_tracer,
                memory_prefix=_mem.get("sentiment", ""),
                _test_llm=_llms.get("sentiment"),
            )

    # --- Performance weights for aggregation (D-02) ---
    perf_weights = get_weights(data_dir=data_dir)

    # --- Voting: collect valid signals from all non-contrarian agents ---
    candidate_signals = [fundamental.signal, technical.signal]
    if risk_analysis is not None:
        candidate_signals.append(risk_analysis.signal)
    if macro_analysis is not None:
        candidate_signals.append(macro_analysis.signal)
    if sentiment_analysis is not None:
        candidate_signals.append(sentiment_analysis.signal)

    valid_signals = [s for s in candidate_signals if s is not None]
    decision = confidence_weighted_vote(valid_signals)

    # --- Contrarian (second-pass, always after voting) ---
    if run_contrarian:
        from hifi.agents.contrarian_agent import (
            _build_ensemble_context,
            run_contrarian_analysis,
        )

        agent_summaries = []
        for sig in valid_signals:
            agent_summaries.append({
                "agent_type": sig.agent_type,
                "decision": sig.decision,
                "confidence": sig.confidence,
                "rationale": sig.rationale,
                "key_concern": sig.key_concern,
            })

        ensemble_context = _build_ensemble_context(
            ticker=ticker,
            as_of_date=as_of_date,
            agent_summaries=agent_summaries,
            collective_decision=decision.collective_decision,
            collective_confidence=decision.collective_confidence,
        )

        with trace_context(trace_id):
            contrarian_analysis = run_contrarian_analysis(
                ticker=ticker,
                as_of_date=as_of_date,
                ensemble_context=ensemble_context,
                tracer=_tracer,
                _test_llm=_llms.get("contrarian"),
            )

    # --- Run all four aggregation methods (D-02, D-06) ---
    method_comparison = run_all_methods(
        signals=candidate_signals,
        contrarian=contrarian_analysis,
        weights=perf_weights,
    )

    # Log divergence when methods produce different decisions (D-06)
    decisions_by_method = {k: v.collective_decision for k, v in method_comparison.items()}
    unique_decisions = {d for d in decisions_by_method.values() if d is not None}
    if len(unique_decisions) > 1:
        logger.info(
            "Method divergence for %s %s: %s",
            ticker,
            as_of_date,
            decisions_by_method,
        )

    latency_ms = round((time.monotonic() - start) * 1000, 1)

    output = EnsembleOutput(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamental_analysis=fundamental,
        technical_analysis=technical,
        ensemble_decision=decision,
        latency_ms=latency_ms,
        risk_analysis=risk_analysis,
        macro_analysis=macro_analysis,
        sentiment_analysis=sentiment_analysis,
        contrarian_analysis=contrarian_analysis,
        signals=valid_signals,
        aggregation_method="confidence_weighted",
        method_comparison=method_comparison,
    )

    # Verification covers fundamental and technical (Phase 5/6 layer unchanged)
    verification = verify_ensemble(output)
    log_verification_scores(_tracer, trace_id, verification, decision)

    _tracer.flush()

    return output


def run_debate_ensemble(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
    agents: list[str] | None = None,
    use_graphrag: bool = False,
    max_rounds: int = 1,
    _debate_llm: object | None = None,
    memory_prefixes: dict[str, str] | None = None,
    _test_llms: dict | None = None,
) -> EnsembleOutput:
    """
    Run ensemble with structured debate before final vote (P12-E3-T3, P13-E2-T3, DJ-065, DJ-074).

    max_rounds=1 (default) preserves Phase 12 single-round Oxford behaviour.
    max_rounds>1 invokes run_debate_multi_round() with vote-stability convergence (DJ-074).

    Steps
    -----
    1. Run all agents independently (same logic as run_ensemble).
    2. Call run_debate_round() on the initial valid signals.
    3. If debate_skipped: final vote on initial signals.
       Else: final vote on revised_signals from the transcript.
    4. Run contrarian on the final signals + preliminary decision.
    5. run_all_methods on final signals.
    6. Attach debate_transcript to EnsembleOutput and return.

    Parameters
    ----------
    ticker, as_of_date, snapshot_json, data_dir, tracer, use_rag, agents, use_graphrag :
        Identical semantics to run_ensemble().
    _debate_llm : object | None
        Optional LLM override passed to run_debate_round() for deterministic testing.
        Not part of the production API (prefixed _).
    """
    if use_graphrag:
        assert not use_rag, "use_rag and use_graphrag are mutually exclusive (DJ-068)"

    from hifi.collective.debate import run_debate_multi_round, run_debate_round

    _tracer = tracer if tracer is not None else get_tracer()
    trace_id = _tracer.start_trace(
        "run_debate_ensemble", ticker=ticker, as_of_date=as_of_date
    )

    active = _ALL_AGENTS if agents is None else list(agents)
    run_contrarian = "contrarian" in active
    voting_agents = [a for a in active if a != "contrarian"]

    start = time.monotonic()
    _mem = memory_prefixes or {}
    _llms = _test_llms or {}

    graphrag_ctx = _build_graphrag_context(ticker, data_dir) if use_graphrag else ""

    risk_analysis = None
    macro_analysis = None
    sentiment_analysis = None
    contrarian_analysis = None

    with trace_context(trace_id):
        fundamental = run_analysis(
            ticker=ticker,
            as_of_date=as_of_date,
            snapshot_json=snapshot_json,
            data_dir=data_dir,
            tracer=_tracer,
            use_rag=use_rag,
            retrieved_context=graphrag_ctx,
            memory_prefix=_mem.get("fundamental", ""),
            _test_llm=_llms.get("fundamental"),
        )

        technical = run_technical_analysis(
            ticker=ticker,
            as_of_date=as_of_date,
            data_dir=data_dir,
            tracer=_tracer,
            use_rag=use_rag,
            retrieved_context=graphrag_ctx,
            memory_prefix=_mem.get("technical", ""),
            _test_llm=_llms.get("technical"),
        )

        if "risk" in voting_agents:
            from hifi.agents.risk_agent import run_risk_analysis
            risk_analysis = run_risk_analysis(
                ticker=ticker,
                as_of_date=as_of_date,
                data_dir=data_dir,
                tracer=_tracer,
                memory_prefix=_mem.get("risk", ""),
                _test_llm=_llms.get("risk"),
            )

        if "macro" in voting_agents:
            from hifi.agents.macro_agent import run_macro_analysis
            macro_analysis = run_macro_analysis(
                ticker=ticker,
                as_of_date=as_of_date,
                data_dir=data_dir,
                tracer=_tracer,
                memory_prefix=_mem.get("macro", ""),
                _test_llm=_llms.get("macro"),
            )

        if "sentiment" in voting_agents:
            from hifi.agents.sentiment_agent import run_sentiment_analysis
            sentiment_analysis = run_sentiment_analysis(
                ticker=ticker,
                as_of_date=as_of_date,
                data_dir=data_dir,
                tracer=_tracer,
                memory_prefix=_mem.get("sentiment", ""),
                _test_llm=_llms.get("sentiment"),
            )

    # --- Performance weights ---
    perf_weights = get_weights(data_dir=data_dir)

    # --- Initial signals ---
    initial_candidate = [fundamental.signal, technical.signal]
    if risk_analysis is not None:
        initial_candidate.append(risk_analysis.signal)
    if macro_analysis is not None:
        initial_candidate.append(macro_analysis.signal)
    if sentiment_analysis is not None:
        initial_candidate.append(sentiment_analysis.signal)

    initial_valid = [s for s in initial_candidate if s is not None]

    # --- Debate (single or multi-round, DJ-074) ---
    if max_rounds > 1:
        transcripts = run_debate_multi_round(
            initial_signals=initial_valid,
            ticker=ticker,
            as_of_date=as_of_date,
            max_rounds=max_rounds,
            data_dir=data_dir,
            tracer=_tracer,
            llm=_debate_llm,
        )
        transcript = transcripts[-1]  # final round transcript
    else:
        transcript = run_debate_round(
            signals=initial_valid,
            ticker=ticker,
            as_of_date=as_of_date,
            data_dir=data_dir,
            tracer=_tracer,
            llm=_debate_llm,
        )

    # Final signals: revised if debate ran, initial otherwise
    final_signals = (
        transcript.revised_signals if not transcript.debate_skipped else initial_valid
    )
    decision = confidence_weighted_vote(final_signals)

    # --- Contrarian (runs on final post-debate signals) ---
    if run_contrarian:
        from hifi.agents.contrarian_agent import (
            _build_ensemble_context,
            run_contrarian_analysis,
        )

        agent_summaries = [
            {
                "agent_type": sig.agent_type,
                "decision": sig.decision,
                "confidence": sig.confidence,
                "rationale": sig.rationale,
                "key_concern": sig.key_concern,
            }
            for sig in final_signals
        ]

        ensemble_context = _build_ensemble_context(
            ticker=ticker,
            as_of_date=as_of_date,
            agent_summaries=agent_summaries,
            collective_decision=decision.collective_decision,
            collective_confidence=decision.collective_confidence,
        )

        with trace_context(trace_id):
            contrarian_analysis = run_contrarian_analysis(
                ticker=ticker,
                as_of_date=as_of_date,
                ensemble_context=ensemble_context,
                tracer=_tracer,
                _test_llm=_llms.get("contrarian"),
            )

    # --- All aggregation methods on post-debate signals ---
    method_comparison = run_all_methods(
        signals=final_signals,
        contrarian=contrarian_analysis,
        weights=perf_weights,
    )

    decisions_by_method = {k: v.collective_decision for k, v in method_comparison.items()}
    unique_decisions = {d for d in decisions_by_method.values() if d is not None}
    if len(unique_decisions) > 1:
        logger.info(
            "Method divergence (debate) for %s %s: %s",
            ticker,
            as_of_date,
            decisions_by_method,
        )

    latency_ms = round((time.monotonic() - start) * 1000, 1)

    output = EnsembleOutput(
        ticker=ticker,
        as_of_date=as_of_date,
        fundamental_analysis=fundamental,
        technical_analysis=technical,
        ensemble_decision=decision,
        latency_ms=latency_ms,
        risk_analysis=risk_analysis,
        macro_analysis=macro_analysis,
        sentiment_analysis=sentiment_analysis,
        contrarian_analysis=contrarian_analysis,
        signals=final_signals,
        aggregation_method="confidence_weighted",
        method_comparison=method_comparison,
        debate_transcript=transcript,
    )

    verification = verify_ensemble(output)
    log_verification_scores(_tracer, trace_id, verification, decision)
    _tracer.flush()

    return output
