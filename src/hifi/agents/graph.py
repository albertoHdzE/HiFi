"""
LangGraph StateGraph for sequential ensemble execution (E3-T3, DJ-089b).

Formalises the causal agent topology:
  fundamental → technical → risk → macro → sentiment → contrarian → END

Each node:
  1. Reads prior agents' summaries from AgentContextStore.
  2. Augments the agent's memory_prefix with the prior-context block.
  3. Calls the agent function.
  4. Stores the result in AgentContextStore.

AgentContextStore is carried as shared mutable state (``context_store`` key)
so all nodes operate on the same run_id accumulator.

Entry point
-----------
``build_sequential_graph()`` returns a compiled LangGraph that can be invoked
as an alternative to ``run_sequential_ensemble()``.

Testing
-------
Topology tests use ``build_sequential_graph().get_graph()`` to verify edges
and confirm no cycles exist.  No LLM or LanceDB required for topology tests.
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Any

from typing_extensions import TypedDict

from hifi.knowledge.agent_context import (
    CANONICAL_ORDER,
    AgentContextRecord,
    AgentContextStore,
    format_prior_context,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------


class SequentialEnsembleState(TypedDict):
    """Shared mutable state flowing through the sequential ensemble graph."""

    # Run configuration (set at invocation)
    ticker: str
    as_of_date: str
    snapshot_json: str
    data_dir: str | None
    use_rag: bool
    use_graphrag: bool
    memory_prefixes: dict[str, str]
    context_store: object | None   # AgentContextStore (mutable, shared)
    run_id: str
    _test_llms: dict[str, object]  # agent_type → mock LLM (for testing)
    _tracer: object | None

    # Agent outputs (populated as the graph progresses)
    fundamental_analysis: object | None
    technical_analysis: object | None
    risk_analysis: object | None
    macro_analysis: object | None
    sentiment_analysis: object | None
    contrarian_analysis: object | None


# ---------------------------------------------------------------------------
# Summary extraction helper
# ---------------------------------------------------------------------------


def _extract_summary(analysis: Any, agent_type: str) -> str:
    """Extract a ≤300-character rationale summary from an analysis object."""
    rationale = ""
    try:
        if hasattr(analysis, "signal") and analysis.signal is not None:
            rationale = getattr(analysis.signal, "rationale", "") or ""
        if not rationale and hasattr(analysis, "rationale"):
            rationale = getattr(analysis, "rationale", "") or ""
    except Exception:
        pass
    if rationale:
        return rationale[:300]
    return f"{agent_type} analysis complete (no rationale)"


def _build_context_record(
    run_id: str,
    ticker: str,
    date: str,
    agent_type: str,
    analysis: Any,
) -> AgentContextRecord | None:
    """Build an AgentContextRecord from an analysis object; return None on failure."""
    try:
        import datetime as _dt

        signal = getattr(analysis, "signal", None)
        if signal is None:
            return None
        return AgentContextRecord(
            run_id=run_id,
            ticker=ticker,
            date=date,
            agent_type=agent_type,
            analysis_summary=_extract_summary(analysis, agent_type),
            decision=signal.decision,
            confidence=float(signal.confidence),
            created_at=_dt.datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        logger.warning("Could not build context record for %s: %s", agent_type, exc)
        return None


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def _augmented_prefix(
    state: SequentialEnsembleState,
    agent_type: str,
) -> str:
    """Return the memory prefix augmented with prior-agent context."""
    store: AgentContextStore | None = state.get("context_store")  # type: ignore[assignment]
    existing = (state.get("memory_prefixes") or {}).get(agent_type, "")
    if store is None:
        return existing
    run_id = state.get("run_id", "")
    records = store.read_prior(run_id, agent_type)
    ctx_block = format_prior_context(records, state["ticker"], state["as_of_date"])
    if not ctx_block:
        return existing
    if existing:
        return ctx_block + "\n\n" + existing
    return ctx_block


def _store_context(
    state: SequentialEnsembleState,
    agent_type: str,
    analysis: Any,
) -> None:
    """Write agent result to the context store if the store is available."""
    store: AgentContextStore | None = state.get("context_store")  # type: ignore[assignment]
    if store is None:
        return
    record = _build_context_record(
        run_id=state.get("run_id", ""),
        ticker=state["ticker"],
        date=state["as_of_date"],
        agent_type=agent_type,
        analysis=analysis,
    )
    if record is not None:
        store.write(record)


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


def _fundamental_node(state: SequentialEnsembleState) -> dict:
    from hifi.agents.fundamental_agent import run_analysis
    from hifi.observability.tracing import get_tracer

    _llms = state.get("_test_llms") or {}
    _tracer = state.get("_tracer") or get_tracer()
    analysis = run_analysis(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        snapshot_json=state["snapshot_json"],
        data_dir=state.get("data_dir"),
        tracer=_tracer,
        use_rag=state.get("use_rag", False),
        memory_prefix=_augmented_prefix(state, "fundamental"),
        _test_llm=_llms.get("fundamental"),
    )
    _store_context(state, "fundamental", analysis)
    return {"fundamental_analysis": analysis}


def _technical_node(state: SequentialEnsembleState) -> dict:
    from hifi.agents.technical_agent import run_technical_analysis
    from hifi.observability.tracing import get_tracer

    _llms = state.get("_test_llms") or {}
    _tracer = state.get("_tracer") or get_tracer()
    analysis = run_technical_analysis(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        data_dir=state.get("data_dir"),
        tracer=_tracer,
        use_rag=state.get("use_rag", False),
        memory_prefix=_augmented_prefix(state, "technical"),
        _test_llm=_llms.get("technical"),
    )
    _store_context(state, "technical", analysis)
    return {"technical_analysis": analysis}


def _risk_node(state: SequentialEnsembleState) -> dict:
    from hifi.agents.risk_agent import run_risk_analysis
    from hifi.observability.tracing import get_tracer

    _llms = state.get("_test_llms") or {}
    _tracer = state.get("_tracer") or get_tracer()
    analysis = run_risk_analysis(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        data_dir=state.get("data_dir"),
        tracer=_tracer,
        memory_prefix=_augmented_prefix(state, "risk"),
        _test_llm=_llms.get("risk"),
    )
    _store_context(state, "risk", analysis)
    return {"risk_analysis": analysis}


def _macro_node(state: SequentialEnsembleState) -> dict:
    from hifi.agents.macro_agent import run_macro_analysis
    from hifi.observability.tracing import get_tracer

    _llms = state.get("_test_llms") or {}
    _tracer = state.get("_tracer") or get_tracer()
    analysis = run_macro_analysis(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        data_dir=state.get("data_dir"),
        tracer=_tracer,
        memory_prefix=_augmented_prefix(state, "macro"),
        _test_llm=_llms.get("macro"),
    )
    _store_context(state, "macro", analysis)
    return {"macro_analysis": analysis}


def _sentiment_node(state: SequentialEnsembleState) -> dict:
    from hifi.agents.sentiment_agent import run_sentiment_analysis
    from hifi.observability.tracing import get_tracer

    _llms = state.get("_test_llms") or {}
    _tracer = state.get("_tracer") or get_tracer()
    analysis = run_sentiment_analysis(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        data_dir=state.get("data_dir"),
        tracer=_tracer,
        memory_prefix=_augmented_prefix(state, "sentiment"),
        _test_llm=_llms.get("sentiment"),
    )
    _store_context(state, "sentiment", analysis)
    return {"sentiment_analysis": analysis}


def _contrarian_node(state: SequentialEnsembleState) -> dict:
    from hifi.agents.contrarian_agent import (
        _build_ensemble_context,
        run_contrarian_analysis,
    )
    from hifi.collective.voting import confidence_weighted_vote
    from hifi.observability.tracing import get_tracer

    _llms = state.get("_test_llms") or {}
    _tracer = state.get("_tracer") or get_tracer()

    # Collect valid signals from voting agents
    analyses = [
        state.get("fundamental_analysis"),
        state.get("technical_analysis"),
        state.get("risk_analysis"),
        state.get("macro_analysis"),
        state.get("sentiment_analysis"),
    ]
    valid_signals = [
        a.signal for a in analyses if a is not None and getattr(a, "signal", None) is not None
    ]
    decision = confidence_weighted_vote(valid_signals)

    agent_summaries = [
        {
            "agent_type": sig.agent_type,
            "decision": sig.decision,
            "confidence": sig.confidence,
            "rationale": sig.rationale,
            "key_concern": sig.key_concern,
        }
        for sig in valid_signals
    ]

    ensemble_context = _build_ensemble_context(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        agent_summaries=agent_summaries,
        collective_decision=decision.collective_decision,
        collective_confidence=decision.collective_confidence,
    )

    # Augment ensemble_context with prior-agent context block from store
    store: AgentContextStore | None = state.get("context_store")  # type: ignore[assignment]
    if store is not None:
        run_id = state.get("run_id", "")
        records = store.read_prior(run_id, "contrarian")
        ctx_block = format_prior_context(records, state["ticker"], state["as_of_date"])
        if ctx_block:
            ensemble_context = ctx_block + "\n\n" + ensemble_context

    analysis = run_contrarian_analysis(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        ensemble_context=ensemble_context,
        tracer=_tracer,
        _test_llm=_llms.get("contrarian"),
    )
    return {"contrarian_analysis": analysis}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_sequential_graph() -> Any:
    """
    Compile and return the sequential ensemble LangGraph.

    The returned graph can be invoked with a ``SequentialEnsembleState`` dict.
    Topology: fundamental → technical → risk → macro → sentiment → contrarian → END.

    Used primarily for:
    - Formal topology declaration and cycle detection at compile time.
    - Alternative invocation path to ``run_sequential_ensemble()``.
    - Topology tests (``build_sequential_graph().get_graph()``).
    """
    from langgraph.graph import END, StateGraph

    _node_fns = {
        "fundamental": _fundamental_node,
        "technical": _technical_node,
        "risk": _risk_node,
        "macro": _macro_node,
        "sentiment": _sentiment_node,
        "contrarian": _contrarian_node,
    }

    workflow: StateGraph = StateGraph(SequentialEnsembleState)

    # Register nodes in CANONICAL_ORDER
    for agent in CANONICAL_ORDER:
        workflow.add_node(agent, _node_fns[agent])

    # Connect in CANONICAL_ORDER: fundamental → technical → … → contrarian → END
    workflow.set_entry_point(CANONICAL_ORDER[0])
    for i in range(len(CANONICAL_ORDER) - 1):
        workflow.add_edge(CANONICAL_ORDER[i], CANONICAL_ORDER[i + 1])
    workflow.add_edge(CANONICAL_ORDER[-1], END)

    return workflow.compile()
