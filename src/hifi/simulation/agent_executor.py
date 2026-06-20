"""
Per-agent execution functions for the agent-first sequential sweep (E0-T2, DJ-106).

Extracts per-agent execution logic from run_sequential_ensemble() so each agent
can be run independently across all tickers before loading the next model:

  1. load_model(fundamental_model_id)
  2. for each ticker: run_agent_pass("fundamental", ticker, ...)
  3. unload_model(fundamental_model_id)
  4. load_model(technical_model_id)
  5. ...repeat for all 6 agents...
  6. for each ticker: aggregate_agent_outputs(ticker, ...)

Storage
-------
Per-agent full analysis objects are saved to:
    {data_dir}/runs/{run_id}/{ticker}_{agent_type}.json

AgentContextStore (LanceDB) stores the <=300-char summary for inter-agent context
injection, allowing subsequent agents to read prior agents' decisions.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hifi.collective.schemas import EnsembleOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _runs_dir(data_dir: str, run_id: str) -> Path:
    """Return directory for per-agent JSON outputs for a given run."""
    return Path(data_dir) / "runs" / run_id


def _agent_json_path(data_dir: str, run_id: str, ticker: str, agent_type: str) -> Path:
    return _runs_dir(data_dir, run_id) / f"{ticker}_{agent_type}.json"


def _save_analysis(
    data_dir: str, run_id: str, ticker: str, agent_type: str, analysis: Any
) -> None:
    """Persist full analysis object as JSON sidecar (fail-safe)."""
    try:
        path = _agent_json_path(data_dir, run_id, ticker, agent_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(analysis.model_dump_json(), encoding="utf-8")
    except Exception as exc:
        logger.warning(
            "Failed to save analysis JSON for %s/%s: %s", ticker, agent_type, exc
        )


def _load_analysis(
    data_dir: str, run_id: str, ticker: str, agent_type: str
) -> dict | None:
    """Load stored analysis JSON dict, or None if missing/corrupt."""
    try:
        path = _agent_json_path(data_dir, run_id, ticker, agent_type)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Failed to load analysis JSON for %s/%s: %s", ticker, agent_type, exc
        )
        return None


def _read_stored_signals(
    data_dir: str, run_id: str, ticker: str, date: str
) -> list[dict]:
    """
    Read stored agent signal dicts from JSON sidecars.

    Used by run_agent_pass() for the contrarian agent to build ensemble context
    from all previously-completed agent passes.
    """
    from hifi.knowledge.agent_context import CANONICAL_ORDER

    signals = []
    for agent_type in CANONICAL_ORDER:
        if agent_type == "contrarian":
            break
        data = _load_analysis(data_dir, run_id, ticker, agent_type)
        if data is None:
            continue
        sig = data.get("signal")
        if not isinstance(sig, dict):
            continue
        signals.append({
            "agent_type": sig.get("agent_type", agent_type),
            "decision": sig.get("decision", "Hold"),
            "confidence": float(sig.get("confidence", 0.5)),
            "rationale": sig.get("rationale", "") or f"{agent_type} analysis complete",
            "key_concern": sig.get("key_concern", "") or "See analysis summary",
        })
    return signals


# ---------------------------------------------------------------------------
# AgentContextStore helpers
# ---------------------------------------------------------------------------


def _store_context(
    store: Any,
    run_id: str,
    ticker: str,
    date: str,
    agent_type: str,
    analysis: Any,
) -> None:
    """Write summary to AgentContextStore for next-agent context injection (fail-safe)."""
    try:
        from hifi.knowledge.agent_context import AgentContextRecord

        sig = getattr(analysis, "signal", None)
        if sig is None:
            return
        rationale = getattr(sig, "rationale", "") or ""
        summary = rationale[:300] if rationale else f"{agent_type} analysis complete"
        record = AgentContextRecord(
            run_id=run_id,
            ticker=ticker,
            date=date,
            agent_type=agent_type,
            analysis_summary=summary,
            decision=sig.decision,
            confidence=float(sig.confidence),
            created_at=datetime.now(UTC).isoformat(),
        )
        store.write(record)
    except Exception as exc:
        logger.warning(
            "Failed to store context for %s/%s: %s", ticker, agent_type, exc
        )


def _build_memory_prefix(
    store: Any,
    run_id: str,
    agent_type: str,
    ticker: str,
    date: str,
    condition: str,
) -> str:
    """
    Return inter-agent context prefix based on condition.

    - "parallel": no inter-agent context (independent passes)
    - "full" / "no-memory" / "homogeneous": read prior-agent summaries from store
    """
    if condition == "parallel":
        return ""
    try:
        from hifi.knowledge.agent_context import format_prior_context

        records = store.read_prior(run_id, agent_type)
        return format_prior_context(records, ticker, date)
    except Exception as exc:
        logger.warning(
            "Failed to read prior context for %s/%s: %s", ticker, agent_type, exc
        )
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent_pass(
    agent_type: str,
    ticker: str,
    date: str,
    condition: str,
    run_id: str,
    data_dir: str,
    db_path: str,
    snapshot_json: str | None = None,
    context_namespace: str = "hifi-dev-context",
    _test_llm: object | None = None,
    _test_store: object | None = None,
) -> Any:
    """
    Run exactly ONE agent for one ticker and store the result.

    Parameters
    ----------
    agent_type : str
        One of: "fundamental", "technical", "risk", "macro", "sentiment", "contrarian".
    ticker : str
        Ticker symbol.
    date : str
        ISO 8601 evaluation date (e.g. "2022-01-31").
    condition : str
        Ablation condition: "full" | "parallel" | "homogeneous" | "no-memory".
        "parallel" disables inter-agent context; other values enable it.
    run_id : str
        Deterministic run identifier (shared across all agents for this date).
    data_dir : str
        Root data directory (e.g. "data").
    db_path : str
        LanceDB path for AgentContextStore (e.g. "data/knowledge.lance").
    snapshot_json : str | None
        JSON-serialized FundamentalsSnapshot. Auto-built as minimal snapshot when
        None and agent_type is "fundamental".
    context_namespace : str
        LanceDB namespace for AgentContextStore (default "hifi-dev-context").
    _test_llm : object | None
        Stub LLM for deterministic testing.
    _test_store : object | None
        Pre-built AgentContextStore for testing (skips LanceDB construction).

    Returns
    -------
    Analysis object (FundamentalAnalysis | TechnicalAnalysis | RiskAnalysis |
    MacroAnalysis | SentimentAnalysis | ContrarianAnalysis).
    """
    start = time.monotonic()

    # Build or reuse AgentContextStore
    if _test_store is not None:
        store = _test_store
    else:
        from hifi.knowledge.agent_context import AgentContextStore
        store = AgentContextStore(namespace=context_namespace, db_path=db_path)

    memory_prefix = _build_memory_prefix(store, run_id, agent_type, ticker, date, condition)

    analysis: Any

    if agent_type == "fundamental":
        from hifi.agents.fundamental_agent import run_analysis
        from hifi.simulation.snapshot import build_minimal_snapshot

        _snap = snapshot_json or build_minimal_snapshot(ticker, date)
        analysis = run_analysis(
            ticker=ticker,
            as_of_date=date,
            snapshot_json=_snap,
            data_dir=data_dir,
            memory_prefix=memory_prefix,
            _test_llm=_test_llm,
        )

    elif agent_type == "technical":
        from hifi.agents.technical_agent import run_technical_analysis

        analysis = run_technical_analysis(
            ticker=ticker,
            as_of_date=date,
            data_dir=data_dir,
            memory_prefix=memory_prefix,
            _test_llm=_test_llm,
        )

    elif agent_type == "risk":
        from hifi.agents.risk_agent import run_risk_analysis

        analysis = run_risk_analysis(
            ticker=ticker,
            as_of_date=date,
            data_dir=data_dir,
            memory_prefix=memory_prefix,
            _test_llm=_test_llm,
        )

    elif agent_type == "macro":
        from hifi.agents.macro_agent import run_macro_analysis

        analysis = run_macro_analysis(
            ticker=ticker,
            as_of_date=date,
            data_dir=data_dir,
            memory_prefix=memory_prefix,
            _test_llm=_test_llm,
        )

    elif agent_type == "sentiment":
        from hifi.agents.sentiment_agent import run_sentiment_analysis

        analysis = run_sentiment_analysis(
            ticker=ticker,
            as_of_date=date,
            data_dir=data_dir,
            memory_prefix=memory_prefix,
            _test_llm=_test_llm,
        )

    elif agent_type == "contrarian":
        from hifi.agents.contrarian_agent import (
            _build_ensemble_context,
            run_contrarian_analysis,
        )
        from hifi.collective.voting import confidence_weighted_vote

        # Build preliminary voting result from stored non-contrarian signals
        stored = _read_stored_signals(data_dir, run_id, ticker, date)
        collective_decision: str = "Hold"
        collective_confidence: float = 0.5
        if stored:
            try:
                from hifi.agents.schemas import AgentSignal

                sigs = [
                    AgentSignal(
                        ticker=ticker,
                        as_of_date=date,
                        decision=s["decision"],
                        confidence=s["confidence"],
                        rationale=s["rationale"],
                        key_concern=s["key_concern"],
                        model_id="aggregate",
                        agent_type=s["agent_type"],
                    )
                    for s in stored
                ]
                vote = confidence_weighted_vote(sigs)
                collective_decision = vote.collective_decision or "Hold"
                collective_confidence = vote.collective_confidence
            except Exception as exc:
                logger.warning("Preliminary vote failed for contrarian: %s", exc)

        ensemble_context = _build_ensemble_context(
            ticker=ticker,
            as_of_date=date,
            agent_summaries=stored,
            collective_decision=collective_decision,
            collective_confidence=collective_confidence,
        )
        if memory_prefix:
            ensemble_context = memory_prefix + "\n\n" + ensemble_context

        analysis = run_contrarian_analysis(
            ticker=ticker,
            as_of_date=date,
            ensemble_context=ensemble_context,
            _test_llm=_test_llm,
        )

    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")

    elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    logger.debug(
        "run_agent_pass agent=%s ticker=%s date=%s condition=%s elapsed_ms=%.1f",
        agent_type, ticker, date, condition, elapsed_ms,
    )

    # Persist full analysis JSON for later aggregation
    _save_analysis(data_dir, run_id, ticker, agent_type, analysis)

    # Write inter-agent context summary (contrarian has no signal, skipped automatically)
    _store_context(store, run_id, ticker, date, agent_type, analysis)

    return analysis


def aggregate_agent_outputs(
    ticker: str,
    date: str,
    run_id: str,
    db_path: str,
    context_namespace: str = "hifi-dev-context",
) -> EnsembleOutput:
    """
    Aggregate stored per-agent analysis JSONs into a full EnsembleOutput.

    Reads per-agent JSON files written by run_agent_pass() from
    ``{data_dir}/runs/{run_id}/{ticker}_{agent_type}.json`` and reconstructs
    EnsembleOutput via the standard confidence-weighted voting machinery.

    ``data_dir`` is derived from ``db_path`` (its parent directory).

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    date : str
        ISO 8601 evaluation date.
    run_id : str
        The run identifier used by the preceding run_agent_pass() calls.
    db_path : str
        LanceDB path — used to derive data_dir as Path(db_path).parent.
    context_namespace : str
        Kept for signature symmetry; unused in aggregation.

    Returns
    -------
    EnsembleOutput
        Full ensemble output reconstructed from stored per-agent analyses.

    Raises
    ------
    RuntimeError
        If fundamental or technical analysis files are missing.
    """
    from hifi.agents.schemas import (
        ContrarianAnalysis,
        FundamentalAnalysis,
        MacroAnalysis,
        RiskAnalysis,
        SentimentAnalysis,
        TechnicalAnalysis,
    )
    from hifi.collective.performance_store import get_weights
    from hifi.collective.schemas import EnsembleOutput
    from hifi.collective.voting import confidence_weighted_vote, run_all_methods

    data_dir = str(Path(db_path).parent)
    start = time.monotonic()

    def _load(agent_type: str, cls: type) -> Any:
        data = _load_analysis(data_dir, run_id, ticker, agent_type)
        if data is None:
            return None
        try:
            return cls.model_validate(data)
        except Exception as exc:
            logger.warning(
                "Failed to deserialize %s/%s: %s", agent_type, ticker, exc
            )
            return None

    fundamental = _load("fundamental", FundamentalAnalysis)
    technical = _load("technical", TechnicalAnalysis)
    risk_analysis = _load("risk", RiskAnalysis)
    macro_analysis = _load("macro", MacroAnalysis)
    sentiment_analysis = _load("sentiment", SentimentAnalysis)
    contrarian_analysis = _load("contrarian", ContrarianAnalysis)

    if fundamental is None or technical is None:
        raise RuntimeError(
            f"Missing fundamental or technical analysis for {ticker}/{date} "
            f"run_id={run_id}. Ensure run_agent_pass() completed for both agents."
        )

    # Reconstruct candidate signals (None signals filtered in voting)
    candidate: list[Any] = [fundamental.signal, technical.signal]
    for ana in (risk_analysis, macro_analysis, sentiment_analysis):
        if ana is not None:
            candidate.append(ana.signal)

    valid_signals = [s for s in candidate if s is not None]
    decision = confidence_weighted_vote(valid_signals)

    perf_weights = get_weights(data_dir=data_dir)
    method_comparison = run_all_methods(
        signals=candidate,
        contrarian=contrarian_analysis,
        weights=perf_weights,
    )

    latency_ms = round((time.monotonic() - start) * 1000, 1)

    return EnsembleOutput(
        ticker=ticker,
        as_of_date=date,
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
