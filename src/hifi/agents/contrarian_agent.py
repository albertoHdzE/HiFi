"""
Contrarian Agent for HiFi (P8-E5).

Second-pass critic: receives all other agents' outputs and the preliminary
ensemble decision, then produces an adversarial stress test (DJ-033).

Design differences from other agents:
- No LangGraph graph (direct LLM call — no MCP tools needed)
- No AgentSignal (does not vote in confidence_weighted_vote)
- Receives formatted ensemble_context string rather than MCP tool outputs
- Always runs LAST in run_ensemble() after voting is complete

Model selection
---------------
Controlled by HIFI_CONTRARIAN_MODEL env var (DJ-032).
Default: mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled.
Reasoning-distilled model: max_tokens=4096 required to avoid JSON truncation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from hifi.agents.lm_client import make_llm
from hifi.agents.schemas import ContrarianAnalysis
from hifi.observability.tracing import AbstractTracer, get_tracer, trace_context

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "contrarian_v1"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{_PROMPT_VERSION}.md"
_DEFAULT_CONTRARIAN_MODEL = "mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled"
_RETRY_MSG = (
    "Your previous response was not valid JSON or was missing required fields. "
    "Produce ONLY the JSON object with the fields: "
    "alternative_thesis (string), risk_scenario (string), "
    "counterargument (string), confidence (0.0-1.0). "
    "No other text."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contrarian_model() -> str:
    return os.environ.get("HIFI_CONTRARIAN_MODEL", _DEFAULT_CONTRARIAN_MODEL)


def _load_prompt_template() -> tuple[str, str]:
    raw = _PROMPT_PATH.read_text(encoding="utf-8")
    parts = raw.split("## User", maxsplit=1)
    system_block = parts[0].replace("## System", "").strip()
    user_block = parts[1].strip() if len(parts) > 1 else ""
    return system_block, user_block


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [line for line in lines if not line.startswith("```")]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _build_ensemble_context(
    ticker: str,
    as_of_date: str,
    agent_summaries: list[dict],
    collective_decision: str | None,
    collective_confidence: float,
) -> str:
    """
    Format the ensemble context string for the Contrarian Agent.

    Includes each contributing agent's decision/confidence/rationale plus
    the preliminary collective decision and confidence.
    """
    ctx: dict = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "preliminary_collective_decision": collective_decision,
        "preliminary_collective_confidence": round(collective_confidence, 3),
        "contributing_agents": agent_summaries,
    }
    return json.dumps(ctx, indent=2)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_contrarian_analysis(
    ticker: str,
    as_of_date: str,
    ensemble_context: str,
    tracer: AbstractTracer | None = None,
    _test_llm: object | None = None,
) -> ContrarianAnalysis:
    """
    Run the Contrarian Agent for one ticker on one date.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    as_of_date : str
        ISO 8601 analysis date (e.g. "2023-03-31").
    ensemble_context : str
        JSON string containing all other agents' outputs and the preliminary
        collective decision. Built by run_ensemble() via _build_ensemble_context().
    tracer : AbstractTracer | None
        Observability tracer.

    Returns
    -------
    ContrarianAnalysis
        Adversarial stress test. Does not contain an AgentSignal and does not
        affect the collective_decision in Phase 8 (DJ-033).
    """
    _tracer = tracer if tracer is not None else get_tracer()
    trace_id = _tracer.start_trace(
        "contrarian_agent", ticker=ticker, as_of_date=as_of_date
    )
    handler = _tracer.get_callback_handler(trace_id)
    # Only add config when tracing is active (keeps untraced call signature same).
    _kw = {"config": {"callbacks": [handler]}} if handler is not None else {}

    start = time.monotonic()
    system_text, user_template = _load_prompt_template()
    user_text = user_template.format(
        ticker=ticker,
        as_of_date=as_of_date,
        ensemble_context=ensemble_context,
    )

    llm = _test_llm if _test_llm is not None else make_llm(_contrarian_model(), max_tokens=4096)
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]

    with trace_context(trace_id):
        response = llm.invoke(messages, **_kw)
        raw = response.content

        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning("Contrarian first parse failed for %s. Retrying.", ticker)
            retry_response = (
                _test_llm if _test_llm is not None else llm
            ).invoke([
                HumanMessage(content=raw),
                HumanMessage(content=_RETRY_MSG),
            ], **_kw)
            raw = retry_response.content
            parsed = _extract_json(raw)

    _tracer.flush()
    latency_ms = (time.monotonic() - start) * 1000

    if parsed is None:
        logger.error("Contrarian agent failed to produce valid JSON for %s", ticker)
        return ContrarianAnalysis(
            alternative_thesis="Parse failure: could not extract JSON from LLM response.",
            risk_scenario="Unknown.",
            counterargument="Unknown.",
            confidence=0.0,
            prompt_version=_PROMPT_VERSION,
            latency_ms=round(latency_ms, 1),
        )

    try:
        return ContrarianAnalysis(
            alternative_thesis=parsed.get("alternative_thesis", "Not provided."),
            risk_scenario=parsed.get("risk_scenario", "Not provided."),
            counterargument=parsed.get("counterargument", "Not provided."),
            confidence=float(parsed.get("confidence", 0.0)),
            prompt_version=_PROMPT_VERSION,
            latency_ms=round(latency_ms, 1),
        )
    except Exception as exc:
        logger.error("ContrarianAnalysis build failed for %s: %s", ticker, exc)
        return ContrarianAnalysis(
            alternative_thesis="Schema validation error.",
            risk_scenario="Unknown.",
            counterargument="Unknown.",
            confidence=0.0,
            prompt_version=_PROMPT_VERSION,
            latency_ms=round(latency_ms, 1),
        )
