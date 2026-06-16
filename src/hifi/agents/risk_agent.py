"""
Risk Analyst Agent for HiFi (P8-E2).

LangGraph graph that calls the get_risk_metrics MCP tool and asks a local LLM
(via LM Studio) to assess investment risk based exclusively on risk metrics.

Graph structure
---------------
call_mcp_tools -> generate_analysis -> parse_output -> END

Information restriction: risk metrics only (hist_vol, beta, max_drawdown, Sharpe,
VaR). No fundamentals, technical indicators, or macro data. This restriction is
the primary diversity mechanism relative to the Technical Agent (David §10.3).

Model selection
---------------
Controlled by HIFI_RISK_MODEL env var (DJ-032). Default: google/gemma-3-4b.
Non-reasoning model: max_tokens=1024 is sufficient.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict

from hifi.agents.lm_client import make_llm
from hifi.agents.mcp_client import call_tool
from hifi.agents.schemas import AgentSignal, RiskAnalysis
from hifi.observability.tracing import AbstractTracer, get_tracer, trace_context

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "risk_v1"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{_PROMPT_VERSION}.md"
_DEFAULT_RISK_MODEL = "google/gemma-3-4b"
_RETRY_MSG = (
    "Your previous response was not valid JSON or was missing required fields. "
    "Produce ONLY the JSON object with the fields: "
    "decision (Buy/Hold/Sell), confidence (0.0-1.0), rationale (string), "
    "key_concern (string), risk_assessment (string), "
    "recommended_position_size (float or null). "
    "No other text."
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class RiskAnalystState(TypedDict, total=False):
    ticker: str
    as_of_date: str
    data_dir: str
    tool_results: dict
    llm_response: str
    signal: AgentSignal | None
    risk_assessment: str | None
    recommended_position_size: float | None
    error: str | None
    start_time: float
    memory_prefix: str  # P13-E4-T3: in-context decision history prefix (DJ-076)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _risk_model() -> str:
    return os.environ.get("HIFI_RISK_MODEL", _DEFAULT_RISK_MODEL)


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


def _build_risk_signal(
    parsed: dict,
    ticker: str,
    as_of_date: str,
    model_id: str,
    call_ids: list[str],
    data_gaps: list[str],
) -> tuple[AgentSignal | None, str | None, float | None]:
    """
    Construct an AgentSignal + risk_assessment + recommended_position_size.

    Returns (signal, risk_assessment, recommended_position_size) or (None, None, None).
    """
    try:
        signal = AgentSignal(
            ticker=ticker,
            as_of_date=as_of_date,
            decision=parsed["decision"],
            confidence=float(parsed["confidence"]),
            rationale=parsed["rationale"],
            key_concern=parsed["key_concern"],
            data_gaps=data_gaps,
            call_ids=call_ids,
            model_id=model_id,
            agent_type="risk",
        )
        risk_assessment = parsed.get("risk_assessment", "")
        pos_size = parsed.get("recommended_position_size")
        if pos_size is not None:
            pos_size = float(pos_size)
        return signal, risk_assessment, pos_size
    except Exception as exc:
        logger.warning("RiskAnalysis signal build failed: %s", exc)
        return None, None, None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def call_mcp_tools_node(state: RiskAnalystState) -> dict:
    """
    Call get_risk_metrics — the only tool the Risk Agent is permitted to use.

    Information restriction enforces diversity from the Technical Agent (David §10.3).
    """
    ticker = state["ticker"]
    as_of_date = state["as_of_date"]
    data_dir = state.get("data_dir") or os.environ.get("HIFI_DATA_DIR", "data")

    def _call(tool: str, params: dict) -> dict:
        try:
            return call_tool(tool, params, data_dir=data_dir)
        except Exception as exc:
            logger.warning("MCP tool %s failed: %s", tool, exc)
            return {"error": "COMPUTATION_ERROR", "detail": str(exc)}

    risk_metrics = _call(
        "get_risk_metrics",
        {"ticker": ticker, "date": as_of_date},
    )

    return {"tool_results": {"risk_metrics": risk_metrics}}


def generate_analysis_node(state: RiskAnalystState) -> dict:
    """Fill the risk prompt template and call the LM Studio model."""
    tool_results = state["tool_results"]

    data_gaps: list[str] = []
    for result_dict in tool_results.values():
        for k, v in result_dict.items():
            if v is None and k not in ("call_id", "error", "detail"):
                data_gaps.append(k)
    data_gaps_list = ", ".join(data_gaps) if data_gaps else "none"

    system_text, user_template = _load_prompt_template()
    user_text = user_template.format(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        risk_metrics=json.dumps(tool_results.get("risk_metrics", {}), indent=2),
        data_gaps_list=data_gaps_list,
    )

    # P13-E4-T3: prepend agent memory prefix when available (DJ-076)
    memory_prefix = state.get("memory_prefix", "")
    if memory_prefix:
        user_text = memory_prefix + "\n\n" + user_text

    llm = make_llm(_risk_model(), max_tokens=1024)
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    response = llm.invoke(messages)
    return {"llm_response": response.content}


def parse_output_node(state: RiskAnalystState) -> dict:
    """Parse the LLM response into an AgentSignal + risk fields. One retry on failure."""
    ticker = state["ticker"]
    as_of_date = state["as_of_date"]
    tool_results = state["tool_results"]
    llm_response = state.get("llm_response", "")

    call_ids = [
        r["call_id"]
        for r in tool_results.values()
        if isinstance(r, dict) and "call_id" in r
    ]
    data_gaps = [
        k
        for result_dict in tool_results.values()
        for k, v in result_dict.items()
        if v is None and k not in ("call_id", "error", "detail")
    ]

    llm = make_llm(_risk_model(), max_tokens=1024)
    model_id = llm.model_name

    def _try_parse(text: str):
        parsed = _extract_json(text)
        if parsed is None:
            return None, None, None
        return _build_risk_signal(parsed, ticker, as_of_date, model_id, call_ids, data_gaps)

    signal, risk_assessment, pos_size = _try_parse(llm_response)
    if signal is not None:
        return {
            "signal": signal,
            "risk_assessment": risk_assessment,
            "recommended_position_size": pos_size,
        }

    logger.warning("First parse attempt failed for %s. Retrying.", ticker)
    retry_response = llm.invoke([
        HumanMessage(content=llm_response),
        HumanMessage(content=_RETRY_MSG),
    ])
    signal, risk_assessment, pos_size = _try_parse(retry_response.content)
    if signal is not None:
        return {
            "signal": signal,
            "risk_assessment": risk_assessment,
            "recommended_position_size": pos_size,
            "llm_response": retry_response.content,
        }

    return {
        "error": (
            f"Failed to parse RiskAnalysis after retry. "
            f"Last response: {retry_response.content[:200]}"
        )
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _should_abort(state: RiskAnalystState) -> str:
    return "abort" if state.get("error") else "continue"


def build_risk_graph():
    """Build and compile the Risk Analyst LangGraph (3-node, no conditional RAG)."""
    from langgraph.graph import END, StateGraph

    builder = StateGraph(RiskAnalystState)
    builder.add_node("call_mcp_tools", call_mcp_tools_node)
    builder.add_node("generate_analysis", generate_analysis_node)
    builder.add_node("parse_output", parse_output_node)

    builder.set_entry_point("call_mcp_tools")
    builder.add_conditional_edges(
        "call_mcp_tools",
        _should_abort,
        {"continue": "generate_analysis", "abort": END},
    )
    builder.add_edge("generate_analysis", "parse_output")
    builder.add_edge("parse_output", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_risk_analysis(
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    memory_prefix: str = "",
) -> RiskAnalysis:
    """
    Run the Risk Analyst Agent for one ticker on one date.

    Parameters
    ----------
    ticker : str
        Ticker symbol (must match a Parquet file in data_dir/market/).
    as_of_date : str
        ISO 8601 analysis date (e.g. "2023-03-31").
    data_dir : str | None
        Path to the data root directory. Defaults to HIFI_DATA_DIR env var.
    tracer : AbstractTracer | None
        Observability tracer.

    Returns
    -------
    RiskAnalysis
        Full risk analysis including AgentSignal (or None on failure) and raw
        risk metrics tool result.
    """
    _tracer = tracer if tracer is not None else get_tracer()
    trace_id = _tracer.start_trace(
        "risk_agent", ticker=ticker, as_of_date=as_of_date
    )
    handler = _tracer.get_callback_handler(trace_id)
    config = {"callbacks": [handler]} if handler is not None else {}

    start = time.monotonic()
    graph = build_risk_graph()

    effective_data_dir = data_dir or os.environ.get("HIFI_DATA_DIR", "data")
    initial_state: RiskAnalystState = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "data_dir": effective_data_dir,
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "risk_assessment": None,
        "recommended_position_size": None,
        "error": None,
        "start_time": start,
        "memory_prefix": memory_prefix,
    }

    with trace_context(trace_id):
        final_state = graph.invoke(initial_state, config=config)

    _tracer.flush()
    latency_ms = (time.monotonic() - start) * 1000

    tool_results = final_state.get("tool_results") or {}

    return RiskAnalysis(
        signal=final_state.get("signal"),
        risk_assessment=final_state.get("risk_assessment") or "",
        recommended_position_size=final_state.get("recommended_position_size"),
        risk_metrics=tool_results.get("risk_metrics", {}),
        prompt_version=_PROMPT_VERSION,
        latency_ms=round(latency_ms, 1),
    )
