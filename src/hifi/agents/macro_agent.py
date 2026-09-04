"""
Macro Analyst Agent for HiFi (P8-E3).

LangGraph graph that calls the get_macro_snapshot MCP tool and asks a local LLM
(via LM Studio) to assess the macroeconomic regime and its equity implications.

Graph structure
---------------
call_mcp_tools -> generate_analysis -> parse_output -> END

Information restriction: macro snapshot only (fed_funds_rate, CPI, unemployment,
yield curve, VIX, GDP). No access to company-specific data. This creates genuine
information-space diversity from the Fundamental and Technical Agents (David §10.3).

Model selection
---------------
Controlled by HIFI_MACRO_MODEL env var (DJ-032).
Default: qwen3.5-27b-claude-4.6-opus-reasoning-distilled-qx64-hi-mlx.
Reasoning-distilled model: max_tokens=4096 required to avoid JSON truncation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict

from hifi.agents.json_parsing import extract_json, message_text
from hifi.agents.lm_client import DEEPSEEK_R1_DISTILL_32B, ChatModel, make_llm
from hifi.agents.mcp_client import call_tool
from hifi.agents.schemas import AgentSignal, MacroAnalysis
from hifi.observability.tracing import AbstractTracer, get_tracer, trace_context

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "macro_v1"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{_PROMPT_VERSION}.md"
_DEFAULT_MACRO_MODEL = DEEPSEEK_R1_DISTILL_32B
_RETRY_MSG = (
    "Your previous response was not valid JSON or was missing required fields. "
    "Produce ONLY the JSON object with the fields: "
    "decision (Buy/Hold/Sell), confidence (0.0-1.0), rationale (string), "
    "key_concern (string), regime_assessment (string), macro_rationale (string). "
    "No other text."
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class MacroAnalystState(TypedDict, total=False):
    ticker: str
    as_of_date: str
    data_dir: str
    tool_results: dict
    llm_response: str
    model_id: str                 # set by generate_analysis_node; read by parse_output_node
    _test_llm: ChatModel | None      # DI: injected by tests only; bypasses make_llm()
    signal: AgentSignal | None
    regime_assessment: str | None
    macro_rationale: str | None
    error: str | None
    start_time: float
    memory_prefix: str  # P13-E4-T3: in-context decision history prefix (DJ-076)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _macro_model() -> str:
    return os.environ.get("HIFI_MACRO_MODEL", _DEFAULT_MACRO_MODEL)


def _load_prompt_template() -> tuple[str, str]:
    raw = _PROMPT_PATH.read_text(encoding="utf-8")
    parts = raw.split("## User", maxsplit=1)
    system_block = parts[0].replace("## System", "").strip()
    user_block = parts[1].strip() if len(parts) > 1 else ""
    return system_block, user_block


#: One definition for every agent (DJ-140). Aliased rather than renamed
#: at the call sites so this file's diff shows the removal, not a sweep.
_extract_json = extract_json


def _build_macro_signal(
    parsed: dict,
    ticker: str,
    as_of_date: str,
    model_id: str,
    call_ids: list[str],
    data_gaps: list[str],
) -> tuple[AgentSignal | None, str | None, str | None]:
    """
    Construct AgentSignal + regime_assessment + macro_rationale.

    Returns (signal, regime_assessment, macro_rationale) or (None, None, None).
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
            agent_type="macro",
        )
        regime = parsed.get("regime_assessment", "")
        macro_rat = parsed.get("macro_rationale", "")
        return signal, regime, macro_rat
    except Exception as exc:
        logger.warning("MacroAnalysis signal build failed: %s", exc)
        return None, None, None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def call_mcp_tools_node(state: MacroAnalystState) -> dict:
    """
    Call get_macro_snapshot — the only tool the Macro Agent is permitted to use.

    No company-specific data is fetched. The ticker is passed to the prompt for
    framing only (macro analysis is ticker-agnostic at this level).
    """
    as_of_date = state["as_of_date"]
    data_dir = state.get("data_dir") or os.environ.get("HIFI_DATA_DIR", "data")

    def _call(tool: str, params: dict) -> dict:
        try:
            return call_tool(tool, params, data_dir=data_dir)
        except Exception as exc:
            logger.warning("MCP tool %s failed: %s", tool, exc)
            return {"error": "COMPUTATION_ERROR", "detail": str(exc)}

    macro_snapshot = _call("get_macro_snapshot", {"date": as_of_date})
    return {"tool_results": {"macro_snapshot": macro_snapshot}}


def generate_analysis_node(state: MacroAnalystState) -> dict:
    """Fill the macro prompt template and call the LM Studio model."""
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
        macro_snapshot=json.dumps(tool_results.get("macro_snapshot", {}), indent=2),
        data_gaps_list=data_gaps_list,
    )

    # P13-E4-T3: prepend agent memory prefix when available (DJ-076)
    memory_prefix = state.get("memory_prefix", "")
    if memory_prefix:
        user_text = memory_prefix + "\n\n" + user_text

    _test_llm = state.get("_test_llm")
    llm = _test_llm if _test_llm is not None else make_llm(_macro_model(), max_tokens=4096)
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    response = llm.invoke(messages)
    return {"llm_response": message_text(response.content), "model_id": llm.model_name}


def parse_output_node(state: MacroAnalystState) -> dict:
    """Parse the LLM response into AgentSignal + macro fields. One retry on failure."""
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

    model_id = state.get("model_id", "")
    _test_llm = state.get("_test_llm")

    def _try_parse(text: str):
        parsed = _extract_json(text)
        if parsed is None:
            return None, None, None
        return _build_macro_signal(parsed, ticker, as_of_date, model_id, call_ids, data_gaps)

    signal, regime, macro_rat = _try_parse(llm_response)
    if signal is not None:
        return {
            "signal": signal,
            "regime_assessment": regime,
            "macro_rationale": macro_rat,
        }

    logger.warning("First parse attempt failed for %s macro. Retrying.", ticker)
    retry_llm = _test_llm if _test_llm is not None else make_llm(_macro_model(), max_tokens=4096)
    retry_response = retry_llm.invoke([
        HumanMessage(content=llm_response),
        HumanMessage(content=_RETRY_MSG),
    ])
    signal, regime, macro_rat = _try_parse(message_text(retry_response.content))
    if signal is not None:
        return {
            "signal": signal,
            "regime_assessment": regime,
            "macro_rationale": macro_rat,
            "llm_response": message_text(retry_response.content),
        }

    return {
        "error": (
            f"Failed to parse MacroAnalysis after retry. "
            f"Last response: {message_text(retry_response.content)[:200]}"
        )
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _should_abort(state: MacroAnalystState) -> str:
    return "abort" if state.get("error") else "continue"


def build_macro_graph():
    """Build and compile the Macro Analyst LangGraph."""
    from langgraph.graph import END, StateGraph

    builder = StateGraph(MacroAnalystState)
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


def run_macro_analysis(
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    memory_prefix: str = "",
    _test_llm: ChatModel | None = None,
) -> MacroAnalysis:
    """
    Run the Macro Analyst Agent for one ticker on one date.

    The ticker is used for prompt framing; the macro snapshot itself is
    ticker-agnostic (same macro environment for all stocks on a given date).

    Parameters
    ----------
    ticker : str
        Ticker symbol (used for prompt framing only).
    as_of_date : str
        ISO 8601 analysis date (e.g. "2023-03-31").
    data_dir : str | None
        Path to the data root directory. Defaults to HIFI_DATA_DIR env var.
    tracer : AbstractTracer | None
        Observability tracer.

    Returns
    -------
    MacroAnalysis
        Full macro analysis including AgentSignal (or None on failure) and raw
        macro snapshot tool result.
    """
    _tracer = tracer if tracer is not None else get_tracer()
    trace_id = _tracer.start_trace(
        "macro_agent", ticker=ticker, as_of_date=as_of_date
    )
    handler = _tracer.get_callback_handler(trace_id)
    config = {"callbacks": [handler]} if handler is not None else {}

    start = time.monotonic()
    graph = build_macro_graph()

    effective_data_dir = data_dir or os.environ.get("HIFI_DATA_DIR", "data")
    initial_state: MacroAnalystState = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "data_dir": effective_data_dir,
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "regime_assessment": None,
        "macro_rationale": None,
        "error": None,
        "start_time": start,
        "memory_prefix": memory_prefix,
        "_test_llm": _test_llm,
    }

    with trace_context(trace_id):
        final_state = graph.invoke(initial_state, config=config)

    _tracer.flush()
    latency_ms = (time.monotonic() - start) * 1000

    tool_results = final_state.get("tool_results") or {}

    return MacroAnalysis(
        signal=final_state.get("signal"),
        regime_assessment=final_state.get("regime_assessment") or "",
        rationale=final_state.get("macro_rationale") or "",
        macro_snapshot=tool_results.get("macro_snapshot", {}),
        prompt_version=_PROMPT_VERSION,
        latency_ms=round(latency_ms, 1),
    )
