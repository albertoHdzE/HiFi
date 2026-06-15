"""
Fundamental Analyst Agent for HiFi (P3-E3, P3-E4).

LangGraph graph that calls four Phase 2 MCP tools and asks a local LLM (via LM Studio)
to interpret the results as a structured investment opinion.

Graph structure
---------------
load_snapshot -> call_mcp_tools -> generate_analysis -> parse_output -> END

State flow
----------
Each node reads from and writes to FundamentalistState, a TypedDict. Nodes are pure
functions from state to state update dict.

Prompt versioning
-----------------
Prompt templates live in src/hifi/agents/prompts/. The version identifier
(e.g. "fundamental_v1") is embedded in FundamentalAnalysis.prompt_version so that
analysis outputs can be compared across prompt iterations in Phase 10.

Parse-and-retry
---------------
The LLM is instructed to produce only a JSON object. If the first response is not
valid JSON (or is missing required fields), one retry is sent with an explicit
correction instruction. If the second attempt also fails, the graph sets an error
in state and terminates. This is documented rather than silently swallowed so the
baseline metrics capture the failure rate accurately.

Data directory
--------------
Controlled by HIFI_DATA_DIR environment variable (same as Phase 2 MCP server).
At test time, this is monkeypatched to the fixtures directory.
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
from hifi.agents.schemas import AgentSignal, FundamentalAnalysis
from hifi.data.schemas import FundamentalsSnapshot
from hifi.observability.tracing import AbstractTracer, get_tracer, trace_context

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "fundamental_v1"
_PROMPT_V2_VERSION = "fundamental_v2"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{_PROMPT_VERSION}.md"
_PROMPT_V2_PATH = Path(__file__).parent / "prompts" / f"{_PROMPT_V2_VERSION}.md"
_RETRY_MSG = (
    "Your previous response was not valid JSON or was missing required fields. "
    "Produce ONLY the JSON object with the fields: "
    "decision (Buy/Hold/Sell), confidence (0.0-1.0), rationale (string), "
    "key_concern (string). No other text."
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class FundamentalistState(TypedDict, total=False):
    ticker: str
    as_of_date: str               # ISO 8601
    snapshot_json: str            # serialised FundamentalsSnapshot
    data_dir: str                 # HIFI_DATA_DIR for MCP server subprocess
    tool_results: dict            # populated by call_mcp_tools_node
    retrieved_context: str        # SEC filing passages (empty if use_rag=False)
    llm_response: str             # raw LLM output (last attempt)
    signal: AgentSignal | None
    error: str | None
    start_time: float             # wall-clock start for latency measurement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompt_template() -> tuple[str, str]:
    """Return (system_text, user_template) from the v1 prompt markdown file."""
    raw = _PROMPT_PATH.read_text(encoding="utf-8")
    # Split on "## User" -- first block is system, second is user template
    parts = raw.split("## User", maxsplit=1)
    system_block = parts[0].replace("## System", "").strip()
    user_block = parts[1].strip() if len(parts) > 1 else ""
    return system_block, user_block


def _load_v2_prompt_template() -> tuple[str, str]:
    """Return (system_text, user_template) from the v2 (RAG-enabled) prompt markdown file."""
    raw = _PROMPT_V2_PATH.read_text(encoding="utf-8")
    parts = raw.split("## User", maxsplit=1)
    system_block = parts[0].replace("## System", "").strip()
    user_block = parts[1].strip() if len(parts) > 1 else ""
    return system_block, user_block


def _extract_json(text: str) -> dict | None:
    """
    Extract a JSON object from the LLM response text.

    The model may wrap its JSON in markdown code fences. This function strips
    fences and attempts to parse the first {...} block found.
    """
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [line for line in lines if not line.startswith("```")]
        text = "\n".join(inner).strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _build_signal(
    parsed: dict,
    ticker: str,
    as_of_date: str,
    model_id: str,
    call_ids: list[str],
    data_gaps: list[str],
) -> AgentSignal | None:
    """Construct an AgentSignal from a parsed LLM dict. Returns None on validation error."""
    try:
        return AgentSignal(
            ticker=ticker,
            as_of_date=as_of_date,
            decision=parsed["decision"],
            confidence=float(parsed["confidence"]),
            rationale=parsed["rationale"],
            key_concern=parsed["key_concern"],
            data_gaps=data_gaps,
            call_ids=call_ids,
            model_id=model_id,
            agent_type="fundamental",
        )
    except Exception as exc:
        logger.warning("AgentSignal validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def load_snapshot_node(state: FundamentalistState) -> dict:
    """
    Validate that snapshot_json is a parseable FundamentalsSnapshot.

    Returns an error state if the JSON is malformed. Otherwise passes through.
    This node is lightweight -- snapshot_json is passed in by the caller.
    """
    try:
        FundamentalsSnapshot.model_validate_json(state["snapshot_json"])
        return {}
    except Exception as exc:
        return {"error": f"Invalid snapshot_json: {exc}"}


def call_mcp_tools_node(state: FundamentalistState) -> dict:
    """
    Call four Phase 2 MCP tools and collect results.

    Tools called: get_financial_ratios, get_growth_metrics, get_valuation_context,
    get_macro_snapshot.
    """
    ticker = state["ticker"]
    as_of_date = state["as_of_date"]
    data_dir = state.get("data_dir") or os.environ.get("HIFI_DATA_DIR", "data")
    snapshot_json = state["snapshot_json"]

    def _call(tool: str, params: dict) -> dict:
        try:
            return call_tool(tool, params, data_dir=data_dir)
        except Exception as exc:
            logger.warning("MCP tool %s failed: %s", tool, exc)
            return {"error": "COMPUTATION_ERROR", "detail": str(exc)}

    ratios = _call("get_financial_ratios", {
        "ticker": ticker,
        "date": as_of_date,
        "snapshot_json": snapshot_json,
    })
    growth = _call("get_growth_metrics", {"snapshot_json": snapshot_json})
    valuation = _call("get_valuation_context", {
        "ticker": ticker,
        "date": as_of_date,
        "snapshot_json": snapshot_json,
    })
    macro = _call("get_macro_snapshot", {"date": as_of_date})

    return {
        "tool_results": {
            "financial_ratios": ratios,
            "growth_metrics": growth,
            "valuation_context": valuation,
            "macro_snapshot": macro,
        }
    }


def retrieve_context_node(state: FundamentalistState) -> dict:
    """
    Retrieve relevant SEC filing passages from the knowledge store.

    Calls the knowledge MCP server's retrieve_context tool. On any failure
    (server unavailable, store empty, network error) returns "" so the graph
    continues with v1 prompt. This is the fail-open RAG pattern.
    """
    ticker = state["ticker"]
    data_dir = state.get("data_dir") or os.environ.get("HIFI_DATA_DIR", "data")
    try:
        result = call_tool(
            "retrieve_context",
            {"query": f"financial analysis {ticker}", "ticker": ticker, "top_k": 5},
            data_dir=data_dir,
            server_module="hifi.mcp.knowledge_server",
        )
        passages = result.get("passages", [])
        if passages:
            lines = []
            for p in passages:
                lines.append(
                    f"[{p['rank']}] {ticker} / {p['filing_type']} / {p['section']} / {p['period']}"
                )
                lines.append(p["text"])
                lines.append("---")
            context = "\n".join(lines)
        else:
            context = ""
    except Exception as exc:
        logger.warning("retrieve_context_node failed for %s: %s", ticker, exc)
        context = ""
    return {"retrieved_context": context}


def generate_analysis_node(state: FundamentalistState) -> dict:
    """
    Fill the prompt template and call the LM Studio model.

    Selects v2 (RAG-enabled) prompt when retrieved_context is non-empty;
    falls back to v1 otherwise. Returns the raw LLM response string.
    """
    tool_results = state["tool_results"]
    retrieved_context = state.get("retrieved_context", "")

    # Build data_gaps: fields with None values across all tool results
    data_gaps: list[str] = []
    for result_dict in tool_results.values():
        for k, v in result_dict.items():
            if v is None and k not in ("call_id", "error", "detail"):
                data_gaps.append(k)
    data_gaps_list = ", ".join(data_gaps) if data_gaps else "none"

    if retrieved_context:
        system_text, user_template = _load_v2_prompt_template()
        user_text = user_template.format(
            ticker=state["ticker"],
            as_of_date=state["as_of_date"],
            financial_ratios=json.dumps(tool_results["financial_ratios"], indent=2),
            growth_metrics=json.dumps(tool_results["growth_metrics"], indent=2),
            valuation_context=json.dumps(tool_results["valuation_context"], indent=2),
            macro_snapshot=json.dumps(tool_results["macro_snapshot"], indent=2),
            data_gaps_list=data_gaps_list,
            retrieved_context=retrieved_context,
        )
    else:
        system_text, user_template = _load_prompt_template()
        user_text = user_template.format(
            ticker=state["ticker"],
            as_of_date=state["as_of_date"],
            financial_ratios=json.dumps(tool_results["financial_ratios"], indent=2),
            growth_metrics=json.dumps(tool_results["growth_metrics"], indent=2),
            valuation_context=json.dumps(tool_results["valuation_context"], indent=2),
            macro_snapshot=json.dumps(tool_results["macro_snapshot"], indent=2),
            data_gaps_list=data_gaps_list,
        )

    _ft_url   = os.environ.get("HIFI_FUNDAMENTAL_FINETUNE_URL")
    _ft_model = os.environ.get("HIFI_FUNDAMENTAL_FINETUNE_MODEL")
    if _ft_url and _ft_model:
        llm = make_llm(model=_ft_model, base_url=_ft_url)
    elif _ft_url:
        llm = make_llm(base_url=_ft_url)
    else:
        llm = make_llm()
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    response = llm.invoke(messages)
    return {"llm_response": response.content}


def parse_output_node(state: FundamentalistState) -> dict:
    """
    Parse the LLM response into an AgentSignal.

    Attempts once with the response from generate_analysis_node. If parsing fails,
    sends a correction message to the LLM and tries once more. Sets error in state
    if both attempts fail.
    """
    ticker = state["ticker"]
    as_of_date = state["as_of_date"]
    tool_results = state["tool_results"]
    llm_response = state.get("llm_response", "")

    # Collect call_ids from tool results
    call_ids = [
        r["call_id"]
        for r in tool_results.values()
        if isinstance(r, dict) and "call_id" in r
    ]

    # Collect data_gaps for the signal
    data_gaps: list[str] = []
    for result_dict in tool_results.values():
        for k, v in result_dict.items():
            if v is None and k not in ("call_id", "error", "detail"):
                data_gaps.append(k)

    _ft_url   = os.environ.get("HIFI_FUNDAMENTAL_FINETUNE_URL")
    _ft_model = os.environ.get("HIFI_FUNDAMENTAL_FINETUNE_MODEL")
    if _ft_url and _ft_model:
        llm = make_llm(model=_ft_model, base_url=_ft_url)
    elif _ft_url:
        llm = make_llm(base_url=_ft_url)
    else:
        llm = make_llm()
    model_id = llm.model_name

    def _try_parse(text: str) -> AgentSignal | None:
        parsed = _extract_json(text)
        if parsed is None:
            return None
        return _build_signal(parsed, ticker, as_of_date, model_id, call_ids, data_gaps)

    signal = _try_parse(llm_response)
    if signal is not None:
        return {"signal": signal}

    # First parse failed -- send a correction request
    logger.warning("First parse attempt failed for %s. Retrying.", ticker)
    retry_response = llm.invoke([
        HumanMessage(content=llm_response),
        HumanMessage(content=_RETRY_MSG),
    ])
    signal = _try_parse(retry_response.content)
    if signal is not None:
        return {"signal": signal, "llm_response": retry_response.content}

    return {
        "error": f"Failed to parse AgentSignal after retry. "
                 f"Last response: {retry_response.content[:200]}"
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _should_abort(state: FundamentalistState) -> str:
    """Conditional edge: abort graph if an error was set by a prior node."""
    return "abort" if state.get("error") else "continue"


def build_fundamental_graph(use_rag: bool = False):
    """
    Build and compile the Fundamental Analyst LangGraph.

    When use_rag=True the graph includes retrieve_context_node between
    call_mcp_tools and generate_analysis:
        load_snapshot -> call_mcp_tools -> retrieve_context -> generate_analysis -> parse_output

    When use_rag=False (default) the graph is identical to Phase 6:
        load_snapshot -> call_mcp_tools -> generate_analysis -> parse_output
    """
    from langgraph.graph import END, StateGraph

    builder = StateGraph(FundamentalistState)
    builder.add_node("load_snapshot", load_snapshot_node)
    builder.add_node("call_mcp_tools", call_mcp_tools_node)
    builder.add_node("generate_analysis", generate_analysis_node)
    builder.add_node("parse_output", parse_output_node)

    builder.set_entry_point("load_snapshot")

    builder.add_conditional_edges(
        "load_snapshot",
        _should_abort,
        {"continue": "call_mcp_tools", "abort": END},
    )

    if use_rag:
        builder.add_node("retrieve_context", retrieve_context_node)
        builder.add_edge("call_mcp_tools", "retrieve_context")
        builder.add_edge("retrieve_context", "generate_analysis")
    else:
        builder.add_edge("call_mcp_tools", "generate_analysis")

    builder.add_edge("generate_analysis", "parse_output")
    builder.add_edge("parse_output", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_analysis(
    ticker: str,
    as_of_date: str,
    snapshot_json: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    use_rag: bool = False,
    retrieved_context: str = "",
) -> FundamentalAnalysis:
    """
    Run the Fundamental Analyst Agent for one ticker on one date.

    Parameters
    ----------
    ticker : str
        Ticker symbol (must match a Parquet file in data_dir/market/).
    as_of_date : str
        ISO 8601 analysis date (e.g. "2023-03-31").
    snapshot_json : str
        JSON-serialised FundamentalsSnapshot (from Phase 1 data acquisition).
    data_dir : str | None
        Path to the data root directory. Defaults to HIFI_DATA_DIR env var.
    tracer : AbstractTracer | None
        Observability tracer. Defaults to get_tracer() (NoOpTracer when
        LANGFUSE_ENABLED=false; LangFuseTracer otherwise). Pass an explicit
        tracer from run_ensemble() to share the parent trace context.

    Returns
    -------
    FundamentalAnalysis
        Full analysis including AgentSignal and raw MCP tool results.
        On parse failure: signal is None and error is set in the state dict
        (the FundamentalAnalysis is constructed from whatever state is available).
    """
    _tracer = tracer if tracer is not None else get_tracer()
    trace_id = _tracer.start_trace(
        "fundamental_agent", ticker=ticker, as_of_date=as_of_date
    )
    handler = _tracer.get_callback_handler(trace_id)
    config = {"callbacks": [handler]} if handler is not None else {}

    start = time.monotonic()
    graph = build_fundamental_graph(use_rag=use_rag)

    effective_data_dir = data_dir or os.environ.get("HIFI_DATA_DIR", "data")
    initial_state: FundamentalistState = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "snapshot_json": snapshot_json,
        "data_dir": effective_data_dir,
        "tool_results": {},
        "retrieved_context": retrieved_context,
        "llm_response": "",
        "signal": None,
        "error": None,
        "start_time": start,
    }

    with trace_context(trace_id):
        final_state = graph.invoke(initial_state, config=config)

    _tracer.flush()
    latency_ms = (time.monotonic() - start) * 1000

    tool_results = final_state.get("tool_results") or {}
    retrieved_context = final_state.get("retrieved_context", "")
    used_prompt_version = _PROMPT_V2_VERSION if retrieved_context else _PROMPT_VERSION

    return FundamentalAnalysis(
        signal=final_state.get("signal"),
        financial_ratios=tool_results.get("financial_ratios", {}),
        growth_metrics=tool_results.get("growth_metrics", {}),
        valuation_context=tool_results.get("valuation_context", {}),
        macro_snapshot=tool_results.get("macro_snapshot", {}),
        prompt_version=used_prompt_version,
        latency_ms=round(latency_ms, 1),
    )
