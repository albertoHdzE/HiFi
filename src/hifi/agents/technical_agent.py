"""
Technical Analyst Agent for HiFi (P4-E1).

LangGraph graph that calls two Phase 2 MCP tools (get_technical_indicators and
get_risk_metrics) and asks a local LLM (via LM Studio) to interpret the results
as a structured investment opinion based exclusively on price-derived data.

Graph structure
---------------
call_mcp_tools -> generate_analysis -> parse_output -> END

No load_snapshot_node: the Technical Agent does not use a FundamentalsSnapshot.
Its inputs are ticker, as_of_date, and the Parquet files in the data directory
(accessed via MCP). This is the primary information restriction that creates
genuine diversity from the Fundamental Agent (David §10.3).

Model selection
---------------
Controlled by HIFI_TECHNICAL_MODEL env var. Default: the MoE Claude-distilled
model (DJ-016). Override for testing with any LM Studio model.
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
from hifi.agents.schemas import AgentSignal, TechnicalAnalysis

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "technical_v1"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{_PROMPT_VERSION}.md"
_DEFAULT_TECHNICAL_MODEL = "mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled"
_RETRY_MSG = (
    "Your previous response was not valid JSON or was missing required fields. "
    "Produce ONLY the JSON object with the fields: "
    "decision (Buy/Hold/Sell), confidence (0.0-1.0), rationale (string), "
    "key_concern (string), time_horizon (short-term/medium-term/long-term). "
    "No other text."
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class TechnicalAnalystState(TypedDict, total=False):
    ticker: str
    as_of_date: str            # ISO 8601
    data_dir: str              # HIFI_DATA_DIR for MCP server subprocess
    tool_results: dict         # populated by call_mcp_tools_node
    llm_response: str          # raw LLM output (last attempt)
    signal: AgentSignal | None
    time_horizon: str | None   # "short-term" | "medium-term" | "long-term"
    error: str | None
    start_time: float          # wall-clock start for latency measurement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _technical_model() -> str:
    """Return the Technical Agent model from environment, or default (DJ-016)."""
    return os.environ.get("HIFI_TECHNICAL_MODEL", _DEFAULT_TECHNICAL_MODEL)


def _load_prompt_template() -> tuple[str, str]:
    """Return (system_text, user_template) from the prompt markdown file."""
    raw = _PROMPT_PATH.read_text(encoding="utf-8")
    parts = raw.split("## User", maxsplit=1)
    system_block = parts[0].replace("## System", "").strip()
    user_block = parts[1].strip() if len(parts) > 1 else ""
    return system_block, user_block


def _extract_json(text: str) -> dict | None:
    """
    Extract a JSON object from the LLM response text.

    Strips markdown code fences and attempts to parse the first {...} block.
    """
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


def _build_technical_signal(
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
            agent_type="technical",
        )
    except Exception as exc:
        logger.warning("AgentSignal validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def call_mcp_tools_node(state: TechnicalAnalystState) -> dict:
    """
    Call two Phase 2 MCP tools restricted to price-derived data.

    Tools called: get_technical_indicators, get_risk_metrics.
    No fundamentals, valuation, or macro tools are called -- this restriction
    is the primary diversity mechanism (David §10.3).
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

    technical_indicators = _call(
        "get_technical_indicators",
        {"ticker": ticker, "date": as_of_date, "window": 20},
    )
    risk_metrics = _call(
        "get_risk_metrics",
        {"ticker": ticker, "date": as_of_date},
    )

    return {
        "tool_results": {
            "technical_indicators": technical_indicators,
            "risk_metrics": risk_metrics,
        }
    }


def generate_analysis_node(state: TechnicalAnalystState) -> dict:
    """
    Fill the technical prompt template and call the LM Studio model.

    Returns the raw LLM response string.
    """
    tool_results = state["tool_results"]
    system_text, user_template = _load_prompt_template()

    # Build data_gaps: fields with None values across all tool results
    data_gaps: list[str] = []
    for result_dict in tool_results.values():
        for k, v in result_dict.items():
            if v is None and k not in ("call_id", "error", "detail"):
                data_gaps.append(k)
    data_gaps_list = ", ".join(data_gaps) if data_gaps else "none"

    user_text = user_template.format(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        technical_indicators=json.dumps(tool_results["technical_indicators"], indent=2),
        risk_metrics=json.dumps(tool_results["risk_metrics"], indent=2),
        data_gaps_list=data_gaps_list,
    )

    llm = make_llm(_technical_model(), max_tokens=4096)
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    response = llm.invoke(messages)
    return {"llm_response": response.content}


def parse_output_node(state: TechnicalAnalystState) -> dict:
    """
    Parse the LLM response into an AgentSignal + time_horizon.

    Extracts time_horizon from the parsed dict before constructing AgentSignal
    (time_horizon is stored in TechnicalAnalysis, not in AgentSignal).
    One retry is attempted if parsing fails.
    """
    ticker = state["ticker"]
    as_of_date = state["as_of_date"]
    tool_results = state["tool_results"]
    llm_response = state.get("llm_response", "")

    call_ids = [
        r["call_id"]
        for r in tool_results.values()
        if isinstance(r, dict) and "call_id" in r
    ]

    data_gaps: list[str] = []
    for result_dict in tool_results.values():
        for k, v in result_dict.items():
            if v is None and k not in ("call_id", "error", "detail"):
                data_gaps.append(k)

    llm = make_llm(_technical_model(), max_tokens=4096)
    model_id = llm.model_name

    def _try_parse(text: str) -> tuple[AgentSignal | None, str | None]:
        parsed = _extract_json(text)
        if parsed is None:
            return None, None
        time_horizon = parsed.get("time_horizon")
        sig = _build_technical_signal(
            parsed, ticker, as_of_date, model_id, call_ids, data_gaps
        )
        return sig, time_horizon

    signal, time_horizon = _try_parse(llm_response)
    if signal is not None:
        return {"signal": signal, "time_horizon": time_horizon}

    logger.warning("First parse attempt failed for %s. Retrying.", ticker)
    retry_response = llm.invoke([
        HumanMessage(content=llm_response),
        HumanMessage(content=_RETRY_MSG),
    ])
    signal, time_horizon = _try_parse(retry_response.content)
    if signal is not None:
        return {
            "signal": signal,
            "time_horizon": time_horizon,
            "llm_response": retry_response.content,
        }

    return {
        "error": (
            f"Failed to parse AgentSignal after retry. "
            f"Last response: {retry_response.content[:200]}"
        )
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _should_abort(state: TechnicalAnalystState) -> str:
    """Conditional edge: abort graph if an error was set by a prior node."""
    return "abort" if state.get("error") else "continue"


def build_technical_graph():
    """
    Build and compile the Technical Analyst LangGraph.

    Graph: call_mcp_tools -> generate_analysis -> parse_output -> END
    No load_snapshot node (Technical Agent needs no FundamentalsSnapshot).
    """
    from langgraph.graph import END, StateGraph

    builder = StateGraph(TechnicalAnalystState)
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


def run_technical_analysis(
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
) -> TechnicalAnalysis:
    """
    Run the Technical Analyst Agent for one ticker on one date.

    Parameters
    ----------
    ticker : str
        Ticker symbol (must match a Parquet file in data_dir/market/).
    as_of_date : str
        ISO 8601 analysis date (e.g. "2023-03-31").
    data_dir : str | None
        Path to the data root directory. Defaults to HIFI_DATA_DIR env var.

    Returns
    -------
    TechnicalAnalysis
        Full analysis including AgentSignal (or None on failure) and raw MCP
        tool results for every field in the rationale to be traced back to a
        specific deterministic computation.
    """
    start = time.monotonic()
    graph = build_technical_graph()

    effective_data_dir = data_dir or os.environ.get("HIFI_DATA_DIR", "data")
    initial_state: TechnicalAnalystState = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "data_dir": effective_data_dir,
        "tool_results": {},
        "llm_response": "",
        "signal": None,
        "time_horizon": None,
        "error": None,
        "start_time": start,
    }

    final_state = graph.invoke(initial_state)
    latency_ms = (time.monotonic() - start) * 1000

    tool_results = final_state.get("tool_results") or {}

    return TechnicalAnalysis(
        signal=final_state.get("signal"),
        technical_indicators=tool_results.get("technical_indicators", {}),
        risk_metrics=tool_results.get("risk_metrics", {}),
        time_horizon=final_state.get("time_horizon"),
        prompt_version=_PROMPT_VERSION,
        latency_ms=round(latency_ms, 1),
    )
