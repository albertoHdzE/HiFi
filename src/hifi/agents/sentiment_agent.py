"""
Sentiment Analyst Agent for HiFi (P8-E4).

Uses RAG (SEC filings via knowledge_server) as its ONLY information source.
No numerical MCP tools are called. This creates genuine information-space
diversity from all other agents (DJ-034).

Unlike the Technical and Fundamental Agents, there is no `use_rag` flag here --
retrieval always runs. If retrieved_context is empty (no filing passages found),
the agent returns a default "Insufficient Data" signal without calling the LLM
(fail-open design per DJ-038).

Model selection
---------------
Controlled by HIFI_SENTIMENT_MODEL env var (DJ-032).
Default: qwen2.5-coder-32b-instruct-mlx.
Non-reasoning model: max_tokens=1024 is sufficient.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from hifi.agents.lm_client import make_llm
from hifi.agents.mcp_client import call_tool
from hifi.agents.schemas import AgentSignal, SentimentAnalysis
from hifi.observability.tracing import AbstractTracer, get_tracer, trace_context

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "sentiment_v1"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{_PROMPT_VERSION}.md"
_DEFAULT_SENTIMENT_MODEL = "qwen2.5-coder-32b-instruct-mlx"
_INSUFFICIENT_DATA_MODEL = "sentiment-default"
_RETRY_MSG = (
    "Your previous response was not valid JSON or was missing required fields. "
    "Produce ONLY the JSON object with the fields: "
    "decision (Buy/Hold/Sell), confidence (0.0-1.0), rationale (string), "
    "key_concern (string), sentiment_summary (string), notable_signals (list of strings). "
    "No other text."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentiment_model() -> str:
    return os.environ.get("HIFI_SENTIMENT_MODEL", _DEFAULT_SENTIMENT_MODEL)


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


def _default_insufficient_signal(ticker: str, as_of_date: str) -> AgentSignal:
    """
    Return a neutral default signal when no SEC filing context is available.

    decision=Hold, confidence=0.0 communicates to the ensemble that this agent
    has no information to contribute (rather than conveying false conviction).
    """
    return AgentSignal(
        ticker=ticker,
        as_of_date=as_of_date,
        decision="Hold",
        confidence=0.0,
        rationale="Insufficient SEC filing data for sentiment analysis.",
        key_concern="No qualitative context available from SEC filings.",
        data_gaps=["retrieved_context"],
        call_ids=[],
        model_id=_INSUFFICIENT_DATA_MODEL,
        agent_type="sentiment",
    )


def _retrieve_context(ticker: str, as_of_date: str, data_dir: str) -> str:
    """
    Query the knowledge MCP server for SEC filing passages.

    Fail-open: any error returns "" so the agent returns the default signal.
    The query is tuned for qualitative/narrative content (management tone,
    forward guidance) rather than numerical data (DJ-034).
    """
    query = (
        f"{ticker} management outlook guidance forward-looking statements risks "
        f"revenue growth margin services"
    )
    try:
        result = call_tool(
            "retrieve_context",
            {"query": query, "ticker": ticker, "top_k": 5},
            data_dir=data_dir,
            server_module="hifi.mcp.knowledge_server",
        )
        passages = result.get("passages", [])
        if passages:
            lines = []
            for p in passages:
                lines.append(
                    f"[{p['rank']}] {ticker} / {p['filing_type']} / "
                    f"{p['section']} / {p['period']}"
                )
                lines.append(p["text"])
                lines.append("---")
            return "\n".join(lines)
        return ""
    except Exception as exc:
        logger.warning("retrieve_context failed for %s: %s", ticker, exc)
        return ""


def _call_llm_for_sentiment(
    ticker: str,
    as_of_date: str,
    retrieved_context: str,
    model_id: str,
) -> tuple[AgentSignal | None, str, list[str]]:
    """
    Call the LLM with the retrieved context.

    Returns (signal, sentiment_summary, notable_signals) or (None, "", []) on failure.
    """
    system_text, user_template = _load_prompt_template()
    user_text = user_template.format(
        ticker=ticker,
        as_of_date=as_of_date,
        retrieved_context=retrieved_context,
    )

    llm = make_llm(_sentiment_model(), max_tokens=1024)

    def _try_parse(text: str):
        parsed = _extract_json(text)
        if parsed is None:
            return None, "", []
        try:
            signal = AgentSignal(
                ticker=ticker,
                as_of_date=as_of_date,
                decision=parsed["decision"],
                confidence=float(parsed["confidence"]),
                rationale=parsed["rationale"],
                key_concern=parsed["key_concern"],
                data_gaps=[],
                call_ids=[],
                model_id=model_id,
                agent_type="sentiment",
            )
            summary = parsed.get("sentiment_summary", "")
            notable = parsed.get("notable_signals", [])
            if not isinstance(notable, list):
                notable = []
            return signal, summary, notable
        except Exception as exc:
            logger.warning("Sentiment signal build failed: %s", exc)
            return None, "", []

    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    response = llm.invoke(messages)
    signal, summary, notable = _try_parse(response.content)
    if signal is not None:
        return signal, summary, notable

    logger.warning("First sentiment parse attempt failed for %s. Retrying.", ticker)
    retry_response = llm.invoke([
        HumanMessage(content=response.content),
        HumanMessage(content=_RETRY_MSG),
    ])
    return _try_parse(retry_response.content)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_sentiment_analysis(
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
) -> SentimentAnalysis:
    """
    Run the Sentiment Analyst Agent for one ticker on one date.

    Retrieval always runs (no use_rag flag). If no filing passages are found,
    the agent returns a default "Insufficient Data" signal without calling the LLM.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    as_of_date : str
        ISO 8601 analysis date (e.g. "2023-03-31").
    data_dir : str | None
        Path to the data root directory. Defaults to HIFI_DATA_DIR env var.
    tracer : AbstractTracer | None
        Observability tracer.

    Returns
    -------
    SentimentAnalysis
        Full sentiment analysis including AgentSignal (or None on failure) and
        qualitative assessment fields.
    """
    _tracer = tracer if tracer is not None else get_tracer()
    trace_id = _tracer.start_trace(
        "sentiment_agent", ticker=ticker, as_of_date=as_of_date
    )

    start = time.monotonic()
    effective_data_dir = data_dir or os.environ.get("HIFI_DATA_DIR", "data")

    with trace_context(trace_id):
        retrieved_context = _retrieve_context(ticker, as_of_date, effective_data_dir)

        if not retrieved_context:
            # Fail-open: no SEC filing data available
            default_signal = _default_insufficient_signal(ticker, as_of_date)
            _tracer.flush()
            latency_ms = (time.monotonic() - start) * 1000
            return SentimentAnalysis(
                signal=default_signal,
                sentiment_summary="Insufficient Data: no SEC filing passages retrieved.",
                notable_signals=[],
                prompt_version=_PROMPT_VERSION,
                latency_ms=round(latency_ms, 1),
            )

        llm_model_id = make_llm(_sentiment_model(), max_tokens=1024).model_name
        signal, sentiment_summary, notable_signals = _call_llm_for_sentiment(
            ticker, as_of_date, retrieved_context, llm_model_id
        )

    _tracer.flush()
    latency_ms = (time.monotonic() - start) * 1000

    return SentimentAnalysis(
        signal=signal,
        sentiment_summary=sentiment_summary or "Parse failed.",
        notable_signals=notable_signals,
        prompt_version=_PROMPT_VERSION,
        latency_ms=round(latency_ms, 1),
    )
