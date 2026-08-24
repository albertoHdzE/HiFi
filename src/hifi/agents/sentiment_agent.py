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
Default: qwen2.5-coder-32b-instruct-mlx (DJ-087 — reverted from gemma-4-e4b after
  DJ-086 diagnostic confirmed E4B has a chat-template failure in LM Studio: it echoes
  the user prompt instead of generating a response for AAPL/JPM; 12B-it fails to load.
  qwen2.5-coder-32b is reliable (SGR=0.167 baseline) and will be the Sentiment base
  until an E4B/12B serving fix is available or a new model is tested.)
Non-reasoning model: max_tokens=1024 is sufficient.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from hifi.agents.lm_client import GEMMA3_12B, make_llm
from hifi.agents.mcp_client import call_tool
from hifi.agents.schemas import AgentSignal, SentimentAnalysis
from hifi.observability.tracing import AbstractTracer, get_tracer, trace_context

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "sentiment_v1"
_PROMPT_PATH = Path(__file__).parent / "prompts" / f"{_PROMPT_VERSION}.md"
_DEFAULT_SENTIMENT_MODEL = GEMMA3_12B  # E0-T3: Gemma 3 12B (DJ-089)
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
    Retrieve SEC filing passages for the sentiment agent.

    Two sources, tried in order:

    1. The knowledge MCP server's vector search over ingested chunks.
    2. The EDGAR MD&A corpus (DJ-120 fallback).

    The fallback exists because the vector store's ``chunks_a`` table was only
    ever populated for three tickers, while the EDGAR table holds 209,722 MD&A
    chunks for all 98. With no fallback the agent hit its "Insufficient Data"
    path on 97% of passes and returned Hold at confidence 0.0 every time — a
    silent constant inside an ensemble whose entire purpose is disagreement.

    Both paths are queried with the same narrative-tuned query (DJ-034), which
    also keeps the EDGAR chunks distinct from the head-of-document slice the
    fundamental agent receives from the same filing.

    Fail-open: any error returns "" so the agent returns the default signal.
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
    except Exception as exc:
        logger.warning("retrieve_context failed for %s: %s", ticker, exc)

    try:
        from hifi.knowledge.edgar_retriever import retrieve_mda_context

        return retrieve_mda_context(
            ticker=ticker,
            as_of_date=as_of_date,
            db_path=str(Path(data_dir) / "knowledge.lance"),
            query=query,
        )
    except Exception as exc:
        logger.warning("EDGAR MD&A fallback failed for %s: %s", ticker, exc)
        return ""


def _call_llm_for_sentiment(
    ticker: str,
    as_of_date: str,
    retrieved_context: str,
    memory_prefix: str = "",
    _test_llm: object | None = None,
    callbacks: list | None = None,
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

    # P13-E4-T3: prepend agent memory prefix when available (DJ-076)
    if memory_prefix:
        user_text = memory_prefix + "\n\n" + user_text

    llm = _test_llm if _test_llm is not None else make_llm(_sentiment_model(), max_tokens=1024)
    model_id = llm.model_name

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

    # LangFuse tracing: attach the callback so this direct llm.invoke is traced
    # like the LangGraph agents (DJ-116). Only add config when tracing is active
    # so the untraced call signature is unchanged.
    _kw = {"config": {"callbacks": callbacks}} if callbacks else {}
    messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    response = llm.invoke(messages, **_kw)
    signal, summary, notable = _try_parse(response.content)
    if signal is not None:
        return signal, summary, notable

    logger.warning("First sentiment parse attempt failed for %s. Retrying.", ticker)
    retry_response = llm.invoke([
        HumanMessage(content=response.content),
        HumanMessage(content=_RETRY_MSG),
    ], **_kw)
    return _try_parse(retry_response.content)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_sentiment_analysis(
    ticker: str,
    as_of_date: str,
    data_dir: str | None = None,
    tracer: AbstractTracer | None = None,
    memory_prefix: str = "",
    _test_llm: object | None = None,
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
    handler = _tracer.get_callback_handler(trace_id)
    callbacks = [handler] if handler is not None else None

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

        signal, sentiment_summary, notable_signals = _call_llm_for_sentiment(
            ticker, as_of_date, retrieved_context, memory_prefix,
            _test_llm=_test_llm, callbacks=callbacks,
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
