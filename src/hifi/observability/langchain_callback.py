"""Minimal LangChain callback that records LLM generations to LangFuse.

Replaces langfuse v2's bundled CallbackHandler, which imports legacy
langchain.schema paths removed in langchain 1.x. This handler uses only
stable langchain_core APIs and the langfuse v2 client's generation()
method, so prompts, completions, model ids, and token usage land in the
trace regardless of langchain version.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class HiFiLangfuseCallbackHandler(BaseCallbackHandler):
    """Logs each LLM call as a LangFuse generation on an existing trace."""

    def __init__(self, client: Any, trace_id: str) -> None:
        self._client = client
        self._trace_id = trace_id
        self._runs: dict[Any, dict] = {}

    # -- start ---------------------------------------------------------

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._start(run_id, prompts, kwargs)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        flat = [
            {"role": getattr(m, "type", "unknown"), "content": str(getattr(m, "content", m))}
            for batch in messages for m in batch
        ]
        self._start(run_id, flat, kwargs)

    def _start(self, run_id, input_payload, kwargs) -> None:
        params = kwargs.get("invocation_params") or {}
        self._runs[run_id] = {
            "start": datetime.now(UTC),
            "input": input_payload,
            "model": params.get("model") or params.get("model_name"),
        }

    # -- end -----------------------------------------------------------

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        try:
            texts = [g.text for batch in response.generations for g in batch]
            usage = (response.llm_output or {}).get("token_usage") or {}
            self._client.generation(
                trace_id=self._trace_id,
                name="llm-call",
                start_time=run["start"],
                end_time=datetime.now(UTC),
                model=run["model"],
                input=run["input"],
                output=texts[0] if len(texts) == 1 else texts,
                usage={
                    "input": usage.get("prompt_tokens"),
                    "output": usage.get("completion_tokens"),
                    "total": usage.get("total_tokens"),
                } if usage else None,
            )
        except Exception as exc:
            logger.warning("LangFuse generation logging failed: %s", exc)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        try:
            self._client.generation(
                trace_id=self._trace_id,
                name="llm-call",
                start_time=run["start"],
                end_time=datetime.now(UTC),
                model=run["model"],
                input=run["input"],
                level="ERROR",
                status_message=str(error),
            )
        except Exception as exc:
            logger.warning("LangFuse error logging failed: %s", exc)
