"""
LM Studio client for HiFi agents (P3-E1).

Wraps langchain-openai's ChatOpenAI to point at a local LM Studio instance.
LM Studio exposes an OpenAI-compatible REST API; no cloud calls are made.

Configuration
-------------
HIFI_LM_STUDIO_URL : str
    Base URL for the LM Studio API (default: http://localhost:1234/v1).
    Override in tests by setting this environment variable before importing.

Usage
-----
    from hifi.agents.lm_client import make_llm

    llm = make_llm()            # uses default model from config
    llm = make_llm("google/gemma-3-4b")  # override model for fast iteration
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from langchain_openai import ChatOpenAI
from pydantic import SecretStr


@runtime_checkable
class ChatModel(Protocol):
    """What an agent actually needs from a language model (DJ-142).

    Every agent node accepted its model as ``_test_llm: object | None``, so the
    ``if _test_llm is not None: llm = _test_llm`` branch joined with the
    ``make_llm()`` branch to ``object`` — and each of the six agents then had
    three type errors saying ``"object" has no attribute "invoke"``.

    Annotating ``ChatOpenAI`` would have silenced them by claiming something
    untrue: the injected doubles are fakes, not ChatOpenAI instances, and
    requiring them to be would make the seam useless. What the nodes call is
    ``invoke`` and ``model_name``, and this says exactly that. ``ChatOpenAI``
    satisfies it structurally; so does any double the tests build.
    """

    model_name: str

    def invoke(self, input: Any, **kwargs: Any) -> Any: ...


_DEFAULT_URL = "http://localhost:1234/v1"
_DEFAULT_MODEL = "qwen2.5-coder-32b-instruct-mlx"

# ---------------------------------------------------------------------------
# Phase 14 target model identifiers (DJ-089, E0-T1)
# Load ONE at a time in LM Studio; run E0-T2 diagnostic per model.
# ---------------------------------------------------------------------------
LLAMA_33_70B = "llama-3.3-70b-instruct"                              # Fundamental
MISTRAL_SMALL_32 = "mistral-small-3.2-24b-instruct-2506-mlx"         # Risk
DEEPSEEK_R1_DISTILL_32B = "deepseek-r1-distill-qwen-32b"             # Macro
GEMMA3_12B = "gemma-3-12b-it"                                        # Sentiment


def lm_studio_url() -> str:
    """Return the LM Studio base URL from environment (default: localhost:1234)."""
    return os.environ.get("HIFI_LM_STUDIO_URL", _DEFAULT_URL)


def make_llm(
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    base_url: str | None = None,
) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance pointed at the local LM Studio server.

    Parameters
    ----------
    model : str
        LM Studio model ID. Default: qwen2.5-coder-32b-instruct-mlx (DJ-014).
    temperature : float
        Sampling temperature. 0.0 for deterministic output (structured JSON).
    max_tokens : int
        Maximum tokens in the completion. 1024 is sufficient for AgentSignal JSON.
    base_url : str | None
        Override the LM Studio base URL. When None, uses HIFI_LM_STUDIO_URL env var.

    Returns
    -------
    ChatOpenAI
        Configured to call the local LM Studio endpoint.
    """
    # max_tokens is a pydantic alias on ChatOpenAI, and api_key coerces to
    # SecretStr; both were verified against the installed langchain-openai
    # (max_tokens=1024 round-trips, api_key becomes SecretStr). The stubs are
    # stricter than the library, and this is the factory every agent's model
    # comes from — not somewhere to change kwargs to satisfy a type checker.
    return ChatOpenAI(
        model=model,
        base_url=base_url if base_url is not None else lm_studio_url(),
        api_key=SecretStr("lm-studio"),  # LM Studio ignores it; openai requires it
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore[call-arg]
    )
