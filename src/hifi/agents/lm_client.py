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

from langchain_openai import ChatOpenAI

_DEFAULT_URL = "http://localhost:1234/v1"
_DEFAULT_MODEL = "qwen2.5-coder-32b-instruct-mlx"


def lm_studio_url() -> str:
    """Return the LM Studio base URL from environment (default: localhost:1234)."""
    return os.environ.get("HIFI_LM_STUDIO_URL", _DEFAULT_URL)


def make_llm(
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
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

    Returns
    -------
    ChatOpenAI
        Configured to call the local LM Studio endpoint.
    """
    return ChatOpenAI(
        model=model,
        base_url=lm_studio_url(),
        api_key="lm-studio",  # LM Studio ignores the key; required by openai client
        temperature=temperature,
        max_tokens=max_tokens,
    )
