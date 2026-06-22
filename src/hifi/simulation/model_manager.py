"""
LM Studio model loading/unloading manager (E0-T3, DJ-106).

Extracted from scripts/run_phase14_e0_full.py for reuse by the smoke test
and production orchestrator scripts.

Public API
----------
load_model(model_id, timeout_s=600) -> bool
unload_model(model_id) -> None
model_is_loaded(model_id) -> bool
get_loaded_ids() -> set[str]

LM Studio Management API endpoints:
  GET  /api/v0/models                        — list models with state
  POST /api/v0/models/load  {"identifier": id, "ttl": 0}  — load
  POST /api/v0/models/unload {"identifier": id}            — unload

The lms CLI path is ~/.lmstudio/bin/lms (confirmed working, DJ-089).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.request

logger = logging.getLogger(__name__)

_LM_BASE = "http://127.0.0.1:1234"
_LMS = os.path.expanduser("~/.lmstudio/bin/lms")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lm_api(
    method: str,
    path: str,
    body: dict | None = None,
    timeout_s: int = 30,
) -> dict:
    url = f"{_LM_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


def _lms_run(*args: str, timeout_s: int = 900) -> tuple[int, str]:
    """Run lms CLI command; return (returncode, combined stdout+stderr)."""
    cmd = [_LMS, *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    return result.returncode, (result.stdout + result.stderr).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_loaded_ids() -> set[str]:
    """
    Return the set of model IDs currently loaded in LM Studio.

    Returns empty set on any connection error (LM Studio not running).
    """
    try:
        result = _lm_api("GET", "/api/v0/models")
        return {m["id"] for m in result.get("data", []) if m.get("state") == "loaded"}
    except Exception:
        return set()


def model_is_loaded(model_id: str) -> bool:
    """
    Return True if model_id is currently loaded.

    Matches exact model ID or substring (LM Studio may truncate long IDs).
    """
    loaded = get_loaded_ids()
    return model_id in loaded or any(
        model_id in lid or lid in model_id for lid in loaded
    )


def load_model(model_id: str, timeout_s: int = 600, context_length: int | None = None) -> bool:
    """
    Load a model via the lms CLI.

    If the model is already loaded, returns True immediately without reloading.

    Parameters
    ----------
    model_id : str
        LM Studio model identifier
        (e.g. "mlx-community/Llama-3.3-70B-Instruct-4bit").
    timeout_s : int
        Maximum seconds to wait for lms CLI to return (default 600).
    context_length : int | None
        Override the model's default context window size (tokens). When None,
        LM Studio uses its built-in default. Pass an explicit value (e.g. 8192)
        for models whose default is too small for the intended prompts.

    Returns
    -------
    bool
        True if model loaded successfully (or was already loaded).
        False on lms CLI failure.
    """
    if model_is_loaded(model_id):
        logger.info("Model already loaded: %s", model_id)
        return True

    logger.info("Loading %s via lms CLI ...", model_id)
    t0 = time.monotonic()
    extra_args = ["-c", str(context_length)] if context_length is not None else []
    rc, out = _lms_run("load", model_id, "-y", *extra_args, timeout_s=timeout_s)
    elapsed = int(time.monotonic() - t0)

    if rc == 0:
        logger.info("Model loaded in %ds: %s", elapsed, model_id)
        return True

    logger.error("lms load failed (rc=%d, %ds): %s", rc, elapsed, out[:200])
    return False


def unload_model(model_id: str) -> None:
    """
    Unload a model via the lms CLI.

    Logs a warning on failure but does not raise.

    Parameters
    ----------
    model_id : str
        LM Studio model identifier to unload.
    """
    rc, out = _lms_run("unload", model_id, timeout_s=60)
    if rc == 0:
        logger.info("Model unloaded: %s", model_id)
    else:
        logger.warning("lms unload rc=%d for %s: %s", rc, model_id, out[:100])
