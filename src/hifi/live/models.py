"""Which model each agent runs on, and proof that it can answer.

Model heterogeneity is the treatment, not a deployment detail: the ensemble's
diversity comes from five different model families reasoning over the same
evidence, so ``_AGENT_CONFIG`` is part of the experimental design and is
reproduced verbatim in the record.

There is exactly one way for an agent to obtain a model — LM Studio serves the
name given here — and no fallback. See ``_setup_agent_model`` for why that
constraint is load-bearing rather than an omission.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


# The LM Studio OpenAI-compatible endpoint every agent is served from. There is
# no second base URL any more: the fine-tune servers on ports 1235 and 1236 were
# retired with DJ-124 and their fallback branches removed (see _setup_agent_model).
_LM_STUDIO_URL = "http://localhost:1234/v1"


# Standard model config (full / parallel / no-memory conditions).
# Tuples: (agent_type, lms_model_id, env_var, load_timeout_s, ctx_len | None)
# lms_model_id and env_var are REQUIRED. They were optional while a None meant
# "route to the fine-tune server"; that route is gone (DJ-135).
# ctx_len: override lms load -c <n>. Gemma 12B's default (~4096) is too small for
# tickers with long EDGAR passages (prompt + output ≈ 4,357 tokens for AAPL).
_AGENT_CONFIG: list[tuple[str, str, str, int, int | None]] = [
    # fmt: (agent_type, lms_model_id, env_var, load_timeout_s, ctx_len_override)
    ("fundamental", "llama-3.3-70b-instruct",      "HIFI_FUNDAMENTAL_MODEL", 600, None),
    # Technical runs on the BASE model, not the technical_v2 LoRA (DJ-124).
    # The adapter, and the fine-tune server that served it, are gone: there is no
    # longer any value of this field that can reach them (DJ-135).
    #
    # Measured on 2026-08-18, same 15 tickers, same date, same indicators, same
    # prompt — the adapter is the only difference:
    #   technical_v2 : Buy 15/15, confidence 0.70 every time  (zero variance)
    #   base qwen2.5 : Hold 9, Buy 3, Sell 3, confidences 0.65/0.75/0.85
    #
    # This reproduces the project's own Phase 12.1 finding (OQ-M02: "diversity
    # preserved under fine-tuning? NO — 100% entropy degradation, 0.367 -> 0.000;
    # technical_v2 + fundamental_v1 vote unanimously Buy"). technical_v1 was
    # rejected outright at DJ-058 (GR 1.000 -> 0.000) and technical_v2's GR gate
    # was never formally tested — W2 was skipped — yet v2 shipped into the live
    # experiment anyway.
    #
    # A constant member contributes no information to an ensemble and mechanically
    # drives measured herding to 1.0, which is fatal to an experiment whose
    # dependent variable IS disagreement.
    ("technical",   "qwen2.5-coder-32b-instruct-mlx", "HIFI_TECHNICAL_MODEL",  300, None),
    ("risk",        "mistral-small-3.2-24b-instruct-2506-mlx",
                                                    "HIFI_RISK_MODEL",        300, None),
    ("macro",       "deepseek-r1-distill-qwen-32b", "HIFI_MACRO_MODEL",       600, None),
    ("sentiment",   "gemma-3-12b-it",               "HIFI_SENTIMENT_MODEL",   300, 8192),
    ("contrarian",  "mlx-qwen3.5-35b-a3b",          "HIFI_CONTRARIAN_MODEL",  300, None),
]


# Homogeneous model config (Phase 13 qwen-dominant baseline, DJ-096).
_HOMOGENEOUS_AGENT_CONFIG: list[tuple[str, str, str, int, int | None]] = [
    ("fundamental", "qwen2.5-coder-32b-instruct-mlx",  "HIFI_FUNDAMENTAL_MODEL", 300, None),
    ("technical",   "qwen2.5-coder-32b-instruct-mlx",  "HIFI_TECHNICAL_MODEL",   300, None),
    ("risk",        "google/gemma-3-4b",                "HIFI_RISK_MODEL",        120, None),
    ("macro",       "mlx-qwen3.5-35b-a3b",             "HIFI_MACRO_MODEL",       600, None),
    ("sentiment",   "qwen2.5-coder-32b-instruct-mlx",  "HIFI_SENTIMENT_MODEL",   300, None),
    ("contrarian",  "mlx-qwen3.5-35b-a3b",             "HIFI_CONTRARIAN_MODEL",  600, None),
]


def _agent_config_for_condition(
    condition: str,
) -> list[tuple[str, str, str, int, int | None]]:
    return _HOMOGENEOUS_AGENT_CONFIG if condition == "homogeneous" else _AGENT_CONFIG


def _probe_chat_model(base_url: str, model_id: str, timeout: int = 60) -> tuple[str, str]:
    """Send one minimal system+user request. Returns (status, detail) (DJ-120).

    status is "ok" | "unusable" | "inconclusive".

    Selection by name is necessary but not sufficient — it cannot detect a model
    that is served yet unusable. The 2026-08-04 outage was exactly that shape:
    the request was well-formed and the server healthy, but the model rejected
    the system role, and we only learned that 98 identical 404s later.

    The tri-state matters, and is not fastidiousness. A first probe of a COLD 32B
    model timed out at 120 s in testing; treating that as failure would abort
    healthy nights — strictly worse than the bug being fixed. So the caller must
    abort only on positive evidence of wrongness:

      4xx  -> "unusable"      server parsed the request and refused it: the model
                              or the prompt shape is wrong. Definitive.
      5xx  -> "inconclusive"  server-side trouble, says nothing about our choice.
      timeout / connection error -> "inconclusive"  cold weights or a busy GPU.

    Absence of evidence is not evidence of absence: an inconclusive probe warns
    and proceeds, and the run surfaces any real problem on its own.
    """
    import urllib.error as _ue  # noqa: PLC0415
    import urllib.request as _ur  # noqa: PLC0415

    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": "ping"},
            {"role": "user", "content": "ping"},
        ],
        "max_tokens": 1,
    }).encode()
    req = _ur.Request(f"{base_url}/chat/completions", data=payload,
                      headers={"Content-Type": "application/json"})
    try:
        with _ur.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return "unusable", f"HTTP {resp.status}"
            return "ok", "ok"
    except _ue.HTTPError as exc:
        detail = f"HTTP {exc.code}"
        with contextlib.suppress(Exception):  # body is best-effort diagnostics
            detail = f"{detail}: {exc.read().decode()[:200]}"
        # 4xx is a verdict on our request; 5xx is a verdict on the server.
        return ("unusable" if 400 <= exc.code < 500 else "inconclusive"), detail
    except Exception as exc:
        return "inconclusive", str(exc)


def _setup_agent_model(
    agent_type: str,
    lms_model_id: str,
    env_var: str,
    load_timeout: int,
    context_length: int | None = None,
) -> bool:
    """
    Load the agent's model in LM Studio and set its env var.

    Returns True if the agent is ready to run.

    There is exactly one way for an agent to get a model: LM Studio serves the
    name given in ``_AGENT_CONFIG`` on :1234. No fallback exists (DJ-135). If the
    named model will not load, the pass fails.

    That is a deliberate design constraint, not an omission. This function used
    to carry two fallbacks to the mlx_lm fine-tune servers on ports 1235/1236,
    and they were the shape of DJ-124: a substitution that succeeded, returned
    True, and let the night proceed while the sidecars recorded a model nobody
    had chosen. A pass that cannot run the model it declares must stop, because
    an ensemble whose members are silently not the configured members answers a
    different question than the experiment asked.
    """
    from hifi.simulation.model_manager import load_model, model_is_loaded  # noqa: PLC0415

    assert lms_model_id is not None, (
        f"{agent_type} names no model in _AGENT_CONFIG. Every agent must declare "
        "the model it runs on; a None here used to mean 'route to the fine-tune "
        "server', which is exactly the substitution DJ-124 recorded."
    )
    assert env_var is not None

    # No agent may carry a fine-tune URL (DJ-124, DJ-135). The agents read
    # HIFI_{AGENT}_FINETUNE_URL unconditionally, so a value left by a shell
    # export or a stale .env would silently send every request to a retired
    # adapter while the logs claimed the configured model was in use. Nothing in
    # this file sets these any more; the scrub defends against the environment.
    for _stale_var in (f"HIFI_{agent_type.upper()}_FINETUNE_URL",
                       f"HIFI_{agent_type.upper()}_FINETUNE_MODEL"):
        stale = os.environ.pop(_stale_var, None)
        if stale:
            logger.warning("Cleared %s=%s; %s runs on the configured LM Studio "
                           "model %s (DJ-124)", _stale_var, stale, agent_type,
                           lms_model_id)

    if model_is_loaded(lms_model_id):
        os.environ[env_var] = lms_model_id
        logger.info("Model already loaded: %s", lms_model_id)
        return _probe_or_fail(agent_type, lms_model_id)

    # Evict stale models before loading (handles variant IDs that substring-match
    # at detection but fail to unload, e.g. mlx-qwen3.5-35b-a3b-claude-4.6-opus-
    # reasoning-distilled blocking Llama-70B at condition boundaries).
    from hifi.simulation.model_manager import unload_all  # noqa: PLC0415
    unload_all()

    logger.info("Loading %s ...", lms_model_id)
    t0 = time.monotonic()
    ok = load_model(lms_model_id, timeout_s=load_timeout, context_length=context_length)
    elapsed = int(time.monotonic() - t0)

    if ok:
        os.environ[env_var] = lms_model_id
        logger.info("Loaded %s (%ds)", lms_model_id, elapsed)
        return _probe_or_fail(agent_type, lms_model_id)

    logger.error(
        "Failed to load %s (%ds); %s passes are ABORTED. No substitute model is "
        "used: an ensemble member running an unchosen model is worse than a "
        "missing one, because the sidecar would look valid.",
        lms_model_id, elapsed, agent_type,
    )
    return False


def _probe_or_fail(agent_type: str, model_id: str) -> bool:
    """Confirm the loaded model answers a system+user request before the sweep.

    DJ-120's second lesson, generalised. The 2026-08-04 outage was a model that
    was *served and healthy* but rejected the system role; we learned it 98
    identical 404s later, after the night was already lost. One probe of
    ``max_tokens=1`` costs under a second and converts that into an immediate
    abort. Previously this guard ran only against the fine-tune servers — the
    path that has since been retired — leaving the path that actually runs every
    night unprotected.

    Only positive evidence of wrongness aborts: an inconclusive probe (cold 32B
    weights, busy GPU) warns and proceeds. See ``_probe_chat_model``.
    """
    status, detail = _probe_chat_model(_LM_STUDIO_URL, model_id)
    if status == "unusable":
        logger.error("%s model %s is served but unusable: %s. Aborting the pass "
                     "rather than routing every ticker at a model that will "
                     "refuse them.", agent_type, model_id, detail)
        return False
    if status == "inconclusive":
        logger.warning("%s model %s probe inconclusive (%s); proceeding.",
                       agent_type, model_id, detail)
    return True
