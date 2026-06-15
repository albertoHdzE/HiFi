"""
Per-agent LLM calls for Oxford debate phases (P12-E3-T4, DJ-065).

Provides three standalone functions (challenge_node, respond_node, revise_node)
that implement the three debate phases of the Oxford 1-round protocol.

These are called by run_debate_round() in debate.py and are NOT added to the
existing agent LangGraph graphs — this preserves the tested Phase 3-8 graph
structure (see DJ-065 rationale in PHASE_12_CONTEXT.md).

Model selection follows the same env-var pattern as the production agents:
  fundamental : HIFI_FUNDAMENTAL_FINETUNE_URL/MODEL or qwen2.5-coder-32b
  technical   : HIFI_TECHNICAL_FINETUNE_URL, HIFI_TECHNICAL_MODEL or default
  risk        : HIFI_RISK_MODEL or google/gemma-3-4b
  macro       : HIFI_MACRO_MODEL or qwen3.5-27b
  sentiment   : HIFI_SENTIMENT_MODEL or qwen2.5-coder-32b

Output token limits: challenge=512, response=256, revision=1024.

Each function accepts an optional `llm` parameter for deterministic testing
without a live LM Studio instance.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from hifi.agents.lm_client import make_llm
from hifi.agents.schemas import AgentSignal
from hifi.collective.debate import DebateTurn

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "agents" / "prompts"

_DEFAULT_BASE_MODEL = "qwen2.5-coder-32b-instruct-mlx"

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_prompt(name: str) -> tuple[str, str]:
    """Return (system_text, user_template) from a prompt markdown file."""
    path = _PROMPTS_DIR / f"{name}.md"
    raw = path.read_text(encoding="utf-8")
    parts = raw.split("## User", maxsplit=1)
    system_block = parts[0].replace("## System", "").strip()
    user_block = parts[1].strip() if len(parts) > 1 else ""
    return system_block, user_block


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _make_debate_llm(agent_type: str, max_tokens: int) -> ChatOpenAI:
    """Return the appropriate LLM for an agent type during debate phases."""
    if agent_type == "fundamental":
        ft_url = os.environ.get("HIFI_FUNDAMENTAL_FINETUNE_URL")
        ft_model = os.environ.get("HIFI_FUNDAMENTAL_FINETUNE_MODEL")
        if ft_url and ft_model:
            return make_llm(ft_model, max_tokens=max_tokens, base_url=ft_url)
        if ft_url:
            return make_llm(_DEFAULT_BASE_MODEL, max_tokens=max_tokens, base_url=ft_url)
        return make_llm(_DEFAULT_BASE_MODEL, max_tokens=max_tokens)

    if agent_type == "technical":
        ft_url = os.environ.get("HIFI_TECHNICAL_FINETUNE_URL")
        model = os.environ.get(
            "HIFI_TECHNICAL_MODEL",
            "mlx-qwen3.5-35b-a3b-claude-4.6-opus-reasoning-distilled",
        )
        return (
            make_llm(model, max_tokens=max_tokens, base_url=ft_url)
            if ft_url
            else make_llm(model, max_tokens=max_tokens)
        )

    if agent_type == "risk":
        model = os.environ.get("HIFI_RISK_MODEL", "google/gemma-3-4b")
        return make_llm(model, max_tokens=max_tokens)

    if agent_type == "macro":
        model = os.environ.get(
            "HIFI_MACRO_MODEL",
            "qwen3.5-27b-claude-4.6-opus-reasoning-distilled-qx64-hi-mlx",
        )
        return make_llm(model, max_tokens=max_tokens)

    # sentiment + unknown agent types
    model = os.environ.get("HIFI_SENTIMENT_MODEL", _DEFAULT_BASE_MODEL)
    return make_llm(model, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# JSON extraction (consistent with existing agent parse helpers)
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM response text."""
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


# ---------------------------------------------------------------------------
# Debate phase nodes
# ---------------------------------------------------------------------------


def challenge_node(
    signal: AgentSignal,
    majority_decision: str,
    majority_count: int,
    total_agents: int,
    llm: ChatOpenAI | None = None,
) -> DebateTurn:
    """
    Generate a challenge argument for a minority-position agent.

    The agent cites specific evidence from its original analysis to argue
    against the majority position. Output is free-form text (max 512 tokens).

    Parameters
    ----------
    signal : AgentSignal
        The agent's initial signal (provides rationale and key_concern).
    majority_decision : str
        The majority vote (Buy/Hold/Sell) being challenged.
    majority_count : int
        Number of agents holding the majority position.
    total_agents : int
        Total number of voting agents.
    llm : ChatOpenAI | None
        Optional LLM override. When None, selects by agent_type env vars.

    Returns
    -------
    DebateTurn
        Phase="challenge" turn with the generated argument.
    """
    system_text, user_template = _load_prompt("challenge_v1")
    user_text = user_template.format(
        agent_type=signal.agent_type,
        ticker=signal.ticker,
        as_of_date=signal.as_of_date,
        agent_decision=signal.decision,
        agent_confidence=f"{signal.confidence:.2f}",
        agent_rationale=signal.rationale,
        agent_key_concern=signal.key_concern,
        majority_decision=majority_decision,
        majority_count=majority_count,
        total_agents=total_agents,
    )

    _llm = llm if llm is not None else _make_debate_llm(signal.agent_type, max_tokens=512)
    model_id = _llm.model_name

    try:
        response = _llm.invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        argument = (
            response.content.strip()
            or f"Challenging {majority_decision}: {signal.key_concern}"
        )
    except Exception as exc:
        logger.warning(
            "challenge_node LLM failed for %s/%s: %s",
            signal.agent_type, signal.ticker, exc,
        )
        argument = (
            f"Challenge: {signal.agent_type} maintains {signal.decision}."
            f" {signal.key_concern}"
        )

    return DebateTurn(
        agent_type=signal.agent_type,
        phase="challenge",
        argument=argument,
        model_id=model_id,
    )


def respond_node(
    signal: AgentSignal,
    challenge_turns: list[DebateTurn],
    majority_decision: str,
    llm: ChatOpenAI | None = None,
) -> DebateTurn:
    """
    Generate a response to minority challenges for a majority-position agent.

    The agent acknowledges the strongest challenge and explains why the
    majority position still holds. Output is free-form text (max 256 tokens).

    Parameters
    ----------
    signal : AgentSignal
        The majority agent's initial signal.
    challenge_turns : list[DebateTurn]
        All challenge turns from minority agents.
    majority_decision : str
        The majority position being defended.
    llm : ChatOpenAI | None
        Optional LLM override for testing.

    Returns
    -------
    DebateTurn
        Phase="response" turn with the rebuttal argument.
    """
    challenges_text = "\n".join(
        f"- {t.agent_type}: {t.argument}" for t in challenge_turns
    ) or "No specific challenges raised."

    system_text, user_template = _load_prompt("response_v1")
    user_text = user_template.format(
        agent_type=signal.agent_type,
        ticker=signal.ticker,
        as_of_date=signal.as_of_date,
        agent_decision=signal.decision,
        agent_confidence=f"{signal.confidence:.2f}",
        agent_rationale=signal.rationale,
        majority_decision=majority_decision,
        challenges_text=challenges_text,
    )

    _llm = llm if llm is not None else _make_debate_llm(signal.agent_type, max_tokens=256)
    model_id = _llm.model_name

    try:
        response = _llm.invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        argument = response.content.strip() or f"{majority_decision} position maintained."
    except Exception as exc:
        logger.warning(
            "respond_node LLM failed for %s/%s: %s",
            signal.agent_type, signal.ticker, exc,
        )
        argument = f"Response: {signal.agent_type} maintains {signal.decision}. {signal.rationale}"

    return DebateTurn(
        agent_type=signal.agent_type,
        phase="response",
        argument=argument,
        model_id=model_id,
    )


def revise_node(
    signal: AgentSignal,
    challenge_turns: list[DebateTurn],
    response_turns: list[DebateTurn],
    majority_decision: str,
    llm: ChatOpenAI | None = None,
) -> tuple[DebateTurn, AgentSignal]:
    """
    Revise an agent's position after seeing the full debate transcript.

    Each agent (minority and majority) sees both the challenge and response
    turns, then decides whether to maintain or revise its decision.

    The revised AgentSignal is used for final voting in run_debate_ensemble().
    On LLM failure or JSON parse failure, the original signal is returned
    unchanged (fail-open: preserves pre-debate diversity).

    Parameters
    ----------
    signal : AgentSignal
        The agent's original signal.
    challenge_turns : list[DebateTurn]
        Challenge turns from minority agents.
    response_turns : list[DebateTurn]
        Response turns from majority agents.
    majority_decision : str
        The initial majority position.
    llm : ChatOpenAI | None
        Optional LLM override for testing.

    Returns
    -------
    tuple[DebateTurn, AgentSignal]
        (revision_turn, revised_signal)
        revision_turn.revised_decision and revised_signal.decision are identical.
    """
    transcript_parts: list[str] = []
    for t in challenge_turns:
        transcript_parts.append(f"CHALLENGE ({t.agent_type}): {t.argument}")
    for t in response_turns:
        transcript_parts.append(f"RESPONSE ({t.agent_type}): {t.argument}")
    debate_transcript = "\n".join(transcript_parts) or "No debate content."

    system_text, user_template = _load_prompt("revision_v1")
    user_text = user_template.format(
        agent_type=signal.agent_type,
        ticker=signal.ticker,
        as_of_date=signal.as_of_date,
        agent_decision=signal.decision,
        agent_confidence=f"{signal.confidence:.2f}",
        agent_rationale=signal.rationale,
        agent_key_concern=signal.key_concern,
        majority_decision=majority_decision,
        debate_transcript=debate_transcript,
    )

    _llm = llm if llm is not None else _make_debate_llm(signal.agent_type, max_tokens=1024)
    model_id = _llm.model_name

    parsed: dict | None = None
    try:
        response = _llm.invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=user_text),
        ])
        parsed = _extract_json(response.content)
    except Exception as exc:
        logger.warning(
            "revise_node LLM failed for %s/%s: %s",
            signal.agent_type, signal.ticker, exc,
        )

    if parsed is None:
        # Fall back to original signal — no revision
        revision_turn = DebateTurn(
            agent_type=signal.agent_type,
            phase="revision",
            argument=signal.rationale,
            revised_decision=signal.decision,
            revised_confidence=signal.confidence,
            model_id=model_id,
        )
        return revision_turn, signal

    # Extract revised values with fallbacks to original
    new_decision = parsed.get("decision", signal.decision)
    if new_decision not in ("Buy", "Hold", "Sell"):
        new_decision = signal.decision

    try:
        new_confidence = float(parsed.get("confidence", signal.confidence))
        new_confidence = max(0.0, min(1.0, new_confidence))
    except (TypeError, ValueError):
        new_confidence = signal.confidence

    new_rationale = str(parsed.get("rationale", signal.rationale)).strip() or signal.rationale
    new_key_concern = (
        str(parsed.get("key_concern", signal.key_concern)).strip() or signal.key_concern
    )

    revised_signal = AgentSignal(
        ticker=signal.ticker,
        as_of_date=signal.as_of_date,
        decision=new_decision,
        confidence=new_confidence,
        rationale=new_rationale,
        key_concern=new_key_concern,
        data_gaps=signal.data_gaps,
        call_ids=signal.call_ids,
        model_id=model_id,
        agent_type=signal.agent_type,
    )

    revision_turn = DebateTurn(
        agent_type=signal.agent_type,
        phase="revision",
        argument=new_rationale,
        revised_decision=new_decision,
        revised_confidence=new_confidence,
        model_id=model_id,
    )

    return revision_turn, revised_signal
