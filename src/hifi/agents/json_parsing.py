"""One definition of "read the model's answer as JSON" (DJ-140).

Every agent has to turn a language model's free text into a decision object, and
until now every agent had its own copy of the code that does it — six agents
plus the debate path, seven functions, functionally identical and differing only
in their docstrings. That is the shape DJ-135 removed from the ensemble roster,
for the same reason: a parser that is copied is a parser that drifts, and two
agents that disagree about what counts as a parseable response are a confound in
a study whose dependent variable is agent disagreement.

The guard below is not cosmetic either. ``json.loads`` returns whatever the JSON
contained — a list, a number, a bare string are all valid JSON — while every
caller immediately does ``parsed.get("decision")``. A model that answered
``["Buy", "Sell"]`` produced a list, and the caller raised AttributeError inside
the agent's own try/except, which recorded a parse failure naming neither the
cause nor the text. Returning None for "valid JSON, wrong shape" puts that case
on the same path as unparseable output, where the retry logic already lives.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON **object** from LLM response text.

    Strips markdown code fences, tries a direct parse, then falls back to the
    first ``{...}`` span in the text. Returns None when nothing parses, and also
    when what parses is not an object — a list or a scalar is valid JSON and
    still not an answer any caller can use.
    """
    text = text.strip()

    # Models frequently wrap the object in ```json ... ```.
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [line for line in lines if not line.startswith("```")]
        text = "\n".join(inner).strip()

    try:
        return _as_object(json.loads(text))
    except json.JSONDecodeError:
        pass

    # Prose around the object: take the widest {...} span and try that.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return _as_object(json.loads(text[start:end + 1]))
        except json.JSONDecodeError:
            pass

    return None


def _as_object(parsed: Any) -> dict[str, Any] | None:
    """A parsed value the callers can use, or None."""
    return parsed if isinstance(parsed, dict) else None


def message_text(content: str | list[str | dict[str, Any]] | Any) -> str:
    """Flatten a LangChain message's ``content`` to plain text (DJ-142).

    ``BaseMessage.content`` is ``str | list[str | dict]``: providers may answer
    with content *blocks* rather than a string. Every agent passed it straight
    into ``extract_json``, which begins ``text.strip()`` — so a block response
    would raise AttributeError inside the agent's own try/except and be recorded
    as a parse failure naming neither the cause nor the text, exactly the way a
    list from ``json.loads`` used to be.

    Local LM Studio models return strings today, which is why nothing has broken.
    That is a property of the current serving stack, not of the interface, and it
    is not one the experiment should depend on.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # The convention across providers: {"type": "text", "text": ...}
                value = block.get("text") or block.get("content") or ""
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(p for p in parts if p)
    return str(content)
