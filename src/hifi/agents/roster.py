"""The agent roster: one definition of who is in the ensemble, and in what order.

Why this module exists
----------------------
Before DJ-135 the roster was written out as a list literal in six places —
``knowledge/agent_context.py``, ``agents/ensemble_runner.py``,
``analytics/decision_audit.py``, ``scripts/run_phase15_orchestrator.py``,
``scripts/run_phase15_walkforward.py`` and ``scripts/verify_agent_repair.py``.
Four of them listed six agents and two listed five, which was *correct* (the two
were counting voters) but indistinguishable from a copy that had drifted. Adding
a seventh agent, or promoting the contrarian to a voter, would have required
finding all six by hand, and any one missed would fail silently: an analytics
module that iterates five agents over six sidecars simply reports on five.

The distinction the duplicates were encoding is real and is preserved here as
two names rather than two literals.

CANONICAL_ORDER
    All six passes, in execution order. The sequential ensemble runs them in
    this order and each agent sees the summaries of those before it, so the
    order is causal, not cosmetic.

VOTING_AGENTS
    The five that emit a decision counted by ``confidence_weighted_vote``. The
    contrarian reviews the others and contributes no standalone vote, so it must
    never appear in a tally, a herding measure or a disagreement entropy — its
    inclusion would bias every diversity metric the experiment depends on.
"""

from __future__ import annotations

#: All agent passes, in causal execution order.
CANONICAL_ORDER: list[str] = [
    "fundamental",
    "technical",
    "risk",
    "macro",
    "sentiment",
    "contrarian",
]

#: The agent whose pass reviews the others rather than voting.
NON_VOTING: str = "contrarian"

#: The five agents whose decisions are counted in the collective vote.
VOTING_AGENTS: list[str] = [a for a in CANONICAL_ORDER if a != NON_VOTING]

__all__ = ["CANONICAL_ORDER", "NON_VOTING", "VOTING_AGENTS"]
