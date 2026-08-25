"""Strategy personalities — deterministic postures over recorded votes (Phase 20).

The owner-pressure idea: the same night's agent votes, replayed through
different managerial postures — aggressive, conservative, careful — produce
different collective decisions. This module is the decision-layer realization:

  - **Deterministic arithmetic** on (decision, confidence) vectors. No LLM
    calls, no prompt changes: the running ablation is untouched.
  - **Shadow-first**: results are computed nightly by
    ``scripts/run_personality_shadow.py`` from stored ensembles and written to
    ``shadow_personality.jsonl``. A personality is never traded until weeks of
    shadow evidence justify promoting it to a real arm via pre-registration.

Postures (v1 heuristics — tunable, but changes are protocol events):

  BASELINE      the live behavior: plain confidence-weighted plurality.
  AGGRESSIVE    half of the Hold mass leans toward entry when any Buy
                conviction exists; ties resolve toward action.
  CONSERVATIVE  a Buy must exceed both rival options by >= 25% margin,
                otherwise it degrades to Hold.
  CAREFUL       Conservative's entry rule plus a quarter of the Hold mass
                leaning to Sell (risk-lean) and amplified exits.

Conventions inherited from ``collective.voting``: scores are confidence sums
per option; ties resolve to Hold; empty input is Hold at zero confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

_OPTIONS = ("Buy", "Hold", "Sell")


@dataclass(frozen=True)
class PersonalityProfile:
    name: str
    hold_to_buy: float = 0.0     # fraction of Hold mass that leans to entry
    hold_to_sell: float = 0.0    # fraction of Hold mass that leans to exit
    buy_margin: float = 1.0      # Buy must exceed rivals by this factor (>1 enables)


BASELINE = PersonalityProfile("baseline")
AGGRESSIVE = PersonalityProfile("aggressive", hold_to_buy=0.5)
CONSERVATIVE = PersonalityProfile("conservative", buy_margin=1.25)
CAREFUL = PersonalityProfile("careful", hold_to_sell=0.25, buy_margin=1.25)

PERSONALITIES: dict[str, PersonalityProfile] = {
    p.name: p for p in (BASELINE, AGGRESSIVE, CONSERVATIVE, CAREFUL)
}


def posture_vote(votes: list[tuple[str, float]],
                 profile: PersonalityProfile = BASELINE) -> dict:
    """Aggregate (decision, confidence) votes under a personality posture.

    Returns {"decision": str, "confidence": float, "scores": {option: score}}.
    Deterministic: identical input and profile always yield identical output.
    """
    scores = {k: 0.0 for k in _OPTIONS}
    for decision, confidence in votes:
        if decision in scores and confidence > 0:
            scores[decision] += float(confidence)

    s_b, s_h, s_s = scores["Buy"], scores["Hold"], scores["Sell"]
    # The aggressive lean requires existing Buy conviction: without it, a
    # unanimous-Hold book would silently become a Buy — posture must not
    # manufacture conviction nobody expressed.
    h_to_buy = profile.hold_to_buy if s_b > 0 else 0.0
    adj = {
        "Buy": s_b + h_to_buy * s_h,
        "Hold": s_h * (1.0 - h_to_buy - profile.hold_to_sell),
        "Sell": s_s + profile.hold_to_sell * s_h,
    }
    total = sum(adj.values())
    if total <= 0:
        return {"decision": "Hold", "confidence": 0.0, "scores": adj}

    winner = max(_OPTIONS, key=lambda k: (adj[k], k == "Hold"))
    # max() with that key breaks value ties toward Hold, matching voting.py's
    # tie convention only when values are exactly equal across all three;
    # explicit two-way ties below keep the same conservatism.
    top = adj[winner]
    tied = [k for k in _OPTIONS if abs(adj[k] - top) < 1e-12]
    if len(tied) > 1:
        winner = "Hold"

    if (winner == "Buy" and profile.buy_margin > 1.0):
        rival = max(adj["Hold"], adj["Sell"])
        if rival > 0 and adj["Buy"] < profile.buy_margin * rival:
            winner = "Hold"

    return {
        "decision": winner,
        "confidence": round(adj[winner] / total, 4),
        "scores": {k: round(v, 6) for k, v in adj.items()},
    }
