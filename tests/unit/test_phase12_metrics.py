"""
Unit tests for compute_factorial_summary() and _herding_coefficient() from
scripts/archive/run_phase12_evaluation.py (P12-E4-T2).

No LLM calls. All inputs are synthetic, analytically derived.

Scientific grounding (David SS5.6.3, DJ-067):
- herding_coefficient α_t = majority_count / N.  Range [1/N, 1.0].
- interaction_effect = (D−B) − (C−A): positive ↔ fine-tuning amplifies debate benefit.
- OQ-M02 threshold: < 10% disagreement_entropy degradation relative to Condition A.
- herding assessment threshold: > 0.10 absolute increase (A → C) flags debate-induced herding.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loading (script, not package — imported via importlib)
# ---------------------------------------------------------------------------

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "archive" / "run_phase12_evaluation.py")


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("run_phase12_evaluation", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_eval_module()
_herding_coefficient = _mod._herding_coefficient
_interaction_effect = _mod._interaction_effect
compute_factorial_summary = _mod.compute_factorial_summary


# ---------------------------------------------------------------------------
# _herding_coefficient (David §5.6.3)
# ---------------------------------------------------------------------------


def test_herding_empty_returns_zero():
    """Empty agent list is not a valid ensemble, returns 0.0 sentinel."""
    assert _herding_coefficient([]) == 0.0


def test_herding_unanimous():
    """All agents in agreement → α = 1.0."""
    assert _herding_coefficient(["Buy", "Buy", "Buy", "Buy", "Buy"]) == pytest.approx(1.0)


def test_herding_two_thirds_majority():
    """2/3 agents share the plurality → α = 2/3."""
    result = _herding_coefficient(["Buy", "Buy", "Sell"])
    assert result == pytest.approx(2 / 3)


def test_herding_three_fifths_majority():
    """3/5 agents vote with plurality → α = 0.6."""
    result = _herding_coefficient(["Buy", "Buy", "Buy", "Sell", "Hold"])
    assert result == pytest.approx(3 / 5)


def test_herding_equal_split():
    """Perfectly split vote (2 agents, different votes) → α = 0.5."""
    result = _herding_coefficient(["Buy", "Sell"])
    assert result == pytest.approx(0.5)


def test_herding_single_agent():
    """Single agent is always unanimous with itself → α = 1.0."""
    assert _herding_coefficient(["Hold"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _interaction_effect (DJ-067 interaction term)
# ---------------------------------------------------------------------------


def test_interaction_positive():
    """(D−B) − (C−A) > 0: fine-tuning amplifies debate benefit."""
    # A=0.5, B=0.6, C=0.7, D=0.9 → (0.9−0.6) − (0.7−0.5) = 0.3 − 0.2 = 0.1
    assert _interaction_effect(0.5, 0.6, 0.7, 0.9) == pytest.approx(0.1)


def test_interaction_negative():
    """(D−B) − (C−A) < 0: debate benefits base models more than fine-tuned."""
    # A=0.5, B=0.8, C=0.7, D=0.8 → (0.8−0.8) − (0.7−0.5) = 0.0 − 0.2 = −0.2
    assert _interaction_effect(0.5, 0.8, 0.7, 0.8) == pytest.approx(-0.2)


def test_interaction_zero():
    """Additive effects (no interaction) → 0.0."""
    # A=0.5, B=0.6, C=0.6, D=0.7 → (0.7−0.6) − (0.6−0.5) = 0.1 − 0.1 = 0.0
    assert _interaction_effect(0.5, 0.6, 0.6, 0.7) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Helpers for compute_factorial_summary tests
# ---------------------------------------------------------------------------


def _run(
    entropy: float,
    dispersion: float,
    herding: float,
    vote_delta: str | None = None,
    debate_skipped: bool | None = None,
    n_changed: int | None = None,
) -> dict:
    """Build a synthetic metric dict matching the shape from _extract_metrics()."""
    return {
        "disagreement_entropy": entropy,
        "opinion_dispersion": dispersion,
        "herding_coefficient": herding,
        "collective_decision": "Buy",
        "n_valid_signals": 2,
        "vote_delta": vote_delta,
        "debate_skipped": debate_skipped,
        "n_agents_changed_vote": n_changed,
    }


def _ur(n: int, entropy: float, herding: float, **kwargs) -> list[dict]:
    """n identical runs with given entropy and herding (uniform_runs shorthand)."""
    return [_run(entropy=entropy, dispersion=0.5, herding=herding, **kwargs) for _ in range(n)]


# ---------------------------------------------------------------------------
# compute_factorial_summary — structural correctness
# ---------------------------------------------------------------------------


def test_summary_condition_run_counts():
    """n_runs matches input list lengths for each condition."""
    all_metrics = {
        "A": _ur(10, entropy=0.8, herding=0.6),
        "B": _ur(10, entropy=0.7, herding=0.7),
        "C": _ur(
            10, entropy=0.75, herding=0.65,
            vote_delta="unchanged", debate_skipped=True,
        ),
        "D": _ur(
            10, entropy=0.9, herding=0.8,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    for cond in "ABCD":
        assert summary["conditions"][cond]["n_runs"] == 10


def test_summary_mean_herding_correct():
    """Mean herding per condition is the arithmetic mean of input values."""
    # Condition A: 3 runs with herding 0.6, 0.8, 1.0 → mean = 0.8
    all_metrics = {
        "A": [
            _run(entropy=0.8, dispersion=0.5, herding=0.6),
            _run(entropy=0.8, dispersion=0.5, herding=0.8),
            _run(entropy=0.8, dispersion=0.5, herding=1.0),
        ],
        "B": _ur(3, entropy=0.7, herding=0.9),
        "C": _ur(3, entropy=0.7, herding=0.7, vote_delta="unchanged", debate_skipped=True),
        "D": _ur(
            3, entropy=0.9, herding=0.8,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    assert summary["conditions"]["A"]["mean_herding_coefficient"] == pytest.approx(0.8)
    assert summary["conditions"]["B"]["mean_herding_coefficient"] == pytest.approx(0.9)


def test_summary_mean_entropy_correct():
    """Mean disagreement_entropy per condition is correctly computed."""
    all_metrics = {
        "A": [
            _run(entropy=1.0, dispersion=0.5, herding=0.5),
            _run(entropy=0.5, dispersion=0.5, herding=0.5),
        ],
        "B": _ur(2, entropy=0.6, herding=0.7),
        "C": _ur(2, entropy=0.7, herding=0.6, vote_delta="unchanged", debate_skipped=True),
        "D": _ur(
            2, entropy=0.9, herding=0.8,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    assert summary["conditions"]["A"]["mean_disagreement_entropy"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# compute_factorial_summary — interaction effect correctness
# ---------------------------------------------------------------------------


def test_summary_interaction_effect_positive():
    """
    Positive entropy interaction: fine-tuning amplifies debate's diversity benefit.

    A: entropy=0.5, B: entropy=0.6 (FT increases diversity by 0.1)
    C: entropy=0.7, D: entropy=0.9 (FT+debate increases diversity by 0.2)
    Interaction = (0.9−0.6) − (0.7−0.5) = 0.3 − 0.2 = +0.1
    """
    all_metrics = {
        "A": _ur(5, entropy=0.5, herding=0.8),
        "B": _ur(5, entropy=0.6, herding=0.7),
        "C": _ur(
            5, entropy=0.7, herding=0.6,
            vote_delta="unchanged", debate_skipped=True,
        ),
        "D": _ur(
            5, entropy=0.9, herding=0.4,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    assert summary["interaction_effects"]["disagreement_entropy"] == pytest.approx(0.1)
    # FT reduces herding more when combined with debate → negative herding interaction
    assert summary["interaction_effects"]["herding_coefficient"] < 0


def test_summary_interaction_effect_sign_consistency():
    """Interaction effect is zero when all conditions share the same metric value."""
    # All conditions identical → (D-B)-(C-A) = 0.0 for all metrics
    all_metrics = {cond: _ur(3, entropy=0.5, herding=0.5) for cond in "ABCD"}
    summary = compute_factorial_summary(all_metrics)
    assert summary["interaction_effects"]["disagreement_entropy"] == pytest.approx(0.0)
    assert summary["interaction_effects"]["herding_coefficient"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_factorial_summary — OQ-M02 diversity preservation
# ---------------------------------------------------------------------------


def test_oq_m02_finetune_preserved_when_small_degradation():
    """Fine-tuning degrades entropy by < 10% → diversity_preserved_finetune = True."""
    # entropy_a=1.0, entropy_b=0.95 → degradation = (1.0−0.95)/1.0 = 5% < 10%
    all_metrics = {
        "A": _ur(5, entropy=1.0, herding=0.5),
        "B": _ur(5, entropy=0.95, herding=0.55),
        "C": _ur(
            5, entropy=0.9, herding=0.6,
            vote_delta="unchanged", debate_skipped=True,
        ),
        "D": _ur(
            5, entropy=0.98, herding=0.52,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    assert summary["oq_m02"]["diversity_preserved_finetune"] is True
    assert summary["oq_m02"]["finetune_effect_entropy_degradation"] == pytest.approx(0.05)


def test_oq_m02_finetune_not_preserved_when_large_degradation():
    """Fine-tuning degrades entropy by > 10% → diversity_preserved_finetune = False."""
    # entropy_a=1.0, entropy_b=0.80 → degradation = 20% > 10%
    all_metrics = {
        "A": _ur(5, entropy=1.0, herding=0.5),
        "B": _ur(5, entropy=0.80, herding=0.7),
        "C": _ur(
            5, entropy=0.9, herding=0.6,
            vote_delta="unchanged", debate_skipped=True,
        ),
        "D": _ur(
            5, entropy=0.85, herding=0.65,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    assert summary["oq_m02"]["diversity_preserved_finetune"] is False
    assert summary["oq_m02"]["finetune_effect_entropy_degradation"] == pytest.approx(0.20)


def test_oq_m02_debate_preserved_when_small_degradation():
    """Debate degrades entropy by < 10% → diversity_preserved_debate = True."""
    # entropy_a=1.0, entropy_c=0.92 → debate degradation = 8% < 10%
    all_metrics = {
        "A": _ur(5, entropy=1.0, herding=0.5),
        "B": _ur(5, entropy=0.95, herding=0.55),
        "C": _ur(
            5, entropy=0.92, herding=0.6,
            vote_delta="unchanged", debate_skipped=True,
        ),
        "D": _ur(
            5, entropy=0.98, herding=0.52,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    assert summary["oq_m02"]["diversity_preserved_debate"] is True


def test_oq_m02_zero_entropy_baseline_no_crash():
    """entropy_a = 0.0 (all unanimous) → degradation = 0.0, no ZeroDivisionError."""
    all_metrics = {cond: _ur(3, entropy=0.0, herding=1.0) for cond in "ABCD"}
    summary = compute_factorial_summary(all_metrics)
    assert summary["oq_m02"]["finetune_effect_entropy_degradation"] == pytest.approx(0.0)
    assert summary["oq_m02"]["debate_effect_entropy_degradation"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_factorial_summary — vote delta distribution
# ---------------------------------------------------------------------------


def test_summary_vote_delta_distribution():
    """vote_delta Counter correctly tallies converged/diverged/unchanged labels."""
    debate_runs = [
        _run(entropy=0.5, dispersion=0.5, herding=0.6,
             vote_delta="converged", debate_skipped=False),
        _run(entropy=0.5, dispersion=0.5, herding=0.6,
             vote_delta="converged", debate_skipped=False),
        _run(entropy=0.5, dispersion=0.5, herding=0.5,
             vote_delta="unchanged", debate_skipped=True),
        _run(entropy=0.5, dispersion=0.5, herding=0.4,
             vote_delta="diverged", debate_skipped=False),
    ]
    all_metrics = {
        "A": _ur(4, entropy=0.8, herding=0.6),
        "B": _ur(4, entropy=0.7, herding=0.7),
        "C": debate_runs,
        "D": debate_runs,
    }
    summary = compute_factorial_summary(all_metrics)
    dist = summary["conditions"]["C"]["vote_delta_distribution"]
    assert dist["converged"] == 2
    assert dist["unchanged"] == 1
    assert dist["diverged"] == 1


def test_summary_vote_delta_none_excluded():
    """Runs with vote_delta=None (no-debate conditions) are excluded from distribution."""
    all_metrics = {
        "A": _ur(3, entropy=0.8, herding=0.6),    # vote_delta=None
        "B": _ur(3, entropy=0.7, herding=0.7),    # vote_delta=None
        "C": _ur(3, entropy=0.7, herding=0.65, vote_delta="unchanged", debate_skipped=True),
        "D": _ur(
            3, entropy=0.9, herding=0.8,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    # A and B have no debate; their vote_delta_distribution should be empty Counter
    assert dict(summary["conditions"]["A"]["vote_delta_distribution"]) == {}
    assert dict(summary["conditions"]["B"]["vote_delta_distribution"]) == {}


# ---------------------------------------------------------------------------
# compute_factorial_summary — herding assessment
# ---------------------------------------------------------------------------


def test_herding_assessment_induces_herding():
    """herding_c − herding_a > 0.10 → debate_induces_herding = True."""
    all_metrics = {
        "A": _ur(5, entropy=0.8, herding=0.50),
        "B": _ur(5, entropy=0.7, herding=0.60),
        "C": _ur(
            5, entropy=0.7, herding=0.65,
            vote_delta="converged", debate_skipped=False,
        ),
        "D": _ur(
            5, entropy=0.9, herding=0.80,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    assert summary["herding_assessment"]["debate_induces_herding"] is True
    assert summary["herding_assessment"]["herding_increase_A_to_C"] == pytest.approx(0.15)


def test_herding_assessment_no_herding():
    """herding_c − herding_a <= 0.10 → debate_induces_herding = False."""
    all_metrics = {
        "A": _ur(5, entropy=0.8, herding=0.60),
        "B": _ur(5, entropy=0.7, herding=0.65),
        "C": _ur(
            5, entropy=0.7, herding=0.65,
            vote_delta="unchanged", debate_skipped=True,
        ),
        "D": _ur(
            5, entropy=0.9, herding=0.70,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    assert summary["herding_assessment"]["debate_induces_herding"] is False
    assert summary["herding_assessment"]["herding_increase_A_to_C"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# compute_factorial_summary — debate participation rate
# ---------------------------------------------------------------------------


def test_debate_participation_rate():
    """debate_participation_rate = fraction of non-skipped runs in debate conditions."""
    # 3 runs: 2 participated (debate_skipped=False), 1 skipped → rate = 2/3
    debate_runs = [
        _run(entropy=0.6, dispersion=0.5, herding=0.6,
             vote_delta="converged", debate_skipped=False),
        _run(entropy=0.6, dispersion=0.5, herding=0.6,
             vote_delta="converged", debate_skipped=False),
        _run(entropy=0.6, dispersion=0.5, herding=0.8,
             vote_delta="unchanged", debate_skipped=True),
    ]
    all_metrics = {
        "A": _ur(3, entropy=0.8, herding=0.6),
        "B": _ur(3, entropy=0.7, herding=0.7),
        "C": debate_runs,
        "D": debate_runs,
    }
    summary = compute_factorial_summary(all_metrics)
    rate = summary["conditions"]["C"]["debate_participation_rate"]
    assert rate == pytest.approx(2 / 3)


def test_debate_participation_rate_none_when_no_debate():
    """Non-debate conditions return None for debate_participation_rate."""
    all_metrics = {
        "A": _ur(3, entropy=0.8, herding=0.6),
        "B": _ur(3, entropy=0.7, herding=0.7),
        "C": _ur(3, entropy=0.7, herding=0.65, vote_delta="unchanged", debate_skipped=True),
        "D": _ur(
            3, entropy=0.9, herding=0.8,
            vote_delta="converged", debate_skipped=False,
        ),
    }
    summary = compute_factorial_summary(all_metrics)
    # Condition A has no debate_skipped flags → rate = None
    assert summary["conditions"]["A"]["debate_participation_rate"] is None


# ---------------------------------------------------------------------------
# compute_factorial_summary — empty conditions
# ---------------------------------------------------------------------------


def test_summary_empty_condition_no_crash():
    """Missing condition keys (partial runs) do not raise; results are floats."""
    all_metrics: dict = {
        "A": _ur(3, entropy=0.8, herding=0.6),
    }
    summary = compute_factorial_summary(all_metrics)
    # Must not raise; interaction_effects must be a dict of floats
    assert "interaction_effects" in summary
    ie = summary["interaction_effects"]
    assert isinstance(ie["herding_coefficient"], float)
    assert isinstance(ie["disagreement_entropy"], float)
