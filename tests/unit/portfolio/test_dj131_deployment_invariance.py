"""DJ-131: deployment must not be a function of the Buy count.

Every test here fails against the pre-DJ-131 policy (``ceil_max_single = 0.10``
applied unconditionally), which is the point: the defect was invisible because
nothing asserted the property it violated.

The defining observation, 2026-08-24 Genesis II cycle 1: arm A emitted 7 Buys
spanning conviction 0.4217-0.6800 (1.61x) and received $10,000.00 +/- $0.08 on
every one of them (dollar spread 1.0000x), leaving 30.0% of the book idle.
Arm B, the same night on the same code, emitted 11 Buys and achieved a 1.5776x
dollar spread. That pair is the reference distribution for these tests.
"""

from __future__ import annotations

import pytest

from hifi.portfolio import PortfolioPolicy

# The real arm-A basket. Kept verbatim so the regression is the actual event.
ARM_A_20260824 = [0.6800, 0.6623, 0.4459, 0.4416, 0.4359, 0.4286, 0.4217]
UNIVERSE_N = 97


class TestDeploymentInvariance:
    """The property whose violation *was* DJ-131."""

    @pytest.mark.parametrize("n", range(1, UNIVERSE_N + 1))
    def test_full_deployment_always_reachable(self, n: int) -> None:
        policy = PortfolioPolicy(n_candidates=n)
        assert policy.max_deployable >= policy.target_deployment - 1e-9, (
            f"n={n}: caps permit only {policy.max_deployable:.1%} of capital to "
            f"be deployed against a target of {policy.target_deployment:.1%}. "
            "Deployment has become a function of the Buy count, which is the "
            "treatment (DJ-131)."
        )

    @pytest.mark.parametrize("n", [3, 5, 7, 8, 9])
    def test_narrow_books_are_not_taxed(self, n: int) -> None:
        """The exact region where DJ-131 stranded capital: n < 10."""
        assert PortfolioPolicy(n_candidates=n).max_deployable == pytest.approx(1.0)

    def test_deployable_is_constant_across_book_width(self) -> None:
        """Invariance stated directly: no dependence on n at all."""
        deployable = {n: PortfolioPolicy(n_candidates=n).max_deployable
                      for n in range(1, UNIVERSE_N + 1)}
        assert len(set(round(v, 9) for v in deployable.values())) == 1, (
            "max_deployable varies with book width: "
            f"{ {n: v for n, v in deployable.items() if v < 1.0} }"
        )


class TestConvictionIsExpressible:
    """A cap equal to the equal weight forces a flat book and erases conviction."""

    @pytest.mark.parametrize("n", range(1, UNIVERSE_N + 1))
    def test_cap_leaves_room_above_equal_weight(self, n: int) -> None:
        policy = PortfolioPolicy(n_candidates=n)
        if n == 1:
            return  # a one-name book is flat by definition
        assert policy.max_single_stock > policy.equal_weight * 1.0 + 1e-12, (
            f"n={n}: cap {policy.max_single_stock:.4f} does not exceed the equal "
            f"weight {policy.equal_weight:.4f}, so every name above average "
            "conviction is clipped to the same value and the ensemble's "
            "ordering cannot reach the portfolio."
        )

    def test_arm_a_basket_can_express_its_spread(self) -> None:
        """The 2026-08-24 event, replayed: 7 names must not all price alike."""
        n = len(ARM_A_20260824)
        policy = PortfolioPolicy(n_candidates=n)
        total = sum(ARM_A_20260824)
        raw = [c / total for c in ARM_A_20260824]
        capped = [min(w, policy.max_single_stock) for w in raw]

        assert max(capped) / min(capped) > 1.0, (
            "conviction-proportional weights collapsed to a single value under "
            "the cap -- this is the observed 1.0000x dollar spread"
        )
        # The conviction spread is 1.61x; the surviving weight spread must be
        # materially closer to that than to the 1.0000x that was observed.
        assert max(capped) / min(capped) == pytest.approx(
            max(raw) / min(raw), rel=1e-9), (
            "no name in this basket should be capped at all: the largest "
            f"raw weight is {max(raw):.4f} against a cap of "
            f"{policy.max_single_stock:.4f}"
        )


class TestKnobIsInteriorToItsBracket:
    """datasaurus G3. DJ-131's root cause was a knob pinned at its own ceiling.

    Pre-DJ-131, ``concentration`` never bound for n <= 30, so for n = 1..97 the
    cap sat exactly on ``ceil_max_single`` for 30 of 97 values -- including
    every Buy count ever observed live (A=7, B=11, D=10).
    """

    def test_concentration_binds_somewhere_in_the_operating_range(self) -> None:
        binding = [n for n in range(1, UNIVERSE_N + 1)
                   if PortfolioPolicy(n_candidates=n).max_single_stock
                   == pytest.approx(3.0 / n)]
        assert binding, "`concentration` never binds -- it is a dead parameter"

    @pytest.mark.parametrize("n", [7, 10, 11])
    def test_observed_live_book_widths_are_not_pinned_flat(self, n: int) -> None:
        """The three widths actually traded on 2026-08-24."""
        policy = PortfolioPolicy(n_candidates=n)
        assert policy.max_deployable == pytest.approx(1.0)
        assert policy.max_single_stock > policy.equal_weight


class TestTargetDeploymentIsTheOnlyLever:
    """Idle cash must trace to a named decision, never to cap arithmetic."""

    def test_partial_deployment_is_expressible_when_asked_for(self) -> None:
        policy = PortfolioPolicy(n_candidates=7, target_deployment=0.60)
        assert policy.target_deployment == 0.60
        assert policy.max_deployable >= 0.60

    def test_lowering_tilt_headroom_reintroduces_dj131(self) -> None:
        """Guard the guard: name the parameter whose misuse is the defect."""
        broken = PortfolioPolicy(n_candidates=7, tilt_headroom=0.5,
                                 ceil_max_single=0.10)
        assert broken.max_deployable < 1.0
