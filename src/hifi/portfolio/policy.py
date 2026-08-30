"""Single source of truth for portfolio construction limits (DJ-122).

Why this exists
---------------
The constraints ``max_single_stock=0.05``, ``max_sector=0.20`` and
``min_position=0.01`` were absolute constants, hardcoded independently in
``run_phase16_live.run_mcp_pipeline`` and as defaults on
``compose_portfolio``. They bore no relation to how many securities were
actually being chosen from, which produced two failures:

1. **Silent cash drag.** With a narrow or sector-concentrated buy list the
   caps bind long before the capital is deployed. Eight Buys in one sector
   invest 20% of the book and strand the other 80% — with no error anywhere.
   Measured before this change: 8 Buys -> 20% deployed, 16 -> 40%, 30 -> 70%.

2. **Meaningless caps at the other end.** Across 98 names the equal weight is
   1.02%, so a 5% single-stock cap is 5x the natural weight and never binds.
   The same number is simultaneously too tight at N=8 and inert at N=98.

Expressing limits as *multiples of the equal weight* makes one knob govern
every book width. ``concentration=3.0`` means "no position may exceed three
times its equal-weight share" and carries the same meaning whether the arm is
choosing among 8 names or 98. That is the universal control point: the limits
follow the universe instead of being restated beside it.

Design note — why not simply loosen the constants
-------------------------------------------------
Because the binding constraint must not be a function of the *treatment*. The
arms differ in how many names they select, so a fixed absolute cap taxes a
diversified arm and leaves a concentrated one untouched — exactly the
invariance failure recorded for the circuit breaker in DJ-119. A relative
policy applies identically regardless of how many names an arm picks.

DJ-131 — the design note above was violated by this module's own constant
--------------------------------------------------------------------------
DJ-122 replaced the sector-driven cash drag with a relative cap and then
bounded that cap with a new absolute constant, ``ceil_max_single = 0.10``. The
bound reintroduced the very failure this module exists to remove, at a
different ``n``:

* the deployment ceiling became ``min(1, 0.10 x n)``, so any arm emitting
  fewer than ten Buys could not invest its capital at all — 7 Buys stranded
  30% of the book;
* below ``n = 10`` every name hit the same cap, so conviction ordering was
  erased. Observed 2026-08-24: arm A allocated $10,000.00 +/- $0.08 to all
  seven of its names across a 1.61x conviction spread, while arm B (11 names,
  same night, same code) achieved a 1.5776x dollar spread.

Worse, the relative rule never bound in live operation. ``concentration``
takes effect only when ``3/n < 0.10``, i.e. ``n > 30``; for ``n = 1..97`` the
cap sat exactly on its ceiling for 30 of 97 values, and every observed live
Buy count (A=7, B=11, D=10) fell inside that region. The knob was pinned at
its own bracket and never inspected.

The lesson generalises beyond this file, and is the reason the paragraph above
was left in place rather than rewritten: **a recorded principle does not
enforce itself.** DJ-124 found a rejected LoRA still wired up while its
rejection sat in a bitácora; this is the same pattern one layer down, with the
principle in a docstring and the contradiction seventy lines below it.

The repair separates the two decisions the cap had been conflating. The cap
expresses *do not concentrate*; it must not also express *do not invest*. How
much capital is deployed is now an explicit, named decision
(``target_deployment``), and every cap is a multiple of the equal weight, so
full deployment is always reachable and conviction tilt is always expressible.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PortfolioPolicy"]


@dataclass(frozen=True)
class PortfolioPolicy:
    """Position limits derived from the number of investable candidates.

    Parameters
    ----------
    n_candidates : int
        Number of securities the portfolio is being built from. Use the count
        of actionable (Buy) signals, not the whole universe: the limits should
        reflect what is actually selectable on the day.
    concentration : float
        Maximum single-position weight, as a multiple of the equal weight.
        3.0 permits meaningful conviction tilts while keeping any one name
        from dominating.
    sector_slack : float
        Maximum sector weight, as a multiple of that sector's equal-weight
        share. Above 1.0 so a genuinely sector-heavy signal set is not forced
        into cash, which is what the old flat 20% did.
    min_position_frac : float
        Minimum position weight, as a fraction of the equal weight. Positions
        below it are dropped and redistributed — this exists to avoid dust,
        not to shape the portfolio.
    floor_max_single : float
        Absolute floor under ``max_single_stock``. At large N the relative cap
        becomes very small; this keeps a single name investable.
    ceil_max_single : float
        Absolute ceiling on a single position, applied **only where it is not
        the binding constraint on deployment**. At large N this is the hard
        concentration limit and behaves as before. At small N, where 10% of the
        book is at or below the equal weight, enforcing it would forbid full
        investment rather than forbid concentration — so it yields to
        ``tilt_headroom``. See the DJ-131 note in the module docstring.
    tilt_headroom : float
        Minimum cap, as a multiple of the equal weight. Guarantees two things
        no absolute constant can: full deployment is always *reachable*
        (``2.0 x 1/n x n = 2.0 >= 1.0``), and conviction tilt is always
        *expressible* (a cap equal to the equal weight would force a flat
        book). Lowering this below 1.0 reintroduces DJ-131.
    target_deployment : float
        Fraction of capital the arm intends to invest. **This is the
        deployment decision, and it is the only thing permitted to make it.**
        Any idle cash must be traceable to this field or to a named risk
        control — never to cap arithmetic, which is how DJ-131 stranded 30% of
        arm A's book with no error anywhere.

        Held at 1.0 (owner decision, 2026-08-27): deploy the capital, and let
        risk valuation and capital allocation restrain it, not a blunt limit.
        A consequence worth stating: a three-name book at 1.0 puts ~33% in
        each name, so concentration risk is delegated to the risk manager by
        design.
    floor_max_sector : float
        Absolute floor under ``max_sector``, so a narrow book is not trapped
        in cash by its own sector composition.
    """

    n_candidates: int
    concentration: float = 3.0
    sector_slack: float = 1.5
    min_position_frac: float = 0.25
    floor_max_single: float = 0.02
    ceil_max_single: float = 0.10
    tilt_headroom: float = 2.0
    target_deployment: float = 1.0
    floor_max_sector: float = 0.25

    def __post_init__(self) -> None:
        if self.n_candidates < 0:
            raise ValueError("n_candidates must be non-negative")
        if self.concentration <= 0 or self.sector_slack <= 0:
            raise ValueError("concentration and sector_slack must be positive")

    @property
    def equal_weight(self) -> float:
        """The weight each candidate receives under equal allocation."""
        return 1.0 / self.n_candidates if self.n_candidates > 0 else 0.0

    @property
    def max_single_stock(self) -> float:
        """Per-position cap, always expressed as a multiple of the equal weight.

        The cap is ``concentration`` x equal weight, bounded above by the
        absolute ceiling and below by ``tilt_headroom`` x equal weight. Stating
        every bound as a multiple is what makes the result invariant to book
        width: whatever ``n`` is, the cap is at least ``tilt_headroom`` times
        the equal share, so ``max_single_stock * n >= tilt_headroom >= 1``
        and full deployment is always reachable.

        The absolute ceiling therefore binds only where it is genuinely a
        concentration limit (large ``n``), never where it would silently
        become a deployment limit (small ``n``) — the DJ-131 defect.
        """
        if self.n_candidates <= 0:
            return self.ceil_max_single
        eq = self.equal_weight
        # All three bounds in "multiples of the equal weight" so they compare.
        ceiling_mult = max(self.tilt_headroom, self.ceil_max_single / eq)
        mult = min(self.concentration, ceiling_mult)
        # Clamped at 1.0: a weight above the whole book is not a cap, it is a
        # number that will leak into the composer as a nonsense constraint.
        # Reachable at n <= 2, where tilt_headroom x equal weight exceeds 1.
        return min(1.0, max(self.floor_max_single, mult * eq))

    @property
    def max_deployable(self) -> float:
        """Upper bound on the fraction of capital the caps permit investing.

        Exists so the invariant can be asserted rather than inferred. Under
        DJ-131 this silently equalled ``min(1, 0.10 * n)``; it must now be
        1.0 for every ``n``, and any shortfall is a defect, not a preference.
        """
        if self.n_candidates <= 0:
            return 0.0
        return min(1.0, self.max_single_stock * self.n_candidates)

    @property
    def min_position(self) -> float:
        """Dust threshold: a fraction of the equal weight.

        Kept strictly below ``max_single_stock`` so the two can never invert
        and silently empty the portfolio.
        """
        if self.n_candidates <= 0:
            return 0.0
        return min(self.min_position_frac * self.equal_weight,
                   self.max_single_stock * 0.5)

    def max_sector(self, n_in_largest_sector: int | None = None) -> float:
        """Sector cap.

        With ``n_in_largest_sector`` the cap tracks the actual composition of
        the buy list: a set that is genuinely 60% one sector is allowed to be
        sector-heavy rather than forced into cash. Without it, falls back to
        the floor.
        """
        if not self.n_candidates or not n_in_largest_sector:
            return self.floor_max_sector
        share = n_in_largest_sector / self.n_candidates
        return min(1.0, max(self.floor_max_sector, self.sector_slack * share))

    def as_constraints(
        self,
        capital: float,
        current_capital: float = 0.0,
        available_cash: float | None = None,
        n_in_largest_sector: int | None = None,
    ) -> dict:
        """Build the constraints dict consumed by ``run_pipeline``.

        This is the one place the pipeline's limit vocabulary is assembled, so
        call sites cannot drift apart again.
        """
        return {
            "max_single_stock": self.max_single_stock,
            "max_sector": self.max_sector(n_in_largest_sector),
            "min_position": self.min_position,
            "target_deployment": self.target_deployment,
            "capital": capital,
            "current_capital": current_capital,
            "available_cash": available_cash,
        }

    def describe(self) -> str:
        """One-line summary for run logs and the report's provenance panel.

        ``max_deployable`` is printed because its silent collapse was the
        DJ-131 defect: a log line that showed only the cap looked healthy
        while 30% of the book sat idle.
        """
        return (
            f"policy(n={self.n_candidates}, equal_weight={self.equal_weight * 100:.2f}%, "
            f"max_single={self.max_single_stock * 100:.2f}% "
            f"({self.max_single_stock / self.equal_weight:.2f}x eq), "
            f"max_deployable={self.max_deployable * 100:.1f}%, "
            f"target_deployment={self.target_deployment * 100:.0f}%, "
            f"min_position={self.min_position * 100:.2f}%, "
            f"max_sector>={self.floor_max_sector * 100:.0f}%)"
            if self.n_candidates > 0 else "policy(n=0, no candidates)"
        )
