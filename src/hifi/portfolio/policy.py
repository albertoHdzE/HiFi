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
        Absolute ceiling. At small N the relative cap explodes (3 x 1/2 =
        150%); this is the hard concentration limit that always applies. Held
        at 10%: this is a diversified multi-name strategy study, and a single
        position above a tenth of the book would make arm-level returns a
        story about one company rather than about ensemble architecture.
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
        """Per-position cap: ``concentration`` x equal weight, bounded."""
        if self.n_candidates <= 0:
            return self.ceil_max_single
        raw = self.concentration * self.equal_weight
        return min(self.ceil_max_single, max(self.floor_max_single, raw))

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
            "capital": capital,
            "current_capital": current_capital,
            "available_cash": available_cash,
        }

    def describe(self) -> str:
        """One-line summary for run logs and the report's provenance panel."""
        return (
            f"policy(n={self.n_candidates}, equal_weight={self.equal_weight * 100:.2f}%, "
            f"max_single={self.max_single_stock * 100:.2f}%, "
            f"min_position={self.min_position * 100:.2f}%, "
            f"max_sector>={self.floor_max_sector * 100:.0f}%)"
        )
