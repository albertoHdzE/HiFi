"""Tests for the universe-derived portfolio policy (DJ-122).

The hardcoded 5% / 20% / 1% limits were absolutes restated at every call site
and unrelated to how many names were being chosen from. They stranded capital
on a narrow book (8 Buys in one sector deployed 20% of the account) while
being inert across 98 names. Worse for the experiment, a fixed absolute cap
binds harder on a diversified arm than a concentrated one — so the constraint
became a function of the treatment, the same invariance failure recorded for
the circuit breaker in DJ-119.
"""

from __future__ import annotations

import pytest

from hifi.portfolio import PortfolioPolicy


class TestScaling:
    @pytest.mark.parametrize("n", [1, 2, 8, 16, 30, 50, 98, 200])
    def test_min_position_never_exceeds_max_single(self, n):
        """If these inverted, every position would be simultaneously too big
        and too small and the book would silently empty."""
        p = PortfolioPolicy(n)
        assert p.min_position < p.max_single_stock

    @pytest.mark.parametrize("n", [10, 16, 30, 50, 98, 200])
    def test_full_deployment_possible_once_the_book_is_wide_enough(self, n):
        """From 10 candidates up, the caps must not stand between the strategy
        and a fully invested book.

        This is the property whose absence caused the bug: with a flat 5% cap,
        8 Buys could reach only 40% invested and the rest sat in cash with no
        error raised anywhere. Measured before the fix: 8 Buys -> 20% deployed,
        16 -> 40%, 30 -> 70%.
        """
        p = PortfolioPolicy(n)
        assert n * p.max_single_stock >= 1.0

    @pytest.mark.parametrize("n", [1, 2, 5, 7, 9])
    def test_very_narrow_books_deploy_fully(self, n):
        """REVERSED 2026-08-27 (DJ-131). Was `test_very_narrow_books_
        deliberately_hold_cash`, asserting n=1 -> 10%, n=2 -> 20%, n=5 -> 50%.

        The original reasoning is preserved here because it was sound and is
        the reason this took two months to surface:

            "Below 10 candidates the 10% ceiling binds, and that is intended.
            If the ensemble finds only one Buy, the answer is a 10% position
            and 90% cash - not a single-name bet. Arm returns must remain a
            story about ensemble architecture, not about one company. Cash is
            the correct residual of low conviction, not a bug."

        Two things were bundled in that behaviour, and only one of them was
        intended:

        1. **Cash on narrow books** - genuinely intended, and now reversed by
           owner decision (2026-08-27): deploy the capital, and let risk
           valuation and capital allocation restrain it, not a blunt cap. The
           concentration concern above is real and is delegated to the risk
           manager, which is the layer that can actually see correlation.
        2. **Conviction erasure and treatment-dependence** - never intended,
           and concealed by (1). Because the same ceiling bound every name
           below n=10, arm A allocated an identical $10,000 to all seven of
           its holdings across a 1.61x conviction spread on 2026-08-24, and
           the size of the cash penalty was a monotone function of the Buy
           count, which *is* the treatment.

        The deployment decision now lives in `target_deployment`, so a future
        owner who wants cash on narrow books sets that field rather than
        rediscovering it as an emergent property of cap arithmetic.
        """
        p = PortfolioPolicy(n)
        assert n * p.max_single_stock >= p.target_deployment - 1e-9

    def test_cap_tightens_as_universe_widens(self):
        caps = [PortfolioPolicy(n).max_single_stock for n in (8, 16, 30, 98)]
        assert caps == sorted(caps, reverse=True)

    def test_relative_to_equal_weight_in_the_normal_range(self):
        p = PortfolioPolicy(98)
        assert p.max_single_stock == pytest.approx(3.0 * p.equal_weight)

    def test_absolute_ceiling_yields_to_tilt_headroom_on_narrow_books(self):
        """REVISED 2026-08-27 (DJ-131). Was `test_absolute_ceiling_applies_on_
        narrow_books`, asserting the cap equals 0.10 at n=2 and n=8.

        The concern it encoded is still real: 3 x equal weight explodes at
        small n (3 x 1/2 = 150%). But bounding it with an absolute constant
        made the ceiling *below* the equal weight for narrow books, which
        stops being a concentration limit and becomes a deployment limit:
        at n=8, 10% is 0.8x the equal share, so every name is clipped to the
        same value and 20% of the book cannot be invested at all.

        The bound is now `tilt_headroom` x equal weight, which caps the
        explosion (2x, not 3x) while keeping the cap above the equal share so
        conviction remains expressible and the capital remains investable.
        """
        for n in (2, 8):
            p = PortfolioPolicy(n)
            assert p.max_single_stock == pytest.approx(
                min(1.0, p.tilt_headroom * p.equal_weight))
            assert p.max_single_stock > p.equal_weight or n == 1
        # The explosion the original test guarded against is still guarded.
        assert PortfolioPolicy(2).max_single_stock <= 1.0

    def test_absolute_floor_applies_on_very_wide_books(self):
        assert PortfolioPolicy(1000).max_single_stock == 0.02


class TestSectorCap:
    def test_tracks_actual_composition(self):
        """A genuinely sector-heavy buy list should be allowed to be sector
        heavy rather than forced into cash by a flat 20%."""
        p = PortfolioPolicy(30)
        assert p.max_sector(5) == pytest.approx(0.25)    # floor
        assert p.max_sector(10) == pytest.approx(0.50)   # 1.5 x 10/30
        assert p.max_sector(30) == pytest.approx(1.0)    # all one sector

    def test_falls_back_to_floor_without_composition(self):
        assert PortfolioPolicy(30).max_sector() == 0.25

    def test_never_exceeds_one(self):
        assert PortfolioPolicy(10).max_sector(10) <= 1.0


class TestEdgeCases:
    def test_zero_candidates_is_safe(self):
        p = PortfolioPolicy(0)
        assert p.equal_weight == 0.0
        assert p.min_position == 0.0
        assert p.max_single_stock == p.ceil_max_single

    def test_negative_candidates_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            PortfolioPolicy(-1)

    def test_non_positive_knobs_rejected(self):
        with pytest.raises(ValueError):
            PortfolioPolicy(10, concentration=0)
        with pytest.raises(ValueError):
            PortfolioPolicy(10, sector_slack=-1)

    def test_frozen(self):
        """Immutable so a policy cannot be mutated between the compose and
        allocate stages of a single run."""
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            PortfolioPolicy(10).n_candidates = 20


class TestConstraintsDict:
    def test_supplies_every_key_the_pipeline_reads(self):
        c = PortfolioPolicy(30).as_constraints(
            capital=100_000.0, current_capital=5_000.0,
            available_cash=95_000.0, n_in_largest_sector=6,
        )
        assert set(c) == {
            "max_single_stock", "max_sector", "min_position",
            "target_deployment",  # DJ-131: the deployment decision, made explicit
            "capital", "current_capital", "available_cash",
        }
        assert c["capital"] == 100_000.0
        assert c["available_cash"] == 95_000.0
        assert c["max_sector"] == pytest.approx(0.30)

    def test_describe_is_one_line(self):
        d = PortfolioPolicy(98).describe()
        assert "\n" not in d
        assert "n=98" in d


# ---------------------------------------------------------------------------
# Sector cap must see positions already held (DJ-122)
# ---------------------------------------------------------------------------


class TestSectorCapWithExistingHoldings:
    """Held names consume sector budget even when they are not being traded.

    Arm A held NVDA (Information Technology) as a Hold, so the risk layer
    never saw it while allocating 30 Buys. Every individual check passed and
    the combined book still landed at 21.86% IT against a 20% cap.
    """

    @staticmethod
    def _cap(new, sectors, max_sector, existing=None):
        from hifi.mcp.portfolio_composer import _apply_sector_cap
        return _apply_sector_cap(new, sectors, max_sector, existing)

    def test_held_position_consumes_sector_budget(self):
        new = {"AAPL": 0.10, "MSFT": 0.10}
        sectors = {"AAPL": "IT", "MSFT": "IT", "NVDA": "IT"}
        out = self._cap(new, sectors, 0.20, {"NVDA": 0.05})
        # 0.20 budget less 0.05 already held -> 0.15 for the new names.
        assert sum(out.values()) == pytest.approx(0.15)

    def test_matches_old_behaviour_when_nothing_is_held(self):
        new = {"AAPL": 0.15, "MSFT": 0.15}
        sectors = {"AAPL": "IT", "MSFT": "IT"}
        assert sum(self._cap(new, sectors, 0.20).values()) == pytest.approx(0.20)

    def test_sector_already_full_gets_no_new_allocation(self):
        """Must not produce a negative budget or a negative weight."""
        out = self._cap(
            {"AAPL": 0.10}, {"AAPL": "IT", "NVDA": "IT"}, 0.20, {"NVDA": 0.25}
        )
        assert out["AAPL"] == 0.0

    def test_other_sectors_unaffected(self):
        new = {"AAPL": 0.10, "JPM": 0.10}
        sectors = {"AAPL": "IT", "JPM": "Financials", "NVDA": "IT"}
        out = self._cap(new, sectors, 0.20, {"NVDA": 0.15})
        assert out["JPM"] == pytest.approx(0.10)
        assert out["AAPL"] == pytest.approx(0.05)

    def test_held_name_also_in_new_allocation_is_not_double_counted(self):
        """A name being reallocated is already represented in `new`."""
        new = {"NVDA": 0.15}
        out = self._cap(new, {"NVDA": "IT"}, 0.20, {"NVDA": 0.05})
        assert out["NVDA"] == pytest.approx(0.15)

    def test_end_to_end_through_compose_portfolio(self):
        import json

        from hifi.mcp.portfolio_composer import compose_portfolio

        signals = [
            {"ticker": t, "decision": "Buy", "confidence": 0.7, "sector": "IT"}
            for t in ("AAPL", "MSFT", "ORCL")
        ]
        w = compose_portfolio(
            json.dumps(signals),
            max_single_stock=0.10, max_sector=0.20, min_position=0.01,
            existing_weights={"NVDA": 0.05},
            existing_sectors={"NVDA": "IT"},
        )
        assert sum(w.values()) + 0.05 <= 0.20 + 1e-9
