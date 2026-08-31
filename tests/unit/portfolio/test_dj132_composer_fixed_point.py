"""DJ-132: every constraint must hold simultaneously, not in turn.

The pre-DJ-132 composer applied the per-stock cap, the sector cap and the
minimum position exactly once each, in that order, and never looked again.
Each step is individually correct and each invalidates the ones before it, so
the returned book satisfied every constraint *at the moment that constraint was
applied* and none of them jointly.

Three failures were reproduced on the real 2026-08-24 arm-A basket before the
repair; each has a regression here, with the measured number in the assertion
message so a future reader sees the actual event rather than a description of
it. The fuzz test at the bottom is the one that generalises: it asserts the
post-conditions over random baskets, which is the check whose absence let a
25.92%-vs-25.00% sector breach sit in a returned portfolio.
"""

from __future__ import annotations

import json
import random

import pytest

from hifi.mcp.portfolio_composer import (
    PortfolioConstraintError,
    _solve_to_fixed_point,
    compose_portfolio,
)
from hifi.portfolio import PortfolioPolicy

# The real arm-A basket, 2026-08-24. Conviction spread 1.6125x.
ARM_A_CONF = [0.6800, 0.6623, 0.4459, 0.4416, 0.4359, 0.4286, 0.4217]
ARM_A_SECTORS = [
    "Information Technology", "Information Technology", "Financials",
    "Health Care", "Energy", "Utilities", "Industrials",
]


def _signals(confidences, sectors):
    return json.dumps([
        {"ticker": f"T{i}", "decision": "Buy", "confidence": c, "sector": s}
        for i, (c, s) in enumerate(zip(confidences, sectors, strict=True))
    ])


def _sector_totals(weights, sectors_by_ticker):
    totals: dict[str, float] = {}
    for t, v in weights.items():
        s = sectors_by_ticker[t]
        totals[s] = totals.get(s, 0.0) + v
    return totals


class TestResidualReachesTheBook:
    """Capping is subtractive; without a fill step the book silently shrinks."""

    def test_arm_a_basket_deploys_its_target(self):
        policy = PortfolioPolicy(n_candidates=7)
        weights = compose_portfolio(
            _signals(ARM_A_CONF, ARM_A_SECTORS),
            max_single_stock=policy.max_single_stock,
            max_sector=policy.max_sector(n_in_largest_sector=2),
            min_position=policy.min_position,
            target_deployment=policy.target_deployment,
        )
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6), (
            "the one-pass composer deployed 86.8231% of this basket and sent "
            "13.1769% to cash with no decision recorded anywhere"
        )
        assert len(weights) == 7, "no ensemble pick may be discarded silently"

    def test_every_candidate_survives_when_constraints_allow(self):
        policy = PortfolioPolicy(n_candidates=7)
        weights = compose_portfolio(
            _signals(ARM_A_CONF, ARM_A_SECTORS),
            max_single_stock=policy.max_single_stock,
            max_sector=policy.max_sector(n_in_largest_sector=2),
            min_position=policy.min_position,
        )
        assert set(weights) == {f"T{i}" for i in range(7)}


class TestConvictionOrderingSurvivesTheConstraints:
    """The sector cap inverted the ensemble's ranking, which is worse than
    stranding cash: the portfolio expressed the opposite of the signal."""

    def test_ordering_matches_confidence_when_no_sector_cap_binds(self):
        policy = PortfolioPolicy(n_candidates=7)
        weights = compose_portfolio(
            _signals(ARM_A_CONF, ARM_A_SECTORS),
            max_single_stock=policy.max_single_stock,
            max_sector=policy.max_sector(n_in_largest_sector=2),
            min_position=policy.min_position,
        )
        by_weight = [t for t, _ in sorted(weights.items(), key=lambda kv: -kv[1])]
        by_conf = [f"T{i}" for i in sorted(range(7), key=lambda i: -ARM_A_CONF[i])]
        assert by_weight == by_conf, (
            "the one-pass composer gave the 0.6800-conviction name 12.66% and "
            "the 0.4459-conviction name 12.68% -- the ranking came out backwards"
        )

    def test_spread_is_transmitted_not_flattened(self):
        policy = PortfolioPolicy(n_candidates=7)
        weights = compose_portfolio(
            _signals(ARM_A_CONF, ARM_A_SECTORS),
            max_single_stock=policy.max_single_stock,
            max_sector=policy.max_sector(n_in_largest_sector=2),
            min_position=policy.min_position,
        )
        spread = max(weights.values()) / min(weights.values())
        assert spread > 1.3, (
            f"dollar spread {spread:.4f}x against a 1.6125x conviction spread; "
            "live on 2026-08-24 this was 1.0000x"
        )


class TestMinPositionCannotBreachTheCaps:
    """`_apply_min_position` ran last and redistributed straight through the
    limits it had just satisfied."""

    def test_dust_redistribution_respects_the_sector_cap(self):
        # 3 strong names + 2 dust names. Removing the dust pushed sector S2 to
        # 25.92% against a 25.00% cap in the returned book.
        conf = [0.90, 0.88, 0.86, 0.05, 0.05]
        sectors = [f"S{i % 3}" for i in range(5)]
        policy = PortfolioPolicy(n_candidates=5)
        max_sector = 0.25
        weights = compose_portfolio(
            _signals(conf, sectors),
            max_single_stock=policy.max_single_stock,
            max_sector=max_sector,
            min_position=policy.min_position,
        )
        by_ticker = {f"T{i}": s for i, s in enumerate(sectors)}
        worst = max(_sector_totals(weights, by_ticker).values())
        assert worst <= max_sector + 1e-9, (
            f"sector at {worst:.6f} against a {max_sector:.6f} cap; the "
            "one-pass composer returned 0.259192 here"
        )

    def test_dust_redistribution_respects_the_stock_cap(self):
        conf = [0.95, 0.05, 0.05, 0.05]
        sectors = ["A", "B", "C", "D"]
        policy = PortfolioPolicy(n_candidates=4)
        weights = compose_portfolio(
            _signals(conf, sectors),
            max_single_stock=policy.max_single_stock,
            max_sector=1.0,
            min_position=policy.min_position,
        )
        assert max(weights.values()) <= policy.max_single_stock + 1e-9


class TestTargetDeploymentIsHonoured:
    """DJ-131 made deployment an explicit decision; DJ-132 makes something
    actually read it."""

    def test_composer_reads_the_field(self):
        policy = PortfolioPolicy(n_candidates=7, target_deployment=0.60)
        weights = compose_portfolio(
            _signals(ARM_A_CONF, ARM_A_SECTORS),
            max_single_stock=policy.max_single_stock,
            max_sector=policy.max_sector(n_in_largest_sector=2),
            min_position=policy.min_position,
            target_deployment=policy.target_deployment,
        )
        assert sum(weights.values()) == pytest.approx(0.60, abs=1e-6)

    def test_default_is_full_deployment(self):
        weights = compose_portfolio(
            _signals(ARM_A_CONF, ARM_A_SECTORS),
            max_single_stock=0.30, max_sector=0.50, min_position=0.01,
        )
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_shortfall_only_from_a_genuinely_binding_constraint(self):
        """Seven names in one sector against a 25% budget: 25% is the honest
        answer, and the difference from DJ-131 is that it is now the sector
        cap saying so rather than cap-ordering arithmetic."""
        weights = compose_portfolio(
            _signals(ARM_A_CONF, ["Information Technology"] * 7),
            max_single_stock=0.2857, max_sector=0.25, min_position=0.001,
        )
        assert sum(weights.values()) == pytest.approx(0.25, abs=1e-6)


class TestFailsClosed:
    """A book that breaches a risk limit must not reach the broker."""

    def test_unsatisfiable_constraints_return_an_error_not_a_bad_book(self):
        # min_position above max_single_stock: no book can satisfy both.
        result = compose_portfolio(
            _signals(ARM_A_CONF, ARM_A_SECTORS),
            max_single_stock=0.05, max_sector=1.0, min_position=0.40,
        )
        assert "error" in result, (
            "an infeasible constraint set must surface, not silently produce "
            "a portfolio that violates one of the limits"
        )

    def test_solver_raises_rather_than_returning_a_violating_book(self):
        with pytest.raises(PortfolioConstraintError):
            _solve_to_fixed_point(
                confidence={f"T{i}": c for i, c in enumerate(ARM_A_CONF)},
                sectors={f"T{i}": "IT" for i in range(7)},
                max_weight=0.05,
                budgets={"IT": 1.0},
                min_position=0.40,
                target=1.0,
            )


class TestPostConditionsHoldOverRandomBaskets:
    """The check whose absence was DJ-132. Any book the composer returns must
    satisfy every constraint at once, for every input, not for the cases
    someone happened to write a test for."""

    @pytest.mark.parametrize("seed", range(200))
    def test_fuzz(self, seed: int) -> None:
        rng = random.Random(seed)
        n = rng.randint(1, 40)
        sector_pool = [f"S{i}" for i in range(rng.randint(1, 6))]
        conf = [rng.uniform(0.05, 1.0) for _ in range(n)]
        sectors = [rng.choice(sector_pool) for _ in range(n)]

        policy = PortfolioPolicy(n_candidates=n)
        counts: dict[str, int] = {}
        for s in sectors:
            counts[s] = counts.get(s, 0) + 1
        max_sector = policy.max_sector(n_in_largest_sector=max(counts.values()))
        target = rng.choice([1.0, 0.75, 0.5])

        weights = compose_portfolio(
            _signals(conf, sectors),
            max_single_stock=policy.max_single_stock,
            max_sector=max_sector,
            min_position=policy.min_position,
            target_deployment=target,
        )
        if "error" in weights:
            pytest.fail(f"seed={seed} n={n}: solver failed on a feasible book")
        if not weights:
            return

        by_ticker = {f"T{i}": s for i, s in enumerate(sectors)}
        ctx = f"seed={seed} n={n} sectors={len(sector_pool)} target={target}"

        assert max(weights.values()) <= policy.max_single_stock + 1e-9, (
            f"{ctx}: stock cap breached")
        assert min(weights.values()) >= policy.min_position - 1e-9, (
            f"{ctx}: dust below min_position retained")
        worst = max(_sector_totals(weights, by_ticker).values())
        assert worst <= max_sector + 1e-9, f"{ctx}: sector cap breached at {worst:.6f}"
        assert sum(weights.values()) <= target + 1e-9, (
            f"{ctx}: deployed more than the target")
