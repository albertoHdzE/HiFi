"""Phase 20 (DJ-130): personality postures over recorded votes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hifi.collective.personality import (  # noqa: E402
    AGGRESSIVE,
    BASELINE,
    CAREFUL,
    CONSERVATIVE,
    posture_vote,
)


def _scores(votes):
    return posture_vote(votes, BASELINE)["scores"]


class TestBaseline:
    def test_plain_confidence_plurality(self):
        votes = [("Buy", .8), ("Buy", .8), ("Buy", .8), ("Hold", .9), ("Hold", .9)]
        r = posture_vote(votes, BASELINE)
        assert r["decision"] == "Buy"
        assert r["confidence"] == pytest.approx(2.4 / 4.2, abs=1e-3)

    def test_empty_votes_hold_zero_confidence(self):
        r = posture_vote([], BASELINE)
        assert r["decision"] == "Hold"
        assert r["confidence"] == 0.0

    def test_two_way_tie_resolves_hold(self):
        r = posture_vote([("Buy", .5), ("Sell", .5)], BASELINE)
        assert r["decision"] == "Hold"


class TestAggressive:
    def test_lean_boosts_entry_on_marginal_buy(self):
        # sB=2.0, sH=1.7: baseline buys narrowly; aggressive adds half the
        # Hold mass to entry, raising conviction but never flipping a
        # baseline Buy away.
        votes = [("Buy", 1.0), ("Buy", 1.0), ("Hold", .85), ("Hold", .85)]
        base = posture_vote(votes, BASELINE)
        agg = posture_vote(votes, AGGRESSIVE)
        assert base["decision"] == "Buy"
        assert agg["decision"] == "Buy"
        assert agg["scores"]["Buy"] > base["scores"]["Buy"]

    def test_never_manufactures_conviction_from_pure_hold(self):
        """Unanimous Hold must stay Hold under every posture: posture leans
        existing conviction; it cannot invent one."""
        votes = [("Hold", .9)] * 4
        for profile in (BASELINE, AGGRESSIVE, CONSERVATIVE, CAREFUL):
            assert posture_vote(votes, profile)["decision"] == "Hold"


class TestConservative:
    def test_margin_keeps_clear_buy(self):
        # sB=2.4 vs sH=1.8: 1.25 * 1.8 = 2.25 <= 2.4 — Buy survives.
        votes = [("Buy", .8), ("Buy", .8), ("Buy", .8), ("Hold", .9), ("Hold", .9)]
        assert posture_vote(votes, CONSERVATIVE)["decision"] == "Buy"

    def test_marginal_buy_degrades_to_hold(self):
        # sB=2.0 vs sH=1.7: 1.25 * 1.7 = 2.125 > 2.0 — insufficient margin.
        votes = [("Buy", 1.0), ("Buy", 1.0), ("Hold", .85), ("Hold", .85)]
        assert posture_vote(votes, BASELINE)["decision"] == "Buy"
        assert posture_vote(votes, CONSERVATIVE)["decision"] == "Hold"


class TestCareful:
    def test_sell_lean_amplifies_exit_mass(self):
        votes = [("Buy", .6), ("Hold", 1.0), ("Hold", 1.0), ("Sell", .9)]
        cons = posture_vote(votes, CONSERVATIVE)
        care = posture_vote(votes, CAREFUL)
        # Same Hold decision (entry margin fails for both)…
        assert cons["decision"] == care["decision"] == "Hold"
        # …but careful shifts mass toward Sell.
        assert care["scores"]["Sell"] > cons["scores"]["Sell"]
        assert care["scores"]["Hold"] < cons["scores"]["Hold"]

    def test_careful_blocks_unbacked_entry(self):
        votes = [("Buy", .6), ("Hold", 1.0), ("Hold", 1.0), ("Sell", .9)]
        assert posture_vote(votes, CAREFUL)["decision"] == "Hold"


class TestDeterminism:
    def test_identical_inputs_identical_outputs(self):
        votes = [("Buy", .7), ("Hold", .6), ("Sell", .4), ("Buy", .3)]
        for profile in (BASELINE, AGGRESSIVE, CONSERVATIVE, CAREFUL):
            assert posture_vote(votes, profile) == posture_vote(votes, profile)

    def test_unknown_decisions_ignored(self):
        r = posture_vote([("Yolo", .99)], BASELINE)
        assert r["decision"] == "Hold"
        assert r["confidence"] == 0.0
