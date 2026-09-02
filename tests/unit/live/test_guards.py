"""The guards, at the branches that decide whether a night happens.

62% covered before this file, and the uncovered part was not decoration: the
delisting adjustment (DJ-123), the tradability probe, the arm-invariance probe
(DJ-119) and the thread watchdog (DJ-112) were all unexercised. Every one of
them exists because it was written after an incident.

The distinction these tests pin is *which guards block*. It is a scientific
judgement, not a safety preference:

  blocks   check_data_coverage — a starved run produces decision records
           indistinguishable from opinions and they enter the permanent record
  halts    the circuit breakers — one arm stops trading, observation continues
  reports  check_tradability, log_arm_invariance — a delisting and a
           concentrated book are facts about the world, not faults
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from hifi.execution.broker import Position
from hifi.live import accounts, guards, paths


@pytest.fixture
def live_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(paths, "_OUTPUT_DIR", str(tmp_path / "live"))
    return tmp_path


def _pos(ticker, qty=10.0, entry=100.0, value=1000.0, pnl=0.0):
    return Position(ticker=ticker, qty=qty, market_value=value,
                    avg_entry_price=entry, unrealized_pnl=pnl, side="long")


def _executor(equity=100_000.0, last_equity=100_000.0, positions=None,
              assets_exist=True):
    ex = MagicMock()
    acct = MagicMock()
    acct.equity = str(equity)
    acct.last_equity = str(last_equity)
    ex.client.get_account.return_value = acct
    ex.get_positions.return_value = positions or {}
    ex.get_portfolio_value.return_value = equity
    if assets_exist:
        ex.client.get_asset.return_value = MagicMock(tradable=True)
    else:
        ex.client.get_asset.side_effect = RuntimeError("asset not found")
    return ex


def _breakers(root, arm="A"):
    p = root / "live" / arm / "circuit_breakers.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


class TestDailyLossBreaker:
    def test_a_loss_beyond_the_limit_halts(self, live_root):
        ex = _executor(equity=97_000.0, last_equity=100_000.0)  # -3%
        assert guards.check_circuit_breakers("A", ex) is True
        rows = _breakers(live_root)
        assert rows[0]["trigger"] == "daily_loss" and rows[0]["action"] == "halt"

    def test_a_loss_inside_the_limit_does_not(self, live_root):
        ex = _executor(equity=98_500.0, last_equity=100_000.0)  # -1.5%
        assert guards.check_circuit_breakers("A", ex) is False
        assert _breakers(live_root) == []

    def test_a_gain_never_halts(self, live_root):
        assert guards.check_circuit_breakers(
            "A", _executor(equity=110_000.0, last_equity=100_000.0)) is False

    def test_zero_prior_equity_does_not_divide_by_zero(self, live_root):
        assert guards.check_circuit_breakers(
            "A", _executor(equity=100.0, last_equity=0.0)) is False

    def test_a_broker_error_does_not_halt_the_arm(self, live_root, caplog):
        """Fail-open, deliberately. An unreadable account is an infrastructure
        problem; halting on it would let a flaky API silently cost trading days
        and confound the ablation."""
        ex = MagicMock()
        ex.client.get_account.side_effect = RuntimeError("503")
        assert guards.check_circuit_breakers("A", ex) is False
        assert "Circuit breaker check failed" in caplog.text


class TestDelistingIsNotALoss:
    """DJ-123: value that left the book without becoming cash."""

    def _seed_equity(self, root, arm, positions):
        p = root / "live" / arm / "equity.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "decision_date": "2026-08-17",
            "positions": [{"ticker": t, "market_value": v} for t, v in positions],
        }) + "\n")

    def test_a_vanished_position_is_excluded_from_the_daily_change(self, live_root):
        """The literal 2026-08-17 arm-D failure.

        EQR was deleted from Alpaca's asset universe; the holding vanished from
        equity without being credited to cash, and the resulting -3.72% "daily
        loss" halted a healthy arm for a day. The breaker must measure P&L, not
        bookkeeping.
        """
        self._seed_equity(live_root, "D", [("EQR", 3_720.0)])
        ex = _executor(equity=96_280.0, last_equity=100_000.0, assets_exist=False)
        assert guards.check_circuit_breakers("D", ex) is False, (
            "a delisting halted a healthy arm"
        )
        triggers = {r["trigger"] for r in _breakers(live_root, "D")}
        assert "position_removed" in triggers, "the event was not recorded"

    def test_a_sold_position_is_not_mistaken_for_a_vanished_one(self, live_root):
        """A legitimate sale reappears as cash; treating it as vanished would
        turn a real loss into an apparent gain and blind the breaker the other
        way."""
        self._seed_equity(live_root, "D", [("AAPL", 3_720.0)])
        ex = _executor(equity=96_280.0, last_equity=100_000.0, assets_exist=True)
        assert guards.check_circuit_breakers("D", ex) is True

    def test_no_snapshot_history_means_nothing_vanished(self, live_root):
        assert guards._vanished_position_value("A", _executor()) == 0.0

    def test_an_unreadable_position_list_returns_zero(self, live_root, caplog):
        self._seed_equity(live_root, "A", [("EQR", 100.0)])
        ex = MagicMock()
        ex.get_positions.side_effect = RuntimeError("timeout")
        assert guards._vanished_position_value("A", ex) == 0.0
        assert "vanish check" in caplog.text

    def test_a_still_held_symbol_is_not_vanished(self, live_root):
        self._seed_equity(live_root, "A", [("AAPL", 1000.0)])
        ex = _executor(positions={"AAPL": _pos("AAPL")})
        assert guards._vanished_position_value("A", ex) == 0.0


class TestPositionBreakerScalesWithBookWidth:
    """DJ-119: the halt criterion must not be monotone in the treatment."""

    def test_a_big_loss_on_a_tiny_position_only_flags(self, live_root):
        # -20% on 1% of the book costs 0.2% of equity: recorded, not fatal.
        ex = _executor(positions={"X": _pos("X", qty=10, entry=100.0,
                                            value=1_000.0, pnl=-200.0)})
        assert guards.check_circuit_breakers("A", ex) is False
        rows = _breakers(live_root)
        assert rows[0]["action"] == "flag"
        assert rows[0]["trigger"] == "position_loss"

    def test_a_big_loss_on_a_large_position_halts(self, live_root):
        # -20% on 50% of the book costs 10% of equity.
        ex = _executor(positions={"X": _pos("X", qty=500, entry=100.0,
                                            value=50_000.0, pnl=-10_000.0)})
        assert guards.check_circuit_breakers("A", ex) is True
        assert _breakers(live_root)[0]["action"] == "halt"

    def test_a_shallow_loss_is_ignored_entirely(self, live_root):
        ex = _executor(positions={"X": _pos("X", qty=500, entry=100.0,
                                            value=47_500.0, pnl=-2_500.0)})  # -5%
        assert guards.check_circuit_breakers("A", ex) is False
        assert _breakers(live_root) == []

    @pytest.mark.parametrize("qty,entry", [(0, 100.0), (10, 0.0), (-5, 100.0)])
    def test_degenerate_positions_are_skipped_not_divided_by(self, live_root,
                                                             qty, entry):
        ex = _executor(positions={"X": _pos("X", qty=qty, entry=entry, pnl=-100.0)})
        assert guards.check_circuit_breakers("A", ex) is False

    def test_the_flag_row_carries_weight_and_impact(self, live_root):
        # A bare pnl% is not auditable; the science record needs the arithmetic.
        ex = _executor(positions={"X": _pos("X", qty=10, entry=100.0,
                                            value=1_000.0, pnl=-200.0)})
        guards.check_circuit_breakers("A", ex)
        row = _breakers(live_root)[0]
        assert "weight" in row and "impact" in row


class TestDataCoverageBlocks:
    """DJ-120: the one guard that refuses to let the night proceed."""

    def _coverage(self, found, total, layout="nested", last="2026-08-31"):
        rep = {f"T{i}": {"found": True, "layout": layout, "last_date": last}
               for i in range(found)}
        rep.update({f"M{i}": {"found": False, "layout": None, "last_date": None}
                    for i in range(total - found)})
        return rep

    def test_full_coverage_passes(self, live_root):
        with patch("hifi.data.market_store.coverage_report",
                   return_value=self._coverage(97, 97)):
            assert guards.check_data_coverage([f"T{i}" for i in range(97)]) is True

    def test_a_starved_universe_blocks(self, live_root, caplog):
        """83 of 98 resolved to nothing for a month, and the only symptom was an
        unusually bearish arm. Agents render absence as conviction."""
        with patch("hifi.data.market_store.coverage_report",
                   return_value=self._coverage(15, 98)):
            assert guards.check_data_coverage([f"T{i}" for i in range(98)]) is False
        assert "ABORT" in caplog.text
        assert "bearish conviction" in caplog.text

    def test_the_threshold_is_strict(self, live_root):
        # 96/97 is 98.9%, below the 99% floor: one blind ticker is one too many.
        with patch("hifi.data.market_store.coverage_report",
                   return_value=self._coverage(96, 97)):
            assert guards.check_data_coverage([f"T{i}" for i in range(97)]) is False

    def test_stale_legacy_layout_warns_without_blocking(self, live_root, caplog):
        with patch("hifi.data.market_store.coverage_report",
                   return_value=self._coverage(97, 97, layout="flat-legacy")):
            assert guards.check_data_coverage([f"T{i}" for i in range(97)]) is True
        assert "STALE legacy flat layout" in caplog.text

    def test_an_empty_universe_does_not_divide_by_zero(self, live_root):
        with patch("hifi.data.market_store.coverage_report", return_value={}):
            assert guards.check_data_coverage([]) is False


class TestTradabilityReportsButNeverBlocks:
    """DJ-123: the store and the broker can disagree, and on 2026-08-17 they did."""

    def test_untradable_symbols_are_named(self, live_root, caplog, monkeypatch):
        ex = _executor()
        ex.client.get_asset.side_effect = lambda t: (
            MagicMock(tradable=(t != "EQR")))
        monkeypatch.setattr(accounts, "get_executor", lambda a: ex)
        assert guards.check_tradability(["AAPL", "EQR"], "A") == ["EQR"]
        assert "EQR" in caplog.text
        ex.disconnect.assert_called_once()

    def test_a_missing_asset_counts_as_untradable(self, live_root, monkeypatch):
        monkeypatch.setattr(accounts, "get_executor",
                            lambda a: _executor(assets_exist=False))
        assert guards.check_tradability(["GONE"], "A") == ["GONE"]

    def test_an_all_tradable_universe_logs_the_clean_result(self, live_root,
                                                            caplog, monkeypatch):
        monkeypatch.setattr(accounts, "get_executor", lambda a: _executor())
        with caplog.at_level(logging.INFO, logger="hifi.live.guards"):
            assert guards.check_tradability(["AAPL", "MSFT"], "A") == []
        assert "2/2 tickers tradable" in caplog.text

    def test_no_credentials_skips_rather_than_raising(self, live_root, caplog,
                                                      monkeypatch):
        monkeypatch.setattr(accounts, "get_executor", lambda a: None)
        assert guards.check_tradability(["AAPL"], "A") == []
        assert "skipped" in caplog.text


class TestArmInvarianceProbe:
    """DJ-119: every rule that is NOT the treatment must apply equally."""

    def _arm(self, n_positions, equity=100_000.0):
        pos = {f"T{i}": _pos(f"T{i}", value=equity / max(n_positions, 1) * 0.9)
               for i in range(n_positions)}
        return _executor(equity=equity, positions=pos)

    def test_it_reports_width_exposure_and_threshold_per_arm(self, live_root,
                                                             caplog, monkeypatch):
        arms = {"A": self._arm(30), "B": self._arm(3), "C": self._arm(97),
                "D": self._arm(10)}
        monkeypatch.setattr(accounts, "get_executor", lambda a: arms[a])
        with caplog.at_level(logging.INFO, logger="hifi.live.guards"):
            guards.log_arm_invariance(["A", "B", "C", "D"])
        assert "Arm invariance probe" in caplog.text
        for arm in "ABCD":
            assert f"[{arm}]" in caplog.text

    def test_a_wide_deployment_spread_is_called_out(self, live_root, caplog,
                                                    monkeypatch):
        """Raw return is not comparable across arms at this spread — it measures
        exposure, not signal."""
        wide = _executor(equity=100_000.0,
                         positions={"X": _pos("X", value=95_000.0)})
        narrow = _executor(equity=100_000.0,
                           positions={"Y": _pos("Y", value=1_000.0)})
        monkeypatch.setattr(accounts, "get_executor",
                            lambda a: wide if a == "A" else narrow)
        guards.log_arm_invariance(["A", "B"])
        assert "NOT comparable across arms" in caplog.text

    def test_a_failing_arm_does_not_stop_the_probe(self, live_root, monkeypatch):
        broken = MagicMock()
        broken.get_portfolio_value.side_effect = RuntimeError("timeout")
        monkeypatch.setattr(accounts, "get_executor",
                            lambda a: broken if a == "A" else self._arm(5))
        guards.log_arm_invariance(["A", "B"])  # must not raise
        broken.disconnect.assert_called_once()

    def test_no_provisioned_arms_returns_quietly(self, live_root, monkeypatch):
        monkeypatch.setattr(accounts, "get_executor", lambda a: None)
        guards.log_arm_invariance(["A", "B"])  # must not raise

    def test_it_never_blocks(self, live_root, monkeypatch):
        # A genuinely concentrated arm is a legitimate outcome; what must not
        # happen silently is the apparatus CAUSING the concentration.
        monkeypatch.setattr(accounts, "get_executor", lambda a: self._arm(1))
        assert guards.log_arm_invariance(["A"]) is None


class TestHaltBeforeSubmit:
    """DJ-129c: hours of inference sit between the first check and the first order."""

    def test_a_dry_run_never_halts(self, live_root):
        assert guards._halt_before_submit("A", _executor(), is_dry=True,
                                          date="2026-08-31") is False

    def test_a_clean_book_proceeds(self, live_root):
        with patch.object(guards, "check_circuit_breakers", return_value=False):
            assert guards._halt_before_submit("A", _executor(), False, "2026-08-31") \
                is False

    def test_a_trip_records_equity_and_disconnects(self, live_root):
        """A halted arm must still land in the equity curve (DJ-119)."""
        ex = _executor()
        with patch.object(guards, "check_circuit_breakers", return_value=True), \
             patch("hifi.execution.portfolio_recorder.record_account") as rec:
            assert guards._halt_before_submit("A", ex, False, "2026-08-31") is True
        rec.assert_called_once()
        ex.disconnect.assert_called_once()


class TestThreadWatchdog:
    """DJ-112: macOS panics the kernel near ~2048 pthreads; three crashes cost
    three full nights."""

    def test_the_limit_leaves_room_above_a_healthy_run(self):
        assert 100 < guards._THREAD_ABORT < 2048, (
            "a healthy live run sits under ~30 threads; the limit must be well "
            "above that and well below the kernel's ceiling"
        )

    def test_it_starts_a_daemon_thread_that_does_not_block_exit(self):
        import threading
        before = threading.active_count()
        guards._start_thread_watchdog(interval_s=3600)
        after = [t for t in threading.enumerate() if t.name == "thread-watchdog"]
        assert threading.active_count() == before + 1
        assert after[0].daemon, "a non-daemon watchdog would hang every exit"
