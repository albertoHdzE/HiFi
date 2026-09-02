"""Arm identity, per-arm state, and the price inputs a cycle reads.

66% and 62% covered before this. The uncovered parts were the credential
resolution (which decides *which brokerage account* an arm trades), the
high-water mark's rebuild-from-history path (the drawdown breaker's baseline),
and the session-date resolution that decides what date a decision is stamped
with. All three are quiet when wrong.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hifi.execution.broker import Position
from hifi.live import accounts, market, paths


@pytest.fixture
def live_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(paths, "_OUTPUT_DIR", str(tmp_path / "live"))
    return tmp_path


@pytest.fixture(autouse=True)
def _no_ambient_alpaca_keys(monkeypatch):
    """The developer's .env is loaded by the CLI; do not let it leak in here."""
    for suffix in ("", "_A", "_B", "_C", "_D", "_FIRST", "_SECOND", "_THIRD",
                   "_FOURTH"):
        monkeypatch.delenv(f"ALPACA_API_KEY{suffix}", raising=False)
        monkeypatch.delenv(f"ALPACA_SECRET{suffix}", raising=False)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class TestCredentialResolution:
    """Which arm trades which brokerage account. A mistake here is invisible
    and corrupts both arms it touches."""

    @pytest.mark.parametrize("arm,suffix", [
        ("A", "_FIRST"), ("B", "_SECOND"), ("C", "_THIRD"), ("D", "_FOURTH"),
    ])
    def test_each_arm_finds_its_primary_suffix(self, arm, suffix, monkeypatch):
        monkeypatch.setenv(f"ALPACA_API_KEY{suffix}", f"k-{arm}")
        monkeypatch.setenv(f"ALPACA_SECRET{suffix}", f"s-{arm}")
        with patch("hifi.execution.alpaca_executor.AlpacaExecutor") as ex:
            accounts.get_executor(arm)
        assert ex.call_args.kwargs["api_key"] == f"k-{arm}"
        assert ex.call_args.kwargs["paper"] is True, "a live account was constructed"

    def test_suffixes_are_tried_in_order(self, monkeypatch):
        # Both present: the first in the list must win, or renaming an old key
        # silently re-points an arm.
        monkeypatch.setenv("ALPACA_API_KEY_FIRST", "primary")
        monkeypatch.setenv("ALPACA_SECRET_FIRST", "s1")
        monkeypatch.setenv("ALPACA_API_KEY_A", "secondary")
        monkeypatch.setenv("ALPACA_SECRET_A", "s2")
        with patch("hifi.execution.alpaca_executor.AlpacaExecutor") as ex:
            accounts.get_executor("A")
        assert ex.call_args.kwargs["api_key"] == "primary"

    def test_only_arm_a_falls_back_to_unsuffixed_keys(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "bare")
        monkeypatch.setenv("ALPACA_SECRET", "bare-s")
        with patch("hifi.execution.alpaca_executor.AlpacaExecutor") as ex:
            assert accounts.get_executor("A") is not None
        assert ex.call_args.kwargs["api_key"] == "bare"
        for arm in "BCD":
            assert accounts.get_executor(arm) is None, (
                f"arm {arm} fell back to the unsuffixed keys and would trade "
                "arm A's account"
            )

    def test_missing_credentials_return_none_and_warn(self, caplog):
        assert accounts.get_executor("B") is None
        assert "no credentials" in caplog.text.lower()

    def test_a_partial_credential_pair_is_not_used(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY_SECOND", "key-only")
        assert accounts.get_executor("B") is None

    def test_the_executor_is_connected_before_return(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY_THIRD", "k")
        monkeypatch.setenv("ALPACA_SECRET_THIRD", "s")
        with patch("hifi.execution.alpaca_executor.AlpacaExecutor") as cls:
            accounts.get_executor("C")
        cls.return_value.connect.assert_called_once()


# ---------------------------------------------------------------------------
# High-water mark
# ---------------------------------------------------------------------------


class TestHighWaterMark:
    """DJ-129b: the drawdown breaker's baseline must be a real peak."""

    def test_first_run_seeds_from_current_equity(self, live_root):
        assert accounts.update_hwm("A", 100_000.0) == 100_000.0

    def test_it_ratchets_up(self, live_root):
        accounts.update_hwm("A", 100_000.0)
        assert accounts.update_hwm("A", 110_000.0) == 110_000.0

    def test_it_never_ratchets_down(self, live_root):
        accounts.update_hwm("A", 110_000.0)
        assert accounts.update_hwm("A", 90_000.0) == 110_000.0, (
            "a falling market lowered the mark, which is how the -15% control "
            "became unfireable"
        )

    def test_it_rebuilds_from_equity_history_when_the_file_is_missing(self, live_root):
        """An account that once sat at $110k must carry that peak even on the
        first run after the fix, or activating the breaker silently resets it."""
        eq = live_root / "live" / "A" / "equity.jsonl"
        eq.parent.mkdir(parents=True)
        eq.write_text("\n".join(json.dumps({"decision_date": d, "equity": v})
                                for d, v in [("2026-08-24", 100_000.0),
                                             ("2026-08-25", 110_000.0),
                                             ("2026-08-26", 95_000.0)]))
        assert accounts.update_hwm("A", 90_000.0) == 110_000.0

    def test_a_corrupt_hwm_file_falls_back_to_history(self, live_root, caplog):
        paths._hwm_path("A").parent.mkdir(parents=True)
        paths._hwm_path("A").write_text("{not json")
        eq = live_root / "live" / "A" / "equity.jsonl"
        eq.write_text(json.dumps({"equity": 105_000.0}) + "\n")
        assert accounts.update_hwm("A", 90_000.0) == 105_000.0
        assert "Could not read HWM file" in caplog.text

    def test_an_unreadable_equity_history_degrades_to_current_equity(
            self, live_root, caplog):
        eq = live_root / "live" / "A" / "equity.jsonl"
        eq.parent.mkdir(parents=True)
        eq.write_text("not jsonl at all {{{")
        assert accounts.update_hwm("A", 90_000.0) == 90_000.0
        assert "Could not seed HWM" in caplog.text

    def test_history_without_an_equity_column_is_ignored(self, live_root):
        eq = live_root / "live" / "A" / "equity.jsonl"
        eq.parent.mkdir(parents=True)
        eq.write_text(json.dumps({"decision_date": "2026-08-24"}) + "\n")
        assert accounts.update_hwm("A", 90_000.0) == 90_000.0

    def test_the_write_is_atomic(self, live_root):
        # tmp-then-rename: a crash mid-write must not leave a truncated mark.
        accounts.update_hwm("A", 100_000.0)
        assert paths._hwm_path("A").exists()
        assert not paths._hwm_path("A").with_suffix(".json.tmp").exists()
        stored = json.loads(paths._hwm_path("A").read_text())
        assert stored["hwm"] == 100_000.0 and "updated" in stored


class TestAlreadyDecided:
    def test_false_when_no_log_exists(self, live_root):
        assert accounts.already_decided("A", "2026-08-31") is False

    def test_true_only_for_the_matching_date(self, live_root):
        log = paths._decisions_log("A")
        log.parent.mkdir(parents=True)
        log.write_text(json.dumps({"decision_date": "2026-08-27"}) + "\n")
        assert accounts.already_decided("A", "2026-08-27") is True
        assert accounts.already_decided("A", "2026-08-31") is False

    def test_a_corrupt_log_proceeds_rather_than_blocking(self, live_root, caplog):
        """A run that cannot read its log should trade, not silently skip: the
        idempotency key at the broker is the real duplicate defence (DJ-129a)."""
        log = paths._decisions_log("A")
        log.parent.mkdir(parents=True)
        log.write_text("{{{ not json\n")
        assert accounts.already_decided("A", "2026-08-31") is False
        assert "Could not read decision log" in caplog.text

    def test_blank_lines_are_tolerated(self, live_root):
        log = paths._decisions_log("A")
        log.parent.mkdir(parents=True)
        log.write_text("\n" + json.dumps({"decision_date": "2026-08-27"}) + "\n\n")
        assert accounts.already_decided("A", "2026-08-27") is True


class TestShowStatus:
    def test_it_prints_equity_positions_and_recent_decisions(self, live_root, capsys):
        ex = MagicMock()
        acct = MagicMock(equity="100000.0", cash="5000.0", last_equity="99000.0")
        ex.client.get_account.return_value = acct
        ex.get_positions.return_value = {
            "AAPL": Position(ticker="AAPL", qty=10, market_value=2000.0,
                             avg_entry_price=190.0, unrealized_pnl=100.0,
                             side="long")}
        log = paths._decisions_log("A")
        log.parent.mkdir(parents=True)
        log.write_text(json.dumps({"decision_date": "2026-08-27", "n_orders": 3,
                                   "portfolio_value": 100_000.0}) + "\n")

        accounts.show_status("A", ex)
        out = capsys.readouterr().out
        assert "Account A" in out and "parallel ensemble" in out
        assert "AAPL" in out
        assert "2026-08-27: 3 orders" in out
        assert "1,000.00" in out, "daily P&L was not shown"

    def test_an_empty_book_says_so(self, live_root, capsys):
        ex = MagicMock()
        ex.client.get_account.return_value = MagicMock(
            equity="100000.0", cash="100000.0", last_equity="100000.0")
        ex.get_positions.return_value = {}
        accounts.show_status("C", ex)
        assert "No open positions" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Market inputs
# ---------------------------------------------------------------------------


def _write_bars(root, ticker, dates, closes=None):
    p = root / "market" / ticker / "ohlcv.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    closes = closes or [100.0] * len(dates)
    pd.DataFrame({"close": closes}, index=pd.to_datetime(dates)).to_parquet(p)
    return p


class TestUpdateData:
    def test_spy_is_always_fetched_alongside_the_universe(self, live_root, caplog):
        """DJ-130 companion: SPY is not a universe member but is the regime
        classifier's benchmark. Without it the classifier runs on stale bars and
        pins itself at 'neutral' with no error anywhere."""
        with patch("hifi.execution.market_data.update_local_ohlcv",
                   return_value={"AAPL": 1, "SPY": 1}) as upd:
            market.update_data(["AAPL"])
        assert "SPY" in upd.call_args.args[0]

    def test_it_writes_into_the_configured_market_dir(self, live_root):
        with patch("hifi.execution.market_data.update_local_ohlcv",
                   return_value={}) as upd:
            market.update_data(["AAPL"])
        assert str(live_root) in upd.call_args.kwargs["market_dir"]


class TestLastCompletedSession:
    """DJ-121: the store is the authoritative trading calendar."""

    def test_it_reads_the_newest_bar(self, live_root):
        _write_bars(live_root, "AAPL", ["2026-08-27", "2026-08-28"])
        assert market._last_completed_session(["AAPL"]) == "2026-08-28"

    def test_a_weekend_run_resolves_back_to_friday(self, live_root):
        """A Sunday run is Friday's cycle executed late, not a new observation.
        Dating it Sunday would invent a decision on a day with no price and let
        two runs record against the same information."""
        _write_bars(live_root, "AAPL", ["2026-08-26", "2026-08-28"])  # Fri 28th
        assert market._last_completed_session(["AAPL"]) == "2026-08-28"

    def test_one_halted_ticker_cannot_drag_the_date_back(self, live_root):
        _write_bars(live_root, "HALT", ["2026-06-01"])
        _write_bars(live_root, "AAPL", ["2026-08-28"])
        assert market._last_completed_session(["HALT", "AAPL"]) == "2026-08-28"

    def test_no_store_returns_none(self, live_root):
        assert market._last_completed_session(["NOPE"]) is None

    def test_an_empty_parquet_is_skipped(self, live_root):
        p = live_root / "market" / "EMPTY" / "ohlcv.parquet"
        p.parent.mkdir(parents=True)
        pd.DataFrame({"close": []}, index=pd.to_datetime([])).to_parquet(p)
        assert market._last_completed_session(["EMPTY"]) is None

    def test_an_unreadable_parquet_warns_and_continues(self, live_root, caplog):
        bad = live_root / "market" / "BAD" / "ohlcv.parquet"
        bad.parent.mkdir(parents=True)
        bad.write_text("corrupt")
        _write_bars(live_root, "AAPL", ["2026-08-28"])
        with caplog.at_level(logging.WARNING, logger="hifi.live.market"):
            assert market._last_completed_session(["BAD", "AAPL"]) == "2026-08-28"
        assert "Could not read BAD" in caplog.text

    def test_it_samples_only_the_first_few_tickers(self, live_root):
        # 97 parquet reads per cycle for a date is waste; the sample is the point.
        for i in range(10):
            _write_bars(live_root, f"T{i}", ["2026-08-01"])
        _write_bars(live_root, "T9", ["2026-08-28"])
        got = market._last_completed_session([f"T{i}" for i in range(10)], sample=3)
        assert got == "2026-08-01", "the sample bound is not being applied"


class TestLatestPrices:
    def test_it_reads_the_last_close_per_ticker(self, live_root):
        _write_bars(live_root, "AAPL", ["2026-08-27", "2026-08-28"], [199.0, 201.5])
        assert market._latest_prices(["AAPL"]) == {"AAPL": 201.5}

    def test_column_case_is_normalised(self, live_root):
        p = live_root / "market" / "X" / "ohlcv.parquet"
        p.parent.mkdir(parents=True)
        pd.DataFrame({"Close": [50.0]}, index=pd.to_datetime(["2026-08-28"])
                     ).to_parquet(p)
        assert market._latest_prices(["X"]) == {"X": 50.0}

    def test_a_missing_ticker_is_omitted_not_zero(self, live_root):
        # A zero price would size an infinite position.
        _write_bars(live_root, "AAPL", ["2026-08-28"], [200.0])
        prices = market._latest_prices(["AAPL", "GONE"])
        assert "GONE" not in prices
