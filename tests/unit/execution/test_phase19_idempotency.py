"""Phase 19 (DJ-129): idempotent order submission + live high-water mark.

Pins the three genesis-hardening properties:

DJ-129a  A crash at any point of the nightly submit loop, followed by a
         same-evening rerun, must never double-fill: orders carry a
         deterministic client_order_id and ids already at the broker are
         skipped. (The 2026-08-17 arm-A crash filled 37 orders and left no
         decision record — a rerun then resubmitted everything.)
DJ-129b  The -15% drawdown breaker receives a persisted, ratcheting
         high-water mark, not today's equity as its own baseline.
DJ-129c  Breakers re-evaluate immediately before submission.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hifi.live import accounts, cycle, guards, market, paths, strategies  # noqa: E402

# ---------------------------------------------------------------------------
# DJ-129a: deterministic client_order_id
# ---------------------------------------------------------------------------


class TestClientId:
    def test_deterministic_and_side_aware(self):
        a = accounts._client_order_id("A", "2026-08-21", "AAPL", "buy")
        assert a == accounts._client_order_id("A", "2026-08-21", "AAPL", "buy")
        assert a != accounts._client_order_id("A", "2026-08-21", "AAPL", "sell")
        assert a != accounts._client_order_id("B", "2026-08-21", "AAPL", "buy")
        assert a != accounts._client_order_id("A", "2026-08-22", "AAPL", "buy")

    def test_alpaca_charset_and_length(self):
        coid = accounts._client_order_id("D", "2026-08-21", "BRK-B", "sell")
        assert len(coid) <= 48
        assert all(c.isalnum() or c in "-" for c in coid)


class TestExecutorCarriesId:
    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_share_order_sets_client_order_id(self, mock_client_cls):
        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="k", secret_key="s", paper=True)
        ex.connect()
        ex.place_market_order("AAPL", 5, "buy",
                              client_order_id="hifiA-2026-08-21-buy-AAPL")

        req = mock_client_cls.return_value.submit_order.call_args[0][0]
        assert req.client_order_id == "hifiA-2026-08-21-buy-AAPL"

    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_notional_order_sets_client_order_id(self, mock_client_cls):
        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="k", secret_key="s", paper=True)
        ex.connect()
        ex.place_market_order("AAPL", 0, "buy", notional=990.0,
                              client_order_id="coid-x")

        req = mock_client_cls.return_value.submit_order.call_args[0][0]
        assert req.client_order_id == "coid-x"
        assert req.notional == 990.0

    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_absent_id_leaves_request_untouched(self, mock_client_cls):
        """Legacy callers get byte-identical behavior."""
        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="k", secret_key="s", paper=True)
        ex.connect()
        ex.place_market_order("AAPL", 5, "buy")

        req = mock_client_cls.return_value.submit_order.call_args[0][0]
        assert not getattr(req, "client_order_id", None)

    @patch("hifi.execution.alpaca_executor.TradingClient")
    def test_get_client_order_ids_merges_open_and_closed(self, mock_client_cls):
        from hifi.execution.alpaca_executor import AlpacaExecutor

        ex = AlpacaExecutor(api_key="k", secret_key="s", paper=True)
        ex.connect()
        client = mock_client_cls.return_value

        open_o, closed_o, blank = MagicMock(), MagicMock(), MagicMock()
        open_o.client_order_id = "open-1"
        closed_o.client_order_id = "closed-1"
        blank.client_order_id = None
        client.get_orders.side_effect = [[open_o], [closed_o, blank]]

        assert ex.get_client_order_ids() == {"open-1", "closed-1"}
        statuses = [c.args[0].status for c in client.get_orders.call_args_list]
        assert statuses == ["open", "closed"]


class TestExecuteOrdersIdempotency:
    """The crash-rerun replay: the second attempt must not resubmit."""

    @staticmethod
    def _snapshot(tickers):
        class S:
            orders = [{"ticker": t, "side": "BUY", "quantity": 1.0,
                       "estimated_value": 100.0} for t in tickers]
        return S()

    def test_existing_id_at_broker_is_skipped_not_resubmitted(self):
        ex = MagicMock()
        result = MagicMock(status="filled", order_id="o1", filled_avg_price=None)
        ex.place_market_order.return_value = result
        # AAPL was already submitted by the crashed attempt.
        ex.get_client_order_ids.return_value = {
            accounts._client_order_id("A", "2026-08-21", "AAPL", "buy")}

        out = cycle.execute_orders(
            self._snapshot(["AAPL", "MSFT"]), ex,
            dry_run=False, account="A", date="2026-08-21")

        by_ticker = {o["ticker"]: o for o in out}
        assert by_ticker["AAPL"]["status"] == "skipped_duplicate"
        assert by_ticker["MSFT"]["status"] == "filled"
        submitted = ex.place_market_order.call_args_list
        assert len(submitted) == 1
        call = submitted[0]
        assert call.args[:3] == ("MSFT", 1.0, "buy")
        assert call.kwargs["client_order_id"] == \
            accounts._client_order_id("A", "2026-08-21", "MSFT", "buy")

    def test_prefetch_failure_fails_closed_before_any_submit(self):
        ex = MagicMock()
        ex.get_client_order_ids.side_effect = RuntimeError("broker unreachable")

        with pytest.raises(RuntimeError):
            cycle.execute_orders(
                self._snapshot(["AAPL"]), ex,
                dry_run=False, account="A", date="2026-08-21")

        ex.place_market_order.assert_not_called()

    def test_no_date_keeps_legacy_behavior(self):
        """Dry-run and legacy/test callers see exactly today's signatures."""
        ex = MagicMock()
        result = MagicMock(status="accepted", order_id="o", filled_avg_price=None)
        ex.place_market_order.return_value = result

        out = cycle.execute_orders(self._snapshot(["JPM"]), ex, dry_run=False)

        ex.get_client_order_ids.assert_not_called()
        call = ex.place_market_order.call_args
        assert call.args[:3] == ("JPM", 1.0, "buy")
        assert "client_order_id" not in call.kwargs or \
            call.kwargs["client_order_id"] is None
        assert out[0]["status"] == "accepted"


class TestControlStrategyIdempotency:
    @patch.object(market, "_latest_prices")
    def test_duplicate_skipped_without_double_spend(self, mock_prices):
        mock_prices.return_value = {"AAPL": 100.0, "MSFT": 200.0}
        ex = MagicMock()
        ex.get_portfolio_value.return_value = 10_000.0
        ex.get_account_cash.return_value = 10_000.0
        ex.get_positions.return_value = {}
        ex.is_fractionable.return_value = True
        result = MagicMock(status="accepted", order_id="o", qty=None)
        result.filled_avg_price = None
        ex.place_market_order.return_value = result
        ex.get_client_order_ids.return_value = {
            accounts._client_order_id("C", "2026-08-21", "AAPL", "buy")}

        out = strategies.run_control_strategy(["AAPL", "MSFT"], ex, dry_run=False,
                                        account="C", date="2026-08-21")

        by_ticker = {o["ticker"]: o for o in out}
        assert by_ticker["AAPL"]["status"] == "skipped_duplicate"
        assert by_ticker["MSFT"]["status"] == "accepted"
        # Only MSFT was actually submitted.
        assert ex.place_market_order.call_count == 1

    @patch.object(market, "_latest_prices")
    def test_dry_run_never_touches_id_lookup(self, mock_prices):
        mock_prices.return_value = {"AAPL": 100.0}
        ex = MagicMock()
        ex.get_portfolio_value.return_value = 10_000.0
        ex.get_account_cash.return_value = 10_000.0
        ex.get_positions.return_value = {}

        strategies.run_control_strategy(["AAPL"], ex, dry_run=True)

        ex.get_client_order_ids.assert_not_called()


# ---------------------------------------------------------------------------
# DJ-129b: persisted high-water mark
# ---------------------------------------------------------------------------


class TestHighWaterMark:
    def _equity_history(self, tmp_path, account, equities):
        d = tmp_path / "live" / account
        d.mkdir(parents=True, exist_ok=True)
        rows = "".join(json.dumps({"equity": e}) + "\n" for e in equities)
        (d / "equity.jsonl").write_text(rows)

    def test_ratchets_up_and_persists_across_restart(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "_account_dir", lambda a: tmp_path / a)
        monkeypatch.setattr(paths, "_DATA_DIR", str(tmp_path))
        self._equity_history(tmp_path, "A", [100_000.0])

        h1 = accounts.update_hwm("A", current_equity=105_000.0)
        assert h1 == pytest.approx(105_000.0)
        # A "restart" is a fresh call reading only what was persisted.
        h2 = accounts.update_hwm("A", current_equity=95_000.0)
        assert h2 == pytest.approx(105_000.0), "a falling market must not lower HWM"

        stored = json.loads((tmp_path / "A" / "hwm.json").read_text())
        assert stored["hwm"] == pytest.approx(105_000.0)

    def test_seeds_from_historical_max_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "_account_dir", lambda a: tmp_path / a)
        monkeypatch.setattr(paths, "_DATA_DIR", str(tmp_path))
        self._equity_history(tmp_path, "A", [90_000.0, 110_000.0, 98_000.0])

        # First run after the fix, at today's depressed equity: the breaker
        # must still know about the old $110k peak, not reset to today.
        hwm = accounts.update_hwm("A", current_equity=98_000.0)
        assert hwm == pytest.approx(110_000.0)

    def test_drawdown_breacher_now_reachable_end_to_end(self, tmp_path, monkeypatch):
        """The wire this phase exists to repair: real HWM vs today's equity."""
        monkeypatch.setattr(paths, "_account_dir", lambda a: tmp_path / a)
        monkeypatch.setattr(paths, "_DATA_DIR", str(tmp_path))
        self._equity_history(tmp_path, "A", [100_000.0])
        accounts.update_hwm("A", current_equity=100_000.0)

        from hifi.mcp.risk_manager import max_drawdown_breached
        # -16% from the peak must trip; the old caller passed hwm=pv so this
        # comparison was identically zero forever.
        assert max_drawdown_breached(84_000.0, 100_000.0) is True
        assert max_drawdown_breached(90_000.0, 100_000.0) is False


# ---------------------------------------------------------------------------
# DJ-129c: pre-submit breaker re-check
# ---------------------------------------------------------------------------


class TestPreSubmitHalt:
    def test_trips_before_submission_and_records_account(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "_account_dir", lambda a: tmp_path / a)
        ex = MagicMock()
        with patch.object(guards, "check_circuit_breakers", return_value=True), \
             patch("hifi.execution.portfolio_recorder.record_account") as rec:
            halted = guards._halt_before_submit("A", ex, is_dry=False,
                                              date="2026-08-21")
        assert halted is True
        assert rec.called
        ex.disconnect.assert_called_once()

    def test_dry_run_never_halts(self):
        with patch.object(guards, "check_circuit_breakers") as chk:
            assert guards._halt_before_submit("A", MagicMock(), True, "d") is False
        chk.assert_not_called()

    def test_healthy_book_proceeds(self):
        with patch.object(guards, "check_circuit_breakers", return_value=False):
            assert guards._halt_before_submit("A", MagicMock(), False, "d") is False
