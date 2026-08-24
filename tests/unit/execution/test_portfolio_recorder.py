"""Tests for automatic financial-performance capture (DJ-114)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from hifi.execution import portfolio_recorder as rec
from hifi.execution.broker import Position


def _executor(equity, cash, positions, history=None):
    ex = MagicMock()
    ex.get_account_snapshot.return_value = {
        "equity": equity, "last_equity": equity, "cash": cash,
        "buying_power": cash, "long_market_value": equity - cash,
    }
    ex.get_positions.return_value = positions
    ex.get_portfolio_history.return_value = history or {
        "timestamp": [1784160000, 1784246400],
        "equity": [100000.0, 100500.0],
        "profit_loss": [0.0, 500.0],
        "profit_loss_pct": [0.0, 0.005],
    }
    return ex


class TestSnapshot:
    def test_writes_equity_jsonl_with_positions(self, tmp_path):
        pos = {"AAPL": Position("AAPL", 10, 1700.0, 170.0, 50.0, "long")}
        ex = _executor(100000.0, 98300.0, pos)
        snap = rec.snapshot_account(ex, "A", str(tmp_path), decision_date="2026-07-19")

        line = json.loads((tmp_path / "live" / "A" / "equity.jsonl").read_text().strip())
        assert line["equity"] == 100000.0
        assert line["n_positions"] == 1
        assert line["positions"][0]["ticker"] == "AAPL"
        assert line["positions"][0]["unrealized_pnl"] == 50.0
        assert snap["decision_date"] == "2026-07-19"

    def test_appends_across_days(self, tmp_path):
        ex = _executor(100000.0, 100000.0, {})
        rec.snapshot_account(ex, "A", str(tmp_path), "2026-07-19")
        rec.snapshot_account(ex, "A", str(tmp_path), "2026-07-20")
        rows = (tmp_path / "live" / "A" / "equity.jsonl").read_text().strip().splitlines()
        assert len(rows) == 2

    def test_capture_failure_does_not_raise(self, tmp_path):
        ex = MagicMock()
        ex.get_account_snapshot.side_effect = RuntimeError("broker down")
        assert rec.snapshot_account(ex, "A", str(tmp_path)) == {}


class TestPortfolioHistory:
    def test_saves_authoritative_curve(self, tmp_path):
        ex = _executor(100000.0, 100000.0, {})
        n = rec.save_portfolio_history(ex, "B", str(tmp_path))
        assert n == 2
        payload = json.loads((tmp_path / "live" / "B" / "portfolio_history.json").read_text())
        assert payload["equity"] == [100000.0, 100500.0]


class TestAnalysisLoaders:
    def test_load_equity_curve_and_returns(self, tmp_path):
        ex = _executor(100000.0, 100000.0, {})
        rec.save_portfolio_history(ex, "C", str(tmp_path))
        curve = rec.load_equity_curve("C", str(tmp_path))
        assert list(curve.values) == [100000.0, 100500.0]
        rets = rec.load_returns("C", str(tmp_path))
        assert len(rets) == 1
        assert abs(rets.iloc[0] - 0.005) < 1e-9

    def test_load_missing_returns_empty(self, tmp_path):
        assert rec.load_equity_curve("ZZ", str(tmp_path)).empty
