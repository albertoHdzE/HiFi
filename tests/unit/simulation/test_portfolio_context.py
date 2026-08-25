"""Phase 20 (DJ-130): portfolio context for situated agents."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hifi.agents.context import (  # noqa: E402
    CONTEXT_ELIGIBLE_AGENTS,
    build_portfolio_context,
    genesis_date,
    load_book_state,
    write_book_state,
)
from hifi.execution.broker import Position  # noqa: E402


def _executor(equity=100_000.0, cash=95_000.0, positions=None):
    ex = MagicMock()
    ex.get_portfolio_value.return_value = equity
    ex.get_account_cash.return_value = cash
    ex.get_positions.return_value = positions or {}
    return ex


def _init_live(tmp_path, genesis="2026-08-24"):
    (tmp_path / "live").mkdir(parents=True, exist_ok=True)
    (tmp_path / "live" / "genesis_date.txt").write_text(genesis + "\n")


class TestEligibility:
    def test_technical_excluded_from_context(self):
        """Its schema promises price-derived information only (audit finding:
        context contamination must not deepen)."""
        assert "technical" not in CONTEXT_ELIGIBLE_AGENTS
        assert {"fundamental", "risk", "macro", "sentiment",
                "contrarian"} <= CONTEXT_ELIGIBLE_AGENTS


class TestBookState:
    def test_roundtrip_and_math(self, tmp_path):
        _init_live(tmp_path)
        pos = {
            "AAPL": Position("AAPL", 10, 3_000.0, 280.0, 200.0, "long"),
            "MSFT": Position("MSFT", 5, 2_000.0, 410.0, -50.0, "long"),
        }
        book = write_book_state(_executor(equity=100_000.0, cash=95_000.0,
                                          positions=pos), "A", str(tmp_path))

        assert book["exposure"] == pytest.approx(0.05)
        assert book["n_positions"] == 2
        loaded = load_book_state("A", str(tmp_path))
        assert loaded == book
        assert loaded["positions"][0]["ticker"] == "AAPL"  # sorted by value desc

    def test_broker_failure_returns_none_not_raises(self, tmp_path):
        _init_live(tmp_path)
        ex = MagicMock()
        ex.get_portfolio_value.side_effect = RuntimeError("broker down")
        assert write_book_state(ex, "A", str(tmp_path)) is None
        assert load_book_state("A", str(tmp_path)) is None


class TestRenderedContext:
    def _book(self, exposure=0.5, n=3):
        pos = [{"ticker": f"T{i}", "weight": exposure / n,
                "unrealized_pnl_pct": 0.02} for i in range(n)]
        return {"account": "A", "equity": 100_000.0, "cash": 100_000.0 * (1 - exposure),
                "invested": 100_000.0 * exposure, "exposure": exposure,
                "n_positions": n, "positions": pos}

    def test_deployment_phase_flags_cold_start(self, tmp_path):
        _init_live(tmp_path)  # genesis yesterday → age 1 session
        text = build_portfolio_context(self._book(exposure=0.07), "A", str(tmp_path))
        assert "Phase: DEPLOYMENT" in text
        assert "COLD START" in text
        assert "entry abstention" in text

    def test_steady_state_when_fully_deployed(self, tmp_path):
        _init_live(tmp_path)
        text = build_portfolio_context(self._book(exposure=0.9), "A", str(tmp_path))
        assert "Phase: STEADY" in text
        assert "COLD START" not in text

    def test_arm_vs_control_delta_when_both_have_records(self, tmp_path):
        _init_live(tmp_path)
        live = tmp_path / "live"
        rows_a = [{"decision_date": "2026-08-24", "equity": 100_000.0},
                  {"decision_date": "2026-08-25", "equity": 101_000.0}]
        rows_c = [{"decision_date": "2026-08-24", "equity": 100_000.0},
                  {"decision_date": "2026-08-25", "equity": 99_500.0}]
        for acct, rows in (("A", rows_a), ("C", rows_c)):
            d = live / acct
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "equity.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

        text = build_portfolio_context(self._book(), "A", str(tmp_path))
        assert "+1.00%" in text          # A since genesis
        assert "-0.50%" in text          # control C
        assert "+1.50" in text           # delta pp

    def test_insufficient_record_is_said_plainly(self, tmp_path):
        _init_live(tmp_path)
        text = build_portfolio_context(self._book(), "A", str(tmp_path))
        assert "insufficient data" in text

    def test_no_genesis_marker_omits_phase_lines(self, tmp_path):
        text = build_portfolio_context(self._book(), "A", str(tmp_path))
        assert "Phase:" not in text
        assert genesis_date(str(tmp_path)) is None
