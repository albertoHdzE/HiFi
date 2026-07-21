"""Tests for the live 4-arm report builders (DJ-115)."""

from __future__ import annotations

import json
from pathlib import Path

from hifi.analytics import live_report as lr


def _live(tmp: Path, arm: str) -> Path:
    d = tmp / "live" / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def _history(tmp, arm, equity):
    ts = [1784160000 + i * 86400 for i in range(len(equity))]
    (_live(tmp, arm) / "portfolio_history.json").write_text(
        json.dumps({"timestamp": ts, "equity": equity,
                    "profit_loss": [0] * len(equity), "profit_loss_pct": [0] * len(equity)})
    )


def _decisions(tmp, arm, rows):
    p = _live(tmp, arm) / "decisions.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sidecar(tmp, arm, cond, date, ticker, agent_decisions, entropy):
    y, m, _ = date.split("-")
    d = _live(tmp, arm) / "walkforward" / date / cond / y / m
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{ticker}.json").write_text(json.dumps({
        "ticker": ticker, "as_of_date": date,
        "ensemble_decision": {"agent_decisions": agent_decisions, "disagreement_entropy": entropy},
    }))


class TestFinancial:
    def test_equity_curves_rebased(self, tmp_path):
        _history(tmp_path, "C", [100.0, 101.0, 99.0])
        curves = lr.equity_curves(str(tmp_path), rebased=True)
        assert curves["C"].iloc[0] == 100.0
        assert round(curves["C"].iloc[1], 2) == 101.0

    def test_metrics_thin_data_marks_not_enough(self, tmp_path):
        _history(tmp_path, "A", [100.0, 101.0])
        m = lr.metrics_table(str(tmp_path), min_points=20)
        assert not m.loc["A", "enough_data"]
        assert m.loc["A", "sharpe"] is None
        assert m.loc["A", "total_return_pct"] is not None  # simple stat still computed

    def test_metrics_enough_data_populates(self, tmp_path):
        import numpy as np
        eq = list(100.0 * np.cumprod(1 + np.random.default_rng(0).normal(0.001, 0.01, 40)))
        _history(tmp_path, "A", eq)
        m = lr.metrics_table(str(tmp_path), min_points=20)
        assert m.loc["A", "enough_data"]
        assert m.loc["A", "sharpe"] is not None


class TestDecisions:
    def test_signal_distribution_counts(self, tmp_path):
        _decisions(tmp_path, "A", [{
            "decision_date": "2026-07-20", "n_orders": 1,
            "signals": [{"decision": "Buy"}, {"decision": "Sell"}, {"decision": "Sell"}],
        }])
        sd = lr.signal_distribution("A", str(tmp_path))
        assert sd.iloc[0]["Buy"] == 1 and sd.iloc[0]["Sell"] == 2
        assert sd.iloc[0]["n_orders"] == 1

    def test_signal_distribution_empty(self, tmp_path):
        assert lr.signal_distribution("Z", str(tmp_path)).empty


class TestHerding:
    def test_unanimity_and_entropy(self, tmp_path):
        # ticker1: unanimous, ticker2: split
        _sidecar(tmp_path, "A", "parallel", "2026-07-20", "AAA", ["Buy", "Buy", "Buy"], 0.0)
        _sidecar(tmp_path, "A", "parallel", "2026-07-20", "BBB", ["Buy", "Sell", "Hold"], 1.5)
        h = lr.herding_series("A", str(tmp_path))
        row = h[h["decision_date"] == "2026-07-20"].iloc[0]
        assert row["n_tickers"] == 2
        assert row["unanimity"] == 0.5
        assert abs(row["mean_entropy"] - 0.75) < 1e-9

    def test_per_date_retained_across_two_dates(self, tmp_path):
        # date-partitioned layout must keep BOTH dates (issue #2 regression guard)
        _sidecar(tmp_path, "A", "parallel", "2026-07-16", "AAA", ["Buy", "Buy", "Buy"], 0.0)
        _sidecar(tmp_path, "A", "parallel", "2026-07-20", "AAA", ["Buy", "Sell", "Hold"], 1.5)
        h = lr.herding_series("A", str(tmp_path))
        assert set(h["decision_date"]) == {"2026-07-16", "2026-07-20"}

    def test_non_llm_arm_empty(self, tmp_path):
        _live(tmp_path, "C")
        assert lr.herding_series("C", str(tmp_path)).empty
