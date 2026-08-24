"""Tests for the riskbudget external-strategy adapter (DJ-113)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hifi.execution import riskbudget_strategy as rb


def _write_store(tmp_path: Path, ticker: str, dates: list[str], closes: list[float]) -> None:
    d = tmp_path / "market" / ticker
    d.mkdir(parents=True)
    df = pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))
    df.index.name = "Date"
    df.to_parquet(d / "ohlcv.parquet")


class TestPointInTimeCloses:
    def test_reads_nested_store_and_lowercases(self, tmp_path):
        _write_store(tmp_path, "AAPL", ["2026-01-02", "2026-01-03"], [100.0, 101.0])
        out = rb.point_in_time_closes("AAPL", "2026-01-03", str(tmp_path))
        assert out == [100.0, 101.0]

    def test_point_in_time_filter_excludes_future(self, tmp_path):
        _write_store(tmp_path, "AAPL", ["2026-01-02", "2026-01-03", "2026-01-06"],
                     [100.0, 101.0, 102.0])
        out = rb.point_in_time_closes("AAPL", "2026-01-03", str(tmp_path))
        assert out == [100.0, 101.0]  # 2026-01-06 excluded

    def test_missing_ticker_returns_empty(self, tmp_path):
        assert rb.point_in_time_closes("ZZZZ", "2026-01-03", str(tmp_path)) == []

    def test_ordered_oldest_to_newest(self, tmp_path):
        _write_store(tmp_path, "MSFT", ["2026-01-06", "2026-01-02", "2026-01-03"],
                     [3.0, 1.0, 2.0])
        out = rb.point_in_time_closes("MSFT", "2026-01-06", str(tmp_path))
        assert out == [1.0, 2.0, 3.0]


class TestGetRiskbudgetSignals:
    def test_passes_closes_bypass_and_returns_payload(self, tmp_path):
        _write_store(tmp_path, "AAPL", ["2026-01-02", "2026-01-03"], [100.0, 101.0])
        fake = {
            "signals": [{"ticker": "AAPL", "decision": "Hold", "confidence": 0.6,
                         "sector": "Information Technology", "target_exposure": 0.6}],
            "skipped": [], "strategy": "calm_exposure",
            "strategy_version": "1.0.0", "as_of_date": "2026-01-03", "call_id": "abc123",
        }
        # call_tool is imported lazily inside the function; patch at its source.
        with patch("hifi.agents.mcp_client.call_tool", return_value=fake) as m2:
            out = rb.get_riskbudget_signals(
                ["AAPL"], "2026-01-03", str(tmp_path),
                sectors={"AAPL": "Information Technology"},
            )
        assert out["strategy_version"] == "1.0.0"
        assert out["signals"][0]["decision"] == "Hold"
        called = m2.call_args.kwargs
        assert called["tool_name"] == "get_signals"
        assert called["params"]["closes"] == {"AAPL": [100.0, 101.0]}
        assert called["server_module"] == "riskbudget.mcp_server"

    def test_error_payload_returns_empty_signals(self, tmp_path):
        _write_store(tmp_path, "AAPL", ["2026-01-02"], [100.0])
        with patch("hifi.agents.mcp_client.call_tool",
                   return_value={"error": "BOOM", "detail": "x"}):
            out = rb.get_riskbudget_signals(["AAPL"], "2026-01-03", str(tmp_path))
        assert out["signals"] == []
        assert out["error"] == "BOOM"
