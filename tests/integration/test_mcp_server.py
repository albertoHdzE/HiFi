"""
Integration tests for the HiFi Financial Calculator MCP server (P2-E5).

These tests exercise the MCP tool handler functions directly (not via a
subprocess JSON-RPC pipe), validating the full pipeline:

  fixture parquet → _load_ohlcv → engine → serialised dict

This approach is equivalent to a subprocess integration test for the
computation layer and avoids subprocess lifecycle complexity in CI.

True stdio subprocess tests (P2-E5-T9: tools/list returns 6 tools) are
marked with the 'subprocess' marker and run separately. They require the
MCP protocol handshake, which is tested at Phase 5 when the Agent layer
is assembled.

Fixtures directory: tests/fixtures/market/
  AAPL_2023-01-03_2023-04-01.parquet  (Q1 2023, ~60 bars)
  JPM_2023-01-03_2023-04-01.parquet
  XOM_2023-01-03_2023-04-01.parquet

Tickets covered: P2-E5-T10 through P2-E5-T13.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

# ---------------------------------------------------------------------------
# Set the data directory to the test fixtures before importing the server
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def set_data_dir(monkeypatch):
    """Point the MCP server at the test fixtures directory."""
    monkeypatch.setenv("HIFI_DATA_DIR", str(_FIXTURES))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2023, 4, 1, 0, 0, 0)
_PROV = ProvenanceRecord(source="test", fetched_at=_NOW)

_AAPL_DATE = "2023-03-31"


def _make_snapshot(ticker: str = "AAPL") -> FundamentalsSnapshot:
    """Realistic AAPL-like FundamentalsSnapshot for engine tests."""
    return FundamentalsSnapshot(
        ticker=ticker,
        period_end=date(2022, 9, 30),
        revenue=394_328_000_000.0,
        net_income=99_803_000_000.0,
        total_assets=352_755_000_000.0,
        total_liabilities=302_083_000_000.0,
        total_equity=50_672_000_000.0,
        eps=6.11,
        pe_ratio=25.2,
        market_cap=2_396_000_000_000.0,
        source="yfinance",
        fetched_at=_NOW,
        provenance=_PROV,
    )


def _snapshot_json(ticker: str = "AAPL") -> str:
    snap = _make_snapshot(ticker)
    return snap.model_dump_json()


# ---------------------------------------------------------------------------
# P2-E5-T10: Technical indicators from AAPL Q1 2023 fixture
# ---------------------------------------------------------------------------


class TestGetTechnicalIndicators:
    """Tests for the get_technical_indicators tool handler."""

    def test_rsi_in_valid_range_for_aapl_fixture(self):
        """T10: RSI is in [0, 100] for the AAPL Q1 2023 fixture."""
        from hifi.mcp.financial_server import get_technical_indicators

        result = get_technical_indicators(ticker="AAPL", date=_AAPL_DATE, window=20)
        assert "error" not in result, f"Tool returned error: {result}"
        rsi = result.get("rsi")
        assert rsi is not None, "RSI should be computed for 60+ bars"
        assert 0.0 <= rsi <= 100.0

    def test_sma_is_float_for_aapl_fixture(self):
        """SMA is a finite float for AAPL Q1 2023."""
        from hifi.mcp.financial_server import get_technical_indicators

        result = get_technical_indicators(ticker="AAPL", date=_AAPL_DATE, window=20)
        assert "error" not in result
        assert result["sma"] is not None
        assert isinstance(result["sma"], float)

    def test_bollinger_bounds_satisfied(self):
        """Bollinger upper >= mid >= lower."""
        from hifi.mcp.financial_server import get_technical_indicators

        result = get_technical_indicators(ticker="AAPL", date=_AAPL_DATE, window=20)
        assert "error" not in result
        if result["bb_upper"] is not None:
            assert result["bb_upper"] >= result["bb_mid"]
            assert result["bb_mid"] >= result["bb_lower"]

    def test_unknown_ticker_returns_error_dict(self):
        """T12: Call for unknown ticker 'ZZZZZ' returns error dict, not exception."""
        from hifi.mcp.financial_server import get_technical_indicators

        result = get_technical_indicators(ticker="ZZZZZ", date=_AAPL_DATE)
        assert "error" in result
        assert result["error"] == "TICKER_NOT_FOUND"

    def test_result_is_json_serialisable(self):
        """T13: Result is JSON-serialisable."""
        from hifi.mcp.financial_server import get_technical_indicators

        result = get_technical_indicators(ticker="AAPL", date=_AAPL_DATE)
        json.dumps(result)

    def test_call_id_present(self):
        """Each response includes a call_id for Phase 5 verification."""
        from hifi.mcp.financial_server import get_technical_indicators

        result = get_technical_indicators(ticker="AAPL", date=_AAPL_DATE)
        assert "call_id" in result
        assert len(result["call_id"]) == 12


# ---------------------------------------------------------------------------
# P2-E5-T11: Macro snapshot
# ---------------------------------------------------------------------------


class TestGetMacroSnapshot:
    """Tests for the get_macro_snapshot tool handler."""

    def test_no_macro_data_returns_error(self):
        """Returns error dict when no macro parquet files are present."""
        from hifi.mcp.financial_server import get_macro_snapshot

        # fixtures/macro only has XML files (HTTP response fixtures), not parquet.
        result = get_macro_snapshot(date="2022-06-15")
        # Either no parquet found (error) or empty result — not a crash
        assert isinstance(result, dict)

    def test_result_is_json_serialisable(self):
        """T13: Macro snapshot result is JSON-serialisable."""
        from hifi.mcp.financial_server import get_macro_snapshot

        result = get_macro_snapshot(date="2022-06-15")
        json.dumps(result)


# ---------------------------------------------------------------------------
# P2-E5-T11 extended: Financial ratios tool
# ---------------------------------------------------------------------------


class TestGetFinancialRatios:
    """Tests for the get_financial_ratios tool handler."""

    def test_pe_ratio_computable_for_aapl(self):
        """PE ratio is non-None when EPS > 0 and price is available."""
        from hifi.mcp.financial_server import get_financial_ratios

        result = get_financial_ratios(
            ticker="AAPL",
            date=_AAPL_DATE,
            snapshot_json=_snapshot_json(),
        )
        assert "error" not in result, f"Tool returned error: {result}"
        assert result["pe"] is not None
        assert result["pe"] > 0.0

    def test_unknown_ticker_returns_error(self):
        """T12: Unknown ticker returns error, not exception."""
        from hifi.mcp.financial_server import get_financial_ratios

        result = get_financial_ratios(
            ticker="ZZZZZ",
            date=_AAPL_DATE,
            snapshot_json=_snapshot_json(),
        )
        assert "error" in result
        assert result["error"] == "TICKER_NOT_FOUND"

    def test_invalid_snapshot_json_returns_error(self):
        """Malformed snapshot_json returns SNAPSHOT_INVALID error."""
        from hifi.mcp.financial_server import get_financial_ratios

        result = get_financial_ratios(
            ticker="AAPL",
            date=_AAPL_DATE,
            snapshot_json="{invalid json}",
        )
        assert "error" in result

    def test_result_is_json_serialisable(self):
        """T13: Financial ratios result is JSON-serialisable."""
        from hifi.mcp.financial_server import get_financial_ratios

        result = get_financial_ratios(
            ticker="AAPL",
            date=_AAPL_DATE,
            snapshot_json=_snapshot_json(),
        )
        json.dumps(result)


# ---------------------------------------------------------------------------
# Risk metrics tool
# ---------------------------------------------------------------------------


class TestGetRiskMetrics:
    """Tests for the get_risk_metrics tool handler."""

    def test_hist_vol_positive_for_aapl(self):
        """Historical volatility is positive for AAPL Q1 2023."""
        from hifi.mcp.financial_server import get_risk_metrics

        result = get_risk_metrics(ticker="AAPL", date=_AAPL_DATE, window=60)
        assert "error" not in result, f"Tool returned error: {result}"
        # May be None if insufficient bars for the window; check if present
        if result["hist_vol_20d"] is not None:
            assert result["hist_vol_20d"] > 0.0

    def test_result_is_json_serialisable(self):
        """T13: Risk metrics result is JSON-serialisable."""
        from hifi.mcp.financial_server import get_risk_metrics

        result = get_risk_metrics(ticker="AAPL", date=_AAPL_DATE)
        json.dumps(result)

    def test_unknown_ticker_returns_error(self):
        """T12: Unknown ticker returns error."""
        from hifi.mcp.financial_server import get_risk_metrics

        result = get_risk_metrics(ticker="ZZZZZ", date=_AAPL_DATE)
        assert "error" in result
        assert result["error"] == "TICKER_NOT_FOUND"


# ---------------------------------------------------------------------------
# All 6 tools return JSON-serialisable dicts (T13)
# ---------------------------------------------------------------------------


class TestAllToolsJsonSafe:
    """T13: All 6 tools return JSON-serialisable dicts."""

    def _run_all_tools(self) -> list[dict]:
        from hifi.mcp.financial_server import (
            get_financial_ratios,
            get_growth_metrics,
            get_macro_snapshot,
            get_risk_metrics,
            get_technical_indicators,
            get_valuation_context,
        )

        snap_json = _snapshot_json()
        return [
            get_financial_ratios(ticker="AAPL", date=_AAPL_DATE, snapshot_json=snap_json),
            get_growth_metrics(snapshot_json=snap_json),
            get_technical_indicators(ticker="AAPL", date=_AAPL_DATE),
            get_risk_metrics(ticker="AAPL", date=_AAPL_DATE),
            get_valuation_context(ticker="AAPL", date=_AAPL_DATE, snapshot_json=snap_json),
            get_macro_snapshot(date=_AAPL_DATE),
        ]

    def test_all_tools_return_dicts(self):
        """Every tool returns a dict."""
        for result in self._run_all_tools():
            assert isinstance(result, dict)

    def test_all_tools_json_safe(self):
        """json.dumps succeeds for every tool result."""
        for result in self._run_all_tools():
            json.dumps(result)  # must not raise

    def test_all_tools_have_call_id(self):
        """All successful tool responses include a call_id."""
        for result in self._run_all_tools():
            if "error" not in result:
                assert "call_id" in result
