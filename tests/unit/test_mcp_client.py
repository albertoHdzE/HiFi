"""Unit tests for the MCP subprocess client (P3-E1)."""

import os

import pytest

from hifi.agents.mcp_client import call_tool

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MARKET_FIXTURES = os.path.join(FIXTURES_DIR, "market")
MACRO_FIXTURES = os.path.join(FIXTURES_DIR, "macro")


@pytest.fixture
def fixtures_data_dir(tmp_path):
    """
    Create a data dir layout that points market/ and macro/ at the Phase 1 fixtures.

    The MCP server looks for:
        HIFI_DATA_DIR/market/{ticker}_*.parquet
        HIFI_DATA_DIR/macro/*.parquet
    """
    import shutil

    market_dir = tmp_path / "market"
    macro_dir = tmp_path / "macro"
    market_dir.mkdir()
    macro_dir.mkdir()

    # Copy Phase 1 parquet fixtures
    fixtures_root = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    for parquet in os.listdir(os.path.join(fixtures_root, "market")):
        if parquet.endswith(".parquet"):
            shutil.copy(
                os.path.join(fixtures_root, "market", parquet),
                market_dir / parquet,
            )
    for parquet in os.listdir(os.path.join(fixtures_root, "macro")):
        if parquet.endswith(".parquet"):
            shutil.copy(
                os.path.join(fixtures_root, "macro", parquet),
                macro_dir / parquet,
            )
    return str(tmp_path)


def test_call_tool_technical_indicators_returns_dict_with_call_id(fixtures_data_dir):
    result = call_tool(
        tool_name="get_technical_indicators",
        params={"ticker": "AAPL", "date": "2023-03-31", "window": 20},
        data_dir=fixtures_data_dir,
    )
    assert isinstance(result, dict)
    assert "call_id" in result


def test_call_tool_technical_indicators_rsi_in_range(fixtures_data_dir):
    result = call_tool(
        tool_name="get_technical_indicators",
        params={"ticker": "AAPL", "date": "2023-03-31", "window": 20},
        data_dir=fixtures_data_dir,
    )
    rsi = result.get("rsi")
    if rsi is not None:
        assert 0.0 <= rsi <= 100.0


def test_call_tool_unknown_ticker_returns_error(fixtures_data_dir):
    result = call_tool(
        tool_name="get_technical_indicators",
        params={"ticker": "ZZZZZ", "date": "2023-03-31"},
        data_dir=fixtures_data_dir,
    )
    assert "error" in result or "TICKER_NOT_FOUND" in str(result)


def test_call_tool_macro_snapshot_returns_dict_with_call_id(fixtures_data_dir):
    result = call_tool(
        tool_name="get_macro_snapshot",
        params={"date": "2022-06-15"},
        data_dir=fixtures_data_dir,
    )
    assert isinstance(result, dict)
    assert "call_id" in result
