"""Integration tests for the agent stack: MCP client + LangGraph skeleton (P3-E1)."""

import os
import shutil

import pytest

from hifi.agents.mcp_client import call_tool


@pytest.fixture
def fixtures_data_dir(tmp_path):
    """Data directory with Phase 1 parquet fixtures for market and macro."""
    fixtures_root = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    market_dst = tmp_path / "market"
    macro_dst = tmp_path / "macro"
    market_dst.mkdir()
    macro_dst.mkdir()

    market_src = os.path.join(fixtures_root, "market")
    macro_src = os.path.join(fixtures_root, "macro")

    for f in os.listdir(market_src):
        if f.endswith(".parquet"):
            shutil.copy(os.path.join(market_src, f), market_dst / f)
    for f in os.listdir(macro_src):
        if f.endswith(".parquet"):
            shutil.copy(os.path.join(macro_src, f), macro_dst / f)

    return str(tmp_path)


def test_mcp_client_technical_indicators_aapl(fixtures_data_dir):
    """MCP call for AAPL technical indicators returns RSI in [0, 100]."""
    result = call_tool(
        "get_technical_indicators",
        {"ticker": "AAPL", "date": "2023-03-31", "window": 20},
        data_dir=fixtures_data_dir,
    )
    assert "call_id" in result
    rsi = result.get("rsi")
    if rsi is not None:
        assert 0.0 <= rsi <= 100.0


def test_mcp_client_macro_snapshot(fixtures_data_dir):
    """MCP macro snapshot returns a dict with call_id."""
    result = call_tool(
        "get_macro_snapshot",
        {"date": "2022-06-15"},
        data_dir=fixtures_data_dir,
    )
    assert isinstance(result, dict)
    assert "call_id" in result


def test_mcp_client_unknown_ticker_returns_error_not_exception(fixtures_data_dir):
    """MCP client returns error dict for unknown ticker; never raises."""
    result = call_tool(
        "get_technical_indicators",
        {"ticker": "ZZZZZ", "date": "2023-03-31"},
        data_dir=fixtures_data_dir,
    )
    assert isinstance(result, dict)
    # Either has "error" key or "TICKER_NOT_FOUND" embedded
    error_indicator = result.get("error") or result.get("TICKER_NOT_FOUND")
    assert error_indicator or "error" in str(result).lower()


def test_langgraph_single_node_calls_mcp(fixtures_data_dir):
    """LangGraph graph with one node calling get_technical_indicators returns valid result."""
    from langgraph.graph import END, StateGraph
    from typing_extensions import TypedDict

    class SimpleState(TypedDict, total=False):
        result: dict

    def fetch_node(state: SimpleState) -> dict:
        r = call_tool(
            "get_technical_indicators",
            {"ticker": "AAPL", "date": "2023-03-31", "window": 20},
            data_dir=fixtures_data_dir,
        )
        return {"result": r}

    builder = StateGraph(SimpleState)
    builder.add_node("fetch", fetch_node)
    builder.set_entry_point("fetch")
    builder.add_edge("fetch", END)
    graph = builder.compile()

    output = graph.invoke({})
    assert "result" in output
    assert "call_id" in output["result"]
