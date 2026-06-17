"""
Unit tests for hifi-portfolio-composer MCP server (E4-T1, DJ-091).

All tests are deterministic. No LLMs, no network, no LanceDB.
Constraint defaults: max_single_stock=0.05, max_sector=0.20, min_position=0.01.
"""

from __future__ import annotations

import json

from hifi.mcp.portfolio_composer import (
    _apply_min_position,
    _apply_sector_cap,
    _apply_stock_cap,
    compose_portfolio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signals(*args: tuple[str, str, float, str]) -> str:
    """Build signals_json from (ticker, decision, confidence, sector) tuples."""
    return json.dumps(
        [
            {"ticker": t, "decision": d, "confidence": c, "sector": s}
            for t, d, c, s in args
        ]
    )


# ---------------------------------------------------------------------------
# _apply_stock_cap
# ---------------------------------------------------------------------------


def test_stock_cap_no_binding() -> None:
    """Weights well below the cap are unchanged."""
    w = {"A": 0.3, "B": 0.3, "C": 0.4}
    result = _apply_stock_cap(w, 0.5)
    assert abs(result["A"] - 0.3) < 1e-9
    assert abs(result["B"] - 0.3) < 1e-9
    assert abs(result["C"] - 0.4) < 1e-9


def test_stock_cap_single_stock_no_redistribution() -> None:
    """Single stock capped at max_weight; excess flows to cash (no redistribution)."""
    w = {"A": 1.0}
    result = _apply_stock_cap(w, 0.05)
    assert abs(result["A"] - 0.05) < 1e-9
    assert abs(sum(result.values()) - 0.05) < 1e-9


def test_stock_cap_redistributes_to_uncapped() -> None:
    """Excess from capped A is redistributed to uncapped B and C."""
    w = {"A": 0.6, "B": 0.2, "C": 0.2}
    result = _apply_stock_cap(w, 0.5)
    assert abs(result["A"] - 0.5) < 1e-9
    # Excess 0.1 redistributed equally (B and C are equal weight)
    assert abs(result["B"] - 0.25) < 1e-9
    assert abs(result["C"] - 0.25) < 1e-9
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_stock_cap_all_capped_simultaneously() -> None:
    """When all positions hit the cap at once, excess goes to cash."""
    # 3 equal positions, cap at 0.2 (each starts at 0.333)
    w = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    result = _apply_stock_cap(w, 0.2)
    for v in result.values():
        assert abs(v - 0.2) < 1e-9
    assert abs(sum(result.values()) - 0.6) < 1e-9  # 0.4 in cash


# ---------------------------------------------------------------------------
# _apply_sector_cap
# ---------------------------------------------------------------------------


def test_sector_cap_no_binding() -> None:
    """Sector total below max: weights unchanged."""
    w = {"A": 0.1, "B": 0.05}
    sectors = {"A": "IT", "B": "IT"}
    result = _apply_sector_cap(w, sectors, 0.20)
    assert abs(result["A"] - 0.1) < 1e-9
    assert abs(result["B"] - 0.05) < 1e-9


def test_sector_cap_scales_down_overweight_sector() -> None:
    """IT sector at 0.8 is scaled down to 0.20; Financials position unchanged."""
    w = {"A": 0.4, "B": 0.4, "C": 0.2}
    sectors = {"A": "IT", "B": "IT", "C": "Financials"}
    result = _apply_sector_cap(w, sectors, 0.20)
    # IT: scale 0.20/0.80 = 0.25
    assert abs(result["A"] - 0.1) < 1e-9
    assert abs(result["B"] - 0.1) < 1e-9
    assert abs(result["C"] - 0.2) < 1e-9


def test_sector_cap_multiple_sectors() -> None:
    """Both overweight sectors are independently capped."""
    w = {"A": 0.4, "B": 0.4, "C": 0.4, "D": 0.4}
    sectors = {"A": "IT", "B": "IT", "C": "Energy", "D": "Energy"}
    result = _apply_sector_cap(w, sectors, 0.20)
    # Each sector total was 0.8, capped at 0.20 → scale 0.25
    for v in result.values():
        assert abs(v - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# _apply_min_position
# ---------------------------------------------------------------------------


def test_min_position_removes_tiny_and_redistributes() -> None:
    """Tiny position is removed; its weight redistributed to remaining."""
    w = {"A": 0.005, "B": 0.5, "C": 0.495}
    result = _apply_min_position(w, 0.01)
    assert "A" not in result
    assert "B" in result
    assert "C" in result
    # Total weight conserved (A's weight redistributed to B and C)
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_min_position_no_removal() -> None:
    """All positions above min_position: returned unchanged."""
    w = {"A": 0.5, "B": 0.5}
    result = _apply_min_position(w, 0.01)
    assert abs(result["A"] - 0.5) < 1e-9
    assert abs(result["B"] - 0.5) < 1e-9


def test_min_position_empty_when_all_below_threshold() -> None:
    """All positions below min_position returns empty dict."""
    w = {"A": 0.005, "B": 0.003}
    result = _apply_min_position(w, 0.01)
    assert result == {}


def test_min_position_iterative_removal() -> None:
    """Positions pushed below threshold by earlier redistribution are also removed."""
    # A is removed; B gets redistributed weight but then also hits min_position only
    # if its initial weight is very small
    w = {"A": 0.008, "B": 0.012}
    result = _apply_min_position(w, 0.01)
    # A removed (0.008 < 0.01), its 0.008 added to B → B = 0.012 + 0.008 = 0.020
    assert "A" not in result
    assert "B" in result
    assert abs(result["B"] - 0.020) < 1e-9


# ---------------------------------------------------------------------------
# compose_portfolio: edge cases
# ---------------------------------------------------------------------------


def test_all_hold_returns_empty_dict() -> None:
    signals = _signals(("AAPL", "Hold", 0.8, "IT"), ("MSFT", "Hold", 0.7, "IT"))
    result = compose_portfolio(signals)
    assert result == {}


def test_all_sell_returns_empty_dict_long_only() -> None:
    signals = _signals(("AAPL", "Sell", 0.8, "IT"), ("MSFT", "Sell", 0.7, "IT"))
    result = compose_portfolio(signals)
    assert result == {}


def test_empty_signals_returns_empty_dict() -> None:
    result = compose_portfolio(json.dumps([]))
    assert result == {}


def test_invalid_json_returns_error() -> None:
    result = compose_portfolio("not-valid-json{")
    assert "error" in result


def test_non_array_json_returns_error() -> None:
    result = compose_portfolio(json.dumps({"ticker": "AAPL"}))
    assert "error" in result


def test_zero_confidence_signals_excluded() -> None:
    """Signals with zero confidence are skipped."""
    signals = _signals(
        ("AAPL", "Buy", 0.0, "IT"),
        ("MSFT", "Buy", 0.8, "IT"),
    )
    result = compose_portfolio(signals, max_single_stock=1.0, max_sector=1.0, min_position=0.0)
    assert "AAPL" not in result
    assert "MSFT" in result


# ---------------------------------------------------------------------------
# compose_portfolio: core algorithm cases
# ---------------------------------------------------------------------------


def test_three_equal_confidence_buy_equal_weights() -> None:
    """3 Buy signals, equal confidence, relaxed caps -> equal weights summing to 1."""
    signals = _signals(
        ("AAPL", "Buy", 0.8, "IT"),
        ("MSFT", "Buy", 0.8, "Financials"),
        ("JPM", "Buy", 0.8, "Consumer Discretionary"),
    )
    result = compose_portfolio(
        signals,
        max_single_stock=0.5,  # not binding (1/3 < 0.5)
        max_sector=0.8,  # not binding
        min_position=0.01,
    )
    assert set(result.keys()) == {"AAPL", "MSFT", "JPM"}
    for weight in result.values():
        assert abs(weight - 1 / 3) < 1e-6
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_single_buy_signal_weight_equals_max_single_stock() -> None:
    """Single Buy is capped at max_single_stock; remaining is cash."""
    signals = _signals(("AAPL", "Buy", 0.9, "IT"))
    result = compose_portfolio(
        signals, max_single_stock=0.05, max_sector=0.20, min_position=0.01
    )
    assert "AAPL" in result
    assert abs(result["AAPL"] - 0.05) < 1e-9


def test_sector_concentration_capped_at_max_sector() -> None:
    """3 stocks in same sector: aggregate capped at max_sector=0.20."""
    signals = _signals(
        ("AAPL", "Buy", 0.8, "IT"),
        ("MSFT", "Buy", 0.8, "IT"),
        ("NVDA", "Buy", 0.8, "IT"),
    )
    result = compose_portfolio(
        signals,
        max_single_stock=0.5,  # not binding (1/3 < 0.5)
        max_sector=0.20,
        min_position=0.001,  # tiny to not eliminate positions
    )
    it_total = sum(result.values())
    assert abs(it_total - 0.20) < 1e-9


def test_weights_sum_to_1_when_no_binding_caps() -> None:
    """When no caps bind, weights sum to 1.0."""
    signals = _signals(
        ("AAPL", "Buy", 0.6, "IT"),
        ("MSFT", "Buy", 0.4, "Financials"),
    )
    result = compose_portfolio(
        signals,
        max_single_stock=0.8,  # not binding (0.6 < 0.8)
        max_sector=0.9,  # not binding
        min_position=0.01,
    )
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_mixed_signals_only_buys_in_output() -> None:
    """Hold and Sell signals are excluded from the output."""
    signals = _signals(
        ("AAPL", "Buy", 0.8, "IT"),
        ("MSFT", "Hold", 0.7, "IT"),
        ("JPM", "Sell", 0.6, "Financials"),
        ("XOM", "Buy", 0.5, "Energy"),
    )
    result = compose_portfolio(
        signals, max_single_stock=0.5, max_sector=0.8, min_position=0.01
    )
    assert "AAPL" in result
    assert "XOM" in result
    assert "MSFT" not in result
    assert "JPM" not in result


def test_confidence_proportional_weighting() -> None:
    """Higher confidence gets proportionally more weight."""
    signals = _signals(
        ("AAPL", "Buy", 0.9, "IT"),
        ("MSFT", "Buy", 0.1, "Financials"),
    )
    result = compose_portfolio(
        signals,
        max_single_stock=1.0,  # no cap
        max_sector=1.0,  # no cap
        min_position=0.0,  # no minimum
    )
    assert abs(result["AAPL"] - 0.9) < 1e-9
    assert abs(result["MSFT"] - 0.1) < 1e-9


def test_min_position_eliminates_small_allocations() -> None:
    """Positions below min_position are removed and weight redistributed."""
    # 10 equal Buy signals -> each weight = 0.10; min_position=0.15 removes all
    signals = json.dumps(
        [
            {"ticker": f"T{i:02d}", "decision": "Buy", "confidence": 1.0, "sector": f"S{i}"}
            for i in range(10)
        ]
    )
    result = compose_portfolio(
        signals,
        max_single_stock=1.0,
        max_sector=1.0,
        min_position=0.15,
    )
    # All positions are 0.10 < 0.15 -> all removed -> empty
    assert result == {}


# ---------------------------------------------------------------------------
# MCP server structure
# ---------------------------------------------------------------------------


def test_module_is_importable() -> None:
    import hifi.mcp.portfolio_composer  # noqa: F401


def test_mcp_instance_exists() -> None:
    import hifi.mcp.portfolio_composer as srv

    assert srv.mcp is not None


def test_compose_portfolio_is_registered() -> None:
    import hifi.mcp.portfolio_composer as srv

    tool_names = [t.name for t in srv.mcp._tool_manager.list_tools()]
    assert "compose_portfolio" in tool_names
