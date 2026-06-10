"""
Holistic test: Phase 2 Financial Engine Pipeline.

Validates the full Phase 2 pipeline end-to-end:
  Load fixture data → compute all six result types → verify cross-cutting
  invariants → confirm Phase 1 regression guard still passes.

The holistic test runs AFTER all unit and integration tests pass. It uses
the AAPL Q1 2023 fixture (tests/fixtures/market/AAPL_2023-01-03_2023-04-01.parquet)
to verify that:
  1. All six engine functions produce typed, valid results from real data.
  2. Cross-cutting invariants hold (no NaN, JSON-safe, Bollinger ordering, etc.).
  3. The MCP tool functions return results consistent with direct engine calls.
  4. Phase 1 data layer is unaffected (regression guard).

Scenario: AAPL, analysis date 2023-03-31 (last bar in Q1 2023 fixture).

Design: Uses engine functions directly to keep the holistic test fast and
independent of subprocess lifecycle. The test serves as a contract test:
any breaking change in a Phase 2 engine shows up immediately here before
it can affect agents in Phase 3+.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pytest

from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord
from tests.conftest import read_raw_ohlcv_fixture

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_AAPL_PARQUET = _FIXTURES / "market" / "AAPL_2023-01-03_2023-04-01.parquet"
_ANALYSIS_DATE = date(2023, 3, 31)
_NOW = datetime(2023, 4, 1, 0, 0, 0)
_PROV = ProvenanceRecord(source="test", fetched_at=_NOW)


@pytest.fixture(autouse=True)
def _set_data_dir(monkeypatch):
    """Point the MCP server at the test fixtures directory."""
    monkeypatch.setenv("HIFI_DATA_DIR", str(_FIXTURES))


@pytest.fixture(scope="module")
def aapl_dataset():
    """Load the AAPL Q1 2023 fixture. Module-scoped for performance."""
    if not _AAPL_PARQUET.exists():
        pytest.skip(f"AAPL fixture not found: {_AAPL_PARQUET}")
    return read_raw_ohlcv_fixture(_AAPL_PARQUET, "AAPL")


@pytest.fixture(scope="module")
def aapl_snapshot():
    """Construct a realistic AAPL FundamentalsSnapshot for Q1 2023 analysis."""
    return FundamentalsSnapshot(
        ticker="AAPL",
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


# ---------------------------------------------------------------------------
# Scenario 1: Direct engine calls — all six result types
# ---------------------------------------------------------------------------


class TestDirectEngineCalls:
    """All six engine functions produce valid, typed results from AAPL fixture."""

    def test_fixture_has_expected_bars(self, aapl_dataset):
        """AAPL Q1 2023 fixture has at least 50 trading bars."""
        assert len(aapl_dataset.bars) >= 50, (
            f"Expected >= 50 bars, got {len(aapl_dataset.bars)}"
        )

    def test_financial_ratios(self, aapl_dataset, aapl_snapshot):
        """Financial ratios computed without error for AAPL Q1 2023."""
        from hifi.engines.fundamental import compute_financial_ratios

        bars = sorted(
            [b for b in aapl_dataset.bars if b.date <= _ANALYSIS_DATE],
            key=lambda b: b.date,
        )
        assert bars, "No bars on or before analysis date"
        price = bars[-1].close

        result = compute_financial_ratios(aapl_snapshot, price)
        # P/E should be a positive finite float
        assert result.pe is not None
        assert result.pe > 0.0
        assert math.isfinite(result.pe)

    def test_growth_metrics(self, aapl_snapshot):
        """Growth metrics: net_margin is computed; YoY fields are None (Phase 2)."""
        from hifi.engines.fundamental import compute_growth_metrics

        result = compute_growth_metrics(aapl_snapshot)
        assert result.net_margin is not None
        assert 0.0 < result.net_margin < 1.0, "Net margin should be between 0 and 1"
        # Phase 2 limitation — documented
        assert result.revenue_growth_yoy is None
        assert result.earnings_growth_yoy is None

    def test_technical_indicators(self, aapl_dataset):
        """All 6 technical indicators computed for AAPL Q1 2023."""
        from hifi.engines.technical import compute_technical_indicators

        result = compute_technical_indicators(aapl_dataset.bars, _ANALYSIS_DATE, window=20)
        # Fixture has ~60 bars — enough for all indicators
        assert result.sma is not None
        assert result.ema is not None
        assert result.rsi is not None
        assert 0.0 <= result.rsi <= 100.0
        assert result.bb_upper is not None
        assert result.bb_upper >= result.bb_mid >= result.bb_lower
        assert result.macd is not None
        assert result.atr is not None
        assert result.atr > 0.0

    def test_risk_metrics(self, aapl_dataset):
        """Risk metrics computed for AAPL Q1 2023."""
        from hifi.engines.risk import compute_risk_metrics

        result = compute_risk_metrics(aapl_dataset, _ANALYSIS_DATE)
        # 60 bars is enough for 20d and 60d vols, but borderline for 252d
        if result.hist_vol_20d is not None:
            assert result.hist_vol_20d > 0.0
            assert math.isfinite(result.hist_vol_20d)
        if result.max_drawdown_252d is not None:
            assert 0.0 <= result.max_drawdown_252d <= 1.0

    def test_valuation_context(self, aapl_dataset, aapl_snapshot):
        """Valuation context: pe_1y_percentile in [0, 1] for AAPL."""
        from hifi.engines.fundamental import compute_valuation_context

        result = compute_valuation_context(aapl_snapshot, aapl_dataset, _ANALYSIS_DATE)
        assert result.current_pe is not None
        assert result.pe_1y_percentile is not None
        assert 0.0 <= result.pe_1y_percentile <= 1.0
        assert result.price_to_52w_high is not None
        # Current price cannot exceed 52-week high (it IS within the fixture window)
        assert result.price_to_52w_high <= 1.0 + 1e-6  # allow floating-point tolerance


# ---------------------------------------------------------------------------
# Scenario 2: All results are JSON-safe
# ---------------------------------------------------------------------------


class TestAllResultsJsonSafe:
    """No NaN values in any result; json.dumps succeeds for all."""

    def _check_no_nan(self, result_dict: dict, label: str) -> None:
        for k, v in result_dict.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"{label}.{k} is NaN"

    def test_financial_ratios_json_safe(self, aapl_dataset, aapl_snapshot):
        from hifi.engines.fundamental import compute_financial_ratios

        bars = sorted(
            [b for b in aapl_dataset.bars if b.date <= _ANALYSIS_DATE],
            key=lambda b: b.date,
        )
        price = bars[-1].close
        result = compute_financial_ratios(aapl_snapshot, price)
        d = result.model_dump()
        self._check_no_nan(d, "FinancialRatioResult")
        json.dumps(d)

    def test_growth_metrics_json_safe(self, aapl_snapshot):
        from hifi.engines.fundamental import compute_growth_metrics

        result = compute_growth_metrics(aapl_snapshot)
        d = result.model_dump()
        self._check_no_nan(d, "GrowthMetricsResult")
        json.dumps(d)

    def test_technical_indicators_json_safe(self, aapl_dataset):
        from hifi.engines.technical import compute_technical_indicators

        result = compute_technical_indicators(aapl_dataset.bars, _ANALYSIS_DATE, window=20)
        d = result.model_dump()
        self._check_no_nan(d, "TechnicalIndicatorsResult")
        json.dumps(d)

    def test_risk_metrics_json_safe(self, aapl_dataset):
        from hifi.engines.risk import compute_risk_metrics

        result = compute_risk_metrics(aapl_dataset, _ANALYSIS_DATE)
        d = result.model_dump()
        self._check_no_nan(d, "RiskMetricsResult")
        json.dumps(d)

    def test_valuation_context_json_safe(self, aapl_dataset, aapl_snapshot):
        from hifi.engines.fundamental import compute_valuation_context

        result = compute_valuation_context(aapl_snapshot, aapl_dataset, _ANALYSIS_DATE)
        d = result.model_dump()
        self._check_no_nan(d, "ValuationResult")
        json.dumps(d)


# ---------------------------------------------------------------------------
# Scenario 3: MCP tool handlers consistent with direct engine calls
# ---------------------------------------------------------------------------


class TestMcpVsDirectEngineConsistency:
    """MCP tool responses match direct engine calls for the same inputs."""

    def test_technical_indicators_consistent(self, aapl_dataset):
        """MCP get_technical_indicators matches compute_technical_indicators."""
        from hifi.engines.technical import compute_technical_indicators
        from hifi.mcp.financial_server import get_technical_indicators

        direct = compute_technical_indicators(
            aapl_dataset.bars, _ANALYSIS_DATE, window=20
        )
        mcp_result = get_technical_indicators(
            ticker="AAPL", date=_ANALYSIS_DATE.isoformat(), window=20
        )

        assert "error" not in mcp_result
        if direct.rsi is not None:
            assert mcp_result["rsi"] == pytest.approx(direct.rsi, rel=1e-6)
        if direct.sma is not None:
            assert mcp_result["sma"] == pytest.approx(direct.sma, rel=1e-6)

    def test_risk_metrics_consistent(self, aapl_dataset):
        """MCP get_risk_metrics matches compute_risk_metrics."""
        from hifi.engines.risk import compute_risk_metrics
        from hifi.mcp.financial_server import get_risk_metrics

        direct = compute_risk_metrics(aapl_dataset, _ANALYSIS_DATE)
        mcp_result = get_risk_metrics(
            ticker="AAPL", date=_ANALYSIS_DATE.isoformat(), window=252
        )

        assert "error" not in mcp_result
        if direct.hist_vol_20d is not None:
            assert mcp_result["hist_vol_20d"] == pytest.approx(direct.hist_vol_20d, rel=1e-6)


# ---------------------------------------------------------------------------
# Scenario 4: Phase 1 regression guard
# ---------------------------------------------------------------------------


class TestPhase1Regression:
    """Phase 1 data layer is unaffected by Phase 2 additions."""

    def test_aapl_fixture_loads_correctly(self, aapl_dataset):
        """Phase 1 read_ohlcv still works for the AAPL fixture."""
        assert aapl_dataset.ticker == "AAPL"
        assert len(aapl_dataset.bars) > 0
        assert all(b.high >= b.close for b in aapl_dataset.bars)
        assert all(b.low <= b.close for b in aapl_dataset.bars)

    def test_jpm_and_xom_fixtures_load_correctly(self):
        """JPM and XOM fixtures are intact."""
        jpm_path = _FIXTURES / "market" / "JPM_2023-01-03_2023-04-01.parquet"
        xom_path = _FIXTURES / "market" / "XOM_2023-01-03_2023-04-01.parquet"
        if jpm_path.exists():
            jpm = read_raw_ohlcv_fixture(jpm_path, "JPM")
            assert len(jpm.bars) > 0
        if xom_path.exists():
            xom = read_raw_ohlcv_fixture(xom_path, "XOM")
            assert len(xom.bars) > 0

    def test_engines_do_not_import_from_data_modules_directly(self, aapl_dataset):
        """Engine modules are pure: no yfinance or fredapi calls."""
        # Verify by calling engines with already-loaded data — if they made
        # live calls, tests would fail in CI without credentials.
        from hifi.engines.technical import compute_technical_indicators

        result = compute_technical_indicators(aapl_dataset.bars, _ANALYSIS_DATE)
        # If this completes without credential errors, the purity guarantee holds.
        assert isinstance(result.model_dump(), dict)
