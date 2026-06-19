"""
Holistic test for the Phase 4 ensemble pipeline (P4-E6).

Uses DI LLMs for both agents and Phase 1 parquet fixtures.
No live LM Studio or network calls required.

What this test validates:
1. Both agents run independently (no shared state during reasoning)
2. run_ensemble returns structurally valid EnsembleOutput
3. EnsembleOutput is JSON-safe
4. Agreement scenario: both agents agree -> agreement=True, entropy=0.0
5. Disagreement scenario: agents disagree -> agreement=False, confidence-weighted winner
6. Phase 3 regression: Fundamental Agent alone still produces valid FundamentalAnalysis
"""

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.agents.ensemble_runner import run_ensemble
from hifi.agents.fundamental_agent import run_analysis
from hifi.agents.schemas import FundamentalAnalysis
from hifi.collective.schemas import EnsembleOutput
from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_data_dir(tmp_path):
    """Phase 1 parquet fixtures copied to a temp data dir."""
    fixtures_root = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    for subdir in ("market", "macro"):
        dst = tmp_path / subdir
        dst.mkdir()
        src = os.path.join(fixtures_root, subdir)
        for f in os.listdir(src):
            if f.endswith(".parquet"):
                shutil.copy(os.path.join(src, f), dst / f)
    return str(tmp_path)


@pytest.fixture
def aapl_snapshot_json():
    snap = FundamentalsSnapshot(
        ticker="AAPL",
        period_end="2023-03-31",
        eps=6.11,
        market_cap=2_500_000_000_000,
        total_equity=62_146_000_000,
        revenue=394_330_000_000,
        net_income=99_803_000_000,
        total_assets=352_755_000_000,
        total_liabilities=290_437_000_000,
        source="test",
        fetched_at=datetime(2023, 4, 1),
        provenance=ProvenanceRecord(source="test", fetched_at=datetime(2023, 4, 1)),
    )
    return snap.model_dump_json()


# DI LLM responses

_BUY_RESPONSE = json.dumps({
    "decision": "Buy",
    "confidence": 0.80,
    "rationale": "P/E of 28.3 is below its 1-year average. RSI of 42.1 signals recovery.",
    "key_concern": "Macro headwinds from high interest rates.",
    "time_horizon": "medium-term",
})

_HOLD_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.65,
    "rationale": "RSI of 48.0 is neutral territory. Sharpe of 0.82 is moderate.",
    "key_concern": "Max drawdown of -18% in past year is notable.",
    "time_horizon": "short-term",
})

_SELL_RESPONSE = json.dumps({
    "decision": "Sell",
    "confidence": 0.70,
    "rationale": "RSI of 72.0 is overbought. MACD histogram is turning negative.",
    "key_concern": "High volatility hist_vol_252d above 0.40.",
    "time_horizon": "short-term",
})


def _stub_llm(response_content: str, model_name: str = "test-model"):
    class _Stub:
        def invoke(self, messages):
            class _R:
                content = response_content
            return _R()
    stub = _Stub()
    stub.model_name = model_name
    return stub


# ---------------------------------------------------------------------------
# Test 1: EnsembleOutput returned with both analyses
# ---------------------------------------------------------------------------


def test_run_ensemble_returns_valid_output(fixtures_data_dir, aapl_snapshot_json):
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={
            "fundamental": _stub_llm(_BUY_RESPONSE, "fund-model"),
            "technical": _stub_llm(_HOLD_RESPONSE, "tech-model"),
        },
    )

    assert isinstance(output, EnsembleOutput)
    assert output.ticker == "AAPL"
    assert output.fundamental_analysis.signal is not None
    assert output.technical_analysis.signal is not None
    assert output.ensemble_decision.n_valid_signals == 2


# ---------------------------------------------------------------------------
# Test 2: Agreement scenario
# ---------------------------------------------------------------------------


def test_ensemble_agreement_scenario(fixtures_data_dir, aapl_snapshot_json):
    """Both agents return Buy -> agreement=True, entropy=0.0."""
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={
            "fundamental": _stub_llm(_BUY_RESPONSE, "fund-model"),
            "technical": _stub_llm(_BUY_RESPONSE, "tech-model"),
        },
    )

    ed = output.ensemble_decision
    assert ed.collective_decision == "Buy"
    assert ed.agreement is True
    assert ed.disagreement_entropy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: Disagreement scenario (confidence-weighted winner)
# ---------------------------------------------------------------------------


def test_ensemble_disagreement_scenario(fixtures_data_dir, aapl_snapshot_json):
    """fundamental=Buy(0.80), technical=Sell(0.70) -> Buy wins."""
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={
            "fundamental": _stub_llm(_BUY_RESPONSE, "fund-model"),
            "technical": _stub_llm(_SELL_RESPONSE, "tech-model"),
        },
    )

    ed = output.ensemble_decision
    assert ed.agreement is False
    assert ed.disagreement_entropy > 0.0
    assert ed.collective_decision == "Buy"  # 0.80 > 0.70
    assert ed.collective_confidence == pytest.approx(0.80 / 1.50, rel=1e-3)


# ---------------------------------------------------------------------------
# Test 4: JSON safety
# ---------------------------------------------------------------------------


def test_ensemble_output_json_safe(fixtures_data_dir, aapl_snapshot_json):
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={
            "fundamental": _stub_llm(_HOLD_RESPONSE, "fund-model"),
            "technical": _stub_llm(_HOLD_RESPONSE, "tech-model"),
        },
    )
    # Must not raise
    json.dumps(output.model_dump())


# ---------------------------------------------------------------------------
# Test 5: Phase 3 regression guard
# ---------------------------------------------------------------------------


def test_phase3_fundamental_agent_still_passes(fixtures_data_dir, aapl_snapshot_json):
    """
    Phase 3 regression guard: Fundamental Agent alone still produces valid
    FundamentalAnalysis after Phase 4 additions.
    """
    analysis = run_analysis(
        ticker="AAPL",
        as_of_date="2023-03-31",
        snapshot_json=aapl_snapshot_json,
        data_dir=fixtures_data_dir,
        _test_llm=_stub_llm(_BUY_RESPONSE, "fund-model"),
    )

    assert isinstance(analysis, FundamentalAnalysis)
    assert analysis.signal is not None
    assert analysis.signal.decision in ("Buy", "Hold", "Sell")
    assert analysis.prompt_version == "fundamental_v1"
    json.dumps(analysis.model_dump())  # JSON-safe check


# ---------------------------------------------------------------------------
# Test 6: Technical Agent time_horizon stored in TechnicalAnalysis
# ---------------------------------------------------------------------------


def test_technical_analysis_time_horizon_in_output(fixtures_data_dir, aapl_snapshot_json):
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json,
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={
            "fundamental": _stub_llm(_HOLD_RESPONSE, "fund-model"),
            "technical": _stub_llm(_SELL_RESPONSE, "tech-model"),
        },
    )

    # time_horizon from the stub JSON is "short-term"
    assert output.technical_analysis.time_horizon == "short-term"
