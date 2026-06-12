"""
Holistic test for the Phase 9 Collective Decision Engine (P9-E5).

Validates:
1. All four aggregation methods run on every ensemble call
2. method_comparison populated with correct keys and semantics
3. contrarian_adjusted carries discount < 1.0 (contrarian ran)
4. signals contains only non-None AgentSignals
5. aggregation_method == "confidence_weighted"
6. EnsembleOutput JSON round-trip is lossless
7. Backward compat: agents=["fundamental","technical"] still works; 4-key comparison

No live LM Studio required. LLMs are monkeypatched with stub objects.
Sentinel fail-open for sentiment (empty passages). Parquet fixtures from Phase 1.

Contrarian stub uses confidence=0.65:
  - discount = 1 - 0.5*0.65 = 0.675
  - review_flagged = 0.65 > 0.70 → False
"""

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.agents.ensemble_runner import run_ensemble
from hifi.collective.schemas import EnsembleOutput
from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_data_dir(tmp_path):
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


# ---------------------------------------------------------------------------
# Stub LLM responses
# ---------------------------------------------------------------------------

_FUND_RESPONSE = json.dumps({
    "decision": "Buy",
    "confidence": 0.80,
    "rationale": "P/E of 28.3 is below sector average; strong FCF generation.",
    "key_concern": "Fed funds at 4.75 compresses growth multiples.",
})

_TECH_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.65,
    "rationale": "RSI of 48 is neutral; MACD histogram near zero.",
    "key_concern": "ATR elevated, short-term volatility risk.",
    "time_horizon": "short-term",
})

_RISK_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "hist_vol_20d of 0.22 is moderate; Sharpe 0.82 acceptable.",
    "key_concern": "max_drawdown_252d of -0.28 is a tail risk.",
    "risk_assessment": "Moderate volatility regime.",
    "recommended_position_size": 0.05,
})

_MACRO_RESPONSE = json.dumps({
    "decision": "Buy",
    "confidence": 0.55,
    "rationale": "Fed near rate peak; CPI decelerating.",
    "key_concern": "Unemployment uptick could signal slowdown.",
    "regime_assessment": "Late tightening cycle with easing bias.",
    "macro_rationale": "CPI deceleration supports risk recovery.",
})

_CONTRARIAN_RESPONSE = json.dumps({
    "alternative_thesis": "Rate cuts are fully priced; real rates remain restrictive.",
    "risk_scenario": "Credit spread widening triggers 15% equity correction.",
    "counterargument": "Consensus underestimates duration of high-rate environment.",
    "confidence": 0.65,   # discount = 0.675; review_flagged = False
})


def _stub_llm(content: str, model_name: str = "stub-model"):
    class _Stub:
        def invoke(self, _messages):
            class _R:
                pass
            r = _R()
            r.content = content
            return r
    s = _Stub()
    s.model_name = model_name
    return s


@pytest.fixture
def patched_llms(monkeypatch):
    """
    Patch make_llm in all agent modules (fund, tech, risk, macro, contrarian).
    Patch sentiment.call_tool to return empty passages → fail-open Hold/0.0.
    """
    import hifi.agents.contrarian_agent as ca
    import hifi.agents.fundamental_agent as fa
    import hifi.agents.macro_agent as ma
    import hifi.agents.risk_agent as ra
    import hifi.agents.sentiment_agent as sa_mod
    import hifi.agents.technical_agent as ta

    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _stub_llm(_FUND_RESPONSE))
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _stub_llm(_TECH_RESPONSE))
    monkeypatch.setattr(ra, "make_llm", lambda *a, **kw: _stub_llm(_RISK_RESPONSE))
    monkeypatch.setattr(ma, "make_llm", lambda *a, **kw: _stub_llm(_MACRO_RESPONSE))
    monkeypatch.setattr(ca, "make_llm", lambda *a, **kw: _stub_llm(_CONTRARIAN_RESPONSE))
    monkeypatch.setattr(sa_mod, "call_tool", lambda *a, **kw: {"passages": []})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_method_comparison_has_four_keys(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """All four aggregation methods run on every run_ensemble() call."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    assert set(output.method_comparison.keys()) == {
        "majority",
        "confidence_weighted",
        "performance_weighted",
        "contrarian_adjusted",
    }


def test_cw_method_equals_ensemble_decision(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """method_comparison['confidence_weighted'] matches ensemble_decision exactly."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    cw = output.method_comparison["confidence_weighted"]
    ed = output.ensemble_decision
    assert cw.collective_decision == ed.collective_decision
    assert cw.collective_confidence == ed.collective_confidence
    assert cw.n_valid_signals == ed.n_valid_signals
    assert cw.agreement == ed.agreement


def test_signals_non_empty_and_all_valid(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """EnsembleOutput.signals contains only non-None AgentSignals."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    assert len(output.signals) > 0
    for sig in output.signals:
        assert sig is not None
        assert sig.decision in {"Buy", "Hold", "Sell"}
        assert 0.0 <= sig.confidence <= 1.0


def test_aggregation_method_is_confidence_weighted(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)
    assert output.aggregation_method == "confidence_weighted"


def test_contrarian_adjusted_has_discount_below_one(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """When contrarian ran (confidence=0.65), discount = 0.675 < 1.0."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    ca_decision = output.method_comparison["contrarian_adjusted"]
    assert ca_decision.contrarian_confidence_discount < 1.0
    assert ca_decision.contrarian_confidence_discount == pytest.approx(
        1.0 - 0.5 * 0.65
    )


def test_contrarian_adjusted_review_not_flagged_at_065(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """contrarian.confidence = 0.65 < 0.70 → review_flagged = False."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    ca_decision = output.method_comparison["contrarian_adjusted"]
    assert ca_decision.review_flagged is False


def test_contrarian_adjusted_direction_unchanged(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """Contrarian discount never changes the winning direction."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    cw = output.method_comparison["confidence_weighted"]
    ca = output.method_comparison["contrarian_adjusted"]
    assert ca.collective_decision == cw.collective_decision


def test_all_method_decisions_are_valid_options(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """All four methods produce Buy/Hold/Sell (or None if no signals)."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    for method_name, decision in output.method_comparison.items():
        assert decision.collective_decision in {"Buy", "Hold", "Sell", None}, (
            f"{method_name} produced unexpected decision: {decision.collective_decision}"
        )


def test_ensemble_output_json_roundtrip_lossless(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """model_dump_json → model_validate_json is lossless with all Phase 9 fields."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    restored = EnsembleOutput.model_validate_json(output.model_dump_json())

    assert restored.ticker == output.ticker
    assert restored.aggregation_method == output.aggregation_method
    assert len(restored.signals) == len(output.signals)
    assert set(restored.method_comparison.keys()) == set(output.method_comparison.keys())
    assert (
        restored.method_comparison["confidence_weighted"].collective_decision
        == output.method_comparison["confidence_weighted"].collective_decision
    )


def test_backward_compat_two_agent_subset(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """agents=['fundamental','technical']: method_comparison still has 4 keys."""
    output = run_ensemble(
        "AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir,
        agents=["fundamental", "technical"],
    )

    assert isinstance(output, EnsembleOutput)
    assert len(output.method_comparison) == 4
    assert output.risk_analysis is None
    assert output.macro_analysis is None


def test_sentiment_fail_open_still_produces_method_comparison(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """Sentiment fail-open (Hold/0.0) still contributes to method_comparison."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    sa = output.sentiment_analysis
    assert sa is not None
    assert sa.signal is not None
    assert sa.signal.decision == "Hold"
    assert sa.signal.confidence == pytest.approx(0.0)
    # method_comparison still has 4 keys despite one agent having zero confidence
    assert len(output.method_comparison) == 4


def test_performance_weighted_key_present_and_valid(
    patched_llms, fixtures_data_dir, aapl_snapshot_json
):
    """performance_weighted runs with uniform fallback weights (no history file)."""
    output = run_ensemble("AAPL", "2023-03-31", aapl_snapshot_json, fixtures_data_dir)

    pw = output.method_comparison["performance_weighted"]
    assert pw.collective_decision in {"Buy", "Hold", "Sell", None}
    assert 0.0 <= pw.collective_confidence <= 1.0
