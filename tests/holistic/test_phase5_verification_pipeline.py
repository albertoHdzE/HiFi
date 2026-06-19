"""
Holistic test for the Phase 5 verification pipeline (P5-E6-T5, P5-E6-T6).

Uses a synthetic EnsembleOutput with known tool results and crafted rationales
so claim verification outcomes are fully predictable without running live MCP
tools or LM Studio.

What this test validates:
1. Both agent reports are produced with correct agent_type fields.
2. A known-correct claim in the fundamental rationale is marked "verified".
3. A known-incorrect claim in the technical rationale is marked "hallucinated".
4. A known-unknown alias in the fundamental rationale is marked "unresolvable".
5. EnsembleVerificationReport is JSON-safe.
6. Phase 4 regression: run_ensemble still produces valid EnsembleOutput.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import pytest

from hifi.agents.schemas import AgentSignal, FundamentalAnalysis, TechnicalAnalysis
from hifi.collective.schemas import EnsembleDecision, EnsembleOutput
from hifi.verification.verifier import verify_ensemble

# ---------------------------------------------------------------------------
# Synthetic EnsembleOutput
# ---------------------------------------------------------------------------
#
# Known tool result values:
#   financial_ratios: pe=28.3  (call_id="fundcall_001")
#   technical_indicators: rsi=42.1  (call_id="techcall_001")
#   risk_metrics: sharpe_252d=0.82  (call_id="techcall_002")
#
# Fundamental rationale claims:
#   "P/E of 28.3"           -> pe=28.3  -> VERIFIED  (call_id_cited=True)
#   "mystery_metric of 7.5" -> None     -> UNRESOLVABLE
#
# Technical rationale claims:
#   "RSI of 42.1"    -> rsi=42.1  -> VERIFIED  (call_id_cited=True)
#   "Sharpe of 5.0"  -> sharpe_252d=0.82  -> HALLUCINATED (5.0 vs 0.82)


def _make_synthetic_ensemble() -> EnsembleOutput:
    fund_signal = AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Buy",
        confidence=0.80,
        rationale=(
            "P/E of 28.3 is below its trailing average. "
            "mystery_metric of 7.5 also supports the thesis."
        ),
        key_concern="Fed funds rate of 4.75 compresses multiples.",
        call_ids=["fundcall_001"],
        model_id="test-fund-model",
        agent_type="fundamental",
    )

    fundamental_analysis = FundamentalAnalysis(
        signal=fund_signal,
        financial_ratios={"pe": 28.3, "roe": 0.24, "call_id": "fundcall_001"},
        growth_metrics={"net_margin": 0.25, "call_id": "fundcall_002"},
        valuation_context={"pe_1y_percentile": 0.6, "call_id": "fundcall_003"},
        macro_snapshot={"fed_funds_rate": 4.75, "call_id": "fundcall_004"},
        prompt_version="fundamental_v1",
        latency_ms=3000.0,
    )

    tech_signal = AgentSignal(
        ticker="AAPL",
        as_of_date="2023-03-31",
        decision="Hold",
        confidence=0.65,
        rationale=(
            "RSI of 42.1 is in neutral territory. "
            "Sharpe of 5.0 looks exceptionally strong."
        ),
        key_concern="Elevated ATR suggests increased near-term volatility.",
        call_ids=["techcall_001", "techcall_002"],
        model_id="test-tech-model",
        agent_type="technical",
    )

    technical_analysis = TechnicalAnalysis(
        signal=tech_signal,
        technical_indicators={"rsi": 42.1, "sma": 158.0, "call_id": "techcall_001"},
        risk_metrics={"sharpe_252d": 0.82, "hist_vol_252d": 0.25, "call_id": "techcall_002"},
        time_horizon="medium-term",
        prompt_version="technical_v1",
        latency_ms=4000.0,
    )

    # Buy (0.80) vs Hold (0.65) -> disagreement -> entropy > 0
    ensemble_decision = EnsembleDecision(
        collective_decision="Buy",
        collective_confidence=0.551,
        n_valid_signals=2,
        agreement=False,
        disagreement_entropy=0.955,
        opinion_dispersion=0.075,
        agent_decisions=["Buy", "Hold"],
        agent_confidences=[0.80, 0.65],
        winning_score=0.80,
        total_score=1.45,
    )

    return EnsembleOutput(
        ticker="AAPL",
        as_of_date="2023-03-31",
        fundamental_analysis=fundamental_analysis,
        technical_analysis=technical_analysis,
        ensemble_decision=ensemble_decision,
        latency_ms=7500.0,
    )


# ---------------------------------------------------------------------------
# Test 1: Both agent reports produced with correct agent_type
# ---------------------------------------------------------------------------


def test_both_reports_correct_agent_type():
    output = _make_synthetic_ensemble()
    report = verify_ensemble(output)

    assert report.fundamental_report.agent_type == "fundamental"
    assert report.technical_report.agent_type == "technical"
    assert report.ticker == "AAPL"
    assert report.as_of_date == "2023-03-31"


# ---------------------------------------------------------------------------
# Test 2: Known-correct claim (P/E of 28.3) is verified
# ---------------------------------------------------------------------------


def test_known_correct_pe_claim_is_verified():
    """
    Fundamental rationale: "P/E of 28.3". Tool: pe=28.3. call_id="fundcall_001" in signal.
    Expected: status="verified", call_id_cited=True.
    """
    output = _make_synthetic_ensemble()
    report = verify_ensemble(output)

    fr = report.fundamental_report
    pe_results = [r for r in fr.results if r.claim.canonical_field == "pe"]
    assert len(pe_results) >= 1, "P/E claim not extracted from fundamental rationale"

    pe_r = pe_results[0]
    assert pe_r.status == "verified", (
        f"P/E of 28.3 expected verified, got {pe_r.status} "
        f"(tool_value={pe_r.tool_value})"
    )
    assert pe_r.call_id_cited is True


# ---------------------------------------------------------------------------
# Test 3: Known-incorrect claim (Sharpe of 5.0 vs 0.82) is hallucinated
# ---------------------------------------------------------------------------


def test_known_wrong_sharpe_claim_is_hallucinated():
    """
    Technical rationale: "Sharpe of 5.0". Tool: sharpe_252d=0.82.
    |5.0 - 0.82| = 4.18 >> abs tol 0.01 for values <= 1.0.
    Expected: status="hallucinated".
    """
    output = _make_synthetic_ensemble()
    report = verify_ensemble(output)

    tr = report.technical_report
    sharpe_results = [r for r in tr.results if r.claim.canonical_field == "sharpe_252d"]
    assert len(sharpe_results) >= 1, "Sharpe claim not extracted from technical rationale"

    sharpe_r = sharpe_results[0]
    assert sharpe_r.status == "hallucinated", (
        f"Sharpe of 5.0 expected hallucinated, got {sharpe_r.status} "
        f"(tool_value={sharpe_r.tool_value})"
    )


# ---------------------------------------------------------------------------
# Test 4: Known-unknown alias (mystery_metric) is unresolvable
# ---------------------------------------------------------------------------


def test_unknown_alias_is_unresolvable():
    """
    Fundamental rationale: "mystery_metric of 7.5". No canonical mapping.
    Expected: status="unresolvable".
    """
    output = _make_synthetic_ensemble()
    report = verify_ensemble(output)

    fr = report.fundamental_report
    unknown = [r for r in fr.results if r.claim.canonical_field is None]
    assert len(unknown) >= 1, "No unresolvable claim found in fundamental report"

    mystery = next(
        (r for r in unknown if "mystery" in r.claim.field_alias.lower()), None
    )
    assert mystery is not None, "mystery_metric claim not found"
    assert mystery.status == "unresolvable"


# ---------------------------------------------------------------------------
# Test 5: EnsembleVerificationReport is JSON-safe
# ---------------------------------------------------------------------------


def test_ensemble_verification_report_json_safe():
    output = _make_synthetic_ensemble()
    report = verify_ensemble(output)
    dumped = json.dumps(report.model_dump())
    loaded = json.loads(dumped)
    assert loaded["ticker"] == "AAPL"
    assert "fundamental_report" in loaded
    assert "technical_report" in loaded
    assert "contradictions" in loaded


# ---------------------------------------------------------------------------
# Test 6: triggered_by_disagreement matches entropy condition
# ---------------------------------------------------------------------------


def test_triggered_by_disagreement_reflects_entropy():
    output = _make_synthetic_ensemble()
    # entropy = 0.955 > 0 -> triggered
    report = verify_ensemble(output)
    assert report.triggered_by_disagreement is True


def test_triggered_false_when_zero_entropy():
    """Manually build an output with entropy=0 and verify trigger=False."""
    output = _make_synthetic_ensemble()
    # Override ensemble_decision with entropy=0 (agreement)
    ed = output.ensemble_decision
    new_ed = EnsembleDecision(
        collective_decision=ed.collective_decision,
        collective_confidence=ed.collective_confidence,
        n_valid_signals=ed.n_valid_signals,
        agreement=True,
        disagreement_entropy=0.0,
        opinion_dispersion=0.0,
        agent_decisions=ed.agent_decisions,
        agent_confidences=ed.agent_confidences,
        winning_score=ed.winning_score,
        total_score=ed.total_score,
    )
    output_no_disagreement = EnsembleOutput(
        ticker=output.ticker,
        as_of_date=output.as_of_date,
        fundamental_analysis=output.fundamental_analysis,
        technical_analysis=output.technical_analysis,
        ensemble_decision=new_ed,
        latency_ms=output.latency_ms,
    )
    report = verify_ensemble(output_no_disagreement)
    assert report.triggered_by_disagreement is False


# ---------------------------------------------------------------------------
# Test 7 (P5-E6-T6): Phase 4 regression -- run_ensemble still produces valid EnsembleOutput
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_data_dir(tmp_path):
    fixtures_root = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    for subdir in ("market", "macro"):
        dst = tmp_path / subdir
        dst.mkdir()
        src = os.path.join(fixtures_root, subdir)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f.endswith(".parquet"):
                    shutil.copy(os.path.join(src, f), dst / f)
    return str(tmp_path)


def test_phase4_run_ensemble_regression(fixtures_data_dir):
    """Phase 4 regression: run_ensemble still returns valid EnsembleOutput."""
    from hifi.agents.ensemble_runner import run_ensemble
    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord

    _hold = json.dumps({
        "decision": "Hold",
        "confidence": 0.60,
        "rationale": "Market appears fairly balanced at current levels.",
        "key_concern": "Rate sensitivity remains a concern.",
        "time_horizon": "medium-term",
    })

    def _stub(name):
        class _S:
            def invoke(self, _):
                class _R:
                    content = _hold
                return _R()
        s = _S()
        s.model_name = name
        return s

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

    output = run_ensemble(
        "AAPL", "2023-03-31", snap.model_dump_json(),
        fixtures_data_dir, agents=["fundamental", "technical"],
        _test_llms={"fundamental": _stub("fund-model"), "technical": _stub("tech-model")},
    )

    assert isinstance(output, EnsembleOutput)
    assert output.ticker == "AAPL"
    assert output.ensemble_decision.n_valid_signals >= 0
    # Must be JSON-safe
    json.dumps(output.model_dump())
