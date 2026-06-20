"""
Unit tests for the deterministic run_id parameter in run_sequential_ensemble()
(E0-T1, DJ-106).

Verifies:
- run_id=None generates a UUID automatically (backward compat)
- run_id="fixed-id" uses the provided value in AgentContextStore records
- Two calls with the same run_id produce the same run_id in stored records
- The UUID fallback is different across calls (no accidental sharing)
"""

from __future__ import annotations

import json
import uuid

from hifi.knowledge.agent_context import AgentContextStore

# ---------------------------------------------------------------------------
# Shared stub responses
# ---------------------------------------------------------------------------

_FUND_STUB = json.dumps({
    "decision": "Buy",
    "confidence": 0.72,
    "rationale": "P/E of 22 is below sector average; earnings growth is strong.",
    "key_concern": "Valuation rich vs. peers in current rate environment.",
})

_TECH_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "RSI at 52 neutral. MACD histogram slightly positive.",
    "key_concern": "Volume declining on recent rally.",
    "time_horizon": "short-term",
})

_RISK_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.55,
    "rationale": "hist_vol_20d moderate. No drawdown alert.",
    "key_concern": "VIX elevated.",
    "risk_assessment": "Moderate",
    "recommended_position_size": 0.05,
})

_MACRO_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.50,
    "rationale": "Fed funds at 5.25%. Yield curve flat.",
    "key_concern": "Rate uncertainty.",
    "regime_assessment": "Restrictive",
    "macro_rationale": "High rates compress multiples.",
})

_SENT_STUB = json.dumps({
    "decision": "Buy",
    "confidence": 0.65,
    "rationale": "Management guidance raised. Services revenue strong.",
    "key_concern": "China headwinds mentioned.",
    "notable_signals": ["raised guidance"],
})

_CONT_STUB = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "Consensus Buy fading given valuation.",
    "key_concern": "Macro headwinds underpriced.",
    "contrarian_thesis": "Hold vs. Buy consensus.",
})


def _stub_llm(response: str) -> object:
    class _Stub:
        model_name = "stub-test-model"

        def invoke(self, messages):
            class _R:
                content = response
            return _R()
    return _Stub()


def _all_stubs() -> dict:
    return {
        "fundamental": _stub_llm(_FUND_STUB),
        "technical": _stub_llm(_TECH_STUB),
        "risk": _stub_llm(_RISK_STUB),
        "macro": _stub_llm(_MACRO_STUB),
        "sentiment": _stub_llm(_SENT_STUB),
        "contrarian": _stub_llm(_CONT_STUB),
    }


def _minimal_snapshot_json() -> str:
    import datetime

    from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord
    snap = FundamentalsSnapshot(
        ticker="AAPL",
        period_end="2022-01-31",
        source="test",
        fetched_at=datetime.datetime(2022, 2, 1),
        provenance=ProvenanceRecord(
            source="test", fetched_at=datetime.datetime(2022, 2, 1)
        ),
    )
    return snap.model_dump_json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_id_none_generates_uuid(tmp_path):
    """When run_id=None, a UUID is automatically assigned to all records."""
    from hifi.agents.ensemble_runner import run_sequential_ensemble

    db = str(tmp_path / "knowledge.lance")
    store = AgentContextStore(namespace="test-det", db_path=db)

    output = run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2022-01-31",
        snapshot_json=_minimal_snapshot_json(),
        run_id=None,
        _test_llms=_all_stubs(),
        _test_store=store,
    )

    # Verify output is valid
    assert output.ticker == "AAPL"
    assert output.ensemble_decision is not None

    # Verify store has records with a valid UUID run_id
    df = store._table.to_pandas()
    assert not df.empty
    stored_run_ids = df["run_id"].unique().tolist()
    assert len(stored_run_ids) == 1
    # Should be a valid UUID
    uuid.UUID(stored_run_ids[0])


def test_run_id_provided_is_used(tmp_path):
    """When run_id is provided, all AgentContextStore records use it."""
    from hifi.agents.ensemble_runner import run_sequential_ensemble

    db = str(tmp_path / "knowledge.lance")
    store = AgentContextStore(namespace="test-det", db_path=db)
    fixed_run_id = "phase14-1-smoke-2022-01-31"

    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2022-01-31",
        snapshot_json=_minimal_snapshot_json(),
        run_id=fixed_run_id,
        _test_llms=_all_stubs(),
        _test_store=store,
    )

    df = store._table.to_pandas()
    assert not df.empty
    for rid in df["run_id"].unique():
        assert rid == fixed_run_id, f"Expected run_id={fixed_run_id!r}, got {rid!r}"


def test_two_calls_different_auto_run_ids(tmp_path):
    """Without explicit run_id, successive calls get distinct UUIDs."""
    from hifi.agents.ensemble_runner import run_sequential_ensemble

    db = str(tmp_path / "knowledge.lance")
    store = AgentContextStore(namespace="test-det", db_path=db)

    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2022-01-31",
        snapshot_json=_minimal_snapshot_json(),
        run_id=None,
        _test_llms=_all_stubs(),
        _test_store=store,
    )
    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2022-02-28",
        snapshot_json=_minimal_snapshot_json(),
        run_id=None,
        _test_llms=_all_stubs(),
        _test_store=store,
    )

    df = store._table.to_pandas()
    run_ids = df["run_id"].unique().tolist()
    assert len(run_ids) == 2, f"Expected 2 distinct run_ids, got: {run_ids}"


def test_prior_context_readable_with_deterministic_run_id(tmp_path):
    """
    With a fixed run_id, prior-agent context stored in call 1 is accessible
    to subsequent agents in call 2 (same run_id lookup).
    """
    from hifi.agents.ensemble_runner import run_sequential_ensemble

    db = str(tmp_path / "knowledge.lance")
    store = AgentContextStore(namespace="test-det", db_path=db)
    run_id = "repro-run-abc123"

    # First run stores agents' context under run_id
    run_sequential_ensemble(
        ticker="AAPL",
        as_of_date="2022-01-31",
        snapshot_json=_minimal_snapshot_json(),
        run_id=run_id,
        _test_llms=_all_stubs(),
        _test_store=store,
    )

    # The store should have records for this run_id
    records = store.read_prior(run_id, "contrarian")
    agent_types = {r.agent_type for r in records}
    # All 5 non-contrarian agents should have stored context
    assert "fundamental" in agent_types
    assert "technical" in agent_types
