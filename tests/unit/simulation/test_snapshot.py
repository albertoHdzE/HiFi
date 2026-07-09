"""
Unit tests for hifi.simulation.snapshot.build_minimal_snapshot (Phase 15, DJ-097).

Covers:
- JSON validity and field presence for representative tickers
- All financial fields are None
- Source and provenance fields are set correctly
- Round-trip FundamentalsSnapshot deserialization
- Parameterized over all 98 PHASE14_UNIVERSE tickers
"""

from __future__ import annotations

import json

import pytest

from hifi.data.universe import PHASE14_UNIVERSE
from hifi.simulation.snapshot import build_minimal_snapshot

_UNIVERSE_TICKERS = [entry["ticker"] for entry in PHASE14_UNIVERSE]
_FINANCIAL_FIELDS = [
    "revenue", "net_income", "total_assets",
    "total_liabilities", "total_equity", "eps", "pe_ratio", "market_cap",
]

# ---------------------------------------------------------------------------
# Representative spot checks
# ---------------------------------------------------------------------------


def test_valid_json_aapl():
    raw = build_minimal_snapshot("AAPL", "2022-01-31")
    data = json.loads(raw)
    assert data["ticker"] == "AAPL"
    assert data["period_end"] == "2022-01-31"


def test_valid_json_jpm():
    raw = build_minimal_snapshot("JPM", "2022-06-30")
    data = json.loads(raw)
    assert data["ticker"] == "JPM"
    assert data["period_end"] == "2022-06-30"


def test_financial_fields_are_null():
    data = json.loads(build_minimal_snapshot("XOM", "2023-03-31"))
    for field in _FINANCIAL_FIELDS:
        assert data[field] is None, f"Expected {field}=None, got {data[field]!r}"


def test_source_is_walk_forward_eval():
    data = json.loads(build_minimal_snapshot("NVDA", "2022-06-30"))
    assert data["source"] == "walk_forward_eval"


def test_provenance_fields():
    data = json.loads(build_minimal_snapshot("BAC", "2023-03-31"))
    prov = data["provenance"]
    assert prov["source"] == "walk_forward_eval"
    assert prov["parameters"]["ticker"] == "BAC"
    assert prov["parameters"]["as_of_date"] == "2023-03-31"


def test_round_trip_deserialization():
    """JSON → FundamentalsSnapshot model must validate without error."""
    from hifi.data.schemas import FundamentalsSnapshot

    raw = build_minimal_snapshot("MSFT", "2022-12-31")
    snap = FundamentalsSnapshot.model_validate_json(raw)
    assert snap.ticker == "MSFT"
    assert snap.revenue is None
    assert snap.source == "walk_forward_eval"


def test_fetched_at_is_present():
    data = json.loads(build_minimal_snapshot("GOOG", "2022-09-30"))
    assert "fetched_at" in data and data["fetched_at"] is not None


# ---------------------------------------------------------------------------
# Parameterized over all 98 universe tickers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", _UNIVERSE_TICKERS)
def test_build_minimal_snapshot_all_tickers(ticker):
    """
    build_minimal_snapshot must return valid JSON for every ticker in PHASE14_UNIVERSE.

    Evaluation dates sampled to cover three distinct regimes:
      2022-01-31 — rate-shock start
      2022-06-30 — mid rate-shock
      2023-03-31 — end of held-out test
    """
    for date in ("2022-01-31", "2022-06-30", "2023-03-31"):
        raw = build_minimal_snapshot(ticker, date)
        data = json.loads(raw)
        assert data["ticker"] == ticker
        assert data["period_end"] == date
        for field in _FINANCIAL_FIELDS:
            assert data[field] is None, (
                f"{ticker}/{date}: expected {field}=None, got {data[field]!r}"
            )
