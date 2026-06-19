"""
Integration tests for Phase 15 walk-forward harness (DJ-097).

Tests cover non-LLM components:
- build_minimal_snapshot: valid JSON, correct ticker/date, None financials
- output_path: directory structure matches expected pattern
- count_completed: counts existing output files correctly
- run_one with _test_llms: full dispatch without real LLM (uses stub fixture)
- scan_walkforward_dir: reads and parses walkforward JSON files
- compute_condition_ic: IC from synthetic walkforward records
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.run_phase15_walkforward import count_completed, output_path

from hifi.simulation.metrics import buy_strength, compute_ic
from hifi.simulation.schedule import WalkForwardPeriod, get_period_dates
from hifi.simulation.snapshot import build_minimal_snapshot

# ---------------------------------------------------------------------------
# build_minimal_snapshot
# ---------------------------------------------------------------------------


def test_build_minimal_snapshot_is_valid_json():
    result = build_minimal_snapshot("AAPL", "2022-01-31")
    data = json.loads(result)
    assert data["ticker"] == "AAPL"
    assert data["period_end"] == "2022-01-31"


def test_build_minimal_snapshot_financials_are_null():
    data = json.loads(build_minimal_snapshot("JPM", "2022-06-30"))
    for field in ["revenue", "net_income", "total_assets",
                  "total_liabilities", "total_equity", "eps", "pe_ratio", "market_cap"]:
        assert data[field] is None, f"Expected {field} to be None"


def test_build_minimal_snapshot_source():
    data = json.loads(build_minimal_snapshot("XOM", "2023-03-31"))
    assert data["source"] == "walk_forward_eval"


def test_build_minimal_snapshot_deserializes_to_model():
    """Round-trip: JSON → FundamentalsSnapshot model validation."""
    from hifi.data.schemas import FundamentalsSnapshot

    raw = build_minimal_snapshot("MSFT", "2022-12-31")
    snap = FundamentalsSnapshot.model_validate_json(raw)
    assert snap.ticker == "MSFT"
    assert snap.revenue is None


# ---------------------------------------------------------------------------
# output_path
# ---------------------------------------------------------------------------


def test_output_path_structure():
    path = output_path("data/walkforward", "full", "2022-01-31", "AAPL")
    assert path == Path("data/walkforward/full/2022/01/AAPL.json")


def test_output_path_held_out_test():
    path = output_path("data/walkforward", "no-memory", "2023-12-31", "JPM")
    assert path == Path("data/walkforward/no-memory/2023/12/JPM.json")


def test_output_path_different_conditions():
    for cond in ["full", "parallel", "homogeneous", "no-memory"]:
        p = output_path("data/walkforward", cond, "2022-06-30", "AAPL")
        assert cond in str(p)


# ---------------------------------------------------------------------------
# count_completed
# ---------------------------------------------------------------------------


def test_count_completed_empty(tmp_path):
    assert count_completed(str(tmp_path), "full", ["AAPL", "JPM"], ["2022-01-31"]) == 0


def test_count_completed_some_files(tmp_path):
    """Two files present → count = 2."""
    for ticker in ["AAPL", "JPM"]:
        p = tmp_path / "full" / "2022" / "01" / f"{ticker}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")

    count = count_completed(
        str(tmp_path), "full",
        tickers=["AAPL", "JPM", "XOM"],
        dates=["2022-01-31"],
    )
    assert count == 2


def test_count_completed_all_files(tmp_path):
    tickers = ["AAPL", "JPM"]
    dates = ["2022-01-31", "2022-02-28"]
    for d in dates:
        year, month, _ = d.split("-")
        for t in tickers:
            p = tmp_path / "full" / year / month / f"{t}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")

    assert count_completed(str(tmp_path), "full", tickers, dates) == 4


# ---------------------------------------------------------------------------
# run_one with stub LLMs (checkpoint + dispatch)
# ---------------------------------------------------------------------------


def _make_stub_llm(decision: str = "Buy", confidence: float = 0.75):
    """Return a fake LLM object that produces a deterministic JSON response."""
    import json as _json
    from unittest.mock import MagicMock

    response_json = _json.dumps({
        "decision": decision,
        "confidence": confidence,
        "rationale": "Stub rationale for test.",
        "key_concern": "None",
        "pe_ratio": None,
        "revenue_growth": None,
        "notable_signals": [],
    })

    mock_llm = MagicMock()
    # model_name must be a real string — agents embed it in AgentSignal.model_id
    mock_llm.model_name = "stub-test-model"
    mock_response = MagicMock()
    mock_response.content = response_json
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def _stub_llms(decision: str = "Buy") -> dict:
    stub = _make_stub_llm(decision)
    return {
        "fundamental": stub,
        "technical": stub,
        "risk": stub,
        "macro": stub,
        "sentiment": stub,
        "contrarian": stub,
    }


@pytest.fixture()
def tmp_walkforward(tmp_path):
    """Fixture providing a temporary output directory."""
    return str(tmp_path / "walkforward")


def test_run_one_creates_output_file(tmp_walkforward, tmp_path):
    """run_one with stub LLMs creates the output JSON file."""
    from scripts.run_phase15_walkforward import run_one

    out = run_one(
        ticker="AAPL",
        date="2022-01-31",
        condition="no-memory",
        data_dir=str(tmp_path),
        output_dir=tmp_walkforward,
        _test_llms=_stub_llms(),
    )
    assert out is not None
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["ticker"] == "AAPL"
    assert data["as_of_date"] == "2022-01-31"


def test_run_one_checkpoint_skips_existing(tmp_walkforward, tmp_path):
    """run_one skips if output file already exists."""
    from scripts.run_phase15_walkforward import run_one

    # Create a fake existing output
    out = output_path(tmp_walkforward, "full", "2022-01-31", "AAPL")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"ticker": "AAPL", "as_of_date": "2022-01-31"}', encoding="utf-8")

    result = run_one(
        ticker="AAPL",
        date="2022-01-31",
        condition="full",
        data_dir=str(tmp_path),
        output_dir=tmp_walkforward,
        _test_llms={},  # no LLMs needed — must skip
    )
    # Returns the existing path without error
    assert result == out


def test_run_one_parallel_condition(tmp_walkforward, tmp_path):
    """Parallel condition also creates output."""
    from scripts.run_phase15_walkforward import run_one

    out = run_one(
        ticker="JPM",
        date="2022-02-28",
        condition="parallel",
        data_dir=str(tmp_path),
        output_dir=tmp_walkforward,
        _test_llms=_stub_llms("Hold"),
    )
    assert out is not None and out.exists()


def test_run_one_no_memory_condition(tmp_walkforward, tmp_path):
    from scripts.run_phase15_walkforward import run_one

    out = run_one(
        ticker="XOM",
        date="2022-03-31",
        condition="no-memory",
        data_dir=str(tmp_path),
        output_dir=tmp_walkforward,
        _test_llms=_stub_llms("Sell"),
    )
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# scan_walkforward_dir
# ---------------------------------------------------------------------------


def _write_ensemble_output(path: Path, ticker: str, date: str, decision: str = "Buy") -> None:
    """Write a minimal EnsembleOutput JSON for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ticker": ticker,
        "as_of_date": date,
        "ensemble_decision": {
            "collective_decision": decision,
            "collective_confidence": 0.75,
            "agreement": True,
            "n_valid_signals": 5,
            "disagreement_entropy": 0.0,
            "opinion_dispersion": 0.0,
            "agent_decisions": [decision] * 5,
            "agent_confidences": [0.75] * 5,
            "winning_score": 3.75,
            "total_score": 3.75,
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_scan_walkforward_dir_finds_files(tmp_path):
    from scripts.compute_phase15_ic import scan_walkforward_dir

    out_dir = str(tmp_path)
    p = tmp_path / "full" / "2022" / "01" / "AAPL.json"
    _write_ensemble_output(p, "AAPL", "2022-01-31", "Buy")

    records = scan_walkforward_dir(out_dir, "full")
    assert len(records) == 1
    assert records[0]["ticker"] == "AAPL"
    assert records[0]["date"] == "2022-01-31"
    assert records[0]["condition"] == "full"


def test_scan_walkforward_dir_date_filter(tmp_path):
    from scripts.compute_phase15_ic import scan_walkforward_dir

    out_dir = str(tmp_path)
    for month, decision in [("01", "Buy"), ("02", "Sell")]:
        p = tmp_path / "full" / "2022" / month / "AAPL.json"
        d = f"2022-{month}-{'31' if month == '01' else '28'}"
        _write_ensemble_output(p, "AAPL", d, decision)

    records = scan_walkforward_dir(out_dir, "full", dates=["2022-01-31"])
    assert len(records) == 1
    assert records[0]["date"] == "2022-01-31"


def test_scan_walkforward_dir_empty_when_no_files(tmp_path):
    from scripts.compute_phase15_ic import scan_walkforward_dir

    records = scan_walkforward_dir(str(tmp_path), "full")
    assert records == []


# ---------------------------------------------------------------------------
# buy_strength from real EnsembleOutput JSON
# ---------------------------------------------------------------------------


def test_buy_strength_from_walkforward_json(tmp_walkforward, tmp_path):
    """buy_strength correctly extracts from a real run_one output."""
    from scripts.run_phase15_walkforward import run_one

    out = run_one(
        ticker="AAPL",
        date="2022-01-31",
        condition="no-memory",
        data_dir=str(tmp_path),
        output_dir=tmp_walkforward,
        _test_llms=_stub_llms("Buy"),
    )
    assert out is not None
    data = json.loads(out.read_text())
    strength = buy_strength(data)
    # Stub LLM returns Buy with confidence 0.75 → buy_strength = 0.75
    assert strength is not None
    assert strength > 0.0


# ---------------------------------------------------------------------------
# IC computation from synthetic data
# ---------------------------------------------------------------------------


def test_ic_from_synthetic_signals():
    """End-to-end IC computation with known signals."""
    # Perfect positive rank correlation
    signals = [0.9, 0.7, 0.5, 0.3, 0.1]
    returns = [0.09, 0.07, 0.05, 0.03, 0.01]
    result = compute_ic(signals, returns)
    assert result.ic == pytest.approx(1.0)
    assert result.n_pairs == 5


def test_period_dates_count_held_out():
    """Held-out test period has exactly 24 dates."""
    dates = get_period_dates(WalkForwardPeriod.HELD_OUT_TEST)
    assert len(dates) == 24
    assert dates[0] == "2022-01-31"
    assert dates[-1] == "2023-12-31"
