"""Phase 20 branch: engine-computed market summaries (scientific-rigor pins).

Every expectation below is hand-derived from the construction of the input
series. The misaligned-VaR cases are the important ones: they encode the
C15 correction — tails must align on DATES, never on position, and a name
without data must renormalize weights, never silently halve the estimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hifi.engines import market_summary as ms  # noqa: E402


def _write_close(tmp_path, ticker, dates, closes):
    d = tmp_path / "market" / ticker
    d.mkdir(parents=True, exist_ok=True)
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    pd.DataFrame({"Close": closes, "Volume": 1.0}, index=idx).to_parquet(
        d / "ohlcv.parquet")


def _business_days(end, periods):
    return pd.bdate_range(end=end, periods=periods)


# ---------------------------------------------------------------------------
# n-session return
# ---------------------------------------------------------------------------


class TestNSessionReturn:
    def test_exact_simple_return(self, tmp_path):
        dates = _business_days("2026-08-21", 25)
        closes = [100.0] + [100.0] * 23 + [110.0]
        _write_close(tmp_path, "X", dates, closes)
        s = ms._load_close(str(tmp_path), "X")
        r = ms._n_session_return(s, pd.Timestamp("2026-08-21"), 20)
        # close(t) / close(t-20) - 1 = 110/100 - 1
        assert r == pytest.approx(0.10)

    def test_insufficient_history_none(self, tmp_path):
        dates = _business_days("2026-08-21", 10)
        _write_close(tmp_path, "X", dates, [100.0] * 10)
        s = ms._load_close(str(tmp_path), "X")
        assert ms._n_session_return(s, pd.Timestamp("2026-08-21"), 20) is None

    def test_no_lookahead_beyond_as_of(self, tmp_path):
        """Bars AFTER as_of must not participate, even if present."""
        dates = list(_business_days("2026-08-21", 25)) + list(
            _business_days("2026-09-30", 3))
        closes = [100.0] + [100.0] * 23 + [110.0] + [500.0, 500.0, 500.0]
        _write_close(tmp_path, "X", dates, closes)
        s = ms._load_close(str(tmp_path), "X")
        r = ms._n_session_return(s, pd.Timestamp("2026-08-21"), 20)
        assert r == pytest.approx(0.10)  # the 500s after as_of are invisible


# ---------------------------------------------------------------------------
# Relative strength vs sector median
# ---------------------------------------------------------------------------


class TestRelativeStrength:
    def test_outperformer_positive_delta(self, tmp_path):
        dates = _business_days("2026-08-21", 22)
        _write_close(tmp_path, "AAA", dates, [100.0] + [100.0] * 20 + [120.0])
        for _i, t in enumerate(("P1", "P2", "P3")):
            _write_close(tmp_path, t, dates, [100.0] + [100.0] * 20 + [100.0])

        from hifi.data.universe import PHASE14_UNIVERSE  # noqa: PLC0415
        PHASE14_UNIVERSE.clear()
        PHASE14_UNIVERSE.extend([
            {"ticker": "AAA", "sector": "Tech"},
            {"ticker": "P1", "sector": "Tech"},
            {"ticker": "P2", "sector": "Tech"},
            {"ticker": "P3", "sector": "Tech"},
        ])
        rel = ms.relative_strength("AAA", None, "2026-08-21", str(tmp_path))
        assert rel["ticker_return"] == pytest.approx(0.20)
        assert rel["peer_median"] == pytest.approx(0.0)
        assert rel["delta_pp"] == pytest.approx(20.0)

    def test_fewer_than_min_peers_is_none(self, tmp_path):
        dates = _business_days("2026-08-21", 22)
        _write_close(tmp_path, "LONELY", dates, [100.0] * 22)
        from hifi.data.universe import PHASE14_UNIVERSE  # noqa: PLC0415
        PHASE14_UNIVERSE.clear()
        PHASE14_UNIVERSE.extend([{"ticker": "LONELY", "sector": "Tech"}])
        assert ms.relative_strength("LONELY", None, "2026-08-21",
                                    str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Book VaR — the C15-correction cases
# ---------------------------------------------------------------------------


class TestBookVarAlignment:
    def test_misaligned_histories_align_on_dates_not_position(self, tmp_path):
        """B crashes in its last 5 sessions; A's series ENDS EARLIER (delisted).

        Positional splicing (the old flaw) would pair A's calm old tail with
        B's crash tail at 50/50 and dilute the estimate. Date intersection
        keeps only sessions both lived through — here the pre-crash stretch —
        so this test pins the INTERSECTION behaviour; the crash-tail coverage
        case is the next test.
        """
        common_dates = list(_business_days("2026-06-30", 40))
        a_dates = common_dates[:35]                      # A delists early
        _write_close(tmp_path, "A", a_dates, [100.0] * len(a_dates))
        b_closes = [100.0] * 35 + [90.0, 90.0, 90.0, 90.0, 90.0]
        _write_close(tmp_path, "B", common_dates, b_closes)

        out = ms.book_var_95({"A": .5, "B": .5}, "2026-06-30", str(tmp_path),
                             window=60)
        # Intersection = first ~34 return-sessions, all flat → VaR ≈ 0,
        # NOT a diluted blend of a crash that never overlapped A.
        assert out is not None
        assert out["var_95_1d"] < 1e-9
        assert set(out["covered_names"]) == {"A", "B"}

    def test_crash_in_overlap_drives_var_with_renormalized_weight(self, tmp_path):
        """A has no data at all → B inherits FULL weight (old code computed
        VaR on half the book).

        Fixture arithmetic: the last 6 closes decline geometrically
        100 → 77, producing 5 consecutive equal returns of
        (77/100)^(1/5) − 1 ≈ −5.117%. Inside the 61-return estimation tail,
        those 5 occupy the bottom 8.2% of observations, so the 5th percentile
        lands squarely on a crash return — hand-derived, not tuned."""
        dates = _business_days("2026-06-30", 70)
        b_closes = [100.0] * 64 + list(np.geomspace(100.0, 77.0, 6))
        _write_close(tmp_path, "B", dates, b_closes)

        out = ms.book_var_95({"GHOST": .5, "B": .5}, "2026-06-30",
                             str(tmp_path), window=60)
        assert out["var_95_1d"] is not None
        assert set(out["covered_names"]) == {"B"}
        assert set(out["renormalized_from"]) == {"GHOST", "B"}
        expected = (77.0 / 100.0) ** (1 / 5) - 1.0
        assert out["var_95_1d"] == pytest.approx(-expected, abs=1e-3)

    def test_insufficient_aligned_sessions_reported_not_guessed(self, tmp_path):
        dates = _business_days("2026-08-21", 10)
        _write_close(tmp_path, "A", dates, [100.0] * 10)
        out = ms.book_var_95({"A": 1.0}, "2026-08-21", str(tmp_path))
        assert out["var_95_1d"] is None
        assert "insufficient aligned history" in out["reason"]

    def test_zero_and_negative_weights_excluded(self, tmp_path):
        dates = _business_days("2026-06-30", 70)
        _write_close(tmp_path, "B", dates, [100.0] * 70)
        out = ms.book_var_95({"DEAD": 0.0, "B": 1.0}, "2026-06-30", str(tmp_path))
        assert out is not None
        assert out["covered_names"] == ["B"]


# ---------------------------------------------------------------------------
# Regime wiring (live paths)
# ---------------------------------------------------------------------------


class TestRegimeSnapshot:
    def _spy_macro(self, tmp_path, spy_last_close=450.0):
        dates = _business_days("2026-08-24", 520)
        base = 400.0
        closes = np.linspace(base, spy_last_close, len(dates))
        _write_close(tmp_path, "SPY", dates, closes.tolist())
        mdates = _business_days("2026-08-24", 520)
        fed = pd.DataFrame({
            "date": mdates,
            "value": [5.33] * len(mdates),
        })
        vix = pd.DataFrame({"date": mdates, "value": [14.0] * len(mdates)})
        (tmp_path / "macro").mkdir(exist_ok=True)
        fed.to_parquet(tmp_path / "macro" / "FEDFUNDS.parquet")
        vix.to_parquet(tmp_path / "macro" / "VIXCLS.parquet")

    def test_bull_low_vol_on_synthetic_inputs(self, tmp_path):
        self._spy_macro(tmp_path)          # +12.5% over 2y, VIX 14, rates flat
        snap = ms.regime_snapshot("2026-08-24", str(tmp_path))
        assert snap["label"] in ("bull_low_vol", "neutral")  # 52w ret >10% & VIX<20
        assert snap["inputs"]["vix"] == "2026-08-24"

    def test_missing_spy_reports_unavailable_not_neutral(self, tmp_path):
        (tmp_path / "macro").mkdir(parents=True)
        pd.DataFrame({"date": ["2026-01-01"], "value": [5.0]}).to_parquet(
            tmp_path / "macro" / "FEDFUNDS.parquet")
        snap = ms.regime_snapshot("2026-08-24", str(tmp_path))
        assert snap["label"] is None
        assert "unavailable" in snap["reason"]
