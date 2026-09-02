"""The macro agent's entire input, previously 27% covered.

``compute_macro_snapshot`` is the whole of what the macro agent sees. When it
returned nothing on 2026-08-31 the agent did not fail — it voted Hold on 193 of
194 passes and said so confidently (DJ-133c). That is the DJ-120 shape: a blinded
agent is indistinguishable from a decisive one, so the only defence is testing
the input path itself.

Two properties carry the weight here and both are about *time*: a value must
never be visible before it was published, and a stale value must not be
laundered into a fresh one. Everything else is arithmetic.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hifi.data.schemas import MacroDataset, MacroIndicator, ProvenanceRecord
from hifi.engines.macro import compute_macro_snapshot


def _ds(series_id: str, obs: list[tuple[str, float]], frequency="monthly") -> MacroDataset:
    now = datetime.now(UTC)
    observations = [
        MacroIndicator(series_id=series_id, date=date.fromisoformat(d), value=v)
        for d, v in obs
    ]
    return MacroDataset(
        series_id=series_id, name=series_id, frequency=frequency, unit="pct",
        observations=observations, source="FRED", fetched_at=now,
        date_from=observations[0].date, date_to=observations[-1].date,
        provenance=ProvenanceRecord(source="FRED", fetched_at=now, parameters={}),
    )


class TestPointInTimeVisibility:
    """A value published after the analysis date must not be visible."""

    def test_a_future_observation_is_not_used(self):
        ds = _ds("FEDFUNDS", [("2026-01-01", 4.0), ("2026-06-01", 5.0)])
        snap = compute_macro_snapshot({"FEDFUNDS": ds}, date(2026, 3, 15))
        assert snap.fed_funds_rate == 4.0, "read a value published in the future"

    def test_the_value_appears_on_its_publication_date(self):
        ds = _ds("FEDFUNDS", [("2026-01-01", 4.0), ("2026-06-01", 5.0)])
        assert compute_macro_snapshot(
            {"FEDFUNDS": ds}, date(2026, 6, 1)).fed_funds_rate == 5.0

    def test_a_date_before_the_series_starts_yields_none(self):
        # None, not a forward-projected first value: there was nothing to know.
        ds = _ds("FEDFUNDS", [("2026-01-01", 4.0)])
        assert compute_macro_snapshot(
            {"FEDFUNDS": ds}, date(2025, 12, 31)).fed_funds_rate is None

    def test_quarterly_gdp_holds_its_value_across_the_quarter(self):
        """Correct, not stale: an investor really did have only that number."""
        ds = _ds("A191RL1Q225SBEA", [("2026-01-01", 2.5), ("2026-04-01", 3.1)],
                 frequency="quarterly")
        for d in (date(2026, 1, 1), date(2026, 2, 14), date(2026, 3, 31)):
            assert compute_macro_snapshot(
                {"A191RL1Q225SBEA": ds}, d).gdp_growth == 2.5
        assert compute_macro_snapshot(
            {"A191RL1Q225SBEA": ds}, date(2026, 4, 1)).gdp_growth == 3.1


class TestMissingSeriesAreNoneNotZero:
    """DJ-133c: unreadable series must surface as absence, never as a number."""

    def test_no_datasets_at_all_gives_every_field_none(self):
        snap = compute_macro_snapshot({}, date(2026, 8, 31))
        for field in ("fed_funds_rate", "unemployment_rate", "yield_10y",
                      "yield_2y", "vix", "gdp_growth", "yield_curve_slope",
                      "cpi_yoy"):
            assert getattr(snap, field) is None, f"{field} was not None"

    def test_one_missing_series_does_not_void_the_others(self):
        snap = compute_macro_snapshot(
            {"VIXCLS": _ds("VIXCLS", [("2026-08-01", 17.4)], "daily")},
            date(2026, 8, 31))
        assert snap.vix == 17.4
        assert snap.fed_funds_rate is None

    @pytest.mark.parametrize("field,series", [
        ("fed_funds_rate", "FEDFUNDS"), ("unemployment_rate", "UNRATE"),
        ("yield_10y", "GS10"), ("yield_2y", "GS2"), ("vix", "VIXCLS"),
        ("gdp_growth", "A191RL1Q225SBEA"),
    ])
    def test_each_direct_field_maps_to_its_fred_series(self, field, series):
        snap = compute_macro_snapshot(
            {series: _ds(series, [("2026-08-01", 1.23)])}, date(2026, 8, 31))
        assert getattr(snap, field) == 1.23


class TestYieldCurveSlope:
    def test_slope_is_ten_year_minus_two_year(self):
        snap = compute_macro_snapshot({
            "GS10": _ds("GS10", [("2026-08-01", 4.30)]),
            "GS2": _ds("GS2", [("2026-08-01", 3.80)]),
        }, date(2026, 8, 31))
        assert snap.yield_curve_slope == pytest.approx(0.50)

    def test_an_inverted_curve_is_negative_not_absolute(self):
        # The sign IS the signal; abs() here would erase a recession indicator.
        snap = compute_macro_snapshot({
            "GS10": _ds("GS10", [("2026-08-01", 3.50)]),
            "GS2": _ds("GS2", [("2026-08-01", 4.20)]),
        }, date(2026, 8, 31))
        assert snap.yield_curve_slope == pytest.approx(-0.70)

    @pytest.mark.parametrize("present", ["GS10", "GS2"])
    def test_one_leg_missing_voids_the_slope(self, present):
        snap = compute_macro_snapshot(
            {present: _ds(present, [("2026-08-01", 4.0)])}, date(2026, 8, 31))
        assert snap.yield_curve_slope is None, (
            "a slope computed against a missing leg is a number with no meaning"
        )


class TestCpiYoY:
    """CPIAUCSL is a raw index; the agent needs a rate."""

    def test_yoy_percentage_from_the_index(self):
        ds = _ds("CPIAUCSL", [("2025-08-01", 300.0), ("2026-08-01", 309.0)])
        snap = compute_macro_snapshot({"CPIAUCSL": ds}, date(2026, 8, 15))
        assert snap.cpi_yoy == pytest.approx(3.0)

    def test_the_raw_index_is_never_reported_as_the_rate(self):
        # The failure mode worth guarding: 309.0 handed to an LLM as "inflation".
        ds = _ds("CPIAUCSL", [("2025-08-01", 300.0), ("2026-08-01", 309.0)])
        snap = compute_macro_snapshot({"CPIAUCSL": ds}, date(2026, 8, 15))
        assert snap.cpi_yoy < 50.0

    def test_deflation_is_negative(self):
        ds = _ds("CPIAUCSL", [("2025-08-01", 300.0), ("2026-08-01", 294.0)])
        snap = compute_macro_snapshot({"CPIAUCSL": ds}, date(2026, 8, 15))
        assert snap.cpi_yoy == pytest.approx(-2.0)

    def test_no_prior_year_anchor_yields_none(self):
        ds = _ds("CPIAUCSL", [("2026-08-01", 309.0)])
        assert compute_macro_snapshot(
            {"CPIAUCSL": ds}, date(2026, 8, 15)).cpi_yoy is None

    def test_zero_prior_index_does_not_divide_by_zero(self):
        ds = _ds("CPIAUCSL", [("2025-08-01", 0.0), ("2026-08-01", 309.0)])
        assert compute_macro_snapshot(
            {"CPIAUCSL": ds}, date(2026, 8, 15)).cpi_yoy is None


class TestDeterminism:
    """David §4.1: same datasets and date -> same result, every run."""

    def test_repeated_calls_agree(self):
        datasets = {
            "FEDFUNDS": _ds("FEDFUNDS", [("2026-01-01", 4.0), ("2026-07-01", 4.5)]),
            "GS10": _ds("GS10", [("2026-07-01", 4.3)]),
            "GS2": _ds("GS2", [("2026-07-01", 3.9)]),
            "CPIAUCSL": _ds("CPIAUCSL", [("2025-08-01", 300.0), ("2026-08-01", 309.0)]),
            "VIXCLS": _ds("VIXCLS", [("2026-08-28", 16.2)], "daily"),
        }
        first = compute_macro_snapshot(datasets, date(2026, 8, 31))
        for _ in range(3):
            assert compute_macro_snapshot(datasets, date(2026, 8, 31)) == first


class TestAgainstTheLiveMacroStore:
    """The seven series the agents actually read must produce a usable snapshot."""

    @pytest.fixture
    def live_datasets(self):
        from pathlib import Path

        from hifi.data.storage import read_macro

        macro_dir = Path("data/macro")
        if not macro_dir.exists():
            pytest.skip("no macro store in this checkout")
        out = {}
        for f in sorted(macro_dir.glob("*.parquet")):
            try:
                ds = read_macro(f)
            except Exception:
                continue  # unreadable files are the subject of the test below
            out[ds.series_id] = ds
        if not out:
            pytest.skip("macro store present but empty")
        return out

    def test_every_stored_series_is_readable(self):
        """DJ-133c: `_load_all_macro` swallows per-file errors and returns {}.

        A single unreadable file is therefore invisible until the agent starts
        voting Hold on everything. This asserts the store parses.
        """
        from pathlib import Path

        from hifi.data.storage import read_macro

        macro_dir = Path("data/macro")
        if not macro_dir.exists():
            pytest.skip("no macro store in this checkout")
        files = sorted(macro_dir.glob("*.parquet"))
        if not files:
            pytest.skip("macro store present but empty")
        broken = []
        for f in files:
            try:
                read_macro(f)
            except Exception as exc:
                broken.append((f.name, str(exc)[:120]))
        assert not broken, (
            "these macro files cannot be parsed by read_macro, so the macro "
            f"agent would silently see NO_MACRO_DATA: {broken}"
        )

    def test_the_snapshot_is_populated_not_empty(self, live_datasets):
        snap = compute_macro_snapshot(live_datasets, date.today())
        populated = {f: getattr(snap, f) for f in
                     ("fed_funds_rate", "unemployment_rate", "yield_10y",
                      "yield_2y", "vix", "cpi_yoy")
                     if getattr(snap, f) is not None}
        assert len(populated) >= 5, (
            f"only {len(populated)} of 6 core macro fields resolved: {populated}. "
            "The macro agent would be reasoning from near-nothing."
        )
