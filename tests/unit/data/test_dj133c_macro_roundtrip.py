"""DJ-133c: macro files must round-trip, and an unreadable set must be loud.

On 2026-08-31 a refresh script rewrote all seven FRED series with a plain
``df.to_parquet()``. ``write_macro`` embeds series_id, name, frequency, unit and
provenance in the Parquet *schema metadata* and ``read_macro`` raises without
it, so every file became unreadable. ``_load_all_macro`` swallowed each failure
as a per-file warning and returned ``{}``; ``get_macro_snapshot`` then reported
NO_MACRO_DATA; and the macro agent voted Hold on 193 of 194 passes, up from 54%
Hold a week earlier.

Nothing failed. An agent simply went blind and kept voting, which is DJ-120's
signature exactly. These tests assert the two properties whose absence allowed
it: the writer's output must be readable, and a directory of files none of
which parse must raise rather than look like an empty directory.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from hifi.data.schemas import MacroDataset, MacroIndicator, ProvenanceRecord
from hifi.data.storage import read_macro, write_macro


def _dataset(series_id: str = "VIXCLS", n: int = 5) -> MacroDataset:
    now = datetime.now(UTC)
    obs = [
        MacroIndicator(series_id=series_id, date=date(2026, 8, i + 1), value=10.0 + i)
        for i in range(n)
    ]
    return MacroDataset(
        series_id=series_id,
        name="CBOE Volatility Index: VIX",
        frequency="daily",
        unit="index",
        observations=obs,
        source="FRED",
        fetched_at=now,
        date_from=obs[0].date,
        date_to=obs[-1].date,
        provenance=ProvenanceRecord(
            source="FRED", fetched_at=now, parameters={"series_id": series_id}
        ),
    )


class TestRoundTrip:
    def test_written_file_is_readable(self, tmp_path):
        p = tmp_path / "VIXCLS.parquet"
        write_macro(_dataset(), p)
        back = read_macro(p)
        assert back.series_id == "VIXCLS"
        assert len(back.observations) == 5

    def test_plain_to_parquet_is_not_readable(self, tmp_path):
        """The exact mistake: pandas drops the schema metadata read_macro needs."""
        p = tmp_path / "VIXCLS.parquet"
        pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-01"]), "value": [10.0]}
        ).to_parquet(p, index=False)
        with pytest.raises(Exception, match="metadata"):
            read_macro(p)

    def test_values_survive_the_round_trip(self, tmp_path):
        p = tmp_path / "VIXCLS.parquet"
        write_macro(_dataset(n=3), p)
        assert [o.value for o in read_macro(p).observations] == [10.0, 11.0, 12.0]


class TestUnreadableDirectoryIsLoud:
    """"Files exist but none parse" is a broken path, not an absence."""

    def test_all_unreadable_raises(self, tmp_path, monkeypatch):
        macro = tmp_path / "macro"
        macro.mkdir()
        for sid in ("VIXCLS", "GS10"):
            pd.DataFrame(
                {"date": pd.to_datetime(["2026-08-01"]), "value": [1.0]}
            ).to_parquet(macro / f"{sid}.parquet", index=False)

        monkeypatch.setenv("HIFI_DATA_DIR", str(tmp_path))
        from hifi.mcp import financial_server

        with pytest.raises(RuntimeError, match="unreadable"):
            financial_server._load_all_macro()

    def test_genuinely_empty_directory_does_not_raise(self, tmp_path, monkeypatch):
        """An empty dir is a legitimate state and must stay distinguishable."""
        (tmp_path / "macro").mkdir()
        monkeypatch.setenv("HIFI_DATA_DIR", str(tmp_path))
        from hifi.mcp import financial_server

        assert financial_server._load_all_macro() == {}

    def test_partial_failure_still_returns_the_readable_ones(self, tmp_path, monkeypatch):
        macro = tmp_path / "macro"
        macro.mkdir()
        write_macro(_dataset("VIXCLS"), macro / "VIXCLS.parquet")
        pd.DataFrame(
            {"date": pd.to_datetime(["2026-08-01"]), "value": [1.0]}
        ).to_parquet(macro / "GS10.parquet", index=False)

        monkeypatch.setenv("HIFI_DATA_DIR", str(tmp_path))
        from hifi.mcp import financial_server

        out = financial_server._load_all_macro()
        assert set(out) == {"VIXCLS"}


class TestLiveMacroDirectoryIsHealthy:
    """Guards the real data, not a fixture. This is the check that would have
    turned a three-day silent outage into an immediate failure."""

    def test_every_live_macro_file_parses(self):
        from pathlib import Path

        files = sorted(Path("data/macro").glob("*.parquet"))
        if not files:
            pytest.skip("no live macro data in this checkout")
        broken = []
        for f in files:
            try:
                read_macro(f)
            except Exception as exc:
                broken.append(f"{f.name}: {exc}")
        assert not broken, f"unreadable macro files: {broken}"
