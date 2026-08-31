"""DJ-133b: the sentiment agent needs an input that changes between votes.

Its only source was SEC filings, which update quarterly, while it votes daily.
Measured on the live record: the context retrieved for AAPL on 2026-08-24 and
2026-08-27 differed by exactly one character -- the echoed ``as_of=`` in the
header -- and the 8,265-character filing body was byte-identical (similarity
0.999879). With no time-varying input the agent was a constant by construction;
it voted Hold on 97/97 tickers on 2026-08-27, and the small wobble in its
output was LLM sampling noise rather than information.

The cutoff tests below are the important ones. The first version of the news
module cut off at midnight Eastern and immediately admitted an after-hours
story published at 20:30 ET into that same day's context. Every summary looked
right -- the article really was published on the decision date -- and it would
have surfaced in review as lookahead.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from hifi.data import news as news_mod
from hifi.data.news import (
    NewsArticle,
    _cutoff,
    _eastern_day,
    fetch_news,
    format_news_block,
)

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class TestTheCutoffIsTheClose:
    def test_cutoff_is_market_close_not_midnight(self):
        c = _cutoff("2026-08-27")
        assert c.astimezone(EASTERN).hour == 16, (
            "the arm prices its book at the close, so the close is the "
            "information cutoff; midnight admits after-hours news"
        )

    def test_after_hours_story_is_excluded(self):
        """The exact article that exposed the bug: 20:30 ET on the decision day."""
        after_hours = datetime(2026, 8, 27, 20, 30, tzinfo=EASTERN)
        assert after_hours.astimezone(UTC) > _cutoff("2026-08-27")

    def test_pre_close_story_is_included(self):
        during = datetime(2026, 8, 27, 15, 59, tzinfo=EASTERN)
        assert during.astimezone(UTC) < _cutoff("2026-08-27")

    def test_cutoff_tracks_daylight_saving(self):
        """A fixed UTC offset would drift an hour twice a year."""
        summer = _cutoff("2026-08-27").astimezone(EASTERN).hour
        winter = _cutoff("2026-01-15").astimezone(EASTERN).hour
        assert summer == winter == 16


class TestDisplayDatesAreEastern:
    def test_evening_story_renders_as_its_trading_day(self):
        """20:30 ET is 00:30 UTC the next day; showing UTC reads as leakage."""
        ts = datetime(2026, 8, 27, 20, 30, tzinfo=EASTERN).astimezone(UTC).isoformat()
        assert ts[:10] == "2026-08-28"          # what the naive version printed
        assert _eastern_day(ts) == "2026-08-27"  # what is true

    def test_malformed_timestamp_degrades_quietly(self):
        assert _eastern_day("not-a-timestamp") == "not-a-timestamp"[:10]


class TestFormatting:
    def test_empty_returns_empty_string_not_a_placeholder(self):
        """A "no news today" placeholder is one more constant to vote on."""
        assert format_news_block("AAPL", "2026-08-27", []) == ""

    def test_block_states_the_cutoff_it_used(self):
        a = NewsArticle(
            created_at=datetime(2026, 8, 27, 10, 0, tzinfo=EASTERN).isoformat(),
            headline="Something happened", source="benzinga",
        )
        block = format_news_block("AAPL", "2026-08-27", [a])
        assert "on or before the 2026-08-27 close" in block
        assert "Something happened" in block
        assert "(2026-08-27)" in block


class TestReproducibility:
    """A stored decision must be auditable against the text the agent saw.
    The vendor's index changes; without a cache the record is not
    reconstructible."""

    def test_cache_is_replayed_without_refetching(self, tmp_path, monkeypatch):
        cache = tmp_path / "news" / "AAPL" / "2026-08-27.json"
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps({"articles": [
            {"created_at": "2026-08-27T10:00:00-04:00", "headline": "Cached story",
             "source": "test", "summary": "", "symbols": ["AAPL"]},
        ]}))

        def _explode():
            raise AssertionError("credentials must not be consulted on a cache hit")

        monkeypatch.setattr(news_mod, "_credentials", _explode)
        out = fetch_news("AAPL", "2026-08-27", data_dir=tmp_path)
        assert [a.headline for a in out] == ["Cached story"]

    def test_corrupt_cache_does_not_crash_the_night(self, tmp_path, monkeypatch):
        cache = tmp_path / "news" / "AAPL" / "2026-08-27.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{ this is not json")
        monkeypatch.setattr(news_mod, "_credentials", lambda: None)
        assert fetch_news("AAPL", "2026-08-27", data_dir=tmp_path) == []


class TestFailsOpen:
    """A news outage must degrade the agent, not take down a trading night
    (DJ-123: one dead ticker aborted an arm mid-execution)."""

    def test_missing_credentials_returns_empty_not_an_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(news_mod, "_credentials", lambda: None)
        assert fetch_news("AAPL", "2026-08-27", data_dir=tmp_path) == []

    def test_vendor_failure_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(news_mod, "_credentials", lambda: ("k", "s"))

        def _boom(*a, **k):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(news_mod, "_cutoff", _boom)
        assert fetch_news("AAPL", "2026-08-27", data_dir=tmp_path) == []


class TestSentimentContextNowVaries:
    """The property whose absence was DJ-133b, asserted on cached fixtures."""

    def _seed(self, tmp_path, date: str, headlines: list[str]) -> None:
        p = tmp_path / "news" / "AAPL" / f"{date}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"articles": [
            {"created_at": f"{date}T10:0{i}:00-04:00", "headline": h,
             "source": "test", "summary": "", "symbols": ["AAPL"]}
            for i, h in enumerate(headlines)
        ]}))

    def test_two_dates_produce_different_blocks(self, tmp_path):
        self._seed(tmp_path, "2026-08-26", ["Chips rally", "Guidance raised"])
        self._seed(tmp_path, "2026-08-27", ["Regulator opens probe"])
        a = format_news_block(
            "AAPL", "2026-08-26", fetch_news("AAPL", "2026-08-26", data_dir=tmp_path))
        b = format_news_block(
            "AAPL", "2026-08-27", fetch_news("AAPL", "2026-08-27", data_dir=tmp_path))
        assert a and b and a != b, (
            "the filing-only context differed by ONE character across four days "
            "(similarity 0.999879); that is what made the agent a constant"
        )

    def test_ordering_is_newest_first(self, tmp_path):
        self._seed(tmp_path, "2026-08-27", ["older", "newer"])
        out = fetch_news("AAPL", "2026-08-27", data_dir=tmp_path)
        assert [a.headline for a in out] == ["newer", "older"]


@pytest.mark.parametrize("date", ["2026-01-15", "2026-06-30", "2026-11-02"])
def test_cutoff_is_always_before_the_next_session(date):
    """Sanity across the year: the cutoff never spills into the following day."""
    c = _cutoff(date).astimezone(EASTERN)
    assert c.strftime("%Y-%m-%d") == date
