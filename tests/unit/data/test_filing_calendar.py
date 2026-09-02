"""The point-in-time gate: what an agent was allowed to know, and when.

19% covered before this file, which is the wrong number for the module that
decides whether the whole study is out-of-sample. ``latest_filed_period`` is the
single function standing between the fundamentals store — indexed by *fiscal
period end*, which is not when the numbers were knowable — and every agent
prompt. If it ever gates on ``period_end`` instead of ``filing_date``, every
result in the paper becomes lookahead-contaminated, and nothing else in the
system would notice: the agents would simply be a little too good.

So these tests are adversarial rather than illustrative. Each one describes a
way the gate could be wrong while still returning a plausible Timestamp.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hifi.data import filing_calendar as fc


def _cal(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """(ticker, form, period_end, filing_date) -> calendar frame."""
    return pd.DataFrame([
        {"ticker": t, "form": f, "period_end": pd.Timestamp(pe),
         "filing_date": pd.Timestamp(fd), "accession": f"acc-{t}-{pe}"}
        for t, f, pe, fd in rows
    ])


#: The literal AEP case from DJ-133a: the March quarter was not public until May.
AEP = _cal([
    ("AEP", "10-Q", "2025-09-30", "2025-11-04"),
    ("AEP", "10-K", "2025-12-31", "2026-02-24"),
    ("AEP", "10-Q", "2026-03-31", "2026-05-06"),
    ("AEP", "10-Q", "2026-06-30", "2026-08-05"),
])


class TestTheGateIsFilingDateNotPeriodEnd:
    """The one thing this module exists to prevent."""

    def test_a_period_that_has_ended_but_not_been_filed_is_invisible(self):
        # 2026-04-15: the March quarter is over, and will not be filed until
        # 2026-05-06. An agent on this date must still be reading December.
        assert fc.latest_filed_period("AEP", "2026-04-15", AEP) == \
            pd.Timestamp("2025-12-31")

    def test_it_becomes_visible_the_day_it_is_filed(self):
        assert fc.latest_filed_period("AEP", "2026-05-06", AEP) == \
            pd.Timestamp("2026-03-31")

    def test_and_not_the_day_before(self):
        assert fc.latest_filed_period("AEP", "2026-05-05", AEP) == \
            pd.Timestamp("2025-12-31")

    @pytest.mark.parametrize("as_of,expected", [
        ("2025-11-03", None),          # nothing filed yet
        ("2025-11-04", "2025-09-30"),  # first filing lands
        ("2026-02-23", "2025-09-30"),
        ("2026-02-24", "2025-12-31"),
        ("2026-05-06", "2026-03-31"),
        ("2026-08-04", "2026-03-31"),
        ("2026-08-05", "2026-06-30"),
    ])
    def test_the_whole_visibility_timeline(self, as_of, expected):
        got = fc.latest_filed_period("AEP", as_of, AEP)
        assert got == (None if expected is None else pd.Timestamp(expected))

    def test_a_late_filing_does_not_reorder_the_answer(self):
        """The answer is the newest PERIOD among filed rows, not the newest filing.

        A restatement or a late 10-K can be filed after a later quarter's 10-Q.
        Taking max(filing_date) instead of max(period_end) would then hand the
        agent an older period and quietly stale its inputs.
        """
        cal = _cal([
            ("X", "10-Q", "2026-03-31", "2026-05-01"),
            ("X", "10-Q", "2026-06-30", "2026-08-01"),
            ("X", "10-K", "2025-12-31", "2026-09-01"),  # filed late, older period
        ])
        assert fc.latest_filed_period("X", "2026-09-15", cal) == \
            pd.Timestamp("2026-06-30")


class TestAbsenceIsReportedNotPaperedOver:
    def test_nothing_filed_yet_returns_none(self):
        # A recently listed company. None is a real state; substituting a stale
        # period would be the DJ-120 failure — rendering absence as data.
        assert fc.latest_filed_period("AEP", "2020-01-01", AEP) is None

    def test_unknown_ticker_returns_none_not_another_company(self):
        assert fc.latest_filed_period("ZZZZ", "2026-08-31", AEP) is None

    def test_empty_calendar_returns_none(self):
        assert fc.latest_filed_period("AEP", "2026-08-31", pd.DataFrame()) is None

    def test_none_calendar_returns_none(self):
        assert fc.latest_filed_period("AEP", "2026-08-31", None) is None

    def test_ticker_matching_is_case_insensitive_on_input_only(self):
        assert fc.latest_filed_period("aep", "2026-08-31", AEP) == \
            pd.Timestamp("2026-06-30")

    @pytest.mark.parametrize("as_of", [
        "2026-05-06", pd.Timestamp("2026-05-06"), __import__("datetime").date(2026, 5, 6),
    ])
    def test_accepts_str_date_and_timestamp_alike(self, as_of):
        assert fc.latest_filed_period("AEP", as_of, AEP) == pd.Timestamp("2026-03-31")


class TestLoadFilingCalendar:
    def test_missing_cache_returns_none_rather_than_raising(self, tmp_path):
        # A caller must be able to degrade loudly; a raise here would abort a
        # night, and a silent empty frame would blind every agent.
        assert fc.load_filing_calendar(data_dir=tmp_path) is None

    def test_unreadable_cache_returns_none(self, tmp_path, caplog):
        bad = tmp_path / "cal.parquet"
        bad.write_text("this is not parquet")
        assert fc.load_filing_calendar(path=bad) is None
        assert "unreadable" in caplog.text.lower()

    def test_round_trips_a_built_calendar(self, tmp_path):
        target = tmp_path / "cal.parquet"
        AEP.to_parquet(target, index=False)
        loaded = fc.load_filing_calendar(path=target)
        assert len(loaded) == 4
        assert fc.latest_filed_period("AEP", "2026-05-06", loaded) == \
            pd.Timestamp("2026-03-31")


class TestScanPeriodic:
    """Only 10-Q/10-K, and only rows that can actually gate something."""

    def _batch(self, forms, reported, filed, acc=None):
        return {"form": forms, "reportDate": reported, "filingDate": filed,
                "accessionNumber": acc or [f"a{i}" for i in range(len(forms))]}

    def test_keeps_only_periodic_forms(self):
        batch = self._batch(
            ["8-K", "10-Q", "S-8", "10-K", "DEF 14A"],
            ["", "2026-03-31", "", "2025-12-31", ""],
            ["2026-04-01", "2026-05-06", "2026-04-02", "2026-02-24", "2026-03-01"])
        rows = fc._scan_periodic("X", batch)
        assert [r["form"] for r in rows] == ["10-Q", "10-K"]

    def test_a_filing_with_no_report_date_is_skipped(self):
        # It cannot be tied to a fiscal period, so it cannot gate anything.
        rows = fc._scan_periodic("X", self._batch(["10-Q"], [""], ["2026-05-06"]))
        assert rows == []

    def test_a_filing_with_no_filing_date_is_skipped(self):
        rows = fc._scan_periodic("X", self._batch(["10-Q"], ["2026-03-31"], [""]))
        assert rows == []

    def test_ragged_arrays_do_not_raise(self):
        # EDGAR occasionally returns short parallel arrays.
        rows = fc._scan_periodic("X", {"form": ["10-Q", "10-Q"],
                                       "reportDate": ["2026-03-31"],
                                       "filingDate": ["2026-05-06"],
                                       "accessionNumber": []})
        assert len(rows) == 1
        assert rows[0]["accession"] == ""

    def test_empty_batch_yields_nothing(self):
        assert fc._scan_periodic("X", {}) == []


class TestPaginationAndSuccession:
    """``filings.recent`` is capped and mixes form types (DJ-133a)."""

    def _client(self, recent, shards=None):
        client = MagicMock()
        client.get_submissions.return_value = {
            "filings": {"recent": recent,
                        "files": [{"name": n} for n in (shards or {})]}}
        client._get_json.side_effect = lambda url: (shards or {})[url.rsplit("/", 1)[-1]]
        return client

    def _periodic(self, n, start_year=2020):
        return {"form": ["10-Q"] * n,
                "reportDate": [f"{start_year + i // 4}-03-31" for i in range(n)],
                "filingDate": [f"{start_year + i // 4}-05-06" for i in range(n)],
                "accessionNumber": [f"acc{start_year}-{i}" for i in range(n)]}

    def test_recent_alone_is_enough_when_it_holds_enough_filings(self):
        client = self._client(self._periodic(8))
        rows = fc._collect_for_cik(client, "AAPL", "0000320193", needed=8)
        assert len(rows) == 8
        client._get_json.assert_not_called()

    def test_paginates_when_recent_is_dominated_by_other_forms(self):
        """The literal JPM/BAC/GS/MS/BLK/PEP/COST failure.

        A heavy 8-K filer pushes its own 10-Qs out of the `recent` window, so
        reading only `recent` gave those names fewer than four filed quarters —
        which silently dropped every TTM ratio for exactly them while the
        balance-sheet ratios kept working.
        """
        recent = {"form": ["8-K"] * 40, "reportDate": [""] * 40,
                  "filingDate": ["2026-01-01"] * 40,
                  "accessionNumber": [f"k{i}" for i in range(40)]}
        client = self._client(recent, shards={"CIK-shard-1.json": self._periodic(8)})
        rows = fc._collect_for_cik(client, "JPM", "0000019617", needed=8)
        assert len(rows) == 8, "pagination did not recover the 10-Qs"

    def test_stops_paginating_once_satisfied(self):
        shards = {"s1.json": self._periodic(8), "s2.json": self._periodic(8, 2010)}
        client = self._client({"form": []}, shards=shards)
        rows = fc._collect_for_cik(client, "X", "1", needed=8)
        assert len(rows) == 8
        assert client._get_json.call_count == 1, "kept fetching after it had enough"

    def test_a_failing_shard_is_skipped_not_fatal(self, caplog):
        client = MagicMock()
        client.get_submissions.return_value = {
            "filings": {"recent": {"form": []},
                        "files": [{"name": "bad.json"}, {"name": "good.json"}]}}
        def _get(url):
            if url.endswith("bad.json"):
                raise RuntimeError("503")
            return self._periodic(8)
        client._get_json.side_effect = _get
        rows = fc._collect_for_cik(client, "X", "1", needed=8)
        assert len(rows) == 8
        assert "Failed to fetch shard" in caplog.text

    def test_a_failing_submissions_call_returns_empty_not_partial(self, caplog):
        client = MagicMock()
        client.get_submissions.side_effect = RuntimeError("connection reset")
        assert fc._collect_for_cik(client, "X", "1", needed=8) == []
        assert "Submissions fetch failed" in caplog.text

    def test_xom_succession_is_declared(self):
        """XOM's ticker resolves to a 2026 shell with no operating history.

        Verified 2026-08-30: "ExxonMobil Holdings Corp" (CIK 2115436) holds the
        ticker; every 10-Q sits under "EXXON MOBIL CORP" (CIK 34088). Without
        the union XOM is silently blind — the DJ-123 shape, a corporate action
        removing an agent's eyesight.
        """
        assert fc._CIK_SUCCESSIONS["XOM"] == ("0000034088",)

    def test_the_pagination_bound_covers_a_ttm_with_slack(self):
        assert fc._MIN_PERIODIC_FILINGS >= 8, (
            "a TTM needs four quarters; the bound must leave slack for the "
            "oldest rows arriving with NaNs"
        )


class TestBuildFilingCalendar:
    """The union-and-dedupe path, without touching the network."""

    #: (period_end month-day, filing month-day) for the four fiscal quarters.
    _QUARTERS = [("03-31", "05-06"), ("06-30", "08-06"),
                 ("09-30", "11-06"), ("12-31", "02-24")]

    def _submissions(self, n, start=2024, prefix="a"):
        reported, filed = [], []
        for i in range(n):
            year = start + i // 4
            pe, fd = self._QUARTERS[i % 4]
            reported.append(f"{year}-{pe}")
            # The Q4 10-K is filed the following February.
            filed.append(f"{year + 1 if fd.startswith('02') else year}-{fd}")
        return {"filings": {"recent": {
            "form": ["10-Q"] * n,
            "reportDate": reported,
            "filingDate": filed,
            "accessionNumber": [f"{prefix}{i}" for i in range(n)],
        }, "files": []}}

    def test_writes_a_sorted_parquet_and_returns_it(self, tmp_path):
        client = MagicMock()
        client.get_submissions.return_value = self._submissions(8)
        with patch("hifi.data.edgar.EdgarFetcher", return_value=client), \
             patch.object(fc, "ticker_to_cik", return_value={"AAA": "0000000001"}):
            df = fc.build_filing_calendar(["AAA"], data_dir=tmp_path)

        assert not df.empty
        assert list(df["ticker"].unique()) == ["AAA"]
        assert df["period_end"].is_monotonic_increasing
        assert (tmp_path / fc.DEFAULT_CALENDAR_PATH).exists()

    def test_a_transition_filing_reported_by_both_ciks_is_counted_once(self, tmp_path):
        """A double-counted quarter would corrupt every TTM built on it."""
        dup = self._submissions(8)
        client = MagicMock()
        client.get_submissions.return_value = dup  # same accessions for both CIKs
        with patch("hifi.data.edgar.EdgarFetcher", return_value=client), \
             patch.object(fc, "ticker_to_cik", return_value={"XOM": "0002115436"}):
            df = fc.build_filing_calendar(["XOM"], data_dir=tmp_path)
        assert len(df) == len(set(df["accession"])), "an accession appears twice"

    def test_a_ticker_with_too_few_filings_warns_loudly(self, tmp_path, caplog):
        client = MagicMock()
        client.get_submissions.return_value = self._submissions(2)
        with patch("hifi.data.edgar.EdgarFetcher", return_value=client), \
             patch.object(fc, "ticker_to_cik", return_value={"NEW": "0000000002"}):
            fc.build_filing_calendar(["NEW"], data_dir=tmp_path)
        assert "only 2 periodic filing(s)" in caplog.text
        assert "_CIK_SUCCESSIONS" in caplog.text, (
            "the warning must name the remedy; a bare count is not actionable"
        )

    def test_tickers_with_no_cik_are_named_not_silently_dropped(self, tmp_path, caplog):
        client = MagicMock()
        client.get_submissions.return_value = self._submissions(8)
        with patch("hifi.data.edgar.EdgarFetcher", return_value=client), \
             patch.object(fc, "ticker_to_cik", return_value={"AAA": "0000000001"}):
            fc.build_filing_calendar(["AAA", "NOPE"], data_dir=tmp_path)
        assert "NOPE" in caplog.text

    def test_an_empty_result_is_an_error_not_an_empty_file(self, tmp_path, caplog):
        with patch("hifi.data.edgar.EdgarFetcher", return_value=MagicMock()), \
             patch.object(fc, "ticker_to_cik", return_value={}):
            df = fc.build_filing_calendar(["AAA"], data_dir=tmp_path)
        assert df.empty
        assert "empty" in caplog.text.lower()
        assert not (tmp_path / fc.DEFAULT_CALENDAR_PATH).exists(), (
            "an empty calendar was cached; the next run would load it and gate "
            "every ticker to None"
        )


class TestTickerToCik:
    def _sec_payload(self):
        return json.dumps({
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
            "2": {"cik_str": 19617, "ticker": "JPM", "title": "JPMorgan"},
        }).encode()

    def _urlopen(self, payload):
        resp = MagicMock()
        resp.read.return_value = payload
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: False
        return resp

    def test_pads_cik_to_ten_digits(self):
        with patch("urllib.request.urlopen", return_value=self._urlopen(self._sec_payload())):
            got = fc.ticker_to_cik(["AAPL"])
        assert got == {"AAPL": "0000320193"}, "EDGAR requires zero-padded CIKs"

    def test_filters_to_requested_tickers(self):
        with patch("urllib.request.urlopen", return_value=self._urlopen(self._sec_payload())):
            got = fc.ticker_to_cik(["AAPL", "JPM"])
        assert set(got) == {"AAPL", "JPM"}

    def test_absent_ticker_is_omitted_not_substituted(self):
        with patch("urllib.request.urlopen", return_value=self._urlopen(self._sec_payload())):
            got = fc.ticker_to_cik(["AAPL", "ZZZZ"])
        assert "ZZZZ" not in got and len(got) == 1

    def test_none_returns_every_listed_ticker(self):
        with patch("urllib.request.urlopen", return_value=self._urlopen(self._sec_payload())):
            assert len(fc.ticker_to_cik(None)) == 3

    def test_sends_a_contact_user_agent(self):
        # EDGAR rejects requests without one, and a rejection here blinds every
        # ticker at once.
        captured = {}
        def _open(req, timeout=None):
            captured["ua"] = req.headers.get("User-agent") or req.headers.get("User-Agent")
            return self._urlopen(self._sec_payload())
        with patch("urllib.request.urlopen", side_effect=_open):
            fc.ticker_to_cik(["AAPL"])
        assert captured["ua"], "no User-Agent header was sent"


class TestAgainstTheRealCalendar:
    """If the built calendar is present, its invariants must hold."""

    @pytest.fixture
    def calendar(self):
        cal = fc.load_filing_calendar(data_dir="data")
        if cal is None or cal.empty:
            pytest.skip("no filing calendar built in this checkout")
        return cal

    def test_no_filing_precedes_its_own_period_end(self, calendar):
        early = calendar[calendar["filing_date"] < calendar["period_end"]]
        assert early.empty, (
            "a filing dated before the period it reports is impossible and would "
            f"open a lookahead window:\n{early.head()}"
        )

    def test_every_universe_ticker_has_four_filed_quarters(self, calendar):
        from hifi.data.universe import PHASE14_UNIVERSE

        counts = calendar.groupby("ticker").size()
        thin = {t["ticker"]: int(counts.get(t["ticker"], 0))
                for t in PHASE14_UNIVERSE if counts.get(t["ticker"], 0) < 4}
        assert not thin, (
            f"these tickers cannot support a TTM ratio: {thin}. Add a "
            "predecessor CIK to _CIK_SUCCESSIONS or retire them."
        )

    def test_no_duplicate_period_per_ticker_and_form(self, calendar):
        dupes = calendar.duplicated(subset=["ticker", "form", "period_end"], keep=False)
        assert not dupes.any(), (
            f"a quarter is counted twice, which corrupts every TTM built on it:\n"
            f"{calendar[dupes].head()}"
        )
