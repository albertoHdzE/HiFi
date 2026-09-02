"""Regime classification — 66% covered, and the uncovered part was the inputs.

``classify_regime`` labels the market state that DJ-130 injects into agent
context. Its failure mode is the quietest in the system: when the SPY paths died
in the DJ-120 store migration the classifier did not error, it returned
"neutral" forever, and "neutral" is a perfectly plausible answer. Nothing
downstream can distinguish a considered neutral from a blind one.

These tests therefore concentrate on the two ways a wrong label is produced
without any error: an input series that silently fails to resolve, and a
threshold applied to a window that does not span what it claims to.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hifi.data import regime


def _spy(n: int = 300, start: float = 100.0, end: float = 100.0,
         end_date: str = "2026-08-31") -> pd.DataFrame:
    idx = pd.bdate_range(end=end_date, periods=n)
    closes = pd.Series(
        [start + (end - start) * i / max(n - 1, 1) for i in range(n)], index=idx)
    return pd.DataFrame({"close": closes})


def _macro(cols: dict[str, list[float]], n: int = 300,
           end_date: str = "2026-08-31") -> pd.DataFrame:
    idx = pd.bdate_range(end=end_date, periods=n)
    return pd.DataFrame({k: pd.Series(v * n if len(v) == 1 else v, index=idx)
                         for k, v in cols.items()})


class TestFiftyTwoWeekReturn:
    def test_it_measures_a_full_trading_year(self):
        spy = _spy(n=300, start=100.0, end=130.0)
        r = regime._spy_52w_return(spy, pd.Timestamp("2026-08-31"))
        assert r is not None and r > 0

    def test_too_little_history_returns_none_not_zero(self):
        # None means "cannot say"; 0.0 would be classified as neutral, which is
        # a claim.
        assert regime._spy_52w_return(_spy(n=100), pd.Timestamp("2026-08-31")) is None

    def test_bars_after_the_as_of_date_are_excluded(self):
        spy = _spy(n=300, start=100.0, end=200.0, end_date="2026-12-31")
        early = regime._spy_52w_return(spy, pd.Timestamp("2026-06-30"))
        late = regime._spy_52w_return(spy, pd.Timestamp("2026-12-31"))
        assert early != late, "the as_of cut had no effect; the future leaked in"

    def test_a_zero_price_a_year_ago_does_not_divide_by_zero(self):
        spy = _spy(n=300)
        spy.iloc[-252] = 0.0
        assert regime._spy_52w_return(spy, pd.Timestamp("2026-08-31")) is None

    def test_the_close_column_is_found_case_insensitively(self):
        spy = _spy(n=300, start=100.0, end=130.0).rename(columns={"close": "Close"})
        assert regime._spy_52w_return(spy, pd.Timestamp("2026-08-31")) is not None


class TestVixResolution:
    def test_an_explicit_vix_column_is_preferred(self):
        macro = _macro({"vix": [35.0]})
        got = regime._vix_value(_spy(), macro, pd.Timestamp("2026-08-31"))
        assert got == pytest.approx(35.0)

    def test_it_falls_back_to_realised_volatility(self):
        """Without a VIX column the classifier estimates from SPY itself, so a
        missing macro series degrades the label rather than voiding it."""
        got = regime._vix_value(_spy(n=100, start=100.0, end=140.0),
                                pd.DataFrame(), pd.Timestamp("2026-08-31"))
        assert got is not None and got >= 0.0

    def test_too_little_price_history_returns_none(self):
        assert regime._vix_value(_spy(n=5), pd.DataFrame(),
                                 pd.Timestamp("2026-08-31")) is None

    def test_a_short_vix_series_falls_back_rather_than_averaging_a_stub(self):
        # Fewer rows than the window: averaging them would silently be a
        # different statistic from the one the threshold was set for.
        macro = _macro({"vix": [35.0]}, n=3)
        got = regime._vix_value(_spy(n=100), macro, pd.Timestamp("2026-08-31"))
        assert got != pytest.approx(35.0)


class TestRateDelta:
    @pytest.mark.parametrize("col", ["fed_funds_rate", "FEDFUNDS", "ffr", "rate"])
    def test_it_accepts_the_known_column_spellings(self, col):
        # The window is 180 CALENDAR days (~128 business days), so the rise has
        # to sit inside it. 300 business days span ~430 calendar days; putting
        # the step at the halfway mark would place both ends of the comparison
        # after the cutoff and measure zero.
        rates = [1.0] * 200 + [4.0] * 100
        assert regime._rate_delta(_macro({col: rates}),
                                  pd.Timestamp("2026-08-31")) == pytest.approx(3.0)

    def test_an_unknown_column_yields_none(self):
        assert regime._rate_delta(_macro({"unemployment": [4.0]}),
                                  pd.Timestamp("2026-08-31")) is None

    def test_an_empty_series_yields_none(self):
        empty = pd.DataFrame({"fedfunds": pd.Series(dtype=float,
                                                    index=pd.DatetimeIndex([]))})
        assert regime._rate_delta(empty, pd.Timestamp("2026-08-31")) is None

    def test_it_measures_over_the_declared_window(self):
        assert regime._RATE_WINDOW_DAYS == 180


class TestClassification:
    def test_a_rate_shock_outranks_the_equity_regimes(self):
        """Checked first because it is the most distinctive: a 2pp move in six
        months dominates whatever the equity return happens to be."""
        spy = _spy(n=300, start=100.0, end=140.0)  # would otherwise be bullish
        macro = _macro({"fedfunds": [0.5] * 200 + [4.0] * 100,
                        "vix": [15.0] * 300})
        assert regime.classify_regime("2026-08-31", spy, macro) == "rate_shock"

    def test_bull_low_vol(self):
        spy = _spy(n=300, start=100.0, end=140.0)
        macro = _macro({"vix": [14.0]})
        assert regime.classify_regime("2026-08-31", spy, macro) == "bull_low_vol"

    def test_a_rising_market_with_high_vol_is_not_bull_low_vol(self):
        spy = _spy(n=300, start=100.0, end=115.0)
        macro = _macro({"vix": [35.0]})
        assert regime.classify_regime("2026-08-31", spy, macro) == "neutral"

    def test_bear_high_vol(self):
        spy = _spy(n=300, start=140.0, end=100.0)
        macro = _macro({"vix": [40.0]})
        assert regime.classify_regime("2026-08-31", spy, macro) == "bear_high_vol"

    def test_recovery_requires_a_bear_year_before_it(self):
        """A strong year alone is a bull market. Recovery is a claim about the
        year before, and asserting it without checking would relabel every
        strong run."""
        # Anchored on where the two lookbacks actually land: 252 bars back for
        # ret_1y and ~504 back for the prior year. Halve over the first of
        # those years, then rise 50% over the second.
        n = 800
        idx = pd.bdate_range(end="2026-08-31", periods=n)
        closes = []
        for i in range(n):
            if i <= n - 505:
                closes.append(200.0)
            elif i <= n - 253:
                span = (i - (n - 505)) / 252
                closes.append(200.0 - 100.0 * span)      # 200 -> 100
            else:
                span = (i - (n - 253)) / 252
                closes.append(100.0 + 50.0 * span)       # 100 -> 150
        spy = pd.DataFrame({"close": pd.Series(closes, index=idx)})
        assert regime.classify_regime("2026-08-31", spy, _macro({"vix": [22.0]})) \
            == "recovery"

    def test_a_strong_year_after_a_flat_year_is_not_recovery(self):
        spy = _spy(n=800, start=100.0, end=190.0)  # steady climb, never bearish
        assert regime.classify_regime("2026-08-31", spy, _macro({"vix": [22.0]})) \
            != "recovery"

    def test_insufficient_history_is_neutral_not_a_guess(self):
        assert regime.classify_regime("2026-08-31", _spy(n=50), pd.DataFrame()) \
            == "neutral"

    def test_every_label_is_in_the_declared_set(self):
        labels = {"bull_low_vol", "bear_high_vol", "rate_shock", "recovery",
                  "neutral"}
        cases = [
            (_spy(n=300, start=100.0, end=140.0), _macro({"vix": [14.0]})),
            (_spy(n=300, start=140.0, end=100.0), _macro({"vix": [40.0]})),
            (_spy(n=50), pd.DataFrame()),
        ]
        for spy, macro in cases:
            assert regime.classify_regime("2026-08-31", spy, macro) in labels


class TestAgainstTheLiveStore:
    """DJ-130: the classifier sat pinned at 'neutral' because its SPY paths died
    in the DJ-120 migration, and nothing anywhere reported it."""

    def test_spy_is_present_and_long_enough_to_classify(self):
        from pathlib import Path

        p = Path("data/market/SPY/ohlcv.parquet")
        if not p.exists():
            pytest.skip("no SPY bars in this checkout")
        df = pd.read_parquet(p)
        assert len(df) >= regime._TRADING_YEAR, (
            f"SPY has {len(df)} bars, fewer than the {regime._TRADING_YEAR} a "
            "52-week return needs; the classifier would return neutral forever"
        )

    def test_the_live_label_is_derived_rather_than_defaulted(self):
        from pathlib import Path

        p = Path("data/market/SPY/ohlcv.parquet")
        if not p.exists():
            pytest.skip("no SPY bars in this checkout")
        spy = pd.read_parquet(p)
        spy.index = pd.to_datetime(spy.index)
        r = regime._spy_52w_return(spy, pd.Timestamp("2026-08-31"))
        assert r is not None, (
            "the 52-week return is unavailable, so every regime label is a "
            "fallback rather than a measurement"
        )
