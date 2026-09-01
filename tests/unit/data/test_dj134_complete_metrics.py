"""DJ-134: every metric the agents' contracts promise must actually arrive.

After DJ-133a restored the fundamental agent's inputs, three ratios were still
absent on 194/194 passes (ev_ebitda, current_ratio, revenue_growth_yoy) and beta
was absent on 194/194 for the risk and technical agents. None of it was a hard
failure; the agents simply reasoned without them and said so in `data_gaps`.

Both causes were wiring, not capability:

* ``compute_beta`` has existed since Phase 2, but neither agent passed
  ``benchmark_ticker``, so the server's default of None made beta None forever.
* EBITDA, cash, current assets/liabilities, COGS and operating income were all
  present in the quarterly statements and were never carried into
  ``FundamentalsSnapshot``, so the ratio engine returned hardcoded None.

The residual gaps are accounting, not defects, and the tests below pin that
distinction so a future reader does not "fix" it: banks and insurers do not
report current assets, current liabilities or cost of revenue, because they have
no operating cycle.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hifi.data.schemas import FundamentalsSnapshot, ProvenanceRecord
from hifi.engines.fundamental import compute_financial_ratios, compute_growth_metrics


def _snap(**over) -> FundamentalsSnapshot:
    now = datetime.now(UTC)
    base = dict(
        ticker="TEST",
        period_end=date(2026, 6, 30),
        revenue=1000.0,
        net_income=100.0,
        total_assets=5000.0,
        total_liabilities=2000.0,
        total_equity=1500.0,
        eps=2.0,
        market_cap=10000.0,
        ebitda=400.0,
        cash_and_equivalents=500.0,
        current_assets=900.0,
        current_liabilities=600.0,
        cost_of_revenue=600.0,
        operating_income=250.0,
        revenue_latest_q=260.0,
        revenue_year_ago_q=200.0,
        net_income_latest_q=30.0,
        net_income_year_ago_q=25.0,
        source="edgar_pointintime",
        fetched_at=now,
        provenance=ProvenanceRecord(source="edgar_pointintime", fetched_at=now,
                                    parameters={}),
    )
    base.update(over)
    return FundamentalsSnapshot(**base)


class TestEvEbitda:
    def test_enterprise_value_adds_debt_and_subtracts_cash(self):
        r = compute_financial_ratios(_snap(), current_price=50.0)
        # EV = 10000 + 2000 - 500 = 11500; / 400 EBITDA
        assert r.ev_ebitda == pytest.approx(11500 / 400)

    def test_negative_ebitda_yields_none_not_a_cheap_looking_multiple(self):
        r = compute_financial_ratios(_snap(ebitda=-400.0), current_price=50.0)
        assert r.ev_ebitda is None, (
            "a negative denominator produces a negative multiple that reads as "
            "'cheap' and means nothing"
        )

    def test_missing_market_cap_voids_it(self):
        assert compute_financial_ratios(
            _snap(market_cap=None), current_price=50.0).ev_ebitda is None

    def test_absent_debt_and_cash_are_treated_as_zero(self):
        r = compute_financial_ratios(
            _snap(total_liabilities=None, cash_and_equivalents=None), current_price=50.0)
        assert r.ev_ebitda == pytest.approx(10000 / 400)


class TestCurrentRatio:
    def test_computed_from_current_assets_and_liabilities(self):
        assert compute_financial_ratios(
            _snap(), current_price=50.0).current_ratio == pytest.approx(900 / 600)

    def test_banks_reporting_neither_yield_none(self):
        """JPM, BAC, WFC, GS, MS, CB and AXP report no operating cycle. None is
        the correct answer here, not a defect to be patched with a zero."""
        r = compute_financial_ratios(
            _snap(current_assets=None, current_liabilities=None), current_price=50.0)
        assert r.current_ratio is None


class TestGrowth:
    def test_same_quarter_year_on_year(self):
        g = compute_growth_metrics(_snap())
        assert g.revenue_growth_yoy == pytest.approx((260 - 200) / 200)
        assert g.earnings_growth_yoy == pytest.approx((30 - 25) / 25)

    def test_negative_base_yields_none(self):
        """A swing through zero has no interpretable growth rate. Boeing's
        year-ago quarter was a loss; -150% would be a number, not a fact."""
        g = compute_growth_metrics(_snap(net_income_year_ago_q=-25.0))
        assert g.earnings_growth_yoy is None

    def test_zero_base_yields_none(self):
        assert compute_growth_metrics(
            _snap(revenue_year_ago_q=0.0)).revenue_growth_yoy is None

    def test_growth_compares_quarter_to_quarter_not_ttm_to_quarter(self):
        """The guard against a manufactured multi-hundred-percent growth rate:
        the TTM revenue field must never be the numerator here."""
        g = compute_growth_metrics(_snap(revenue=1000.0, revenue_latest_q=260.0,
                                         revenue_year_ago_q=200.0))
        assert g.revenue_growth_yoy < 1.0


class TestMargins:
    def test_gross_and_operating_margin(self):
        g = compute_growth_metrics(_snap())
        assert g.gross_margin == pytest.approx((1000 - 600) / 1000)
        assert g.operating_margin == pytest.approx(250 / 1000)

    def test_missing_cogs_yields_none(self):
        assert compute_growth_metrics(_snap(cost_of_revenue=None)).gross_margin is None


class TestBetaIsRequested:
    """compute_beta existed all along; nobody passed a benchmark."""

    def test_both_agents_define_a_benchmark(self):
        from hifi.agents import risk_agent, technical_agent

        assert risk_agent._BENCHMARK == "SPY"
        assert technical_agent._BENCHMARK == "SPY"

    @pytest.mark.parametrize("module", ["risk_agent", "technical_agent"])
    def test_benchmark_is_passed_to_the_tool(self, module):
        import inspect

        from hifi.agents import risk_agent, technical_agent

        mod = {"risk_agent": risk_agent, "technical_agent": technical_agent}[module]
        src = inspect.getsource(mod)
        assert "benchmark_ticker" in src, (
            f"{module} calls get_risk_metrics without benchmark_ticker, so the "
            "server returns beta=None -- the DJ-134 defect"
        )


class TestLiveCoverage:
    """Guards the real universe. These numbers are the DJ-134 acceptance
    criteria; a regression here means an agent has gone partially blind again."""

    def test_every_ticker_gets_the_non_financial_ratios(self):
        import json

        from hifi.data.universe import PHASE14_UNIVERSE
        from hifi.simulation.snapshot import build_pointintime_snapshot

        missing: dict[str, list[str]] = {}
        n = 0
        for entry in PHASE14_UNIVERSE:
            raw = build_pointintime_snapshot(entry["ticker"], "2026-08-31",
                                             data_dir="data")
            if raw is None:
                continue
            n += 1
            snap = FundamentalsSnapshot.model_validate_json(raw)
            price = json.loads(raw)["provenance"]["parameters"]["price_used"]
            r = compute_financial_ratios(snap, float(price))
            for field in ("pb", "ps", "roe", "roa", "debt_equity"):
                if getattr(r, field) is None:
                    missing.setdefault(field, []).append(entry["ticker"])
        if n == 0:
            pytest.skip("no live fundamentals in this checkout")
        assert not missing, f"universally-computable ratios are missing: {missing}"

    def test_bank_only_gaps_stay_confined_to_financials(self):
        import json

        from hifi.data.universe import PHASE14_UNIVERSE
        from hifi.simulation.snapshot import build_pointintime_snapshot

        sectors = {e["ticker"]: e["sector"] for e in PHASE14_UNIVERSE}
        offenders = []
        n = 0
        for entry in PHASE14_UNIVERSE:
            raw = build_pointintime_snapshot(entry["ticker"], "2026-08-31",
                                             data_dir="data")
            if raw is None:
                continue
            n += 1
            snap = FundamentalsSnapshot.model_validate_json(raw)
            price = json.loads(raw)["provenance"]["parameters"]["price_used"]
            blank = compute_financial_ratios(snap, float(price)).current_ratio is None
            if blank and sectors[entry["ticker"]] != "Financials":
                offenders.append(entry["ticker"])
        if n == 0:
            pytest.skip("no live fundamentals in this checkout")
        assert not offenders, (
            "current_ratio is absent outside Financials, so this is a data-path "
            f"defect rather than bank accounting: {offenders}"
        )
