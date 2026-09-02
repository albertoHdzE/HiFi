"""Aggregation of grounding and hallucination rates — 0% covered before this.

These are numbers that go in the paper. ``mean_hallucination_rate`` is the
measurement surface the Phase 5 → Phase 7 RAG improvement is claimed against, so
an aggregation bug here does not crash anything; it moves a published result.

The subtle one is the unresolvable rate. Unresolvable claims are *measurement
gaps* — an alias the table does not know — and must not count as hallucinations,
or an incomplete alias table reads as a lying agent.
"""

from __future__ import annotations

import pytest

from hifi.verification.metrics import compute_verification_metrics
from hifi.verification.schemas import (
    AgentVerificationReport,
    NumericalClaim,
    VerificationResult,
)


def _result(status: str, cited: bool = True) -> VerificationResult:
    return VerificationResult(
        claim=NumericalClaim(field_alias="P/E", canonical_field="pe",
                             value=1.0, context_snippet="..."),
        status=status, tool_value=1.0, tool_field="pe",
        call_id_cited=cited, tolerance_used=0.01,
    )


def _report(verified=0, hallucinated=0, unresolvable=0, cited=True,
            ticker="AAPL") -> AgentVerificationReport:
    results = ([_result("verified", cited)] * verified
               + [_result("hallucinated")] * hallucinated
               + [_result("unresolvable")] * unresolvable)
    return AgentVerificationReport(
        ticker=ticker, as_of_date="2026-08-31", agent_type="fundamental",
        prompt_version="v1", results=results,
    )


class TestEmptyInput:
    def test_no_reports_gives_neutral_values_not_a_crash(self):
        m = compute_verification_metrics({})
        assert m["n_reports"] == 0
        assert m["mean_hallucination_rate"] == 0.0
        assert m["total_claims"] == 0

    def test_empty_coverage_is_one_not_zero(self):
        # Coverage is 1 - unresolvable_rate. With no claims, nothing failed to
        # resolve; reporting 0.0 would read as "the alias table resolved
        # nothing" and trigger the P5-E2 remediation for no reason.
        assert compute_verification_metrics({})["alias_table_coverage"] == 1.0


class TestRatesAreMeansOverReports:
    def test_hallucination_rate_excludes_unresolvable_from_the_denominator(self):
        """HR = n_hallucinated / (n_claims - n_unresolvable).

        Counting unresolvable claims as denominators would make an incomplete
        alias table look like a more honest agent, which is backwards.
        """
        r = _report(verified=1, hallucinated=1, unresolvable=8)
        assert r.hallucination_rate == pytest.approx(0.5)
        m = compute_verification_metrics({"a": r})
        assert m["mean_hallucination_rate"] == pytest.approx(0.5)

    def test_mean_is_over_reports_not_over_claims(self):
        """One report with many claims must not outvote one with few.

        The unit of analysis is the agent-ticker pass, so a chatty rationale
        cannot dominate the phase-level statistic.
        """
        m = compute_verification_metrics({
            "few": _report(verified=1, hallucinated=1),        # HR 0.5
            "many": _report(verified=100, hallucinated=0),     # HR 0.0
        })
        assert m["mean_hallucination_rate"] == pytest.approx(0.25)

    def test_grounding_rate_is_the_cited_fraction_of_verified(self):
        # An agent can be perfectly accurate and still cite nothing: correct
        # numbers with no audit trail. GR is what separates the two.
        grounded = compute_verification_metrics({"a": _report(verified=4, cited=True)})
        ungrounded = compute_verification_metrics({"a": _report(verified=4, cited=False)})
        assert grounded["mean_grounding_rate"] == pytest.approx(1.0)
        assert ungrounded["mean_grounding_rate"] == pytest.approx(0.0)
        assert ungrounded["mean_hallucination_rate"] == 0.0, (
            "an ungrounded but accurate agent is not a hallucinating one"
        )

    def test_unresolvable_rate_is_per_report_over_all_claims(self):
        m = compute_verification_metrics({"a": _report(verified=6, unresolvable=4)})
        assert m["mean_unresolvable_rate"] == pytest.approx(0.4)
        assert m["alias_table_coverage"] == pytest.approx(0.6)

    def test_a_report_with_no_claims_contributes_zero_not_nan(self):
        m = compute_verification_metrics({"empty": _report(), "a": _report(verified=2)})
        assert m["mean_unresolvable_rate"] == 0.0
        assert m["n_reports"] == 2


class TestTotalsAreSums:
    def test_totals_add_across_reports(self):
        m = compute_verification_metrics({
            "a": _report(verified=3, hallucinated=1, unresolvable=2),
            "b": _report(verified=5, hallucinated=0, unresolvable=1),
        })
        assert m["total_claims"] == 12
        assert m["total_verified"] == 8
        assert m["total_hallucinated"] == 1
        assert m["total_unresolvable"] == 3
        assert m["n_reports"] == 2

    def test_high_hr_reports_are_counted(self):
        # Threshold is 0.25; 3 of 4 resolvable fabricated is 0.75.
        m = compute_verification_metrics({
            "bad": _report(verified=1, hallucinated=3),
            "good": _report(verified=10, hallucinated=0),
        })
        assert m["n_flagged_high_hr"] == 1

    def test_coverage_below_the_phase5_goal_is_visible(self):
        # >10% unresolvable means the alias table needs extending (P5-E2).
        m = compute_verification_metrics({"a": _report(verified=8, unresolvable=2)})
        assert m["alias_table_coverage"] < 0.90


class TestOutputContract:
    """The fixture schema downstream analysis reads."""

    def test_every_documented_key_is_present(self):
        m = compute_verification_metrics({"a": _report(verified=1)})
        assert set(m) == {
            "mean_hallucination_rate", "mean_grounding_rate",
            "mean_unresolvable_rate", "alias_table_coverage", "n_reports",
            "total_claims", "total_verified", "total_hallucinated",
            "total_unresolvable", "n_flagged_high_hr",
        }

    def test_rates_are_rounded_for_stable_fixtures(self):
        m = compute_verification_metrics({f"r{i}": _report(verified=1, hallucinated=1)
                                          for i in range(3)})
        for k in ("mean_hallucination_rate", "mean_grounding_rate",
                  "mean_unresolvable_rate", "alias_table_coverage"):
            assert m[k] == round(m[k], 6), f"{k} would churn the baseline fixture"

    @pytest.mark.parametrize("key", ["n_reports", "total_claims", "total_verified",
                                     "total_hallucinated", "total_unresolvable",
                                     "n_flagged_high_hr"])
    def test_counts_are_ints(self, key):
        assert isinstance(compute_verification_metrics({"a": _report(verified=1)})[key], int)

    def test_coverage_and_unresolvable_rate_are_complementary(self):
        m = compute_verification_metrics({"a": _report(verified=7, unresolvable=3)})
        assert m["alias_table_coverage"] + m["mean_unresolvable_rate"] == \
            pytest.approx(1.0)
