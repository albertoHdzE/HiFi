"""
Verification metrics aggregation (P5-E6).

compute_verification_metrics() aggregates a set of AgentVerificationReport
objects (one per ticker per agent) into phase-level summary statistics. This
is the measurement surface against which Phase 7 (RAG) improvements will be
quantified: a reduction in mean_hallucination_rate between the Phase 5
baseline and the Phase 7 post-RAG run is the primary evidence of improvement.

alias_table_coverage
--------------------
Fraction of all claims that were NOT unresolvable:
    coverage = 1 - (total_unresolvable / total_claims)
When coverage is below 0.90 (i.e. unresolvable_rate > 0.10), the alias table
is missing common field reference patterns and should be extended before
Phase 5 is considered complete (P5-E2 coverage goal).
"""

from __future__ import annotations

from typing import Any

from hifi.verification.schemas import AgentVerificationReport


def compute_verification_metrics(
    reports: dict[str, AgentVerificationReport],
) -> dict[str, Any]:
    """
    Compute phase-level verification metrics across a set of agent reports.

    Parameters
    ----------
    reports : dict[str, AgentVerificationReport]
        Mapping of an identifier (e.g. ticker or "AAPL_fundamental") to an
        AgentVerificationReport. May contain reports from one or both agents
        across multiple tickers.

    Returns
    -------
    dict
        Summary statistics suitable for inclusion in the phase5_verification
        baseline fixture. Keys:
        - mean_hallucination_rate : float -- mean HR across all reports
        - mean_grounding_rate     : float -- mean GR across all reports
        - mean_unresolvable_rate  : float -- mean n_unresolvable / n_claims
        - alias_table_coverage    : float -- 1 - mean_unresolvable_rate
        - n_reports               : int   -- number of reports
        - total_claims            : int   -- sum of n_claims
        - total_verified          : int   -- sum of n_verified
        - total_hallucinated      : int   -- sum of n_hallucinated
        - total_unresolvable      : int   -- sum of n_unresolvable
        - n_flagged_high_hr       : int   -- reports with flag_high_hr=True
    """
    if not reports:
        return {
            "mean_hallucination_rate": 0.0,
            "mean_grounding_rate": 0.0,
            "mean_unresolvable_rate": 0.0,
            "alias_table_coverage": 1.0,
            "n_reports": 0,
            "total_claims": 0,
            "total_verified": 0,
            "total_hallucinated": 0,
            "total_unresolvable": 0,
            "n_flagged_high_hr": 0,
        }

    vals = list(reports.values())
    n = len(vals)

    hrs = [r.hallucination_rate for r in vals]
    grs = [r.grounding_rate for r in vals]

    # Unresolvable rate per report: n_unresolvable / n_claims (0.0 if 0 claims).
    unresolvable_rates = [
        r.n_unresolvable / r.n_claims if r.n_claims > 0 else 0.0
        for r in vals
    ]

    mean_hr = sum(hrs) / n
    mean_gr = sum(grs) / n
    mean_ur = sum(unresolvable_rates) / n

    return {
        "mean_hallucination_rate": round(mean_hr, 6),
        "mean_grounding_rate": round(mean_gr, 6),
        "mean_unresolvable_rate": round(mean_ur, 6),
        "alias_table_coverage": round(1.0 - mean_ur, 6),
        "n_reports": n,
        "total_claims": sum(r.n_claims for r in vals),
        "total_verified": sum(r.n_verified for r in vals),
        "total_hallucinated": sum(r.n_hallucinated for r in vals),
        "total_unresolvable": sum(r.n_unresolvable for r in vals),
        "n_flagged_high_hr": sum(1 for r in vals if r.flag_high_hr),
    }
