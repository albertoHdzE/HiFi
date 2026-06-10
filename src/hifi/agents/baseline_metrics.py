"""
Baseline evaluation metrics for Phase 3 (P3-E4, P3-E5).

Computes the Phase 3 quality metrics from a set of FundamentalAnalysis outputs.
These metrics are the floor against which Phase 4+ improvements are measured.

Metrics
-------
compliance_rate
    Fraction of analyses where signal is not None (valid AgentSignal produced).
hallucinated_numbers
    Total count of numeric values in rationale text that do NOT appear (within
    a 1% tolerance) in any MCP tool result. This is the Phase 3 approximation of
    what Phase 5 will verify rigorously.
data_gaps_acknowledged
    Count of analyses where at least one data_gap field appears in the rationale
    with an acknowledgment phrase ("unavailable", "null", "insufficient", "not available",
    "no data", "missing"). Measures how well the agent handles None inputs.
mean_call_id_coverage
    Mean fraction of tool results that were cited via call_ids in each signal.
    A coverage of 1.0 means the agent cited every tool call in its rationale.
mean_latency_ms
    Average wall-clock time per analysis in milliseconds.
"""

from __future__ import annotations

import re
from typing import Any

from hifi.agents.schemas import FundamentalAnalysis

_ACKNOWLEDGMENT_PHRASES = (
    "unavailable", "null", "insufficient", "not available",
    "no data", "missing", "not provided",
)


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a string."""
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


def number_in_tool_results(value: float, tool_results: dict[str, Any], tol: float = 0.01) -> bool:
    """Return True if value appears within tol (relative) of any numeric tool result value."""
    for v in _flatten_values(tool_results):
        if not isinstance(v, (int, float)):
            continue
        if v == 0.0:
            if abs(value) < 1e-9:
                return True
            continue
        if abs((value - v) / v) <= tol:
            return True
    return False


def _flatten_values(obj: Any) -> list[Any]:
    """Recursively collect all leaf values from a nested dict/list structure."""
    if isinstance(obj, dict):
        result = []
        for v in obj.values():
            result.extend(_flatten_values(v))
        return result
    if isinstance(obj, list):
        result = []
        for item in obj:
            result.extend(_flatten_values(item))
        return result
    return [obj]


def count_hallucinated_numbers(analysis: FundamentalAnalysis) -> int:
    """
    Count numbers in the rationale that do not appear in any MCP tool result.

    A number is considered hallucinated if it does not appear (within 1% tolerance)
    in the flat list of all numeric values returned by the four MCP tools.
    Small integers (0, 1, 2, 3) are excluded because they likely refer to
    ordinal positions or counts rather than financial values.
    """
    if analysis.signal is None:
        return 0
    rationale_numbers = extract_numbers(analysis.signal.rationale)
    tool_flat = analysis.tool_results_flat()
    count = 0
    for n in rationale_numbers:
        if abs(n) <= 3:  # skip small integers that are likely non-financial
            continue
        if not number_in_tool_results(n, tool_flat):
            count += 1
    return count


def data_gap_acknowledged(analysis: FundamentalAnalysis) -> bool:
    """
    Return True if the agent acknowledged at least one data gap in its rationale.

    For each field in data_gaps, check whether the rationale contains the field name
    AND at least one acknowledgment phrase nearby (within the same sentence).
    """
    if analysis.signal is None or not analysis.signal.data_gaps:
        return True  # no gaps to acknowledge
    rationale_lower = analysis.signal.rationale.lower()
    for gap_field in analysis.signal.data_gaps:
        if gap_field.lower() in rationale_lower:
            for phrase in _ACKNOWLEDGMENT_PHRASES:
                if phrase in rationale_lower:
                    return True
    return False


def call_id_coverage(analysis: FundamentalAnalysis) -> float:
    """
    Fraction of MCP tool results that were cited via call_ids in the signal.

    A coverage of 1.0 means the agent cited all four tool calls in its output.
    The total is the number of tool results that had a call_id field.
    """
    if analysis.signal is None:
        return 0.0
    tool_results_with_ids = sum(
        1 for r in [
            analysis.financial_ratios,
            analysis.growth_metrics,
            analysis.valuation_context,
            analysis.macro_snapshot,
        ]
        if isinstance(r, dict) and "call_id" in r
    )
    if tool_results_with_ids == 0:
        return 0.0
    cited = len(analysis.signal.call_ids)
    return min(cited / tool_results_with_ids, 1.0)


def compute_metrics(analyses: dict[str, FundamentalAnalysis]) -> dict[str, Any]:
    """
    Compute baseline metrics over a set of FundamentalAnalysis outputs.

    Parameters
    ----------
    analyses : dict[str, FundamentalAnalysis]
        Mapping of ticker -> FundamentalAnalysis.

    Returns
    -------
    dict
        Metrics dict suitable for inclusion in the phase3_baseline.json fixture.
    """
    if not analyses:
        return {
            "compliance_rate": 0.0,
            "hallucinated_numbers": 0,
            "data_gaps_acknowledged": 0,
            "mean_call_id_coverage": 0.0,
            "mean_latency_ms": 0.0,
            "n_analyses": 0,
        }

    valid = [a for a in analyses.values() if a.signal is not None]
    compliance_rate = len(valid) / len(analyses)

    hallucinated = sum(count_hallucinated_numbers(a) for a in valid)
    gaps_ack = sum(1 for a in valid if data_gap_acknowledged(a))
    coverages = [call_id_coverage(a) for a in valid]
    mean_coverage = sum(coverages) / len(coverages) if coverages else 0.0
    latencies = [a.latency_ms for a in analyses.values() if a.latency_ms is not None]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "compliance_rate": round(compliance_rate, 4),
        "hallucinated_numbers": hallucinated,
        "data_gaps_acknowledged": gaps_ack,
        "mean_call_id_coverage": round(mean_coverage, 4),
        "mean_latency_ms": round(mean_latency, 1),
        "n_analyses": len(analyses),
    }
