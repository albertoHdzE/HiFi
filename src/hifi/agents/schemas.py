"""
Agent output schemas for HiFi (P3-E2, P4-E2).

Every agent produces an AgentSignal. Analysis envelopes (FundamentalAnalysis,
TechnicalAnalysis) wrap AgentSignal with the raw MCP tool results so every number in
the rationale can be traced to a specific deterministic computation.

Design rationale
----------------
AgentSignal.call_ids
    Phase 5 (Verification) uses these to match rationale numbers against tool outputs.

AgentSignal.data_gaps
    Fields that were None in MCP tool results. A rationale that cites a gap field
    without acknowledgment is a hallucination candidate.

TechnicalAnalysis (P4-E1)
    Information-restricted to price-derived data only (technical indicators + risk
    metrics). No access to fundamentals, valuation, or macro. This is the primary
    diversity mechanism for the Phase 4 ensemble (David §10.3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AgentSignal(BaseModel):
    """
    Atomic output of any HiFi agent (David §10.2).

    All agent specializations produce this schema. The collective decision engine
    (Phase 9) aggregates a list of AgentSignal objects -- one per agent per ticker.
    """

    ticker: str
    as_of_date: str  # ISO 8601
    decision: Literal["Buy", "Hold", "Sell"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    key_concern: str
    data_gaps: list[str] = Field(default_factory=list)
    call_ids: list[str] = Field(default_factory=list)
    model_id: str
    agent_type: str = "fundamental"

    @field_validator("rationale", "key_concern")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale and key_concern must not be empty strings")
        return v


class FundamentalAnalysis(BaseModel):
    """
    Full output of the Fundamental Analyst Agent (P3-E4).

    Wraps AgentSignal with raw MCP tool results for traceability.
    """

    signal: AgentSignal | None
    financial_ratios: dict
    growth_metrics: dict
    valuation_context: dict
    macro_snapshot: dict
    prompt_version: str
    latency_ms: float | None = None

    def tool_results_flat(self) -> dict:
        """Return all tool results merged for hallucination checking."""
        merged: dict = {}
        for d in (
            self.financial_ratios,
            self.growth_metrics,
            self.valuation_context,
            self.macro_snapshot,
        ):
            merged.update(d)
        return merged


class TechnicalAnalysis(BaseModel):
    """
    Full output of the Technical Analyst Agent (P4-E1).

    Information-restricted to technical indicators and risk metrics only.
    No access to fundamental or macro data -- restriction enforced by the agent's
    call_mcp_tools_node which only calls get_technical_indicators and get_risk_metrics.
    """

    signal: AgentSignal | None
    technical_indicators: dict
    risk_metrics: dict
    time_horizon: str | None = None  # "short-term" | "medium-term" | "long-term"
    prompt_version: str
    latency_ms: float | None = None

    def tool_results_flat(self) -> dict:
        """Return all tool results merged for hallucination checking."""
        merged: dict = {}
        for d in (self.technical_indicators, self.risk_metrics):
            merged.update(d)
        return merged
