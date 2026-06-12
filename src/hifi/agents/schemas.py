"""
Agent output schemas for HiFi (P3-E2, P4-E2, P8-E1).

Every agent produces an AgentSignal. Analysis envelopes (FundamentalAnalysis,
TechnicalAnalysis, RiskAnalysis, MacroAnalysis, SentimentAnalysis) wrap AgentSignal
with the raw MCP tool results so every number in the rationale can be traced to a
specific deterministic computation.

ContrarianAnalysis (P8-E5) is the exception: it does NOT produce an AgentSignal
because the Contrarian Agent does not vote in the confidence-weighted aggregation.
It is a second-pass critic that stress-tests the consensus (DJ-033).

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

RiskAnalysis (P8-E2)
    Information-restricted to risk metrics only (hist_vol, beta, max_drawdown, Sharpe,
    VaR). Model diversity: google/gemma-3-4b vs Qwen family (DJ-032).

MacroAnalysis (P8-E3)
    Information-restricted to macro snapshot only (fed_funds_rate, CPI, unemployment,
    yield curve, VIX, GDP). Model: qwen3.5-27b reasoning-distilled (DJ-032).

SentimentAnalysis (P8-E4)
    Information source: SEC filings via RAG only (no numerical tools). Qualitative
    analysis of management tone and forward guidance (DJ-034). Fail-open: empty
    retrieval returns a default "Insufficient Data" signal.

ContrarianAnalysis (P8-E5)
    Second-pass agent: receives all other agents' outputs + preliminary decision.
    Produces adversarial stress test, not a Buy/Hold/Sell vote (DJ-033).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class RiskAnalysis(BaseModel):
    """
    Full output of the Risk Analyst Agent (P8-E2).

    Information-restricted to risk metrics only (hist_vol, beta, max_drawdown,
    Sharpe, VaR). No access to fundamental, technical, or macro data.
    Model: google/gemma-3-4b (non-reasoning, max_tokens=1024) — DJ-032.

    risk_assessment is a structured narrative of the stock's risk profile.
    recommended_position_size is the agent's suggested portfolio weight [0, 1].
    """

    signal: AgentSignal | None
    risk_assessment: str
    recommended_position_size: float | None = None
    risk_metrics: dict = Field(default_factory=dict)
    prompt_version: str
    latency_ms: float | None = None

    def tool_results_flat(self) -> dict:
        """Return tool results for hallucination checking."""
        return dict(self.risk_metrics)


class MacroAnalysis(BaseModel):
    """
    Full output of the Macro Analyst Agent (P8-E3).

    Information-restricted to macro snapshot only (fed_funds_rate, CPI,
    unemployment, yield curve, VIX, GDP). No access to company-specific data.
    Model: qwen3.5-27b reasoning-distilled (max_tokens=4096) — DJ-032.

    regime_assessment is the macro regime classification (e.g. "stagflation risk",
    "soft landing", "tightening cycle peak").
    rationale provides the macro reasoning behind the signal.
    """

    signal: AgentSignal | None
    regime_assessment: str
    rationale: str
    macro_snapshot: dict = Field(default_factory=dict)
    prompt_version: str
    latency_ms: float | None = None

    def tool_results_flat(self) -> dict:
        """Return tool results for hallucination checking."""
        return dict(self.macro_snapshot)


class SentimentAnalysis(BaseModel):
    """
    Full output of the Sentiment Analyst Agent (P8-E4).

    Information source: SEC 8-K, 10-K MD&A, and 10-Q MD&A filings via RAG only
    (DJ-034). No access to numerical MCP tools. Qualitative analysis of management
    tone, forward guidance, and language signals.

    Fail-open behaviour: if retrieved_context is empty (no filing passages found),
    the agent returns a default "Insufficient Data" signal (decision=Hold,
    confidence=0.0) without calling the LLM.

    notable_signals are specific statements or phrases flagged from the filings.
    """

    signal: AgentSignal | None
    sentiment_summary: str
    notable_signals: list[str] = Field(default_factory=list)
    prompt_version: str
    latency_ms: float | None = None


class ContrarianAnalysis(BaseModel):
    """
    Full output of the Contrarian Agent (P8-E5).

    The Contrarian Agent is a second-pass critic (DJ-033):
    - Receives all other agents' outputs and the preliminary ensemble decision.
    - Does NOT produce a Buy/Hold/Sell vote (no AgentSignal field).
    - Produces an adversarial stress test of the consensus.
    - Its output is logged but does NOT change the collective_decision in Phase 8.
      (Integration into the decision mechanism is a Phase 9 concern.)

    Model: mlx-qwen3.5-35b reasoning-distilled (max_tokens=4096) — DJ-032.

    alternative_thesis is the bear/bull case opposite to the consensus.
    risk_scenario is a specific adverse scenario with estimated probability.
    counterargument is a structured argument against the dominant position.
    confidence is the Contrarian's conviction in its own view [0, 1].
    """

    alternative_thesis: str
    risk_scenario: str
    counterargument: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    prompt_version: str
    latency_ms: float | None = None

    @model_validator(mode="after")
    def validate_non_empty_strings(self) -> ContrarianAnalysis:
        for field in ("alternative_thesis", "risk_scenario", "counterargument"):
            v = getattr(self, field)
            if not v.strip():
                raise ValueError(f"{field} must not be empty")
        return self
