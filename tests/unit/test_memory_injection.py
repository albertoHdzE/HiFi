"""
Tests for agent memory injection into prompts (P13-E4-T3, DJ-076).

Verifies that when memory_prefix is supplied, each agent's generate_analysis_node
prepends it to the user message before calling the LLM. The memory_prefix itself
is produced by AgentMemoryStore.format_for_prompt(); this test suite validates
the injection plumbing, not the content of the prefix.

Agents covered: risk, macro, fundamental, technical, sentiment.
"""

import json

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MEMORY_PREFIX = (
    "[Agent Memory — last 1 decisions for AAPL]\n"
    "2022-12-31: Buy (confidence=0.75)"
)

_MINIMAL_RISK_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.50,
    "rationale": "hist_vol_20d of 0.25 indicates moderate risk.",
    "key_concern": "Elevated volatility.",
    "risk_assessment": "Moderate regime.",
    "recommended_position_size": 0.05,
})

_MINIMAL_MACRO_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.55,
    "rationale": "Fed funds rate at 5.0% signals tightening.",
    "key_concern": "Yield curve inversion risk.",
    "regime_assessment": "Restrictive monetary policy.",
    "macro_rationale": "High rates constrain equity valuations.",
})

_MINIMAL_FUND_RESPONSE = json.dumps({
    "decision": "Buy",
    "confidence": 0.70,
    "rationale": "P/E of 22 is below sector average.",
    "key_concern": "Revenue growth slowing.",
})

_MINIMAL_TECH_RESPONSE = json.dumps({
    "decision": "Hold",
    "confidence": 0.60,
    "rationale": "RSI at 52 is neutral.",
    "key_concern": "No clear trend signal.",
    "time_horizon": "medium-term",
})

_MINIMAL_SENT_RESPONSE = json.dumps({
    "decision": "Buy",
    "confidence": 0.65,
    "rationale": "MD&A cites record services revenue.",
    "key_concern": "FX headwinds mentioned.",
    "sentiment_summary": "Positive management tone.",
    "notable_signals": ["record services revenue", "FX headwinds cited"],
})


def _capturing_llm(captured: list, response_text: str):
    """Return a stub LLM that captures all messages it receives."""
    class _StubLLM:
        model_name = "test-model"

        def invoke(self, messages):
            captured.extend(messages)

            class _R:
                content = response_text
            return _R()

    return _StubLLM()


# ---------------------------------------------------------------------------
# Risk agent — generate_analysis_node memory injection
# ---------------------------------------------------------------------------

def test_risk_generate_analysis_node_injects_memory_prefix(monkeypatch):
    import hifi.agents.risk_agent as ra
    from hifi.agents.risk_agent import RiskAnalystState, generate_analysis_node

    captured: list = []
    stub = lambda *a, **kw: _capturing_llm(captured, _MINIMAL_RISK_RESPONSE)  # noqa: E731
    monkeypatch.setattr(ra, "make_llm", stub)

    state: RiskAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": {"risk_metrics": {"hist_vol_20d": 0.25, "call_id": "abc"}},
        "llm_response": "",
        "signal": None,
        "risk_assessment": None,
        "recommended_position_size": None,
        "error": None,
        "start_time": 0.0,
        "memory_prefix": _MEMORY_PREFIX,
    }
    generate_analysis_node(state)

    human_msgs = [m for m in captured if hasattr(m, "content") and _MEMORY_PREFIX in m.content]
    assert human_msgs, "memory_prefix not found in any message sent to the LLM"


def test_risk_generate_analysis_node_no_prefix_unchanged(monkeypatch):
    import hifi.agents.risk_agent as ra
    from hifi.agents.risk_agent import RiskAnalystState, generate_analysis_node

    captured: list = []
    stub = lambda *a, **kw: _capturing_llm(captured, _MINIMAL_RISK_RESPONSE)  # noqa: E731
    monkeypatch.setattr(ra, "make_llm", stub)

    state: RiskAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": {"risk_metrics": {"hist_vol_20d": 0.25, "call_id": "abc"}},
        "llm_response": "",
        "signal": None,
        "risk_assessment": None,
        "recommended_position_size": None,
        "error": None,
        "start_time": 0.0,
        "memory_prefix": "",
    }
    generate_analysis_node(state)

    # Prefix text must NOT appear when memory_prefix is empty
    any_has_prefix = any(
        hasattr(m, "content") and _MEMORY_PREFIX in m.content
        for m in captured
    )
    assert not any_has_prefix


def test_risk_run_function_accepts_memory_prefix(monkeypatch):
    """run_risk_analysis() signature accepts memory_prefix without error."""
    import hifi.agents.risk_agent as ra

    captured: list = []
    stub = lambda *a, **kw: _capturing_llm(captured, _MINIMAL_RISK_RESPONSE)  # noqa: E731
    monkeypatch.setattr(ra, "make_llm", stub)
    # Patch graph to avoid MCP calls
    monkeypatch.setattr(ra, "build_risk_graph", lambda: _stub_graph_for(ra, captured, _MINIMAL_RISK_RESPONSE))  # noqa: E501

    from hifi.agents.risk_agent import run_risk_analysis
    result = run_risk_analysis("AAPL", "2023-03-31", memory_prefix=_MEMORY_PREFIX)
    # If signature is wrong this would raise TypeError before returning
    assert result is not None


# ---------------------------------------------------------------------------
# Macro agent — generate_analysis_node memory injection
# ---------------------------------------------------------------------------

def test_macro_generate_analysis_node_injects_memory_prefix(monkeypatch):
    import hifi.agents.macro_agent as ma
    from hifi.agents.macro_agent import MacroAnalystState, generate_analysis_node

    captured: list = []
    monkeypatch.setattr(ma, "make_llm", lambda *a, **kw: _capturing_llm(captured, _MINIMAL_MACRO_RESPONSE))  # noqa: E501

    state: MacroAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": {"macro_snapshot": {"fed_funds_rate": 5.0, "call_id": "xyz"}},
        "llm_response": "",
        "signal": None,
        "regime_assessment": None,
        "macro_rationale": None,
        "error": None,
        "start_time": 0.0,
        "memory_prefix": _MEMORY_PREFIX,
    }
    generate_analysis_node(state)

    human_msgs = [m for m in captured if hasattr(m, "content") and _MEMORY_PREFIX in m.content]
    assert human_msgs, "memory_prefix not found in macro generate_analysis_node messages"


def test_macro_generate_analysis_node_empty_prefix_ok(monkeypatch):
    import hifi.agents.macro_agent as ma
    from hifi.agents.macro_agent import MacroAnalystState, generate_analysis_node

    captured: list = []
    monkeypatch.setattr(ma, "make_llm", lambda *a, **kw: _capturing_llm(captured, _MINIMAL_MACRO_RESPONSE))  # noqa: E501

    state: MacroAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": {"macro_snapshot": {"fed_funds_rate": 5.0, "call_id": "xyz"}},
        "llm_response": "",
        "signal": None,
        "regime_assessment": None,
        "macro_rationale": None,
        "error": None,
        "start_time": 0.0,
        # memory_prefix intentionally omitted — should default to ""
    }
    generate_analysis_node(state)

    # Should not raise; LLM should have been called
    assert len(captured) > 0


# ---------------------------------------------------------------------------
# Fundamental agent — generate_analysis_node memory injection
# ---------------------------------------------------------------------------

def _make_fund_tool_results():
    return {
        "financial_ratios": {"pe_ratio": 22.0, "call_id": "f1"},
        "growth_metrics": {"revenue_growth": 0.05, "call_id": "f2"},
        "valuation_context": {"ev_ebitda": 15.0, "call_id": "f3"},
        "macro_snapshot": {"fed_funds_rate": 5.0, "call_id": "f4"},
    }


def test_fundamental_generate_analysis_node_injects_memory_prefix(monkeypatch):
    import hifi.agents.fundamental_agent as fa
    from hifi.agents.fundamental_agent import FundamentalistState, generate_analysis_node

    captured: list = []
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _capturing_llm(captured, _MINIMAL_FUND_RESPONSE))  # noqa: E501

    state: FundamentalistState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "snapshot_json": "{}",
        "data_dir": "data",
        "tool_results": _make_fund_tool_results(),
        "retrieved_context": "",
        "llm_response": "",
        "signal": None,
        "error": None,
        "start_time": 0.0,
        "memory_prefix": _MEMORY_PREFIX,
    }
    generate_analysis_node(state)

    human_msgs = [m for m in captured if hasattr(m, "content") and _MEMORY_PREFIX in m.content]
    assert human_msgs, "memory_prefix not found in fundamental generate_analysis_node messages"


def test_fundamental_generate_analysis_node_no_prefix_ok(monkeypatch):
    import hifi.agents.fundamental_agent as fa
    from hifi.agents.fundamental_agent import FundamentalistState, generate_analysis_node

    captured: list = []
    monkeypatch.setattr(fa, "make_llm", lambda *a, **kw: _capturing_llm(captured, _MINIMAL_FUND_RESPONSE))  # noqa: E501

    state: FundamentalistState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "snapshot_json": "{}",
        "data_dir": "data",
        "tool_results": _make_fund_tool_results(),
        "retrieved_context": "",
        "llm_response": "",
        "signal": None,
        "error": None,
        "start_time": 0.0,
        "memory_prefix": "",
    }
    generate_analysis_node(state)
    assert len(captured) > 0


# ---------------------------------------------------------------------------
# Technical agent — generate_analysis_node memory injection
# ---------------------------------------------------------------------------

def _make_tech_tool_results():
    return {
        "technical_indicators": {"rsi_14": 52.0, "call_id": "t1"},
        "risk_metrics": {"hist_vol_20d": 0.22, "call_id": "t2"},
    }


def test_technical_generate_analysis_node_injects_memory_prefix(monkeypatch):
    import hifi.agents.technical_agent as ta
    from hifi.agents.technical_agent import TechnicalAnalystState, generate_analysis_node

    captured: list = []
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _capturing_llm(captured, _MINIMAL_TECH_RESPONSE))  # noqa: E501

    state: TechnicalAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": _make_tech_tool_results(),
        "retrieved_context": "",
        "llm_response": "",
        "signal": None,
        "time_horizon": None,
        "error": None,
        "start_time": 0.0,
        "memory_prefix": _MEMORY_PREFIX,
    }
    generate_analysis_node(state)

    human_msgs = [m for m in captured if hasattr(m, "content") and _MEMORY_PREFIX in m.content]
    assert human_msgs, "memory_prefix not found in technical generate_analysis_node messages"


def test_technical_generate_analysis_node_prefix_appears_before_analysis(monkeypatch):
    """Memory prefix must appear before the analytical content (not after)."""
    import hifi.agents.technical_agent as ta
    from hifi.agents.technical_agent import TechnicalAnalystState, generate_analysis_node

    captured: list = []
    monkeypatch.setattr(ta, "make_llm", lambda *a, **kw: _capturing_llm(captured, _MINIMAL_TECH_RESPONSE))  # noqa: E501

    state: TechnicalAnalystState = {
        "ticker": "AAPL",
        "as_of_date": "2023-03-31",
        "data_dir": "data",
        "tool_results": _make_tech_tool_results(),
        "retrieved_context": "",
        "llm_response": "",
        "signal": None,
        "time_horizon": None,
        "error": None,
        "start_time": 0.0,
        "memory_prefix": _MEMORY_PREFIX,
    }
    generate_analysis_node(state)

    human_content = next(
        (m.content for m in captured if hasattr(m, "content") and _MEMORY_PREFIX in m.content),
        None,
    )
    assert human_content is not None
    assert human_content.startswith(_MEMORY_PREFIX), (
        "memory_prefix must be the FIRST content in the user message"
    )


# ---------------------------------------------------------------------------
# Sentiment agent — _call_llm_for_sentiment memory injection
# ---------------------------------------------------------------------------

def test_sentiment_call_llm_injects_memory_prefix(monkeypatch):
    import hifi.agents.sentiment_agent as sa

    captured: list = []
    monkeypatch.setattr(sa, "make_llm", lambda *a, **kw: _capturing_llm(captured, _MINIMAL_SENT_RESPONSE))  # noqa: E501

    from hifi.agents.sentiment_agent import _call_llm_for_sentiment
    _call_llm_for_sentiment(
        ticker="AAPL",
        as_of_date="2023-03-31",
        retrieved_context="Apple services revenue grew 5% YoY.",
        model_id="gemma-4-e4b",
        memory_prefix=_MEMORY_PREFIX,
    )

    human_msgs = [m for m in captured if hasattr(m, "content") and _MEMORY_PREFIX in m.content]
    assert human_msgs, "memory_prefix not found in sentiment _call_llm_for_sentiment messages"


def test_sentiment_call_llm_empty_prefix_ok(monkeypatch):
    import hifi.agents.sentiment_agent as sa

    captured: list = []
    monkeypatch.setattr(sa, "make_llm", lambda *a, **kw: _capturing_llm(captured, _MINIMAL_SENT_RESPONSE))  # noqa: E501

    from hifi.agents.sentiment_agent import _call_llm_for_sentiment
    _call_llm_for_sentiment(
        ticker="AAPL",
        as_of_date="2023-03-31",
        retrieved_context="Apple services revenue grew 5% YoY.",
        model_id="gemma-4-e4b",
        memory_prefix="",
    )
    assert len(captured) > 0


def test_sentiment_run_accepts_memory_prefix(monkeypatch, tmp_path):
    """run_sentiment_analysis() accepts memory_prefix without error."""
    import hifi.agents.sentiment_agent as sa

    passages = [{"rank": 1, "filing_type": "10-K", "section": "MD&A",
                 "period": "2022-09-30", "text": "Apple services revenue grew 5% YoY."}]
    monkeypatch.setattr(sa, "call_tool", lambda *a, **kw: {"passages": passages, "call_id": "x"})
    monkeypatch.setattr(sa, "make_llm", lambda *a, **kw: _capturing_llm([], _MINIMAL_SENT_RESPONSE))

    from hifi.agents.sentiment_agent import run_sentiment_analysis
    result = run_sentiment_analysis(
        "AAPL", "2023-03-31", data_dir=str(tmp_path), memory_prefix=_MEMORY_PREFIX
    )
    assert result is not None
    assert result.signal is not None


# ---------------------------------------------------------------------------
# Ensemble runner — memory_prefixes parameter
# ---------------------------------------------------------------------------

def test_ensemble_run_accepts_memory_prefixes_param():
    """run_ensemble() and run_debate_ensemble() accept memory_prefixes without error.

    We only verify the parameter is accepted (no TypeError). Actual injection
    is tested per-agent above. Full integration test requires LM Studio.
    """
    import inspect

    from hifi.agents.ensemble_runner import run_debate_ensemble, run_ensemble

    sig_run = inspect.signature(run_ensemble)
    assert "memory_prefixes" in sig_run.parameters

    sig_debate = inspect.signature(run_debate_ensemble)
    assert "memory_prefixes" in sig_debate.parameters


def test_ensemble_memory_prefixes_default_is_none():
    import inspect

    from hifi.agents.ensemble_runner import run_ensemble

    param = inspect.signature(run_ensemble).parameters["memory_prefixes"]
    assert param.default is None


# ---------------------------------------------------------------------------
# Stub graph helper (avoids MCP calls in run_function tests)
# ---------------------------------------------------------------------------

def _stub_graph_for(module, captured_list, response_text):
    """Return a minimal stub graph that skips MCP calls and returns a hardcoded state."""
    class _StubGraph:
        def invoke(self, state, config=None):
            return {
                "tool_results": {"risk_metrics": {"hist_vol_20d": 0.25, "call_id": "abc"}},
                "llm_response": response_text,
                "signal": None,
                "risk_assessment": None,
                "recommended_position_size": None,
                "error": None,
            }
    return _StubGraph()
