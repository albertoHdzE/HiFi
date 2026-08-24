"""Tests for per-stock decision traceability (DJ-120)."""

from __future__ import annotations

import json

import pytest

from hifi.analytics import decision_audit as da


def _sidecar(tmp, account, date, ticker, *, tech_error=False, sent_conf=0.7):
    d = tmp / "live" / account / "walkforward" / date / "parallel" / date[:4] / date[5:7]
    d.mkdir(parents=True, exist_ok=True)
    tech_block = {
        "signal": {
            "ticker": ticker, "decision": "Sell" if tech_error else "Buy",
            "confidence": 1.0 if tech_error else 0.7,
            "rationale": "No data available." if tech_error else "Momentum positive.",
            "key_concern": "none", "data_gaps": [], "call_ids": ["abc"],
            "model_id": "/models/Qwen2.5-Coder", "agent_type": "technical",
        },
        "technical_indicators": (
            {"error": "TICKER_NOT_FOUND", "detail": "x", "call_id": "abc"}
            if tech_error else {"sma": 1.0, "call_id": "abc"}
        ),
        "latency_ms": 10.0,
        "prompt_version": "technical_v1",
    }
    payload = {
        "ticker": ticker,
        "as_of_date": date,
        "technical_analysis": tech_block,
        "sentiment_analysis": {
            "signal": {
                "ticker": ticker, "decision": "Hold", "confidence": sent_conf,
                "rationale": "Neutral tone.", "agent_type": "sentiment",
                "model_id": "gemma-3-12b-it", "call_ids": [],
            },
            "latency_ms": 5.0,
        },
        "ensemble_decision": {
            "collective_decision": "Sell" if tech_error else "Hold",
            "collective_confidence": 0.8,
            "agent_decisions": ["Sell", "Hold"] if tech_error else ["Buy", "Hold"],
            "disagreement_entropy": 0.97,
            "agreement": False,
            "opinion_dispersion": 0.3,
        },
    }
    (d / f"{ticker}.json").write_text(json.dumps(payload))


def _decisions(tmp, account, rows):
    p = tmp / "live" / account
    p.mkdir(parents=True, exist_ok=True)
    with (p / "decisions.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture
def store(tmp_path):
    _sidecar(tmp_path, "A", "2026-08-12", "ACN", tech_error=True)
    _sidecar(tmp_path, "A", "2026-08-13", "ACN", tech_error=False)
    _sidecar(tmp_path, "A", "2026-08-13", "NVDA", tech_error=False)
    _decisions(tmp_path, "A", [{
        "decision_date": "2026-08-13", "account": "A", "n_orders": 1,
        "orders": [{"ticker": "NVDA", "side": "buy", "qty": 2.0, "status": "filled"}],
    }])
    return str(tmp_path)


class TestIterSidecars:
    def test_filters_by_ticker_and_date(self, store):
        assert len(list(da.iter_sidecars("A", store))) == 3
        assert len(list(da.iter_sidecars("A", store, ticker="ACN"))) == 2
        assert len(list(da.iter_sidecars("A", store, date="2026-08-13"))) == 2

    def test_missing_arm_yields_nothing(self, store):
        assert list(da.iter_sidecars("C", store)) == []


class TestProvenance:
    def test_failure_rate_per_agent_per_date(self, store):
        pm = da.provenance_matrix("A", store)
        bad = pm[(pm.decision_date == "2026-08-12") & (pm.agent == "technical")].iloc[0]
        assert bad.failed == 1 and bad.failure_rate == 1.0
        good = pm[(pm.decision_date == "2026-08-13") & (pm.agent == "technical")].iloc[0]
        assert good.failed == 0

    def test_agent_behaviour_splits_clean_from_starved(self, store):
        """The diagnostic that exposed DJ-120: the technical agent said Sell
        only when its tools failed and Buy whenever they worked."""
        beh = da.agent_behaviour("A", store).set_index("agent")
        tech = beh.loc["technical"]
        assert tech["when_tools_failed"] == {"Sell": 1}
        assert tech["when_tools_ok"] == {"Buy": 2}

    def test_degenerate_agents_flags_constant_member(self, store):
        """Sentiment holds on every pass here, as it did in production."""
        deg = da.degenerate_agents("A", store)
        assert "sentiment" in set(deg["agent"])
        assert "technical" not in set(deg["agent"])


class TestTrace:
    def test_returns_none_for_absent_pair(self, store):
        assert da.trace("A", "ACN", "1999-01-01", store) is None
        assert da.trace("C", "ACN", "2026-08-13", store) is None

    def test_marks_evidence_incomplete_on_tool_failure(self, store):
        t = da.trace("A", "ACN", "2026-08-12", store)
        assert t["evidence_complete"] is False
        assert t["n_tool_failures"] == 1
        tech = next(a for a in t["agents"] if a["agent"] == "technical")
        assert tech["tools_failed"] == ["technical_indicators"]

    def test_marks_evidence_complete_when_tools_ok(self, store):
        t = da.trace("A", "ACN", "2026-08-13", store)
        assert t["evidence_complete"] is True
        assert t["n_tool_failures"] == 0

    def test_links_decision_to_order(self, store):
        assert da.trace("A", "NVDA", "2026-08-13", store)["order"]["side"] == "buy"
        # A held a view on ACN the same day but no order was placed.
        assert da.trace("A", "ACN", "2026-08-13", store)["order"] is None

    def test_captures_model_and_rationale(self, store):
        t = da.trace("A", "ACN", "2026-08-12", store)
        tech = next(a for a in t["agents"] if a["agent"] == "technical")
        assert tech["model_id"] == "Qwen2.5-Coder"
        assert "No data available" in tech["rationale"]

    def test_agents_in_canonical_order(self, store):
        t = da.trace("A", "ACN", "2026-08-13", store)
        names = [a["agent"] for a in t["agents"]]
        assert names.index("technical") < names.index("sentiment")


class TestFormatTrace:
    def test_warns_when_evidence_incomplete(self, store):
        txt = da.format_trace(da.trace("A", "ACN", "2026-08-12", store))
        assert "INCOMPLETE" in txt
        assert "uninformed" in txt

    def test_explains_absent_order(self, store):
        txt = da.format_trace(da.trace("A", "ACN", "2026-08-13", store))
        assert "none placed" in txt

    def test_handles_none(self):
        assert "no agents" in da.format_trace(None).lower() or \
               "No ensemble sidecar" in da.format_trace(None)


class TestTickerHistory:
    def test_one_row_per_date_with_provenance_column(self, store):
        h = da.ticker_history("A", "ACN", store)
        assert list(h["decision_date"]) == ["2026-08-12", "2026-08-13"]
        assert list(h["n_tool_failures"]) == [1, 0]

    def test_carries_per_agent_votes(self, store):
        h = da.ticker_history("A", "ACN", store).set_index("decision_date")
        assert h.loc["2026-08-12", "a_technical"] == "Sell"
        assert h.loc["2026-08-13", "a_technical"] == "Buy"

    def test_empty_for_unknown_ticker(self, store):
        assert da.ticker_history("A", "ZZZZ", store).empty


class TestDateScoping:
    """Health questions are about the present, not the pooled past (DJ-125).

    Without a date filter, `degenerate_agents` keeps reporting agents that were
    constant only during the DJ-120 starved period. On 2026-08-18 the pooled
    view still flagged fundamental and sentiment while that day alone had no
    agent above a 0.948 modal share — the history was masquerading as the
    current state.
    """

    def test_agent_behaviour_scopes_to_one_date(self, store):
        allb = da.agent_behaviour("A", store).set_index("agent")
        scoped = da.agent_behaviour("A", store, date="2026-08-13").set_index("agent")
        assert allb.loc["technical", "n"] == 3
        assert scoped.loc["technical", "n"] == 2

    def test_degenerate_agents_scopes_to_one_date(self, store):
        """technical is Sell-only on 08-12 and Buy-only on 08-13; constant
        within each date, varied when pooled."""
        pooled = set(da.degenerate_agents("A", store)["agent"])
        d12 = set(da.degenerate_agents("A", store, date="2026-08-12")["agent"])
        assert "technical" not in pooled
        assert "technical" in d12

    def test_unknown_date_is_empty_not_an_error(self, store):
        assert da.agent_behaviour("A", store, date="1999-01-01").empty
        assert da.degenerate_agents("A", store, date="1999-01-01").empty
