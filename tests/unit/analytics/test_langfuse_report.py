"""Tests for LangFuse AI-operations analytics (DJ-116)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hifi.analytics import langfuse_report as lf


def _obs(model, latency, pt, ct, level="DEFAULT"):
    return {"model": model, "latency": latency, "promptTokens": pt,
            "completionTokens": ct, "totalTokens": pt + ct, "level": level,
            "startTime": "2026-07-20T04:00:00Z"}


def _resp(items):
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = {"data": items}
    return r


class TestFetch:
    def test_no_config_returns_empty(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert lf.fetch_generations().empty

    def test_paginates_and_parses(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        page1 = [_obs("llama", 30.0, 2000, 100), _obs("mistral", 4.0, 1000, 150)]
        with patch("requests.get", side_effect=[_resp(page1), _resp([])]):
            df = lf.fetch_generations()
        assert len(df) == 2
        assert set(df["model"]) == {"llama", "mistral"}

    def test_fetch_failure_returns_empty(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        with patch("requests.get", side_effect=RuntimeError("boom")):
            assert lf.fetch_generations().empty


class TestAggregates:
    def _df(self):
        import pandas as pd
        return pd.DataFrame([
            {"model": "llama", "latency": 30.0, "prompt_tokens": 2000,
             "completion_tokens": 100, "total_tokens": 2100, "level": "DEFAULT"},
            {"model": "llama", "latency": 32.0, "prompt_tokens": 2200,
             "completion_tokens": 120, "total_tokens": 2320, "level": "ERROR"},
            {"model": "mistral", "latency": 4.0, "prompt_tokens": 1000,
             "completion_tokens": 150, "total_tokens": 1150, "level": "DEFAULT"},
        ])

    def test_model_usage_table(self):
        u = lf.model_usage_table(self._df())
        assert u.loc["llama", "n_calls"] == 2
        assert u.loc["llama", "total_tokens"] == 4420
        assert u.loc["llama", "error_rate"] == 0.5
        assert u.loc["mistral", "n_calls"] == 1

    def test_ops_summary(self):
        s = lf.llm_ops_summary(self._df())
        assert s["n_calls"] == 3
        assert s["total_tokens"] == 5570
        assert s["n_models"] == 2
        assert s["error_rate"] == round(1 / 3, 4)

    def test_empty_summary(self):
        import pandas as pd
        s = lf.llm_ops_summary(pd.DataFrame())
        assert s["n_calls"] == 0 and s["mean_latency_s"] is None
