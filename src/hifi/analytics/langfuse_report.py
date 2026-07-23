"""LangFuse AI-operations analytics for the live report (DJ-116).

The ensemble sidecars capture *what* each agent decided (the scientific ground
truth). LangFuse captures *how* the LLMs ran — per-call model, token usage,
latency, and errors. This module queries the LangFuse public API and aggregates
that operational telemetry per model/agent for the report's "AI operations"
panel.

Distinct from live_report.herding_series (decision science): this is the compute
/cost/latency/reliability layer. Degrades gracefully — returns empty frames when
LangFuse is unreachable, unconfigured, or has no data.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def _config() -> tuple[str, str, str] | None:
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk:
        return None
    return host, pk, sk


def fetch_generations(max_items: int = 20000, timeout_s: int = 15) -> pd.DataFrame:
    """LLM-call telemetry from LangFuse as a DataFrame.

    Columns: model, latency, prompt_tokens, completion_tokens, total_tokens,
    level, start_time. Empty frame if LangFuse is unconfigured/unreachable.
    """
    import requests  # noqa: PLC0415

    cfg = _config()
    cols = ["model", "latency", "prompt_tokens", "completion_tokens",
            "total_tokens", "level", "start_time"]
    if cfg is None:
        logger.info("LangFuse not configured; skipping AI-ops telemetry.")
        return pd.DataFrame(columns=cols)
    host, pk, sk = cfg

    rows: list[dict] = []
    page = 1
    try:
        while len(rows) < max_items:
            r = requests.get(
                f"{host}/api/public/observations",
                params={"type": "GENERATION", "limit": 100, "page": page},
                auth=(pk, sk), timeout=timeout_s,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                break
            for o in data:
                rows.append({
                    "model": o.get("model"),
                    "latency": o.get("latency"),
                    "prompt_tokens": o.get("promptTokens"),
                    "completion_tokens": o.get("completionTokens"),
                    "total_tokens": o.get("totalTokens"),
                    "level": o.get("level"),
                    "start_time": o.get("startTime"),
                })
            page += 1
    except Exception as exc:
        logger.warning("LangFuse telemetry fetch failed: %s", exc)
        if not rows:
            return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    if "start_time" in df:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    return df


def model_usage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model aggregates: calls, tokens, latency, error rate."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("model", dropna=False)
    out = pd.DataFrame({
        "n_calls": g.size(),
        "total_tokens": g["total_tokens"].sum(min_count=1),
        "mean_prompt_tokens": g["prompt_tokens"].mean().round(0),
        "mean_completion_tokens": g["completion_tokens"].mean().round(0),
        "mean_latency_s": g["latency"].mean().round(2),
        "p95_latency_s": g["latency"].quantile(0.95).round(2),
        "error_rate": g["level"].apply(
            lambda s: round((s == "ERROR").mean(), 4) if len(s) else 0.0),
    })
    return out.sort_values("n_calls", ascending=False)


def llm_ops_summary(df: pd.DataFrame) -> dict:
    """Headline operational totals across all captured LLM calls."""
    if df.empty:
        return {"n_calls": 0, "total_tokens": 0, "n_models": 0,
                "mean_latency_s": None, "error_rate": None, "note": "no LangFuse data"}
    return {
        "n_calls": int(len(df)),
        "total_tokens": int(df["total_tokens"].sum(min_count=1) or 0),
        "n_models": int(df["model"].nunique()),
        "mean_latency_s": (round(float(df["latency"].mean()), 2)
                           if df["latency"].notna().any() else None),
        "error_rate": round(float((df["level"] == "ERROR").mean()), 4),
        "first": str(df["start_time"].min()) if "start_time" in df else None,
        "last": str(df["start_time"].max()) if "start_time" in df else None,
    }
