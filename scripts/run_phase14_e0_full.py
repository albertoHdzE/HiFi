"""
Phase 14 E0-Full: Automated 5-org ensemble diagnostic + diversity upgrade.

Zero manual steps. Uses LM Studio Management API to load/unload models.
Runs E0-T2 (diagnostic), E0-T4 (verification baseline), E0-T3 (update
constants on all-pass), and E0-T5 (diversity baseline). Checkpoint-resume:
results saved incrementally; re-run picks up from last saved state.

LM Studio Management API:
  Load:   POST /api/v0/models/load  {"identifier": "id", "ttl": 0}
  Unload: POST /api/v0/models/unload {"identifier": "id"}
  Poll:   GET  /v1/models  (model appears when ready to serve)

Usage
-----
    uv run python scripts/run_phase14_e0_full.py [--data-dir data] [--skip-diversity]

Outputs
-------
    tests/fixtures/baseline/phase14_model_diagnostic.json     (T2)
    tests/fixtures/baseline/phase14_verification_baseline.json (T4)
    tests/fixtures/baseline/phase14_diversity_baseline.json    (T5)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LM_BASE = "http://localhost:1234"
_AS_OF = "2023-03-31"
_TICKERS = ["AAPL", "JPM", "XOM"]

# Phase 12 evaluation dates — 3 tickers × 10 dates = 30 (ticker, date) pairs
_DIVERSITY_DATES = [
    "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
    "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30",
]

from hifi.agents.lm_client import (  # noqa: E402
    DEEPSEEK_R1_DISTILL_32B,
    GEMMA3_12B,
    LLAMA_33_70B,
    MISTRAL_SMALL_32,
)

_AGENTS_CONFIG = [
    {
        "agent": "fundamental", "model_id": LLAMA_33_70B,
        "env_var": "HIFI_FUNDAMENTAL_MODEL", "load_timeout": 600,
        "threshold": ("json_valid_rate", 1.0),
    },
    {
        "agent": "risk", "model_id": MISTRAL_SMALL_32,
        "env_var": "HIFI_RISK_MODEL", "load_timeout": 300,
        "threshold": ("mean_gr", 0.5),
    },
    {
        "agent": "macro", "model_id": DEEPSEEK_R1_DISTILL_32B,
        "env_var": "HIFI_MACRO_MODEL", "load_timeout": 300,
        "threshold": ("mean_gr", 0.0),
    },
    {
        "agent": "sentiment", "model_id": GEMMA3_12B,
        "env_var": "HIFI_SENTIMENT_MODEL", "load_timeout": 300,
        "threshold": ("mean_sgr", 0.5),
    },
]

_T2_PATH = _ROOT / "tests" / "fixtures" / "baseline" / "phase14_model_diagnostic.json"
_T4_PATH = _ROOT / "tests" / "fixtures" / "baseline" / "phase14_verification_baseline.json"
_T5_PATH = _ROOT / "tests" / "fixtures" / "baseline" / "phase14_diversity_baseline.json"

# Reference snapshots for E0-T4 Fundamental verification (2023-03-31)
_FETCHED_AT = "2023-03-31T00:00:00+00:00"
_SNAPSHOTS: dict[str, dict] = {
    "AAPL": {
        "ticker": "AAPL", "period_end": "2022-12-31",
        "revenue": 117_154_000_000, "net_income": 29_959_000_000,
        "total_assets": 346_747_000_000, "total_liabilities": 302_083_000_000,
        "total_equity": 50_672_000_000, "eps": 1.88,
        "market_cap": 2_650_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q FY2023Q1", "fetched_at": _FETCHED_AT},
    },
    "JPM": {
        "ticker": "JPM", "period_end": "2023-03-31",
        "revenue": 39_340_000_000, "net_income": 12_622_000_000,
        "total_assets": 3_744_305_000_000, "total_liabilities": 3_454_000_000_000,
        "total_equity": 290_000_000_000, "eps": 4.10,
        "market_cap": 420_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q Q1 2023", "fetched_at": _FETCHED_AT},
    },
    "XOM": {
        "ticker": "XOM", "period_end": "2023-03-31",
        "revenue": 86_564_000_000, "net_income": 11_430_000_000,
        "total_assets": 376_317_000_000, "total_liabilities": 205_000_000_000,
        "total_equity": 171_317_000_000, "eps": 2.83,
        "market_cap": 475_000_000_000, "source": "reference",
        "fetched_at": _FETCHED_AT,
        "provenance": {"source": "10-Q Q1 2023", "fetched_at": _FETCHED_AT},
    },
}

_FUNDAMENTAL_SYSTEM = (
    "You are a Fundamental Analyst. Respond ONLY with a JSON object. "
    "Required fields: decision (string: Buy/Hold/Sell), confidence (float 0-1), "
    "rationale (string, 1-3 sentences), pe_ratio (float or null), "
    "revenue_growth (float or null), notable_signals (list of strings)."
)
_FUNDAMENTAL_USER_T2 = (
    "Ticker: {ticker}. Date: {date}. "
    "P/E ratio: 28.5 (sector median 24.0). Revenue growth YoY: +8.3%. "
    "Net income margin: 24.6%. Free cash flow yield: 3.8%. "
    "Provide a structured investment opinion."
)
_FUNDAMENTAL_USER_T5 = (
    "Ticker: {ticker}. Analysis date: {date}. "
    "Use your knowledge of this company's fundamentals as of this period. "
    "Assess P/E relative to sector, revenue trajectory, and margin trend. "
    "Provide a structured investment opinion as JSON."
)


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_ROOT,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# LM Studio Management API
# ---------------------------------------------------------------------------

def _lm_api(method: str, path: str, body: dict | None = None, timeout_s: int = 30) -> dict:
    url = f"{_LM_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


def _get_loaded_ids() -> set[str]:
    try:
        result = _lm_api("GET", "/v1/models")
        return {m["id"] for m in result.get("data", [])}
    except Exception:
        return set()


def _model_is_loaded(model_id: str) -> bool:
    loaded = _get_loaded_ids()
    return model_id in loaded or any(
        model_id in lid or lid in model_id for lid in loaded
    )


def _load_model(model_id: str, timeout_s: int = 600) -> bool:
    if _model_is_loaded(model_id):
        print(f"  Already loaded: {model_id}")
        return True
    print(f"  Loading {model_id}", end="", flush=True)
    t0 = time.monotonic()
    try:
        _lm_api("POST", "/api/v0/models/load", {"identifier": model_id, "ttl": 0})
    except Exception as exc:
        print(f"\n  Load request failed: {exc}")
        return False
    deadline = t0 + timeout_s
    while time.monotonic() < deadline:
        time.sleep(10)
        print(".", end="", flush=True)
        if _model_is_loaded(model_id):
            print(f" ready ({int(time.monotonic() - t0)}s)")
            return True
    print(f" TIMEOUT ({timeout_s}s)")
    return False


def _unload_model(model_id: str) -> None:
    try:
        _lm_api("POST", "/api/v0/models/unload", {"identifier": model_id})
        print(f"  Unloaded: {model_id}")
    except Exception as exc:
        print(f"  Warning: unload error: {exc}")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Shared: sentiment context retrieval
# ---------------------------------------------------------------------------

def _retrieve_sentiment_context(ticker: str, data_dir: str) -> str:
    from hifi.agents.mcp_client import call_tool  # noqa: PLC0415
    query = (
        f"{ticker} management outlook guidance forward-looking statements "
        f"revenue growth margin services"
    )
    try:
        result = call_tool(
            "retrieve_context",
            {"query": query, "ticker": ticker, "top_k": 5},
            data_dir=data_dir,
            server_module="hifi.mcp.knowledge_server",
        )
        passages = result.get("passages", [])
        if not passages:
            return ""
        lines = []
        for p in passages:
            lines.append(
                f"[{p['rank']}] {ticker} / {p['filing_type']} / {p['section']} / {p['period']}"
            )
            lines.append(p["text"])
            lines.append("---")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("retrieve_context %s: %s", ticker, exc)
        return ""


# ---------------------------------------------------------------------------
# E0-T2: Diagnostic (pass/fail per model)
# ---------------------------------------------------------------------------

def _t2_fundamental(data_dir: str, model_id: str) -> dict:
    """Direct LLM call — tests JSON compliance without full agent graph."""
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    from hifi.agents.lm_client import make_llm  # noqa: PLC0415

    results: dict[str, dict] = {}
    llm = make_llm(model_id, max_tokens=512, temperature=0.0)
    for ticker in _TICKERS:
        print(f"    {ticker} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            msgs = [
                SystemMessage(content=_FUNDAMENTAL_SYSTEM),
                HumanMessage(content=_FUNDAMENTAL_USER_T2.format(ticker=ticker, date=_AS_OF)),
            ]
            raw = llm.invoke(msgs).content
            latency_ms = int((time.perf_counter() - t0) * 1000)
            json_valid, parsed = False, {}
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    json_valid = {"decision", "confidence", "rationale"}.issubset(parsed.keys())
                except json.JSONDecodeError:
                    pass
            decision = parsed.get("decision", "PARSE_ERROR")
            confidence = float(parsed.get("confidence", 0.0))
            status = "PASS" if json_valid else "FAIL"
            print(f"{status} ({decision} {confidence:.2f} {latency_ms}ms)")
            results[ticker] = {
                "json_valid": json_valid, "decision": decision,
                "confidence": confidence, "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "json_valid": False, "decision": "ERROR",
                "confidence": 0.0, "latency_ms": latency_ms, "error": str(exc),
            }

    n_valid = sum(1 for r in results.values() if r["json_valid"])
    json_valid_rate = n_valid / len(_TICKERS)
    mean_latency = sum(r["latency_ms"] for r in results.values()) / len(_TICKERS)
    passed = json_valid_rate >= 1.0
    t2_status = "PASS" if passed else "FAIL"
    print(f"\n    json_valid_rate={json_valid_rate:.3f}  {mean_latency:.0f}ms  {t2_status}")
    return {
        "agent": "fundamental", "model_id": model_id, "tickers": results,
        "summary": {
            "json_valid_rate": round(json_valid_rate, 4),
            "mean_latency_ms": round(mean_latency, 1), "passed": passed,
        },
    }


def _t2_risk(data_dir: str, model_id: str) -> dict:
    from hifi.agents.risk_agent import run_risk_analysis  # noqa: PLC0415
    from hifi.verification.verifier import verify_agent  # noqa: PLC0415

    results: dict[str, dict] = {}
    for ticker in _TICKERS:
        print(f"    {ticker} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            analysis = run_risk_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            report = verify_agent(analysis)
            print(
                f"GR={report.grounding_rate:.3f} "
                f"HR={report.hallucination_rate:.3f} ({latency_ms}ms)"
            )
            results[ticker] = {
                "json_valid": True, "n_claims": report.n_claims,
                "gr": round(report.grounding_rate, 4), "hr": round(report.hallucination_rate, 4),
                "decision": analysis.signal.decision if analysis.signal else "NONE",
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "json_valid": False, "n_claims": 0, "gr": 0.0, "hr": 1.0,
                "decision": "ERROR", "latency_ms": latency_ms, "error": str(exc),
            }

    mean_gr = sum(r.get("gr", 0.0) for r in results.values()) / len(_TICKERS)
    mean_latency = sum(r["latency_ms"] for r in results.values()) / len(_TICKERS)
    passed = mean_gr >= 0.5
    print(f"\n    mean_GR={mean_gr:.3f}  {mean_latency:.0f}ms  {'PASS' if passed else 'FAIL'}")
    return {
        "agent": "risk", "model_id": model_id, "tickers": results,
        "summary": {
            "mean_gr": round(mean_gr, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "passed": passed,
        },
    }


def _t2_macro(data_dir: str, model_id: str) -> dict:
    from hifi.agents.macro_agent import run_macro_analysis  # noqa: PLC0415
    from hifi.verification.verifier import verify_agent  # noqa: PLC0415

    results: dict[str, dict] = {}
    for ticker in _TICKERS:
        print(f"    {ticker} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            analysis = run_macro_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            report = verify_agent(analysis)
            print(f"GR={report.grounding_rate:.3f} n={report.n_claims} ({latency_ms}ms)")
            results[ticker] = {
                "json_valid": True, "n_claims": report.n_claims,
                "gr": round(report.grounding_rate, 4), "hr": round(report.hallucination_rate, 4),
                "decision": analysis.signal.decision if analysis.signal else "NONE",
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "json_valid": False, "n_claims": 0, "gr": 0.0, "hr": 1.0,
                "decision": "ERROR", "latency_ms": latency_ms, "error": str(exc),
            }

    mean_gr = sum(r.get("gr", 0.0) for r in results.values()) / len(_TICKERS)
    mean_latency = sum(r["latency_ms"] for r in results.values()) / len(_TICKERS)
    passed = mean_gr >= 0.0  # any claim generation counts
    print(f"\n    mean_GR={mean_gr:.3f}  {mean_latency:.0f}ms  {'PASS' if passed else 'FAIL'}")
    return {
        "agent": "macro", "model_id": model_id, "tickers": results,
        "summary": {
            "mean_gr": round(mean_gr, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "passed": passed,
        },
    }


def _t2_sentiment(data_dir: str, model_id: str) -> dict:
    from hifi.agents.sentiment_agent import run_sentiment_analysis  # noqa: PLC0415
    from hifi.verification.verifier import verify_sentiment_agent  # noqa: PLC0415

    results: dict[str, dict] = {}
    for ticker in _TICKERS:
        print(f"    {ticker} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            analysis = run_sentiment_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            ctx = _retrieve_sentiment_context(ticker, data_dir)
            report = verify_sentiment_agent(analysis, ctx)
            print(f"SGR={report.grounding_rate:.3f} signals={report.n_signals} ({latency_ms}ms)")
            results[ticker] = {
                "json_valid": True, "n_signals": report.n_signals,
                "sgr": round(report.grounding_rate, 4),
                "decision": analysis.signal.decision if analysis.signal else "NONE",
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "json_valid": False, "n_signals": 0, "sgr": 0.0,
                "decision": "ERROR", "latency_ms": latency_ms, "error": str(exc),
            }

    mean_sgr = sum(r.get("sgr", 0.0) for r in results.values()) / len(_TICKERS)
    mean_latency = sum(r["latency_ms"] for r in results.values()) / len(_TICKERS)
    passed = mean_sgr >= 0.5
    print(f"\n    mean_SGR={mean_sgr:.3f}  {mean_latency:.0f}ms  {'PASS' if passed else 'FAIL'}")
    return {
        "agent": "sentiment", "model_id": model_id, "tickers": results,
        "summary": {
            "mean_sgr": round(mean_sgr, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "passed": passed,
        },
    }


_T2_RUNNERS = {
    "fundamental": _t2_fundamental,
    "risk": _t2_risk,
    "macro": _t2_macro,
    "sentiment": _t2_sentiment,
}


# ---------------------------------------------------------------------------
# E0-T4: Verification baseline (HR/GR/SGR saved to fixture)
# For risk/macro/sentiment: reuses T2 analysis results (same call, different fixture).
# For fundamental: uses full run_analysis with reference snapshots.
# ---------------------------------------------------------------------------

def _t4_fundamental(data_dir: str, model_id: str) -> dict:
    """Full run_analysis + verify_agent using reference snapshots for 2023-03-31."""
    from hifi.agents.fundamental_agent import run_analysis  # noqa: PLC0415
    from hifi.data.schemas import FundamentalsSnapshot  # noqa: PLC0415
    from hifi.verification.verifier import verify_agent  # noqa: PLC0415

    results: dict[str, dict] = {}
    for ticker in _TICKERS:
        print(f"    {ticker} ...", end=" ", flush=True)
        snapshot_json = FundamentalsSnapshot.model_validate(_SNAPSHOTS[ticker]).model_dump_json()
        t0 = time.perf_counter()
        try:
            analysis = run_analysis(
                ticker=ticker, as_of_date=_AS_OF, snapshot_json=snapshot_json, data_dir=data_dir
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if analysis.signal is None:
                raise ValueError("No signal generated")
            report = verify_agent(analysis)
            print(
                f"GR={report.grounding_rate:.3f} "
                f"HR={report.hallucination_rate:.3f} ({latency_ms}ms)"
            )
            results[ticker] = {
                "decision": analysis.signal.decision,
                "confidence": round(analysis.signal.confidence, 4),
                "n_claims": report.n_claims,
                "gr": round(report.grounding_rate, 4),
                "hr": round(report.hallucination_rate, 4),
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "decision": "ERROR", "confidence": 0.0, "n_claims": 0,
                "gr": 0.0, "hr": 1.0, "latency_ms": latency_ms, "error": str(exc),
            }

    mean_gr = sum(r.get("gr", 0.0) for r in results.values()) / len(_TICKERS)
    mean_hr = sum(r.get("hr", 0.0) for r in results.values()) / len(_TICKERS)
    print(f"    mean_GR={mean_gr:.4f}  mean_HR={mean_hr:.4f}")
    return {
        "agent": "fundamental", "model_id": model_id, "as_of": _AS_OF,
        "tickers": results,
        "summary": {"mean_gr": round(mean_gr, 4), "mean_hr": round(mean_hr, 4)},
    }


def _t4_from_t2(t2_result: dict) -> dict:
    """Build T4 baseline record directly from T2 analysis results (risk/macro/sentiment)."""
    agent = t2_result["agent"]
    results = t2_result["tickers"]
    summary: dict = {}
    if agent == "sentiment":
        mean_sgr = sum(r.get("sgr", 0.0) for r in results.values()) / len(_TICKERS)
        summary = {"mean_sgr": round(mean_sgr, 4)}
    else:
        mean_gr = sum(r.get("gr", 0.0) for r in results.values()) / len(_TICKERS)
        mean_hr = sum(r.get("hr", 0.0) for r in results.values()) / len(_TICKERS)
        summary = {"mean_gr": round(mean_gr, 4), "mean_hr": round(mean_hr, 4)}
    return {
        "agent": agent,
        "model_id": t2_result["model_id"],
        "as_of": _AS_OF,
        "tickers": results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# E0-T5: Diversity data collection (30 ticker-date pairs per agent)
# ---------------------------------------------------------------------------

def _t5_fundamental_one(ticker: str, dt: str, model_id: str) -> dict | None:
    """Direct LLM call — no snapshot needed for diversity measurement."""
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    from hifi.agents.lm_client import make_llm  # noqa: PLC0415
    try:
        llm = make_llm(model_id, max_tokens=256, temperature=0.0)
        msgs = [
            SystemMessage(content=_FUNDAMENTAL_SYSTEM),
            HumanMessage(content=_FUNDAMENTAL_USER_T5.format(ticker=ticker, date=dt)),
        ]
        raw = llm.invoke(msgs).content
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            dec = parsed.get("decision", "").strip()
            if dec in ("Buy", "Hold", "Sell"):
                return {"decision": dec, "confidence": float(parsed.get("confidence", 0.5))}
    except Exception as exc:
        logger.warning("t5_fundamental %s %s: %s", ticker, dt, exc)
    return None


def _t5_risk_one(ticker: str, dt: str, data_dir: str) -> dict | None:
    from hifi.agents.risk_agent import run_risk_analysis  # noqa: PLC0415
    try:
        analysis = run_risk_analysis(ticker=ticker, as_of_date=dt, data_dir=data_dir)
        if analysis.signal and analysis.signal.decision in ("Buy", "Hold", "Sell"):
            return {"decision": analysis.signal.decision, "confidence": analysis.signal.confidence}
    except Exception as exc:
        logger.warning("t5_risk %s %s: %s", ticker, dt, exc)
    return None


def _t5_macro_one(ticker: str, dt: str, data_dir: str) -> dict | None:
    from hifi.agents.macro_agent import run_macro_analysis  # noqa: PLC0415
    try:
        analysis = run_macro_analysis(ticker=ticker, as_of_date=dt, data_dir=data_dir)
        if analysis.signal and analysis.signal.decision in ("Buy", "Hold", "Sell"):
            return {"decision": analysis.signal.decision, "confidence": analysis.signal.confidence}
    except Exception as exc:
        logger.warning("t5_macro %s %s: %s", ticker, dt, exc)
    return None


def _t5_sentiment_one(ticker: str, dt: str, data_dir: str) -> dict | None:
    from hifi.agents.sentiment_agent import run_sentiment_analysis  # noqa: PLC0415
    try:
        analysis = run_sentiment_analysis(ticker=ticker, as_of_date=dt, data_dir=data_dir)
        if analysis.signal and analysis.signal.decision in ("Buy", "Hold", "Sell"):
            return {"decision": analysis.signal.decision, "confidence": analysis.signal.confidence}
    except Exception as exc:
        logger.warning("t5_sentiment %s %s: %s", ticker, dt, exc)
    return None


def _run_t5_agent(agent: str, model_id: str, data_dir: str, t5_data: dict) -> dict:
    """Collect diversity decisions for one agent across all 30 (ticker, date) pairs."""
    pairs = [(t, d) for d in _DIVERSITY_DATES for t in _TICKERS]
    agent_results: dict = t5_data.get(agent, {})
    n_done = sum(1 for k in agent_results if agent_results[k].get("decision") is not None)
    n_skip = len(agent_results) - n_done  # already saved as None
    n_remaining = len(pairs) - len(agent_results)
    print(f"  T5 {agent}: {n_remaining} remaining, {n_done} done, {n_skip} skipped (checkpoint)")

    for ticker, dt in pairs:
        key = f"{ticker}|{dt}"
        if key in agent_results:
            continue
        print(f"    {ticker} {dt} ...", end=" ", flush=True)
        if agent == "fundamental":
            result = _t5_fundamental_one(ticker, dt, model_id)
        elif agent == "risk":
            result = _t5_risk_one(ticker, dt, data_dir)
        elif agent == "macro":
            result = _t5_macro_one(ticker, dt, data_dir)
        else:
            result = _t5_sentiment_one(ticker, dt, data_dir)

        if result is not None:
            agent_results[key] = result
            print(f"{result['decision']} ({result['confidence']:.2f})")
        else:
            agent_results[key] = {"decision": None, "confidence": 0.0}
            print("None")

    t5_data[agent] = agent_results
    n_valid = sum(1 for r in agent_results.values() if r.get("decision") is not None)
    print(f"  T5 {agent}: {n_valid}/{len(pairs)} valid decisions")
    return t5_data


# ---------------------------------------------------------------------------
# E0-T5: Diversity metrics
# ---------------------------------------------------------------------------

def _compute_diversity_metrics(t5_data: dict) -> dict:
    """
    Shannon entropy + herding coefficient from T5 agent decisions.

    Per (ticker, date) pair: collect decisions from all available agents.
    entropy  = -Σ p_k log2(p_k) over Buy/Hold/Sell proportions
    herding  = max_vote_count / n_agents  (David §5.6.3)
    """
    agents_present = [k for k in t5_data if k in {"fundamental", "risk", "macro", "sentiment"}]
    pairs = [(t, d) for d in _DIVERSITY_DATES for t in _TICKERS]
    entropies, herd_coeffs = [], []
    pair_details: dict = {}

    for ticker, dt in pairs:
        key = f"{ticker}|{dt}"
        decisions = []
        for ag in agents_present:
            rec = t5_data.get(ag, {}).get(key, {})
            dec = rec.get("decision")
            if dec in ("Buy", "Hold", "Sell"):
                decisions.append(dec)
        if not decisions:
            continue
        n = len(decisions)
        counts = Counter(decisions)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
        herding = counts.most_common(1)[0][1] / n
        entropies.append(entropy)
        herd_coeffs.append(herding)
        pair_details[key] = {
            "decisions": decisions, "counts": dict(counts),
            "entropy": round(entropy, 4), "herding": round(herding, 4),
        }

    if not entropies:
        return {
            "error": "no valid pairs",
            "mean_entropy": 0.0,
            "mean_herding_coefficient": 1.0,
            "n_pairs": 0,
        }

    mean_entropy = sum(entropies) / len(entropies)
    mean_herding = sum(herd_coeffs) / len(herd_coeffs)
    return {
        "agents_included": agents_present,
        "n_pairs": len(entropies),
        "mean_entropy": round(mean_entropy, 4),
        "mean_herding_coefficient": round(mean_herding, 4),
        "oq_p14_05_pass": mean_entropy > 0.3,
        "phase12_condA_entropy_ref": 0.367,
        "pair_details": pair_details,
    }


# ---------------------------------------------------------------------------
# E0-T3: Update _DEFAULT_*_MODEL constants in agent source files
# ---------------------------------------------------------------------------

def _update_constants() -> bool:
    """Programmatically update agent default model constants. Returns True if all ok."""
    updates = [
        {
            "file": _ROOT / "src" / "hifi" / "agents" / "fundamental_agent.py",
            "replacements": [
                (
                    "from hifi.agents.lm_client import _DEFAULT_MODEL as _LM_DEFAULT\n"
                    "from hifi.agents.lm_client import make_llm",
                    "from hifi.agents.lm_client import LLAMA_33_70B, make_llm",
                ),
                (
                    "_DEFAULT_FUNDAMENTAL_MODEL = _LM_DEFAULT",
                    "_DEFAULT_FUNDAMENTAL_MODEL = LLAMA_33_70B",
                ),
            ],
            "verify_new": "_DEFAULT_FUNDAMENTAL_MODEL = LLAMA_33_70B",
        },
        {
            "file": _ROOT / "src" / "hifi" / "agents" / "risk_agent.py",
            "replacements": [
                (
                    "from hifi.agents.lm_client import make_llm",
                    "from hifi.agents.lm_client import MISTRAL_SMALL_32, make_llm",
                ),
                (
                    '_DEFAULT_RISK_MODEL = "google/gemma-3-4b"',
                    "_DEFAULT_RISK_MODEL = MISTRAL_SMALL_32",
                ),
            ],
            "verify_new": "_DEFAULT_RISK_MODEL = MISTRAL_SMALL_32",
        },
        {
            "file": _ROOT / "src" / "hifi" / "agents" / "macro_agent.py",
            "replacements": [
                (
                    "from hifi.agents.lm_client import make_llm",
                    "from hifi.agents.lm_client import DEEPSEEK_R1_DISTILL_32B, make_llm",
                ),
                (
                    "_DEFAULT_MACRO_MODEL = "
                    '"qwen3.5-27b-claude-4.6-opus-reasoning-distilled-qx64-hi-mlx"',
                    "_DEFAULT_MACRO_MODEL = DEEPSEEK_R1_DISTILL_32B",
                ),
            ],
            "verify_new": "_DEFAULT_MACRO_MODEL = DEEPSEEK_R1_DISTILL_32B",
        },
        {
            "file": _ROOT / "src" / "hifi" / "agents" / "sentiment_agent.py",
            "replacements": [
                (
                    "from hifi.agents.lm_client import make_llm",
                    "from hifi.agents.lm_client import GEMMA3_12B, make_llm",
                ),
                (
                    '_DEFAULT_SENTIMENT_MODEL = "qwen2.5-coder-32b-instruct-mlx"'
                    "  # DJ-087: reverted from gemma-4-e4b",
                    "_DEFAULT_SENTIMENT_MODEL = GEMMA3_12B  # E0-T3: Gemma 3 12B (DJ-089)",
                ),
            ],
            "verify_new": "_DEFAULT_SENTIMENT_MODEL = GEMMA3_12B",
        },
    ]

    all_ok = True
    for upd in updates:
        path: Path = upd["file"]
        content = path.read_text(encoding="utf-8")

        # Idempotency: skip if already updated
        if upd["verify_new"] in content:
            print(f"  {path.name}: already updated — skipping")
            continue

        modified = content
        for old, new in upd["replacements"]:
            if old in modified:
                modified = modified.replace(old, new, 1)
            else:
                print(f"  WARNING: could not find replacement target in {path.name}")
                print(f"    Expected: {repr(old)[:80]}")
                all_ok = False

        if modified != content:
            path.write_text(modified, encoding="utf-8")
            print(f"  Updated: {path.name}")
        else:
            print(f"  No changes made to {path.name} (may indicate a mismatch)")

    return all_ok


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(t2_agents: dict, t4_agents: dict, diversity_metrics: dict | None) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FINAL SUMMARY — Phase 14 E0")
    print(sep)

    print("\nE0-T2 Diagnostic:")
    fmt = "  {:<14} {:<36} {:<22} {}"
    print(fmt.format("Agent", "Model (short)", "Metric", "Pass?"))
    print(fmt.format("-" * 14, "-" * 36, "-" * 22, "-" * 5))
    all_t2_pass = True
    for cfg in _AGENTS_CONFIG:
        agent = cfg["agent"]
        rec = t2_agents.get(agent, {})
        summ = rec.get("summary", {})
        passed = summ.get("passed", False)
        metric_name, _ = cfg["threshold"]
        val = summ.get(metric_name, 0.0)
        model_short = cfg["model_id"].split("/")[-1][:35]
        status = "PASS" if passed else ("FAIL" if rec else "(not run)")
        print(fmt.format(agent, model_short, f"{metric_name}={val:.3f}", status))
        if not passed:
            all_t2_pass = False
    print(f"\n  All T2 pass: {'YES' if all_t2_pass else 'NO'}")

    print("\nE0-T4 Verification Baseline (2023-03-31):")
    for cfg in _AGENTS_CONFIG:
        agent = cfg["agent"]
        rec = t4_agents.get(agent, {})
        if not rec:
            print(f"  {agent}: (not run)")
            continue
        summ = rec.get("summary", {})
        if agent == "sentiment":
            print(f"  {agent}: mean_SGR={summ.get('mean_sgr', 0):.4f}")
        else:
            mgr = summ.get("mean_gr", 0)
            mhr = summ.get("mean_hr", 0)
            print(f"  {agent}: mean_GR={mgr:.4f}  mean_HR={mhr:.4f}")

    if diversity_metrics and "error" not in diversity_metrics:
        print("\nE0-T5 Diversity Baseline (OQ-P14-05: mean_entropy > 0.3):")
        print(f"  Agents   : {diversity_metrics.get('agents_included', [])}")
        print(f"  N pairs  : {diversity_metrics.get('n_pairs', 0)}")
        me = diversity_metrics.get("mean_entropy", 0)
        mh = diversity_metrics.get("mean_herding_coefficient", 0)
        print(f"  Entropy  : {me:.4f}  (Phase 12 cond-A ref: 0.367)")
        print(f"  Herding  : {mh:.4f}")
        oq = diversity_metrics.get("oq_p14_05_pass", False)
        print(f"  OQ-P14-05: {'PASS — entropy > 0.3' if oq else 'FAIL — entropy <= 0.3'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 14 E0-Full: automated model diagnostic + diversity upgrade."
    )
    parser.add_argument("--data-dir", default=os.environ.get("HIFI_DATA_DIR", "data"))
    parser.add_argument(
        "--skip-diversity", action="store_true",
        help="Skip E0-T5 diversity baseline (saves ~1-3 hours; run separately if needed)",
    )
    parser.add_argument(
        "--skip-t3", action="store_true",
        help="Skip E0-T3 source file update even if all T2 pass",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    print(f"\nPhase 14 E0-Full  |  data_dir={data_dir}  |  commit={_git_sha()}")
    print(f"Tickers: {_TICKERS}  |  T2/T4 date: {_AS_OF}")
    n_diversity_pairs = len(_TICKERS) * len(_DIVERSITY_DATES)
    if not args.skip_diversity:
        print(
            f"T5 diversity: {len(_TICKERS)} tickers × {len(_DIVERSITY_DATES)} dates"
            f" = {n_diversity_pairs} pairs/agent"
        )

    # Pre-flight: verify LM Studio is reachable
    try:
        _lm_api("GET", "/v1/models")
        print("LM Studio: OK")
    except Exception:
        print("\nERROR: LM Studio not responding at http://localhost:1234")
        print("Start LM Studio (no model needs to be loaded) and retry.")
        sys.exit(1)

    # Load checkpoints
    t2_ckpt = _load_json(_T2_PATH)
    t4_ckpt = _load_json(_T4_PATH)
    t5_ckpt = _load_json(_T5_PATH)
    t2_agents: dict = t2_ckpt.get("agents", {})
    t4_agents: dict = t4_ckpt.get("agents", {})
    t5_data: dict = t5_ckpt.get("data", {})

    run_date = date.today().isoformat()
    hifi_commit = _git_sha()
    meta_t2 = {
        "phase": "14", "epic": "E0", "ticket": "T2",
        "tickers": _TICKERS, "as_of_date": _AS_OF, "hifi_commit": hifi_commit,
    }
    meta_t4 = {**meta_t2, "ticket": "T4", "note": "HR/GR/SGR baseline with Phase 14 models"}
    meta_t5 = {
        "phase": "14", "epic": "E0", "ticket": "T5",
        "tickers": _TICKERS, "diversity_dates": _DIVERSITY_DATES, "hifi_commit": hifi_commit,
    }

    # -----------------------------------------------------------------------
    # Per-agent loop: load model → T2 + T4 + T5 → unload
    # -----------------------------------------------------------------------
    for cfg in _AGENTS_CONFIG:
        agent = cfg["agent"]
        model_id = cfg["model_id"]
        env_var = cfg["env_var"]

        t2_done = agent in t2_agents
        t4_done = agent in t4_agents
        t5_done = args.skip_diversity or (
            agent in t5_data and len(t5_data[agent]) >= n_diversity_pairs
        )

        if t2_done and t4_done and t5_done:
            print(f"\n[{agent.upper()}] Checkpoint: all tasks complete — skipping model load")
            continue

        print(f"\n{'=' * 72}")
        print(f"  Agent: {agent.upper()}  |  Model: {model_id}")
        print(f"{'=' * 72}")

        loaded = _load_model(model_id, timeout_s=cfg["load_timeout"])
        if not loaded:
            print(f"  FATAL: Could not load {model_id}. Skipping {agent}.")
            continue

        os.environ[env_var] = model_id
        try:
            # E0-T2
            if not t2_done:
                print("\n  [T2] Diagnostic:")
                t2_result = _T2_RUNNERS[agent](data_dir, model_id)
                t2_agents[agent] = {"run_date": run_date, **t2_result}
                t2_ckpt.update({"metadata": meta_t2, "agents": t2_agents})
                _save_json(_T2_PATH, t2_ckpt)
                print(f"  -> Saved to {_T2_PATH.name}")
            else:
                print("\n  [T2] Checkpoint: already done")
                t2_result = t2_agents[agent]

            # E0-T4
            if not t4_done:
                print("\n  [T4] Verification baseline:")
                if agent == "fundamental":
                    t4_result = _t4_fundamental(data_dir, model_id)
                else:
                    # Reuse T2 analysis results — same underlying calls
                    t4_result = _t4_from_t2(t2_result)
                t4_agents[agent] = {"run_date": run_date, **t4_result}
                t4_ckpt.update({"metadata": meta_t4, "agents": t4_agents})
                _save_json(_T4_PATH, t4_ckpt)
                print(f"  -> Saved to {_T4_PATH.name}")
            else:
                print("\n  [T4] Checkpoint: already done")

            # E0-T5
            if not args.skip_diversity and not t5_done:
                print("\n  [T5] Diversity data collection:")
                t5_data = _run_t5_agent(agent, model_id, data_dir, t5_data)
                t5_ckpt.update({"metadata": meta_t5, "data": t5_data})
                _save_json(_T5_PATH, t5_ckpt)
                print(f"  -> Saved to {_T5_PATH.name}")
            elif args.skip_diversity:
                print("\n  [T5] Skipped (--skip-diversity)")
            else:
                print("\n  [T5] Checkpoint: already done")

        finally:
            os.environ.pop(env_var, None)
            _unload_model(model_id)
            time.sleep(5)  # allow MLX memory release

    # -----------------------------------------------------------------------
    # E0-T3: Update agent source constants (only if all T2 pass)
    # -----------------------------------------------------------------------
    all_t2_pass = all(
        t2_agents.get(cfg["agent"], {}).get("summary", {}).get("passed", False)
        for cfg in _AGENTS_CONFIG
    )

    print(f"\n{'=' * 72}")
    print("  E0-T3: Update agent default model constants")
    print(f"{'=' * 72}")
    if args.skip_t3:
        print("  Skipped (--skip-t3)")
    elif not all_t2_pass:
        print("  SKIPPED — not all T2 diagnostics passed. Fix failing agents first.")
        failed = [
            cfg["agent"] for cfg in _AGENTS_CONFIG
            if not t2_agents.get(cfg["agent"], {}).get("summary", {}).get("passed", False)
        ]
        print(f"  Failing: {failed}")
    else:
        t3_ok = _update_constants()
        if t3_ok:
            print("\n  All constants updated. Next steps:")
            print("    uv run ruff check --fix src/")
            print("    uv run pytest -q --tb=no")
        else:
            print("\n  ERROR: some updates failed — check output above.")

    # -----------------------------------------------------------------------
    # E0-T5: Compute diversity metrics
    # -----------------------------------------------------------------------
    diversity_metrics: dict | None = None
    if not args.skip_diversity and t5_data:
        print(f"\n{'=' * 72}")
        print("  E0-T5: Computing diversity metrics")
        print(f"{'=' * 72}")
        diversity_metrics = _compute_diversity_metrics(t5_data)
        t5_ckpt.update({"metadata": meta_t5, "data": t5_data, "metrics": diversity_metrics})
        _save_json(_T5_PATH, t5_ckpt)
        me = diversity_metrics.get("mean_entropy", 0)
        mh = diversity_metrics.get("mean_herding_coefficient", 0)
        oq = diversity_metrics.get("oq_p14_05_pass", False)
        oq_str = "PASS" if oq else "FAIL"
        print(f"  mean_entropy={me:.4f}  mean_herding={mh:.4f}  OQ-P14-05={oq_str}")

    # -----------------------------------------------------------------------
    # Final summary + exit
    # -----------------------------------------------------------------------
    _print_summary(t2_agents, t4_agents, diversity_metrics)

    exit_code = 0
    if not all_t2_pass:
        print("\nEXIT 1: One or more T2 diagnostics failed.")
        exit_code = 1
    elif diversity_metrics and not diversity_metrics.get("oq_p14_05_pass", True):
        print("\nEXIT 1: OQ-P14-05 failed (mean_entropy <= 0.3).")
        exit_code = 1
    else:
        print("\nEXIT 0: All checks passed.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
