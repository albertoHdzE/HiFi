"""
Phase 14 E0-T2: Per-model diagnostic script.

Tests ONE agent with ONE model loaded in LM Studio (port 1234).
Run sequentially — one model at a time. Load the model in LM Studio,
then invoke this script with --agent matching that model's role.

Usage
-----
    # 1. Load Llama 3.3 70B in LM Studio, then:
    HIFI_FUNDAMENTAL_MODEL=mlx-community/Llama-3.3-70B-Instruct-4bit \\
        uv run python scripts/run_phase14_model_diagnostic.py --agent fundamental

    # 2. Unload Llama. Load Mistral Small 3.2, then:
    HIFI_RISK_MODEL=lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit \\
        uv run python scripts/run_phase14_model_diagnostic.py --agent risk

    # 3. Unload Mistral. Load DeepSeek R1 Distill 32B, then:
    HIFI_MACRO_MODEL=mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit \\
        uv run python scripts/run_phase14_model_diagnostic.py --agent macro

    # 4. Unload DeepSeek. Load Gemma 3 12B, then:
    HIFI_SENTIMENT_MODEL=mlx-community/gemma-3-12b-it-4bit \\
        uv run python scripts/run_phase14_model_diagnostic.py --agent sentiment

Each run appends results to:
    tests/fixtures/baseline/phase14_model_diagnostic.json

Pass/fail thresholds (from Phase 14 plan):
    fundamental : json_valid=True (GR deferred to E0-T4 full baseline)
    risk        : GR >= 0.5
    macro       : GR >= 0.0 (any claim generation counts)
    sentiment   : SGR >= 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_AS_OF = "2023-03-31"
_TICKERS = ["AAPL", "JPM", "XOM"]
_OUTPUT_PATH = (
    _ROOT / "tests" / "fixtures" / "baseline" / "phase14_model_diagnostic.json"
)

_THRESHOLDS = {
    "fundamental": {"metric": "json_valid", "threshold": 1.0},
    "risk":        {"metric": "gr",         "threshold": 0.5},
    "macro":       {"metric": "gr",         "threshold": 0.0},
    "sentiment":   {"metric": "sgr",        "threshold": 0.5},
}


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
# Fundamental: direct LLM call with minimal prompt to test JSON compliance
# ---------------------------------------------------------------------------

_FUNDAMENTAL_SYSTEM = (
    "You are a Fundamental Analyst. Respond ONLY with a JSON object. "
    "Required fields: decision (string: Buy/Hold/Sell), confidence (float 0-1), "
    "rationale (string, 1-3 sentences), pe_ratio (float or null), "
    "revenue_growth (float or null), notable_signals (list of strings)."
)

_FUNDAMENTAL_USER = (
    "Ticker: {ticker}. Date: {date}. "
    "P/E ratio: 28.5 (sector median 24.0). Revenue growth YoY: +8.3%. "
    "Net income margin: 24.6%. Free cash flow yield: 3.8%. "
    "Provide a structured investment opinion."
)


def _run_fundamental_diagnostic(data_dir: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    from hifi.agents.fundamental_agent import _fundamental_model  # noqa: PLC0415
    from hifi.agents.lm_client import make_llm  # noqa: PLC0415

    model_id = _fundamental_model()
    print(f"  Model: {model_id}")
    results = {}

    for ticker in _TICKERS:
        print(f"  {ticker} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            llm = make_llm(model_id, max_tokens=512, temperature=0.0)
            user_text = _FUNDAMENTAL_USER.format(ticker=ticker, date=_AS_OF)
            msgs = [
                SystemMessage(content=_FUNDAMENTAL_SYSTEM),
                HumanMessage(content=user_text),
            ]
            response = llm.invoke(msgs)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            raw = response.content

            # Try to parse JSON
            import re  # noqa: PLC0415
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_valid = False
            parsed = {}
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    required = {"decision", "confidence", "rationale"}
                    json_valid = required.issubset(parsed.keys())
                except json.JSONDecodeError:
                    pass

            decision = parsed.get("decision", "PARSE_ERROR")
            confidence = parsed.get("confidence", 0.0)
            status = "PASS" if json_valid else "FAIL"
            print(f"{status} ({decision}, conf={confidence:.2f}, {latency_ms}ms)")
            results[ticker] = {
                "json_valid": json_valid,
                "decision": decision,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "raw_length": len(raw),
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "json_valid": False,
                "decision": "ERROR",
                "confidence": 0.0,
                "latency_ms": latency_ms,
                "error": str(exc),
            }

    n_valid = sum(1 for r in results.values() if r["json_valid"])
    json_valid_rate = n_valid / len(_TICKERS)
    passed = json_valid_rate >= _THRESHOLDS["fundamental"]["threshold"]
    mean_latency = sum(r["latency_ms"] for r in results.values()) / len(_TICKERS)

    print(f"\n  json_valid_rate: {json_valid_rate:.3f}  ({'PASS' if passed else 'FAIL'})")
    print(f"  mean_latency_ms: {mean_latency:.0f}")

    return {
        "agent": "fundamental",
        "model_id": model_id,
        "tickers": results,
        "summary": {
            "json_valid_rate": round(json_valid_rate, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "passed": passed,
        },
    }


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


def _run_risk_diagnostic(data_dir: str) -> dict:
    from hifi.agents.risk_agent import _risk_model, run_risk_analysis  # noqa: PLC0415
    from hifi.verification.verifier import verify_agent  # noqa: PLC0415

    model_id = _risk_model()
    print(f"  Model: {model_id}")
    results = {}

    for ticker in _TICKERS:
        print(f"  {ticker} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            analysis = run_risk_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            report = verify_agent(analysis)
            print(
                f"claims={report.n_claims} GR={report.grounding_rate:.3f} "
                f"HR={report.hallucination_rate:.3f} ({latency_ms}ms)"
            )
            results[ticker] = {
                "json_valid": True,
                "n_claims": report.n_claims,
                "gr": round(report.grounding_rate, 4),
                "hr": round(report.hallucination_rate, 4),
                "decision": analysis.signal.decision if analysis.signal else "ERROR",
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "json_valid": False, "n_claims": 0, "gr": 0.0, "hr": 1.0,
                "decision": "ERROR", "latency_ms": latency_ms, "error": str(exc),
            }

    mean_gr = sum(r["gr"] for r in results.values()) / len(_TICKERS)
    mean_latency = sum(r["latency_ms"] for r in results.values()) / len(_TICKERS)
    passed = mean_gr >= _THRESHOLDS["risk"]["threshold"]
    print(f"\n  mean_GR: {mean_gr:.3f}  ({'PASS' if passed else 'FAIL'})")
    print(f"  mean_latency_ms: {mean_latency:.0f}")

    return {
        "agent": "risk",
        "model_id": model_id,
        "tickers": results,
        "summary": {
            "mean_gr": round(mean_gr, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "passed": passed,
        },
    }


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------


def _run_macro_diagnostic(data_dir: str) -> dict:
    from hifi.agents.macro_agent import _macro_model, run_macro_analysis  # noqa: PLC0415
    from hifi.verification.verifier import verify_agent  # noqa: PLC0415

    model_id = _macro_model()
    print(f"  Model: {model_id}")
    results = {}

    for ticker in _TICKERS:
        print(f"  {ticker} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            analysis = run_macro_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            report = verify_agent(analysis)
            print(
                f"claims={report.n_claims} GR={report.grounding_rate:.3f} "
                f"HR={report.hallucination_rate:.3f} ({latency_ms}ms)"
            )
            results[ticker] = {
                "json_valid": True,
                "n_claims": report.n_claims,
                "gr": round(report.grounding_rate, 4),
                "hr": round(report.hallucination_rate, 4),
                "decision": analysis.signal.decision if analysis.signal else "ERROR",
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "json_valid": False, "n_claims": 0, "gr": 0.0, "hr": 1.0,
                "decision": "ERROR", "latency_ms": latency_ms, "error": str(exc),
            }

    mean_gr = sum(r["gr"] for r in results.values()) / len(_TICKERS)
    mean_latency = sum(r["latency_ms"] for r in results.values()) / len(_TICKERS)
    passed = mean_gr >= _THRESHOLDS["macro"]["threshold"]
    print(f"\n  mean_GR: {mean_gr:.3f}  ({'PASS' if passed else 'FAIL'})")
    print(f"  mean_latency_ms: {mean_latency:.0f}")

    return {
        "agent": "macro",
        "model_id": model_id,
        "tickers": results,
        "summary": {
            "mean_gr": round(mean_gr, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "passed": passed,
        },
    }


# ---------------------------------------------------------------------------
# Sentiment
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
        logger.warning("retrieve_context failed for %s: %s", ticker, exc)
        return ""


def _run_sentiment_diagnostic(data_dir: str) -> dict:
    from hifi.agents.sentiment_agent import (  # noqa: PLC0415
        _sentiment_model,
        run_sentiment_analysis,
    )
    from hifi.verification.verifier import verify_sentiment_agent  # noqa: PLC0415

    model_id = _sentiment_model()
    print(f"  Model: {model_id}")
    results = {}

    for ticker in _TICKERS:
        print(f"  {ticker} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            analysis = run_sentiment_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            retrieved_context = _retrieve_sentiment_context(ticker, data_dir)
            report = verify_sentiment_agent(analysis, retrieved_context)
            print(
                f"signals={report.n_signals} SGR={report.grounding_rate:.3f} ({latency_ms}ms)"
            )
            results[ticker] = {
                "json_valid": True,
                "n_signals": report.n_signals,
                "sgr": round(report.grounding_rate, 4),
                "decision": analysis.signal.decision if analysis.signal else "ERROR",
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            print(f"ERROR: {exc}")
            results[ticker] = {
                "json_valid": False, "n_signals": 0, "sgr": 0.0,
                "decision": "ERROR", "latency_ms": latency_ms, "error": str(exc),
            }

    mean_sgr = sum(r["sgr"] for r in results.values()) / len(_TICKERS)
    mean_latency = sum(r["latency_ms"] for r in results.values()) / len(_TICKERS)
    passed = mean_sgr >= _THRESHOLDS["sentiment"]["threshold"]
    print(f"\n  mean_SGR: {mean_sgr:.3f}  ({'PASS' if passed else 'FAIL'})")
    print(f"  mean_latency_ms: {mean_latency:.0f}")

    return {
        "agent": "sentiment",
        "model_id": model_id,
        "tickers": results,
        "summary": {
            "mean_sgr": round(mean_sgr, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "passed": passed,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_RUNNERS = {
    "fundamental": _run_fundamental_diagnostic,
    "risk":        _run_risk_diagnostic,
    "macro":       _run_macro_diagnostic,
    "sentiment":   _run_sentiment_diagnostic,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 14 E0-T2: per-model diagnostic (one agent at a time)."
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=list(_RUNNERS),
        help="Which agent to diagnose (load its model in LM Studio first).",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", "data"),
        help="Path to data root directory (default: $HIFI_DATA_DIR or 'data').",
    )
    args = parser.parse_args()

    print(f"\nPhase 14 E0-T2 Diagnostic — agent: {args.agent}")
    print(f"Tickers: {_TICKERS}  |  Date: {_AS_OF}")
    print("-" * 60)

    runner = _RUNNERS[args.agent]
    result = runner(args.data_dir)

    # Load existing results (accumulate across runs)
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if _OUTPUT_PATH.exists():
        with open(_OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    existing.setdefault("metadata", {
        "phase": "14",
        "epic": "E0",
        "ticket": "T2",
        "tickers": _TICKERS,
        "as_of_date": _AS_OF,
        "hifi_commit": _git_sha(),
    })
    existing.setdefault("agents", {})
    existing["agents"][args.agent] = {
        "run_date": date.today().isoformat(),
        **result,
    }

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str)

    passed = result["summary"]["passed"]
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {args.agent} diagnostic complete.")
    print(f"Results appended to {_OUTPUT_PATH}")

    if not passed:
        threshold = _THRESHOLDS[args.agent]
        print(
            f"  FAIL: {threshold['metric']} below threshold {threshold['threshold']:.2f}. "
            f"See DJ-086 fallback procedure in PHASE_14_CONTEXT.md."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
