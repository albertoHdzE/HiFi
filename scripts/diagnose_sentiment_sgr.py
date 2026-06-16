"""
Sentiment SGR Diagnostic Tool (P13-E1 prerequisite, DJ-086).

Diagnoses the SGR=0.000 root cause for Gemma 4 Sentiment agent by printing
raw LLM output before JSON parsing, then running grounding checks.

What is measured
----------------
1. Raw LLM response text — what the model actually produced
2. JSON extraction result — what _extract_json() finds (or fails to find)
3. notable_signals grounding — are signal texts verbatim substrings of context?

Tests both Gemma 4 variants in sequence:
  - google/gemma-4-e4b (default, DJ-085)
  - gemma-4-12b-it-mlx (full 12B, confirmed loaded in LM Studio 2026-06-15)

Usage
-----
    uv run python scripts/diagnose_sentiment_sgr.py [--ticker TICKER] [--data-dir DIR]

Decision output
---------------
After both model tests, prints a DJ-086 recommendation:
  - Which model to use going forward as the Sentiment base
  - Whether the SGR issue is a prompt engineering problem or model capability gap
  - Recommended next step (re-run baseline, prompt tuning, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from hifi.agents.lm_client import make_llm  # noqa: E402
from hifi.agents.mcp_client import call_tool  # noqa: E402

_AS_OF = "2023-03-31"
_MODELS_TO_TEST = [
    ("google/gemma-4-e4b", "Gemma 4 E4B (default DJ-085)"),
    ("gemma-4-12b-it-mlx", "Gemma 4 12B-it MLX (full 12B, confirmed loaded)"),
]
_PROMPT_PATH = _ROOT / "src" / "hifi" / "agents" / "prompts" / "sentiment_v1.md"


def _load_prompt_template() -> tuple[str, str]:
    raw = _PROMPT_PATH.read_text(encoding="utf-8")
    parts = raw.split("## User", maxsplit=1)
    system_block = parts[0].replace("## System", "").strip()
    user_block = parts[1].strip() if len(parts) > 1 else ""
    return system_block, user_block


def _retrieve_context(ticker: str, data_dir: str) -> str:
    query = (
        f"{ticker} management outlook guidance forward-looking statements risks "
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
                f"[{p['rank']}] {ticker} / {p['filing_type']} / "
                f"{p['section']} / {p['period']}"
            )
            lines.append(p["text"])
            lines.append("---")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("retrieve_context failed for %s: %s", ticker, exc)
        return ""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [line for line in lines if not line.startswith("```")]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _check_grounding(notable_signals: list[str], retrieved_context: str) -> list[tuple[str, bool]]:
    norm_ctx = retrieved_context.lower().strip()
    results = []
    for sig in notable_signals:
        norm_sig = sig.lower().strip()
        grounded = bool(norm_sig) and (norm_sig in norm_ctx)
        results.append((sig, grounded))
    return results


def diagnose_ticker(ticker: str, model_id: str, data_dir: str) -> None:
    print(f"\n  Retrieved context for {ticker}... ", end="", flush=True)
    retrieved_context = _retrieve_context(ticker, data_dir)
    if not retrieved_context:
        print("EMPTY — no SEC filing passages found. SGR=0 trivially.")
        return
    print(f"OK ({len(retrieved_context)} chars)")

    system_text, user_template = _load_prompt_template()
    user_text = user_template.format(
        ticker=ticker,
        as_of_date=_AS_OF,
        retrieved_context=retrieved_context,
    )

    print(f"  Calling LLM ({model_id}) ... ", end="", flush=True)
    try:
        llm = make_llm(model_id, max_tokens=1024)
        messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
        response = llm.invoke(messages)
        raw_text = response.content
    except Exception as exc:
        print(f"FAILED: {exc}")
        return
    print("OK")

    print("\n  --- RAW LLM OUTPUT (first 1200 chars) ---")
    print(raw_text[:1200])
    if len(raw_text) > 1200:
        print(f"  ... [{len(raw_text) - 1200} more chars truncated]")
    print("  -----------------------------------------")

    parsed = _extract_json(raw_text)
    if parsed is None:
        print("\n  [DIAGNOSIS] _extract_json() returned None — JSON parsing FAILED.")
        print("  Look above for the raw output to understand why:")
        print("  - If the model outputs markdown prose without JSON, it ignores Rule 4.")
        print("  - If it outputs JSON wrapped in unexpected tags, add stripping logic.")
        print("  - If it outputs empty string or refusal, the model capacity is the issue.")
        return

    print(f"\n  [PARSED JSON] keys: {list(parsed.keys())}")
    decision = parsed.get("decision", "<MISSING>")
    confidence = parsed.get("confidence", "<MISSING>")
    notable = parsed.get("notable_signals", [])
    print(f"  decision={decision!r}  confidence={confidence!r}")
    print(f"  notable_signals ({len(notable)} items):")
    for i, sig in enumerate(notable):
        print(f"    [{i}] {sig!r}")

    if not notable:
        print("\n  [DIAGNOSIS] notable_signals is empty → SGR=0 by definition.")
        print("  Root cause: model output complies with JSON schema but provides")
        print("  no notable_signals items. Prompt reinforcement needed.")
        return

    print("\n  [GROUNDING CHECK] (verbatim substring, lowercased):")
    results = _check_grounding(notable, retrieved_context)
    n_grounded = sum(1 for _, g in results if g)
    for sig, grounded in results:
        status = "GROUNDED" if grounded else "NOT GROUNDED"
        truncated = sig[:80] + "..." if len(sig) > 80 else sig
        print(f"    {status}: {truncated!r}")
    sgr = n_grounded / len(results) if results else 0.0
    print(f"  SGR for {ticker}: {n_grounded}/{len(results)} = {sgr:.3f}")

    if sgr == 0.0 and notable:
        print("\n  [DIAGNOSIS] Signals parsed but NOT grounded in retrieved context.")
        print("  Root cause: model generates paraphrases/summaries, NOT verbatim quotes.")
        print("  Fix options:")
        print("    A) Prompt: add 'Copy the exact phrase verbatim from the filing.'")
        print("    B) Metric: relax to fuzzy/edit-distance matching (Phase 14).")
        print("    C) Model: test if 12B produces more faithful quotations.")


def run_diagnosis(ticker: str, data_dir: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"SGR Diagnostic: {ticker} as of {_AS_OF}")
    print("=" * 60)

    results: dict[str, dict] = {}

    for model_id, model_label in _MODELS_TO_TEST:
        print(f"\n[Model: {model_label}]")
        diagnose_ticker(ticker, model_id, data_dir)
        results[model_id] = {"model_label": model_label}

    print(f"\n{'=' * 60}")
    print("DJ-086 DECISION CRITERIA")
    print("=" * 60)
    print("Compare results above to decide:")
    print("  1. If E4B parses JSON correctly but 12B also does: prefer E4B (lighter).")
    print("  2. If E4B fails JSON parsing but 12B succeeds: use 12B as base.")
    print("  3. If neither produces verbatim quotes: prompt engineering is the fix,")
    print("     not model swap. Add explicit 'verbatim copy' instruction to prompt.")
    print("  4. If 12B is unavailable (connection refused): record E4B as only option.")
    print("\nRecord decision as DJ-086 in PHASE_13_CONTEXT.md.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Sentiment Agent SGR=0 root cause (DJ-086)"
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker to diagnose (default: AAPL — had 0 parseable signals)",
    )
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="Run diagnosis on AAPL, JPM, XOM (full baseline set)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HIFI_DATA_DIR", "data"),
        help="Path to data root directory (default: $HIFI_DATA_DIR or 'data')",
    )
    args = parser.parse_args()

    tickers = ["AAPL", "JPM", "XOM"] if args.all_tickers else [args.ticker]
    for t in tickers:
        run_diagnosis(t, args.data_dir)


if __name__ == "__main__":
    main()
