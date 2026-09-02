"""
Sentiment signal diagnostic: loads gemma-3-12b-it via lms CLI, runs sentiment
analysis for JPM and XOM, prints actual signals vs retrieved context.

Usage
-----
    uv run python scripts/diag_sentiment_signals.py [--data-dir data]

Output: for each ticker, prints the retrieved context passages and the
signals Gemma generated, with a verbatim-match check per signal.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_LMS = os.path.expanduser("~/.lmstudio/bin/lms")
_MODEL = "gemma-3-12b-it"
_TICKERS = ["AAPL", "JPM", "XOM"]
_AS_OF = "2023-03-31"


def _lms_run(*args: str, timeout_s: int = 600) -> tuple[int, str]:
    result = subprocess.run(
        [_LMS, *args], capture_output=True, text=True, timeout=timeout_s,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def _load_model() -> bool:
    print(f"Loading {_MODEL} via lms CLI...", flush=True)
    rc, out = _lms_run("load", _MODEL, "-y", timeout_s=600)
    if rc == 0:
        print("  Loaded.")
        return True
    print(f"  FAILED (rc={rc}): {out[:200]}")
    return False


def _unload_model() -> None:
    rc, out = _lms_run("unload", _MODEL, timeout_s=60)
    if rc == 0:
        print(f"Unloaded {_MODEL}.")
    else:
        print(f"Warning: unload rc={rc}: {out[:100]}")


def _retrieve_context(ticker: str, data_dir: str) -> str:
    from hifi.agents.mcp_client import call_tool  # noqa: PLC0415
    query = (
        f"{ticker} management outlook guidance forward-looking statements "
        f"revenue growth margin services"
    )
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


def _run_ticker(ticker: str, data_dir: str) -> None:
    from hifi.agents.sentiment_agent import run_sentiment_analysis  # noqa: PLC0415

    print(f"\n{'='*70}")
    print(f"  {ticker}  |  {_AS_OF}")
    print(f"{'='*70}")

    ctx = _retrieve_context(ticker, data_dir)
    if not ctx:
        print("  !! No context retrieved — would return insufficient-data signal")
        return

    print("\n--- Retrieved context ---")
    for line in ctx.splitlines():
        print(f"  {line}")

    print("\n--- Running sentiment analysis with Gemma 3 12B ---", flush=True)
    os.environ["HIFI_SENTIMENT_MODEL"] = _MODEL
    t0 = time.perf_counter()
    analysis = run_sentiment_analysis(ticker=ticker, as_of_date=_AS_OF, data_dir=data_dir)
    elapsed = int((time.perf_counter() - t0) * 1000)
    print(f"  Latency: {elapsed}ms")

    if analysis.signal is None:
        print("  !! signal=None (parse failure)")
        return

    print(f"  Decision: {analysis.signal.decision}  confidence={analysis.signal.confidence:.2f}")
    print(f"  Rationale: {analysis.signal.rationale[:200]}")
    print(f"  n_signals: {len(analysis.notable_signals)}")

    ctx_lower = ctx.lower()
    print("\n--- Signal verbatim check ---")
    for i, sig in enumerate(analysis.notable_signals, 1):
        sig_lower = sig.lower().strip()
        grounded = bool(sig_lower) and (sig_lower in ctx_lower)
        status = "GROUNDED" if grounded else "NOT GROUNDED"
        print(f"  [{i}] {status}")
        print(f"       Signal : {repr(sig)}")
        if not grounded:
            # Find closest word overlap
            sig_words = set(sig_lower.split())
            ctx_words = set(ctx_lower.split())
            overlap = sig_words & ctx_words
            overlap_ratio = len(overlap) / len(sig_words) if sig_words else 0.0
            n_ov, n_sig = len(overlap), len(sig_words)
            print(f"       Word overlap ratio: {overlap_ratio:.2f} ({n_ov}/{n_sig} words in ctx)")
            if overlap:
                print(f"       Matching words: {sorted(overlap)[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-load", action="store_true",
                        help="Skip lms load (model already loaded)")
    parser.add_argument("--no-unload", action="store_true",
                        help="Skip lms unload after run")
    args = parser.parse_args()

    if not args.no_load and not _load_model():
        sys.exit(1)

    try:
        for ticker in _TICKERS:
            _run_ticker(ticker, args.data_dir)
    finally:
        if not args.no_unload:
            _unload_model()

    print("\nDone.")


if __name__ == "__main__":
    main()
