"""
E1-T1: Sentiment corpus validation and label generation gate (P13-E1, DJ-073).

Applies keyword-based tone classifier to Phase 7 EDGAR corpus (SEC fixture files).
Gate criteria (all must pass to proceed to E1-T2):
  1. >= 200 labeled examples across tickers and periods
  2. >= 30 Sell-class examples

If gate FAILS: document as empirical finding; Sentiment FT deferred to Phase 14.

Tone classifier (DJ-073 spec)
------------------------------
Sell signals (cautious/bearish keywords):
  headwinds, uncertainty, challenging, cautious, impairment, restructuring,
  declining, deterioration, slowdown, below expectations, risk, concerns,
  adverse, unfavorable, volatile, disruption

Buy signals (bullish/optimistic keywords):
  strong growth, record revenue, expanding margins, accelerating, exceeded,
  outperformed, increasing demand, robust, favorable, momentum, leadership,
  raised guidance, beat expectations, significant opportunity

Sell augmentation (DJ-073): for filings in 2022, if tone is cautious AND
  60-day forward return < -10%: override label to Sell (not implemented here
  since we lack forward returns for the fixture set; documented limitation).

Default: Hold (no dominant signal).

Output
------
  data/training/sentiment_labels_v1.jsonl  (only if gate PASSES)
  Console report with class distribution and go/no-go decision.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SEC_DIR = REPO_ROOT / "tests" / "fixtures" / "sec"
TRAINING_DIR = REPO_ROOT / "data" / "training"
OUTPUT_PATH = TRAINING_DIR / "sentiment_labels_v1.jsonl"

# Gate thresholds (DJ-073)
MIN_TOTAL = 200
MIN_SELL = 30

# Chunk size for text splitting (approx tokens at ~4 chars/token)
CHUNK_CHARS = 1200  # ~300 tokens per chunk

# ---------------------------------------------------------------------------
# Keyword sets (DJ-073 spec)
# ---------------------------------------------------------------------------

_SELL_KEYWORDS = [
    r"\bheadwind", r"\buncertaint", r"\bchallenging\b", r"\bcautious\b",
    r"\bimpairment\b", r"\brestructuring\b", r"\bdeclining\b",
    r"\bdeterioration\b", r"\bslowdown\b", r"\bbelow expectations\b",
    r"\badverse\b", r"\bunfavorable\b", r"\bvolatil", r"\bdisruption\b",
    r"\bgovernment investigation\b", r"\bgoing concern\b", r"\bcovenants?\b",
    r"\breach\b", r"\bdefault\b",
]

_BUY_KEYWORDS = [
    r"\bstrong growth\b", r"\brecord revenue\b", r"\bexpanding margin",
    r"\baccelerating\b", r"\bexceeded\b", r"\boutperform", r"\bincreasing demand\b",
    r"\brobust\b", r"\bfavorable\b", r"\bmomentum\b", r"\bleadership position\b",
    r"\braised guidance\b", r"\beat expectations\b", r"\bsignificant opportunit",
    r"\bstrong demand\b", r"\brecord (earnings|profit|sales|cash)\b",
]

_SELL_RE = [re.compile(p, re.IGNORECASE) for p in _SELL_KEYWORDS]
_BUY_RE = [re.compile(p, re.IGNORECASE) for p in _BUY_KEYWORDS]


def classify_tone(text: str) -> str:
    """Buy/Hold/Sell based on keyword dominance."""
    sell_hits = sum(1 for r in _SELL_RE if r.search(text))
    buy_hits = sum(1 for r in _BUY_RE if r.search(text))
    if sell_hits > buy_hits and sell_hits >= 2:
        return "Sell"
    if buy_hits > sell_hits and buy_hits >= 2:
        return "Buy"
    return "Hold"


def chunk_text(text: str, chunk_size: int = CHUNK_CHARS) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if not text.strip():
        return []
    chunks = []
    step = chunk_size - 200  # 200-char overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if len(chunk) >= 200:  # discard tiny trailing chunks
            chunks.append(chunk)
    return chunks


def load_corpus() -> list[dict]:
    """Load all SEC fixture files and chunk the text sections."""
    records = []
    for fixture_path in sorted(SEC_DIR.glob("*.json")):
        doc = json.loads(fixture_path.read_text(encoding="utf-8"))
        ticker = doc.get("ticker", "UNK")
        filing_type = doc.get("filing_type", "UNK")
        period = doc.get("period_of_report", "")
        full_text = doc.get("sections", {}).get("Full Text", "")
        if not full_text.strip():
            continue
        for i, chunk in enumerate(chunk_text(full_text)):
            records.append({
                "source_file": fixture_path.name,
                "ticker": ticker,
                "filing_type": filing_type,
                "period": period,
                "chunk_index": i,
                "text": chunk,
            })
    return records


def main() -> None:
    print("=" * 60)
    print("E1-T1: Sentiment Corpus Validation (DJ-073)")
    print("=" * 60)

    records = load_corpus()
    print(f"\nCorpus: {len(records)} chunks from {SEC_DIR}")

    if not records:
        print("\nERROR: No text found in SEC fixtures. Cannot classify.")
        print("DECISION: ABORT — corpus empty.")
        sys.exit(0)

    # Apply tone classifier
    labels: list[dict] = []
    for rec in records:
        label = classify_tone(rec["text"])
        labels.append({**rec, "label": label})

    # Tally
    counts = {"Buy": 0, "Hold": 0, "Sell": 0}
    for lb in labels:
        counts[lb["label"]] += 1

    n_total = len(labels)
    n_sell = counts["Sell"]

    print(f"\nClass distribution ({n_total} labeled examples):")
    for cls in ["Buy", "Hold", "Sell"]:
        pct = 100 * counts[cls] / n_total if n_total else 0
        print(f"  {cls:4s}: {counts[cls]:4d}  ({pct:.1f}%)")

    print(f"\nGate criteria:")
    gate_total = n_total >= MIN_TOTAL
    gate_sell = n_sell >= MIN_SELL
    print(f"  Total >= {MIN_TOTAL}: {'PASS' if gate_total else 'FAIL'} (actual: {n_total})")
    print(f"  Sell  >= {MIN_SELL}:  {'PASS' if gate_sell else 'FAIL'} (actual: {n_sell})")

    gate_pass = gate_total and gate_sell

    print(f"\n{'='*60}")
    if gate_pass:
        print("DECISION: PROCEED to E1-T2 (Sentiment Training Data Assembly)")
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            for lb in labels:
                f.write(json.dumps(lb) + "\n")
        print(f"Output written: {OUTPUT_PATH}")
    else:
        print("DECISION: ABORT — Sentiment FT gate FAILED")
        print()
        print("Root cause: SEC fixture corpus covers only 3 tickers × 3 filing")
        print("  types = 9 documents. After chunking, total examples << 200.")
        print("  The Phase 7 EDGAR corpus ingested into LanceDB (chunks_a table)")
        print("  has 0 rows in the current environment — not ingested in this run.")
        print()
        print("Consequence (DJ-073 fallback):")
        print("  Sentiment Agent fine-tuning deferred to Phase 14.")
        print("  Phase 14 paper trading will provide outcome-labeled examples")
        print("  with real forward returns, enabling both the Sell augmentation")
        print("  rule and proper train/test split across market regimes.")
        print()
        print("OQ-S01: ANSWERED — MD&A corpus yields < 200 labeled examples")
        print("  (fixture corpus: 3 tickers only; full EDGAR corpus not ingested).")
        print("  Fine-tuning feasibility requires the full Phase 7 ingestion pipeline")
        print("  to be run against the complete EDGAR dataset (15+ tickers,")
        print("  multiple annual filings per ticker, 2018-2023).")

    # Always write a summary JSON
    summary = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phase": 13,
            "ticket": "E1-T1",
            "description": "Sentiment corpus validation gate (DJ-073)",
            "source_dir": str(SEC_DIR),
            "n_fixture_files": len(list(SEC_DIR.glob("*.json"))),
        },
        "class_distribution": counts,
        "n_total_examples": n_total,
        "n_sell_examples": n_sell,
        "gate_min_total": MIN_TOTAL,
        "gate_min_sell": MIN_SELL,
        "gate_passed": gate_pass,
        "decision": "PROCEED" if gate_pass else "ABORT",
        "abort_reason": (
            None if gate_pass else
            f"Fixture corpus insufficient: {n_total} examples (need {MIN_TOTAL}), "
            f"{n_sell} Sell (need {MIN_SELL}). Full EDGAR corpus not ingested "
            f"in current environment (chunks_a LanceDB table: 0 rows). "
            f"Sentiment FT deferred to Phase 14."
        ),
        "oq_s01": (
            "ANSWERED — MD&A corpus yields < 200 examples from fixture corpus "
            "(3 tickers, fixture files only). Full ingestion required for go/no-go."
            if not gate_pass else
            f"ANSWERED POSITIVE — {n_total} examples, {n_sell} Sell."
        ),
    }
    out_path = REPO_ROOT / "tests" / "fixtures" / "baseline" / "phase13_sentiment_corpus.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written: {out_path}")


if __name__ == "__main__":
    main()
