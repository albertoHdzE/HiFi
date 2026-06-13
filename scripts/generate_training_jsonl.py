"""
generate_training_jsonl.py -- Training JSONL generation (P11-E2-T1, DJ-054).

Converts reference strategy Parquets into mlx_lm-compatible JSONL training datasets
for the Technical and Fundamental agents.

Output:
  data/training/technical_max_return_60d.jsonl
  data/training/fundamental_risk_adjusted_60d.jsonl

Class balancing: Hold examples beyond 2x the Buy+Sell count are excluded.
Shuffle: fixed seed=42 for reproducibility.
Target: >= 400 examples per agent.

Usage:
    uv run python scripts/generate_training_jsonl.py [--agent technical|fundamental|both]
                                                     [--horizon 60]
                                                     [--data-dir DIR]
                                                     [--output-dir data/training]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from hifi.models.training_data import format_as_jsonl  # noqa: E402

_DEFAULT_TICKERS = [
    "AAPL", "JPM", "XOM",
    "MSFT", "NVDA", "GOOGL", "BAC", "GS", "CVX",
    "JNJ", "UNH", "AMZN", "WMT", "CAT", "NEE",
]

_TECHNICAL_PROMPT = str(_ROOT / "src" / "hifi" / "agents" / "prompts" / "technical_v1.md")
_FUNDAMENTAL_PROMPT = str(_ROOT / "src" / "hifi" / "agents" / "prompts" / "fundamental_v1.md")
_MIN_EXAMPLES = 400


def _balance_classes(examples: list[dict], seed: int = 42) -> list[dict]:
    """
    Exclude Hold examples beyond 2x the Buy+Sell count for class balance.

    This prevents Hold-heavy datasets from dominating training loss.
    """
    buy_sell = [e for e in examples if _decision(e) in ("Buy", "Sell")]
    holds = [e for e in examples if _decision(e) == "Hold"]
    max_holds = 2 * len(buy_sell)
    rng = random.Random(seed)
    if len(holds) > max_holds:
        holds = rng.sample(holds, max_holds)
    balanced = buy_sell + holds
    rng.shuffle(balanced)
    return balanced


def _decision(example: dict) -> str:
    """Extract decision from assistant message JSON."""
    try:
        assistant = next(m for m in example["messages"] if m["role"] == "assistant")
        return json.loads(assistant["content"]).get("decision", "Hold")
    except Exception:
        return "Hold"


def _generate_agent(
    agent_type: str,
    strategy: str,
    data_dir: str,
    output_dir: str,
    horizon: int,
    output_name: str,
) -> None:
    import pandas as pd

    ref_dir = Path(data_dir) / "reference_strategies" / strategy
    if not ref_dir.exists():
        logger.error("Reference strategy directory not found: %s", ref_dir)
        logger.error("Run: make generate-reference-strategies")
        sys.exit(1)

    prompt_path = _TECHNICAL_PROMPT if agent_type == "technical" else _FUNDAMENTAL_PROMPT
    all_examples: list[dict] = []

    for ticker in _DEFAULT_TICKERS:
        parquet_path = ref_dir / f"{ticker}_{horizon}d.parquet"
        if not parquet_path.exists():
            logger.warning("%s: Parquet not found at %s, skipping", ticker, parquet_path)
            continue

        try:
            labels_df = pd.read_parquet(parquet_path)
            examples = format_as_jsonl(
                labels_df=labels_df,
                ticker=ticker,
                agent_type=agent_type,
                data_dir=data_dir,
                prompt_template_path=prompt_path,
            )
            all_examples.extend(examples)
            logger.info("%s: %d examples", ticker, len(examples))
        except Exception as exc:
            logger.warning("%s: FAILED: %s", ticker, exc)
            continue

    balanced = _balance_classes(all_examples)
    n = len(balanced)

    if n < _MIN_EXAMPLES:
        logger.warning(
            "Only %d examples (target >= %d). Generate more reference strategies.",
            n, _MIN_EXAMPLES
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / output_name
    with open(out_path, "w") as f:
        for ex in balanced:
            f.write(json.dumps(ex) + "\n")

    # Class distribution summary
    decisions = [_decision(e) for e in balanced]
    counts = {d: decisions.count(d) for d in set(decisions)}
    print(f"\n{output_name}: {n} examples {counts}")
    avg_tokens = sum(
        len(str(e)) // 4  # rough token estimate
        for e in balanced
    ) // max(n, 1)
    print(f"  Avg ~{avg_tokens} tokens/example (rough estimate)")
    print(f"  Written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL training data for Phase 11 fine-tuning.")  # noqa: E501
    parser.add_argument("--agent", choices=["technical", "fundamental", "both"], default="both")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--data-dir", default=str(_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(_ROOT / "data" / "training"))
    args = parser.parse_args()

    if args.agent in ("technical", "both"):
        _generate_agent(
            agent_type="technical",
            strategy="max_return",
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            horizon=args.horizon,
            output_name=f"technical_max_return_{args.horizon}d.jsonl",
        )

    if args.agent in ("fundamental", "both"):
        _generate_agent(
            agent_type="fundamental",
            strategy="risk_adjusted",
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            horizon=args.horizon,
            output_name=f"fundamental_risk_adjusted_{args.horizon}d.jsonl",
        )


if __name__ == "__main__":
    main()
