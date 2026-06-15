"""
generate_compliance_examples.py -- Structured output compliance examples (P11-E2-T2).

Generates secondary JSONL training examples from verified Phase 3/5 baseline runs
where HR=0.000 and GR=1.000. These examples teach the model correct JSON schema
and MCP field citation patterns -- targeting Technical Agent's GR=0.667 weakness.

Phase 12 (P12-E0-T1) extension: also produces technical_compliance_v2.jsonl with
>= 200 compliance examples by combining:
  - Phase 4 verified technical outputs (3 examples)
  - Phase 9 collective technical outputs (3 examples)
  - Synthetic format-compliant examples from OHLCV data (remainder to target >= 200)

Source fixtures (read-only):
  tests/fixtures/baseline/phase3_baseline.json
  tests/fixtures/baseline/phase4_ensemble.json (Phase 4 technical)
  tests/fixtures/baseline/phase5_verification.json
  tests/fixtures/baseline/phase9_collective.json (Phase 12 addition)

Output:
  data/training/technical_compliance.jsonl     (Phase 11 original, 3 examples)
  data/training/fundamental_compliance.jsonl   (Phase 11 original)
  data/training/technical_compliance_v2.jsonl  (Phase 12 augmented, >= 200 examples)

Usage:
    uv run python scripts/generate_compliance_examples.py [--output-dir DIR]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PHASE3_FIXTURE = _ROOT / "tests" / "fixtures" / "baseline" / "phase3_baseline.json"
_PHASE4_FIXTURE = _ROOT / "tests" / "fixtures" / "baseline" / "phase4_ensemble.json"
_PHASE5_FIXTURE = _ROOT / "tests" / "fixtures" / "baseline" / "phase5_verification.json"
_PHASE9_FIXTURE = _ROOT / "tests" / "fixtures" / "baseline" / "phase9_collective.json"
_TECHNICAL_PROMPT = _ROOT / "src" / "hifi" / "agents" / "prompts" / "technical_v1.md"
_FUNDAMENTAL_PROMPT = _ROOT / "src" / "hifi" / "agents" / "prompts" / "fundamental_v1.md"
_DATA_DIR = _ROOT / "data"
_V2_TARGET_COUNT = 200


def _parse_system(template_path: Path) -> str:
    """Extract system section from a prompt Markdown template."""
    from hifi.models.training_data import _parse_prompt_template
    system, _ = _parse_prompt_template(str(template_path))
    return system


def _extract_technical_examples(phase4_data: dict) -> list[dict]:
    """
    Extract verified Technical Agent examples from the Phase 4 ensemble fixture.

    Phase 4 format: outputs[ticker]['technical_analysis']['signal'].
    Only includes analyses where the signal is non-None (successful runs).
    """
    system = _parse_system(_TECHNICAL_PROMPT)
    examples = []
    outputs = phase4_data.get("outputs", {})

    for ticker, analysis_data in outputs.items():
        tech = analysis_data.get("technical_analysis") or {}
        if not tech:
            continue

        signal = tech.get("signal")
        if signal is None:
            continue

        # Build the user message from actual MCP tool outputs
        ti = tech.get("technical_indicators", {})
        rm = tech.get("risk_metrics", {})
        as_of_date = signal.get("as_of_date", "")

        user_content = (
            f"Analyze {ticker} as of {as_of_date} using technical and risk data only.\n\n"  # noqa: E501
            f"### Technical Indicators (20-day window)\n```json\n{json.dumps(ti, indent=2)}\n```\n\n"  # noqa: E501
            f"### Risk Metrics (trailing 252 days)\n```json\n{json.dumps(rm, indent=2)}\n```\n\n"
            f"### Data Gaps (fields that returned null -- do not cite these as known values)\n"
            f"{', '.join(signal.get('data_gaps', [])) or 'None'}"
        )

        # Assistant is the actual verified signal output
        assistant_payload = {
            "decision": signal.get("decision", "Hold"),
            "confidence": signal.get("confidence", 0.7),
            "rationale": signal.get("rationale", ""),
            "key_concern": signal.get("key_concern", ""),
            "time_horizon": "medium-term",
        }

        examples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(assistant_payload)},
            ]
        })

    return examples


def _extract_fundamental_examples(phase3_data: dict) -> list[dict]:
    """Extract verified Fundamental Agent examples from Phase 3 baseline fixture.

    Phase 3 fixture format: analyses[ticker] = {signal, financial_ratios, ...} (flat).
    Phase 4+ fixture format: analyses[ticker] = {fundamental_analysis: {signal, ...}}.
    Both formats are handled.
    """
    system = _parse_system(_FUNDAMENTAL_PROMPT)
    examples = []
    analyses = phase3_data.get("analyses", {})

    for ticker, analysis_data in analyses.items():
        # Try nested format first (phase4+), then flat format (phase3)
        fund = analysis_data.get("fundamental_analysis") or {}
        if not fund:
            # Phase 3 flat format: signal is directly on the ticker dict
            signal = analysis_data.get("signal")
            fin_ratios = analysis_data.get("financial_ratios", {})
            growth = analysis_data.get("growth_metrics", {})
            valuation = analysis_data.get("valuation_context", {})
            macro = analysis_data.get("macro_snapshot", {})
        else:
            signal = fund.get("signal")
            fin_ratios = fund.get("financial_ratios", {})
            growth = fund.get("growth_metrics", {})
            valuation = fund.get("valuation_context", {})
            macro = fund.get("macro_snapshot", {})

        if signal is None:
            continue

        as_of_date = signal.get("as_of_date", "")

        user_content = (
            f"Analyze {ticker} as of {as_of_date}.\n\n"
            f"### Financial Ratios\n```json\n{json.dumps(fin_ratios, indent=2)}\n```\n\n"
            f"### Growth Metrics\n```json\n{json.dumps(growth, indent=2)}\n```\n\n"
            f"### Valuation Context\n```json\n{json.dumps(valuation, indent=2)}\n```\n\n"
            f"### Macro Environment\n```json\n{json.dumps(macro, indent=2)}\n```\n\n"
            f"### Data Gaps (fields that returned null -- do not cite these as known values)\n"
            f"{', '.join(signal.get('data_gaps', [])) or 'None'}"
        )

        assistant_payload = {
            "decision": signal.get("decision", "Hold"),
            "confidence": signal.get("confidence", 0.7),
            "rationale": signal.get("rationale", ""),
            "key_concern": signal.get("key_concern", ""),
        }

        examples.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(assistant_payload)},
            ]
        })

    return examples


def _generate_synthetic_technical_compliance(
    data_dir: Path,
    prompt_path: Path,
    target_count: int,
) -> list[dict]:
    """
    Generate synthetic format-compliant technical compliance examples from OHLCV data.

    Computes real technical indicators (RSI, MACD, SMA, ATR, risk metrics) from
    stored OHLCV parquets at quarterly intervals, then formats them using the same
    mlx_lm chat format as the domain training examples.  The decision label is
    derived from max-return labeling (60-day forward return thresholds, DJ-054).

    This supplements the ~6 verified examples extracted from Phase 4/9 fixtures to
    reach the >= 200 target (DJ-061 fix: 0.75% compliance ratio).

    Parameters
    ----------
    data_dir : Path
        Root data directory containing data/market/*.parquet files.
    prompt_path : Path
        Path to technical_v1.md prompt template.
    target_count : int
        Number of synthetic examples to generate.

    Returns
    -------
    list[dict]
        Up to target_count JSONL-ready example dicts.
    """
    if target_count <= 0:
        return []

    from hifi.models.training_data import format_as_jsonl, generate_max_return_labels

    market_dir = data_dir / "market"
    if not market_dir.exists():
        logger.warning("No market/ directory at %s, skipping synthetic generation", data_dir)
        return []

    # Discover tickers from available parquets
    parquet_files = sorted(market_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No OHLCV parquets in %s", market_dir)
        return []

    tickers = sorted({f.name.split("_")[0] for f in parquet_files})
    logger.info("Generating synthetic compliance examples from %d tickers", len(tickers))

    examples: list[dict] = []
    max_per_ticker = max(1, (target_count + len(tickers) - 1) // len(tickers))

    for ticker in tickers:
        if len(examples) >= target_count:
            break

        labels_df = generate_max_return_labels(ticker, str(data_dir))
        if labels_df.empty:
            logger.debug("No labels for %s, skipping", ticker)
            continue

        # Sample approximately quarterly (every ~63 trading days) for diversity
        sampled = labels_df.iloc[::63].head(max_per_ticker)
        if sampled.empty:
            continue

        ticker_examples = format_as_jsonl(
            sampled,
            ticker=ticker,
            agent_type="technical",
            data_dir=str(data_dir),
            prompt_template_path=str(prompt_path),
        )
        examples.extend(ticker_examples)
        logger.debug("Ticker %s: %d synthetic examples", ticker, len(ticker_examples))

    return examples[:target_count]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate structured output compliance examples from Phase 3/5/9 baselines."
    )
    parser.add_argument("--output-dir", default=str(_ROOT / "data" / "training"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 3 fixture (fundamental agent)
    if not _PHASE3_FIXTURE.exists():
        logger.error("Phase 3 fixture not found: %s", _PHASE3_FIXTURE)
        logger.error("Generate it first: make baseline-phase3  (requires LM Studio)")
        sys.exit(1)

    phase3_data = json.loads(_PHASE3_FIXTURE.read_text())

    # Load Phase 4 fixture (technical + fundamental ensemble)
    phase4_data: dict = {}
    if _PHASE4_FIXTURE.exists():
        phase4_data = json.loads(_PHASE4_FIXTURE.read_text())
    else:
        logger.warning("Phase 4 fixture not found -- technical compliance examples will be empty")

    # Technical compliance examples (from phase4 which has technical_analysis)
    tech_examples = _extract_technical_examples(phase4_data)
    tech_out = output_dir / "technical_compliance.jsonl"
    with open(tech_out, "w") as f:
        for ex in tech_examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Technical compliance: %d examples -> %s", len(tech_examples), tech_out)

    # Fundamental compliance examples
    fund_examples = _extract_fundamental_examples(phase3_data)
    fund_out = output_dir / "fundamental_compliance.jsonl"
    with open(fund_out, "w") as f:
        for ex in fund_examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Fundamental compliance: %d examples -> %s", len(fund_examples), fund_out)

    # -----------------------------------------------------------------------
    # Phase 12: technical_compliance_v2.jsonl (>= 200 examples, DJ-061)
    # -----------------------------------------------------------------------
    # Step 1: Extract verified examples from Phase 4 and Phase 9 fixtures
    tech_v2_extracted: list[dict] = []
    tech_v2_extracted.extend(_extract_technical_examples(phase4_data))

    phase9_data: dict = {}
    if _PHASE9_FIXTURE.exists():
        phase9_data = json.loads(_PHASE9_FIXTURE.read_text())
        tech_v2_extracted.extend(_extract_technical_examples(phase9_data))
    else:
        logger.warning("Phase 9 fixture not found -- Phase 9 compliance examples skipped")

    # Deduplicate by assistant content (Phase 4 and Phase 9 may overlap on same ticker/date)
    seen_keys: set[str] = set()
    tech_v2_deduped: list[dict] = []
    for ex in tech_v2_extracted:
        assistant_msgs = [m for m in ex.get("messages", []) if m.get("role") == "assistant"]
        key = assistant_msgs[0]["content"] if assistant_msgs else json.dumps(ex)
        if key not in seen_keys:
            seen_keys.add(key)
            tech_v2_deduped.append(ex)

    # Step 2: Generate synthetic examples from OHLCV data to reach target
    synthetic_needed = max(0, _V2_TARGET_COUNT - len(tech_v2_deduped))
    synthetic_examples = _generate_synthetic_technical_compliance(
        data_dir=_DATA_DIR,
        prompt_path=_TECHNICAL_PROMPT,
        target_count=synthetic_needed,
    )

    tech_v2_examples = tech_v2_deduped + synthetic_examples
    tech_v2_out = output_dir / "technical_compliance_v2.jsonl"
    with open(tech_v2_out, "w") as f:
        for ex in tech_v2_examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(
        "Technical compliance v2: %d examples (%d extracted, %d synthetic) -> %s",
        len(tech_v2_examples),
        len(tech_v2_deduped),
        len(synthetic_examples),
        tech_v2_out,
    )

    print(f"\nCompliance examples: technical={len(tech_examples)}, fundamental={len(fund_examples)}")  # noqa: E501
    print(f"technical_v2={len(tech_v2_examples)} (target >= {_V2_TARGET_COUNT})")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
