"""
generate_compliance_examples.py -- Structured output compliance examples (P11-E2-T2).

Generates secondary JSONL training examples from verified Phase 3/5 baseline runs
where HR=0.000 and GR=1.000. These examples teach the model correct JSON schema
and MCP field citation patterns -- targeting Technical Agent's GR=0.667 weakness.

Source fixtures (read-only):
  tests/fixtures/baseline/phase3_baseline.json
  tests/fixtures/baseline/phase5_verification.json

Output:
  data/training/technical_compliance.jsonl
  data/training/fundamental_compliance.jsonl

Usage:
    uv run python scripts/generate_compliance_examples.py [--data-dir DIR]
                                                          [--output-dir DIR]
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
_TECHNICAL_PROMPT = _ROOT / "src" / "hifi" / "agents" / "prompts" / "technical_v1.md"
_FUNDAMENTAL_PROMPT = _ROOT / "src" / "hifi" / "agents" / "prompts" / "fundamental_v1.md"


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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate structured output compliance examples from Phase 3/5 baselines."
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

    print(f"\nCompliance examples: technical={len(tech_examples)}, fundamental={len(fund_examples)}")  # noqa: E501
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
