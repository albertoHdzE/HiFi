"""
check_env.py — prerequisite validator for HiFi Makefile targets.

Exits with code 0 on success, 1 on failure (with a clear error message).
Called by the Makefile before scripts that have live-service dependencies.

Usage
-----
    uv run python scripts/check_env.py --check langfuse
    uv run python scripts/check_env.py --check lm-studio
    uv run python scripts/check_env.py --check phase4-fixture
    uv run python scripts/check_env.py --check phase5-fixture
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_langfuse() -> list[str]:
    """Verify LangFuse SDK credentials are present in the environment."""
    errors = []
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        if not os.environ.get(var):
            errors.append(
                f"{var} is not set. "
                "Export it or run: source docker/langfuse/.env"
            )
    return errors


def check_lm_studio() -> list[str]:
    """Verify LM Studio is reachable and serving the /v1/models endpoint."""
    base_url = os.environ.get("HIFI_LM_STUDIO_URL", "http://localhost:1234/v1")
    models_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=3) as resp:
            if resp.status != 200:
                return [f"LM Studio returned HTTP {resp.status} at {models_url}"]
        return []
    except urllib.error.URLError as exc:
        return [
            f"LM Studio not reachable at {models_url}: {exc.reason}",
            "Start LM Studio and load a model before running this target.",
        ]
    except OSError as exc:
        return [f"LM Studio connectivity check failed: {exc}"]


def check_phase4_fixture() -> list[str]:
    """Verify the Phase 4 ensemble fixture exists."""
    path = PROJECT_ROOT / "tests" / "fixtures" / "baseline" / "phase4_ensemble.json"
    if not path.exists():
        return [
            f"Phase 4 fixture not found: {path}",
            "Generate it first: make baseline-phase4  (requires LM Studio)",
        ]
    return []


def check_phase5_fixture() -> list[str]:
    """Verify the Phase 5 verification fixture exists."""
    path = PROJECT_ROOT / "tests" / "fixtures" / "baseline" / "phase5_verification.json"
    if not path.exists():
        return [
            f"Phase 5 fixture not found: {path}",
            "Generate it first: make baseline-phase5",
        ]
    return []


def check_sec_fixtures() -> list[str]:
    """Verify SEC EDGAR fixture files exist in tests/fixtures/sec/.

    Expects at least 9 files (3 tickers x 3 filing types) recorded by
    scripts/record_sec_fixtures.py.
    """
    sec_dir = PROJECT_ROOT / "tests" / "fixtures" / "sec"
    if not sec_dir.exists():
        return [
            f"SEC fixtures directory not found: {sec_dir}",
            "Run first (requires internet): make sec-fixtures",
        ]
    files = list(sec_dir.glob("*.json"))
    if len(files) < 9:
        return [
            f"SEC fixtures incomplete: found {len(files)}/9 files in {sec_dir}",
            "Run first (requires internet): make sec-fixtures",
        ]
    return []


def check_phase7_fixture() -> list[str]:
    """Verify the Phase 7 RAG baseline fixture exists."""
    path = PROJECT_ROOT / "tests" / "fixtures" / "baseline" / "phase7_rag_baseline.json"
    if not path.exists():
        return [
            f"Phase 7 fixture not found: {path}",
            "Generate it first: make baseline-phase7",
        ]
    return []


_CHECKS = {
    "langfuse": check_langfuse,
    "lm-studio": check_lm_studio,
    "phase4-fixture": check_phase4_fixture,
    "phase5-fixture": check_phase5_fixture,
    "sec-fixtures": check_sec_fixtures,
    "phase7-fixture": check_phase7_fixture,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate HiFi runtime prerequisites before running scripts."
    )
    parser.add_argument(
        "--check",
        required=True,
        choices=list(_CHECKS),
        metavar="CHECK",
        help=f"Prerequisite to validate: {', '.join(_CHECKS)}",
    )
    args = parser.parse_args()

    errors = _CHECKS[args.check]()
    if errors:
        print(f"[check_env] FAIL ({args.check})")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)

    print(f"[check_env] OK ({args.check})")


if __name__ == "__main__":
    main()
