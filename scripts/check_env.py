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


def check_market_data() -> list[str]:
    """Verify Phase 1 OHLCV Parquet files exist for AAPL, JPM, XOM, SPY."""
    data_dir = os.environ.get("HIFI_DATA_DIR", str(PROJECT_ROOT / "data"))
    market_dir = Path(data_dir) / "market"
    missing = []
    for ticker in ("AAPL", "JPM", "XOM", "SPY"):
        import glob as _glob
        if not _glob.glob(str(market_dir / f"{ticker}_*.parquet")):
            missing.append(ticker)
    if missing:
        return [
            f"Market Parquet files missing for: {', '.join(missing)}",
            "Run first: make acquire-data",
        ]
    return []


def check_phase8_fixture() -> list[str]:
    """Verify the Phase 8 agent population baseline fixture exists."""
    path = PROJECT_ROOT / "tests" / "fixtures" / "baseline" / "phase8_agent_population.json"
    if not path.exists():
        return [
            f"Phase 8 fixture not found: {path}",
            "Generate it first: make baseline-phase8  (requires LM Studio)",
        ]
    return []


def check_phase9_bootstrap() -> list[str]:
    """Verify the Phase 9 performance history bootstrap file exists."""
    data_dir = os.environ.get("HIFI_DATA_DIR", str(PROJECT_ROOT / "data"))
    path = Path(data_dir) / "agent_performance_history.json"
    if not path.exists():
        return [
            f"Phase 9 bootstrap not found: {path}",
            "Generate it first: make bootstrap-phase9  (no LM Studio required)",
        ]
    return []


def check_phase9_fixture() -> list[str]:
    """Verify the Phase 9 collective engine baseline fixture exists."""
    path = PROJECT_ROOT / "tests" / "fixtures" / "baseline" / "phase9_collective.json"
    if not path.exists():
        return [
            f"Phase 9 fixture not found: {path}",
            "Generate it first: make baseline-phase9  (requires LM Studio)",
        ]
    return []


def check_phase10_data() -> list[str]:
    """Verify Phase 10 OHLCV Parquet files exist for all 12 new tickers."""
    data_dir = os.environ.get("HIFI_DATA_DIR", str(PROJECT_ROOT / "data"))
    market_dir = Path(data_dir) / "market"
    new_tickers = (
        "MSFT", "NVDA", "GOOGL", "BAC", "GS", "CVX", "JNJ", "UNH", "AMZN", "WMT", "CAT", "NEE"
    )
    import glob as _glob
    missing = [t for t in new_tickers if not _glob.glob(str(market_dir / f"{t}_*.parquet"))]
    if missing:
        return [
            f"Phase 10 market Parquet files missing for: {', '.join(missing)}",
            "Run first: make acquire-data-phase10",
        ]
    return []


def check_phase10_bootstrap() -> list[str]:
    """Verify the Phase 10 performance history has >= 1000 labeled records."""
    data_dir = os.environ.get("HIFI_DATA_DIR", str(PROJECT_ROOT / "data"))
    path = Path(data_dir) / "agent_performance_history.json"
    if not path.exists():
        return [
            f"Performance history not found: {path}",
            "Run first: make bootstrap  (no LM Studio required)",
        ]
    import json
    try:
        data = json.loads(path.read_text())
        n_labeled = sum(1 for r in data.get("records", []) if r.get("outcome_correct") is not None)
        if n_labeled < 1000:
            return [
                f"Performance history has only {n_labeled} labeled records (need >= 1000).",
                "Run first: make bootstrap  (Phase 10 bootstrap with 15 tickers)",
            ]
    except Exception as exc:
        return [f"Failed to parse performance history: {exc}"]
    return []


def check_phase10_fixture() -> list[str]:
    """Verify the Phase 10 accuracy baseline fixture exists."""
    path = PROJECT_ROOT / "tests" / "fixtures" / "baseline" / "phase10_accuracy.json"
    if not path.exists():
        return [
            f"Phase 10 fixture not found: {path}",
            "Generate it first: make baseline-phase10  (no LM Studio required)",
        ]
    return []


_CHECKS = {
    "langfuse": check_langfuse,
    "lm-studio": check_lm_studio,
    "market-data": check_market_data,
    "phase4-fixture": check_phase4_fixture,
    "phase5-fixture": check_phase5_fixture,
    "sec-fixtures": check_sec_fixtures,
    "phase7-fixture": check_phase7_fixture,
    "phase8-fixture": check_phase8_fixture,
    "phase9-bootstrap": check_phase9_bootstrap,
    "phase9-fixture": check_phase9_fixture,
    "phase10-data": check_phase10_data,
    "phase10-bootstrap": check_phase10_bootstrap,
    "phase10-fixture": check_phase10_fixture,
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
