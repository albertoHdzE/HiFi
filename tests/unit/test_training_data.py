"""
Unit tests for hifi.models.training_data (P11-E1-T4, DJ-054, DJ-059).

Tests label generators and JSONL formatter with synthetic OHLCV Parquet files
written to pytest's tmp_path. No LLM, no live services required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hifi.models.training_data import (
    FineTuneEvaluationResult,
    _load_close_series,
    format_as_jsonl,
    generate_max_return_labels,
    generate_risk_adjusted_labels,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TECHNICAL_PROMPT = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "hifi" / "agents" / "prompts" / "technical_v1.md"
)
_FUNDAMENTAL_PROMPT = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "hifi" / "agents" / "prompts" / "fundamental_v1.md"
)


def _make_ohlcv_parquet(
    tmp_path: Path,
    ticker: str,
    n_days: int = 300,
    start: str = "2018-01-01",
    drift: float = 0.001,
) -> None:
    """
    Write a synthetic OHLCV Parquet to tmp_path/market/{ticker}_2018-01-01.parquet.

    Prices follow a random walk with known drift, seeded at 42.
    """
    dates = pd.bdate_range(start=start, periods=n_days)
    rng = np.random.default_rng(42)
    ret = rng.normal(drift, 0.015, n_days)
    close = 100.0 * (1 + ret).cumprod()
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    volume = rng.integers(1_000_000, 10_000_000, n_days).astype(float)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)
    df.index.name = "Date"

    market_dir = tmp_path / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(market_dir / f"{ticker}_2018-01-01.parquet")


# ---------------------------------------------------------------------------
# generate_max_return_labels tests
# ---------------------------------------------------------------------------


def test_max_return_labels_buy_case(tmp_path: Path) -> None:
    """
    With a strong positive drift, most labels should be 'Buy'.

    Uses a large drift (0.005/day) so forward 60-day returns exceed +2% consistently.
    """
    ticker = "BUYTEST"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=300, drift=0.005)
    df = generate_max_return_labels(ticker, str(tmp_path), horizon_days=60)

    assert not df.empty, "Expected labeled rows"
    assert set(df.columns) >= {"date", "ticker", "label", "forward_return", "horizon_days"}
    buy_rows = df[df["label"] == "Buy"]
    assert len(buy_rows) > 100, f"Expected many Buy rows with strong drift; got {len(buy_rows)}"
    assert all(row["forward_return"] > 0.02 for _, row in buy_rows.iterrows())


def test_max_return_labels_sell_case(tmp_path: Path) -> None:
    """With strong negative drift, most labels should be 'Sell'."""
    ticker = "SELLTEST"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=300, drift=-0.005)
    df = generate_max_return_labels(ticker, str(tmp_path), horizon_days=60)

    assert not df.empty
    sell_rows = df[df["label"] == "Sell"]
    assert len(sell_rows) > 50, f"Expected many Sell rows with negative drift; got {len(sell_rows)}"
    assert all(row["forward_return"] < -0.02 for _, row in sell_rows.iterrows())


def test_max_return_labels_insufficient_data(tmp_path: Path) -> None:
    """
    With fewer than horizon_days+1 rows, no labeled rows are produced.
    """
    ticker = "SHORT"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=30)
    df = generate_max_return_labels(ticker, str(tmp_path), horizon_days=60)
    assert df.empty, "Expected empty DataFrame when data is shorter than horizon"


def test_max_return_labels_no_lookahead_leakage(tmp_path: Path) -> None:
    """
    Each label at index i only uses data up to index i+horizon_days.

    Verify by checking that date[i] + horizon_days trading days corresponds
    to the price used for the forward return.
    """
    ticker = "NOLEAK"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)
    df = generate_max_return_labels(ticker, str(tmp_path), horizon_days=30)

    assert not df.empty
    # All label dates must come before the last possible date
    series = _load_close_series(ticker, str(tmp_path))
    assert series is not None
    last_valid_date = series.index[-(30 + 1)].date()
    assert all(row["date"] <= last_valid_date for _, row in df.iterrows()), (
        "Labels found for dates too close to end of data (lookahead leakage)"
    )


# ---------------------------------------------------------------------------
# generate_risk_adjusted_labels tests
# ---------------------------------------------------------------------------


def test_risk_adjusted_labels_high_sharpe(tmp_path: Path) -> None:
    """
    With a consistently positive drift, rolling Sharpe should exceed 0.8 frequently.
    """
    ticker = "HIGHSHARPE"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=300, drift=0.003)
    df = generate_risk_adjusted_labels(ticker, str(tmp_path), horizon_days=60)

    assert not df.empty
    assert "sharpe_60d" in df.columns
    buy_rows = df[df["label"] == "Buy"]
    # With positive drift, at least some periods should have high Sharpe
    assert len(buy_rows) > 0, "Expected some Buy rows with positive drift"
    assert all(row["sharpe_60d"] > 0.8 for _, row in buy_rows.iterrows())


def test_risk_adjusted_labels_low_sharpe(tmp_path: Path) -> None:
    """
    With negative drift, rolling Sharpe should fall below 0.3 (Sell label).
    """
    ticker = "LOWSHARPE"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=300, drift=-0.003)
    df = generate_risk_adjusted_labels(ticker, str(tmp_path), horizon_days=60)

    assert not df.empty
    sell_rows = df[df["label"] == "Sell"]
    assert len(sell_rows) > 0, "Expected some Sell rows with negative drift"
    assert all(row["sharpe_60d"] < 0.3 for _, row in sell_rows.iterrows())


def test_risk_adjusted_labels_schema(tmp_path: Path) -> None:
    """Risk-adjusted labels DataFrame has expected schema."""
    ticker = "SCHEMA"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)
    df = generate_risk_adjusted_labels(ticker, str(tmp_path))

    assert set(df.columns) >= {"date", "ticker", "label", "sharpe_60d", "forward_return", "horizon_days"}  # noqa: E501
    assert df["label"].isin(["Buy", "Hold", "Sell"]).all()
    assert df["ticker"].eq(ticker).all()


# ---------------------------------------------------------------------------
# format_as_jsonl tests
# ---------------------------------------------------------------------------


def test_format_as_jsonl_structure(tmp_path: Path) -> None:
    """
    format_as_jsonl returns a list of dicts, each with a 'messages' key
    containing system/user/assistant roles.
    """
    ticker = "FMTTEST"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)
    labels_df = generate_max_return_labels(ticker, str(tmp_path), horizon_days=60)
    # Use first 3 rows only for speed
    labels_df = labels_df.head(3)

    examples = format_as_jsonl(
        labels_df=labels_df,
        ticker=ticker,
        agent_type="technical",
        data_dir=str(tmp_path),
        prompt_template_path=str(_TECHNICAL_PROMPT),
    )

    assert isinstance(examples, list)
    assert len(examples) == 3
    for ex in examples:
        assert "messages" in ex
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"]
        for msg in ex["messages"]:
            assert isinstance(msg["content"], str)
            assert len(msg["content"]) > 0


def test_format_as_jsonl_assistant_is_valid_json(tmp_path: Path) -> None:
    """Each assistant message content is parseable as JSON."""
    ticker = "JSONTEST"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)
    labels_df = generate_max_return_labels(ticker, str(tmp_path), horizon_days=60).head(5)

    examples = format_as_jsonl(
        labels_df=labels_df,
        ticker=ticker,
        agent_type="technical",
        data_dir=str(tmp_path),
        prompt_template_path=str(_TECHNICAL_PROMPT),
    )

    for ex in examples:
        assistant = next(m for m in ex["messages"] if m["role"] == "assistant")
        parsed = json.loads(assistant["content"])
        assert "decision" in parsed
        assert parsed["decision"] in ("Buy", "Hold", "Sell")


def test_format_as_jsonl_schema_compliance(tmp_path: Path) -> None:
    """
    Assistant content contains all AgentSignal required fields
    (decision, confidence, rationale, key_concern).
    """
    ticker = "SCHEMAFMT"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)
    labels_df = generate_max_return_labels(ticker, str(tmp_path)).head(3)

    examples = format_as_jsonl(
        labels_df=labels_df,
        ticker=ticker,
        agent_type="technical",
        data_dir=str(tmp_path),
        prompt_template_path=str(_TECHNICAL_PROMPT),
    )

    required_fields = {"decision", "confidence", "rationale", "key_concern"}
    for ex in examples:
        assistant = next(m for m in ex["messages"] if m["role"] == "assistant")
        payload = json.loads(assistant["content"])
        assert required_fields.issubset(payload.keys()), (
            f"Missing fields: {required_fields - payload.keys()}"
        )
        assert 0.0 <= payload["confidence"] <= 1.0


def test_format_as_jsonl_label_matches_decision(tmp_path: Path) -> None:
    """The assistant decision matches the reference strategy label."""
    ticker = "MATCHLBL"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)
    labels_df = generate_max_return_labels(ticker, str(tmp_path)).head(5)

    examples = format_as_jsonl(
        labels_df=labels_df,
        ticker=ticker,
        agent_type="technical",
        data_dir=str(tmp_path),
        prompt_template_path=str(_TECHNICAL_PROMPT),
    )

    for i, (ex, (_, row)) in enumerate(zip(examples, labels_df.iterrows(), strict=False)):
        assistant = next(m for m in ex["messages"] if m["role"] == "assistant")
        decision = json.loads(assistant["content"])["decision"]
        assert decision == row["label"], (
            f"Row {i}: expected decision={row['label']}, got {decision}"
        )


def test_format_as_jsonl_fundamental(tmp_path: Path) -> None:
    """format_as_jsonl works for the fundamental agent with synthetic fundamentals."""
    ticker = "FUND"
    _make_ohlcv_parquet(tmp_path, ticker, n_days=200)
    labels_df = generate_risk_adjusted_labels(ticker, str(tmp_path)).head(3)

    examples = format_as_jsonl(
        labels_df=labels_df,
        ticker=ticker,
        agent_type="fundamental",
        data_dir=str(tmp_path),
        prompt_template_path=str(_FUNDAMENTAL_PROMPT),
    )

    assert len(examples) == 3
    for ex in examples:
        assistant = next(m for m in ex["messages"] if m["role"] == "assistant")
        payload = json.loads(assistant["content"])
        assert "decision" in payload
        assert "rationale" in payload


def test_format_as_jsonl_empty_df(tmp_path: Path) -> None:
    """format_as_jsonl with empty DataFrame returns empty list."""
    import pandas as pd
    empty_df = pd.DataFrame(columns=["date", "ticker", "label", "forward_return", "horizon_days"])
    result = format_as_jsonl(
        labels_df=empty_df,
        ticker="EMPTY",
        agent_type="technical",
        data_dir=str(tmp_path),
        prompt_template_path=str(_TECHNICAL_PROMPT),
    )
    assert result == []


# ---------------------------------------------------------------------------
# FineTuneEvaluationResult schema tests
# ---------------------------------------------------------------------------


def test_finetune_evaluation_result_json_roundtrip() -> None:
    """FineTuneEvaluationResult round-trips through JSON."""
    from datetime import UTC, datetime
    result = FineTuneEvaluationResult(
        ticker="AAPL",
        analysis_date="2023-03-31",
        base_technical_gr=0.667,
        base_fundamental_gr=1.000,
        finetuned_technical_gr=0.750,
        finetuned_fundamental_gr=1.000,
        base_pairwise_diversity=0.5,
        finetuned_pairwise_diversity=0.48,
        base_disagreement_entropy=0.6,
        finetuned_disagreement_entropy=0.58,
        generated_at=datetime.now(UTC).isoformat(),
    )
    serialised = result.model_dump_json()
    restored = FineTuneEvaluationResult.model_validate_json(serialised)
    assert restored.ticker == "AAPL"
    assert restored.finetuned_technical_gr == 0.750


def test_diversity_preserved_true() -> None:
    """diversity_preserved=True when finetuned >= 0.9 * base."""
    from datetime import UTC, datetime
    result = FineTuneEvaluationResult(
        ticker="JPM",
        analysis_date="2023-03-31",
        base_technical_gr=0.667,
        base_fundamental_gr=1.0,
        finetuned_technical_gr=0.72,
        finetuned_fundamental_gr=1.0,
        base_pairwise_diversity=0.5,
        finetuned_pairwise_diversity=0.47,  # 0.47 >= 0.9*0.5=0.45 -> True
        base_disagreement_entropy=0.6,
        finetuned_disagreement_entropy=0.57,
        generated_at=datetime.now(UTC).isoformat(),
    )
    assert result.diversity_preserved is True


def test_diversity_preserved_false() -> None:
    """diversity_preserved=False when finetuned < 0.9 * base."""
    from datetime import UTC, datetime
    result = FineTuneEvaluationResult(
        ticker="XOM",
        analysis_date="2023-03-31",
        base_technical_gr=0.667,
        base_fundamental_gr=1.0,
        finetuned_technical_gr=0.72,
        finetuned_fundamental_gr=1.0,
        base_pairwise_diversity=0.5,
        finetuned_pairwise_diversity=0.40,  # 0.40 < 0.9*0.5=0.45 -> False
        base_disagreement_entropy=0.6,
        finetuned_disagreement_entropy=0.50,
        generated_at=datetime.now(UTC).isoformat(),
    )
    assert result.diversity_preserved is False


def test_gr_improved_technical() -> None:
    """gr_improved_technical=True when delta >= 0.05."""
    from datetime import UTC, datetime
    result = FineTuneEvaluationResult(
        ticker="AAPL",
        analysis_date="2023-03-31",
        base_technical_gr=0.667,
        base_fundamental_gr=1.0,
        finetuned_technical_gr=0.720,   # delta=0.053 >= 0.05 -> True
        finetuned_fundamental_gr=1.0,
        base_pairwise_diversity=0.5,
        finetuned_pairwise_diversity=0.5,
        base_disagreement_entropy=0.6,
        finetuned_disagreement_entropy=0.6,
        generated_at=datetime.now(UTC).isoformat(),
    )
    assert result.gr_improved_technical is True


def test_gr_not_improved_when_delta_small() -> None:
    """gr_improved_technical=False when delta < 0.05."""
    from datetime import UTC, datetime
    result = FineTuneEvaluationResult(
        ticker="AAPL",
        analysis_date="2023-03-31",
        base_technical_gr=0.667,
        base_fundamental_gr=1.0,
        finetuned_technical_gr=0.700,   # delta=0.033 < 0.05 -> False
        finetuned_fundamental_gr=1.0,
        base_pairwise_diversity=0.5,
        finetuned_pairwise_diversity=0.5,
        base_disagreement_entropy=0.6,
        finetuned_disagreement_entropy=0.6,
        generated_at=datetime.now(UTC).isoformat(),
    )
    assert result.gr_improved_technical is False
