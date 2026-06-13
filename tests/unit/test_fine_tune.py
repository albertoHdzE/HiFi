"""
Unit tests for hifi.models.fine_tune (P11-E3-T4, DJ-056).

Tests the Python wrapper logic for mlx_lm subprocess orchestration.
Does NOT test actual mlx_lm training (requires GPU + model download).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hifi.models.fine_tune import (
    check_adapter_quality,
    load_rank_sweep_results,
    optimal_rank_from_sweep,
    run_lora_training,
)

# ---------------------------------------------------------------------------
# run_lora_training tests
# ---------------------------------------------------------------------------


def test_run_lora_training_missing_venv(tmp_path: Path) -> None:
    """Raises RuntimeError with clear message when venv_python does not exist."""
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"messages": []}\n')

    with pytest.raises(RuntimeError, match="finetune venv not found"):
        run_lora_training(
            train_file=str(train_file),
            output_dir=str(tmp_path / "out"),
            venv_python=str(tmp_path / "nonexistent" / "python"),
        )


def test_run_lora_training_missing_train_file(tmp_path: Path) -> None:
    """Raises RuntimeError when training file does not exist."""
    venv_py = tmp_path / "python"
    venv_py.touch()
    venv_py.chmod(0o755)

    with pytest.raises(RuntimeError, match="Training file not found"):
        run_lora_training(
            train_file=str(tmp_path / "nonexistent.jsonl"),
            output_dir=str(tmp_path / "out"),
            venv_python=str(venv_py),
        )


def test_run_lora_training_returns_expected_keys(tmp_path: Path) -> None:
    """
    Mock the subprocess call: assert the return dict has the required keys
    (output_dir, train_loss, n_examples, duration_seconds).
    """
    venv_py = tmp_path / "python"
    venv_py.touch()
    venv_py.chmod(0o755)

    train_file = tmp_path / "train.jsonl"
    # Write 5 example lines
    train_file.write_text(
        '\n'.join(['{"messages": [{"role": "user", "content": "test"}]}'] * 5)
    )

    output_dir = tmp_path / "adapters"

    # Mock subprocess.run to simulate a successful training run
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Iter 100: Train loss 1.234\n"

    # Also need to create a fake adapter file so the existence check passes
    def fake_run(cmd, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapters.json").write_text('{"model_type": "lora"}')
        return mock_result

    with patch("hifi.models.fine_tune.subprocess.run", side_effect=fake_run):
        result = run_lora_training(
            train_file=str(train_file),
            output_dir=str(output_dir),
            lora_rank=8,
            num_iters=10,
            venv_python=str(venv_py),
        )

    assert "output_dir" in result
    assert "train_loss" in result
    assert "n_examples" in result
    assert "duration_seconds" in result
    assert result["n_examples"] == 5


def test_run_lora_training_no_adapters_produced(tmp_path: Path) -> None:
    """Raises RuntimeError when training completes but no adapter files are produced."""
    venv_py = tmp_path / "python"
    venv_py.touch()
    venv_py.chmod(0o755)

    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"messages": []}\n')

    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("hifi.models.fine_tune.subprocess.run", return_value=mock_result), \
            pytest.raises(RuntimeError, match="no adapter files found"):
            run_lora_training(
                train_file=str(train_file),
                output_dir=str(tmp_path / "empty_out"),
                venv_python=str(venv_py),
            )


# ---------------------------------------------------------------------------
# check_adapter_quality tests
# ---------------------------------------------------------------------------


def test_check_adapter_quality_missing_dir() -> None:
    """Returns False when adapter_dir does not exist."""
    result = check_adapter_quality("/nonexistent/path/adapters")
    assert result is False


def test_check_adapter_quality_empty_dir(tmp_path: Path) -> None:
    """Returns False when adapter_dir exists but has no adapter files."""
    adapter_dir = tmp_path / "empty_adapters"
    adapter_dir.mkdir()
    result = check_adapter_quality(str(adapter_dir))
    assert result is False


def test_check_adapter_quality_missing_venv(tmp_path: Path) -> None:
    """Returns False when venv_python does not exist."""
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.json").write_text("{}")
    result = check_adapter_quality(str(adapter_dir), venv_python=str(tmp_path / "no_python"))
    assert result is False


def test_check_adapter_quality_generation_success(tmp_path: Path) -> None:
    """Returns True when generation subprocess succeeds with non-empty output."""
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.json").write_text("{}")

    venv_py = tmp_path / "python"
    venv_py.touch()
    venv_py.chmod(0o755)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Hello world"

    with patch("hifi.models.fine_tune.subprocess.run", return_value=mock_result):
        result = check_adapter_quality(str(adapter_dir), venv_python=str(venv_py))

    assert result is True


# ---------------------------------------------------------------------------
# Rank sweep helpers
# ---------------------------------------------------------------------------


def test_optimal_rank_from_sweep_selects_lowest_loss() -> None:
    """optimal_rank_from_sweep selects the rank with lowest train_loss among quality_ok=True."""
    sweep = {
        "4": {"train_loss": 2.5, "quality_ok": True},
        "8": {"train_loss": 2.1, "quality_ok": True},
        "16": {"train_loss": 2.3, "quality_ok": True},
        "32": {"train_loss": None, "quality_ok": False},
    }
    assert optimal_rank_from_sweep(sweep) == 8


def test_optimal_rank_fallback_when_none_qualify() -> None:
    """Defaults to rank 8 when no rank has quality_ok=True."""
    sweep = {
        "4": {"train_loss": None, "quality_ok": False},
        "8": {"train_loss": None, "quality_ok": False},
    }
    assert optimal_rank_from_sweep(sweep) == 8


def test_load_rank_sweep_results_missing(tmp_path: Path) -> None:
    """Returns None when rank_sweep_results.json does not exist."""
    result = load_rank_sweep_results(str(tmp_path))
    assert result is None


def test_load_rank_sweep_results_exists(tmp_path: Path) -> None:
    """Returns parsed JSON when file exists."""
    sweep_data = {"8": {"train_loss": 2.1, "quality_ok": True}}
    sweep_path = tmp_path / "rank_sweep_results.json"
    sweep_path.write_text(json.dumps(sweep_data))

    result = load_rank_sweep_results(str(tmp_path))
    assert result is not None
    assert "8" in result
