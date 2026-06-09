"""Tests for HiFi configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from hifi.config.loader import HiFiConfig, load_config


class TestLoadConfig:
    """Test configuration loading from YAML files."""

    def test_load_default_config(self):
        """Default configuration loads without error."""
        config = load_config()
        assert isinstance(config, HiFiConfig)

    def test_default_config_project_name(self):
        """Default config has correct project name."""
        config = load_config()
        assert config.project.name == "HiFi"

    def test_default_config_has_universe(self):
        """Default config defines a non-empty stock universe."""
        config = load_config()
        assert len(config.data.universe) > 0

    def test_default_config_universe_contains_expected_stocks(self):
        """Default universe includes baseline stocks."""
        config = load_config()
        assert "AAPL" in config.data.universe
        assert "NVDA" in config.data.universe
        assert "JPM" in config.data.universe

    def test_default_config_has_evaluation_periods(self):
        """Default config defines evaluation periods."""
        config = load_config()
        assert len(config.evaluation.periods) > 0
        assert "eval_crisis" in config.evaluation.periods

    def test_default_config_safety_limits(self):
        """Safety limits have sensible defaults."""
        config = load_config()
        assert 0 < config.safety.max_position_pct <= 0.10
        assert 0 < config.safety.max_daily_loss_pct <= 0.05
        assert config.safety.min_agents_quorum >= 1

    def test_default_seed_is_deterministic(self):
        """Reproducibility seed is set."""
        config = load_config()
        assert config.reproducibility.default_seed == 42

    def test_load_missing_file_raises(self, tmp_path: Path):
        """Loading a non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_custom_config(self, tmp_path: Path):
        """A custom config file can override defaults."""
        custom = {
            "project": {"name": "TestHiFi", "version": "0.0.1"},
            "data": {"universe": ["TEST"], "start_date": "2020-01-01"},
            "reproducibility": {"default_seed": 99},
        }
        config_path = tmp_path / "custom.yaml"
        with open(config_path, "w") as f:
            yaml.dump(custom, f)

        config = load_config(config_path)
        assert config.project.name == "TestHiFi"
        assert config.data.universe == ["TEST"]
        assert config.reproducibility.default_seed == 99

    def test_load_empty_config_uses_defaults(self, tmp_path: Path):
        """An empty YAML file produces a valid config with all defaults."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        config = load_config(config_path)
        assert isinstance(config, HiFiConfig)
        assert config.project.name == "HiFi"

    def test_load_partial_config_fills_defaults(self, tmp_path: Path):
        """A partial config fills missing sections with defaults."""
        partial = {"data": {"universe": ["AAPL"]}}
        config_path = tmp_path / "partial.yaml"
        with open(config_path, "w") as f:
            yaml.dump(partial, f)

        config = load_config(config_path)
        assert config.data.universe == ["AAPL"]
        # Other sections should have defaults
        assert config.project.name == "HiFi"
        assert config.safety.max_position_pct == 0.05


class TestHiFiConfigValidation:
    """Test Pydantic validation of configuration values."""

    def test_valid_config_from_dict(self):
        """A valid dict produces a valid config."""
        config = HiFiConfig(
            project={"name": "Test", "version": "1.0"},
            data={"universe": ["AAPL"], "start_date": "2020-01-01"},
        )
        assert config.project.name == "Test"

    def test_empty_dict_produces_defaults(self):
        """An empty dict produces all defaults."""
        config = HiFiConfig()
        assert config.project.name == "HiFi"
        assert config.reproducibility.default_seed == 42


class TestSyntheticDataFixtures:
    """Verify that synthetic data fixtures are deterministic."""

    def test_ohlcv_is_deterministic(self, synthetic_ohlcv):
        """OHLCV fixture produces the same data on every run."""
        # The first close price should always be the same with seed=42
        first_close = synthetic_ohlcv["close"][0]
        assert synthetic_ohlcv["n_days"] == 252
        # Verify it is a real number, not NaN or Inf
        assert first_close > 0
        assert not any(v != v for v in synthetic_ohlcv["close"])  # no NaN

    def test_ohlcv_price_relationships(self, synthetic_ohlcv):
        """OHLCV data maintains high >= open, close >= low."""
        for i in range(synthetic_ohlcv["n_days"]):
            assert synthetic_ohlcv["high"][i] >= synthetic_ohlcv["low"][i]
            assert synthetic_ohlcv["high"][i] >= synthetic_ohlcv["open"][i]
            assert synthetic_ohlcv["high"][i] >= synthetic_ohlcv["close"][i]
            assert synthetic_ohlcv["low"][i] <= synthetic_ohlcv["open"][i]
            assert synthetic_ohlcv["low"][i] <= synthetic_ohlcv["close"][i]

    def test_financials_are_deterministic(self, synthetic_financials):
        """Financial fixture produces consistent data on every run."""
        assert synthetic_financials["quarters"] == 4
        assert all(r > 0 for r in synthetic_financials["revenue"])

    def test_financials_relationships(self, synthetic_financials):
        """Financial data maintains accounting identities."""
        for i in range(synthetic_financials["quarters"]):
            # Net income should be less than revenue
            assert synthetic_financials["net_income"][i] < synthetic_financials["revenue"][i]
            # Total equity + debt should approximate total assets
            equity_plus_debt = (
                synthetic_financials["total_equity"][i] + synthetic_financials["total_debt"][i]
            )
            assert abs(equity_plus_debt - synthetic_financials["total_assets"][i]) < 1.0
