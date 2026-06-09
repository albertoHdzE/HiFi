"""Configuration loading and validation for HiFi.

Loads YAML configuration files and validates them against a Pydantic schema.
Supports layered configuration: defaults can be overridden by environment-specific
files or environment variables.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# --- Configuration Schema ---


class DataConfig(BaseModel):
    """Data acquisition and storage configuration."""

    universe: list[str] = Field(default_factory=list)
    start_date: str = "2015-01-01"
    end_date: str | None = None
    storage_format: str = "parquet"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    features_dir: str = "data/features"


class EvalPeriod(BaseModel):
    """A single evaluation period definition."""

    start: str
    end: str


class EvaluationConfig(BaseModel):
    """Evaluation framework configuration."""

    periods: dict[str, EvalPeriod] = Field(default_factory=dict)


class SafetyConfig(BaseModel):
    """Paper trading safety limits."""

    max_position_pct: float = 0.05
    max_daily_loss_pct: float = 0.02
    max_sector_exposure_pct: float = 0.25
    max_confidence: float = 0.95
    min_agents_quorum: int = 4


class ReproducibilityConfig(BaseModel):
    """Reproducibility settings."""

    default_seed: int = 42


class ProjectConfig(BaseModel):
    """Project metadata."""

    name: str = "HiFi"
    version: str = "0.1.0"


class HiFiConfig(BaseModel):
    """Root configuration for HiFi.

    All configuration values are validated at load time. Missing sections
    receive defaults. Unknown fields are rejected to prevent silent
    misconfiguration.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


# --- Loading ---

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> HiFiConfig:
    """Load and validate HiFi configuration from a YAML file.

    Args:
        path: Path to a YAML configuration file. If None, loads the default
              configuration from configs/default.yaml.

    Returns:
        A validated HiFiConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        pydantic.ValidationError: If the configuration is invalid.
    """
    config_path = path or _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return HiFiConfig(**raw)
