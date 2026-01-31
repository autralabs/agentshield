"""Configuration management for RagShield."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    provider: Literal["local", "openai"] = "local"
    model: str = "all-MiniLM-L6-v2"

    # OpenAI-specific settings
    openai_model: str = "text-embedding-3-small"


class ZEDDConfig(BaseModel):
    """Configuration for ZEDD detector."""

    # Threshold for drift detection (None = auto-calibrate)
    threshold: Optional[float] = None

    # Text cleaning settings
    cleaning_method: Literal["heuristic", "llm"] = "heuristic"

    # LLM cleaning settings (only used if cleaning_method="llm")
    llm_model: str = "gpt-3.5-turbo"


class BehaviorConfig(BaseModel):
    """Configuration for detection behavior."""

    # Action to take when injection is detected
    on_detect: Literal["block", "warn", "flag"] = "flag"

    # Minimum confidence to trigger action
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PerformanceConfig(BaseModel):
    """Configuration for performance tuning."""

    batch_size: int = Field(default=32, ge=1)
    cache_embeddings: bool = True
    cache_dir: Optional[str] = None


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "text"


class ShieldConfig(BaseSettings):
    """
    Main configuration for RagShield.

    Configuration is loaded from (in order of precedence):
    1. Explicit values passed to constructor
    2. Environment variables (RAGSHIELD_* prefix)
    3. YAML config file (if specified)
    4. Default values

    Example YAML config:
        embeddings:
          provider: local
          model: all-MiniLM-L6-v2

        zedd:
          threshold: null  # auto-calibrate
          cleaning_method: heuristic

        behavior:
          on_detect: flag
          confidence_threshold: 0.5
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGSHIELD_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Sub-configurations
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    zedd: ZEDDConfig = Field(default_factory=ZEDDConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ShieldConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShieldConfig:
        """Create configuration from a dictionary."""
        return cls(**data)

    @classmethod
    def load(
        cls,
        config: Optional[Union[ShieldConfig, Dict[str, Any], str, Path]] = None,
    ) -> ShieldConfig:
        """
        Load configuration from various sources.

        Args:
            config: Can be:
                - ShieldConfig instance (returned as-is)
                - dict (parsed as config)
                - str/Path to YAML file
                - None (uses defaults + env vars)

        Returns:
            ShieldConfig instance
        """
        if config is None:
            # Check for config file in environment
            env_path = os.environ.get("RAGSHIELD_CONFIG_PATH")
            if env_path and Path(env_path).exists():
                return cls.from_yaml(env_path)
            return cls()

        if isinstance(config, ShieldConfig):
            return config

        if isinstance(config, dict):
            return cls.from_dict(config)

        if isinstance(config, (str, Path)):
            return cls.from_yaml(config)

        raise TypeError(f"Invalid config type: {type(config)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return self.model_dump()

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Convenience function for getting cache directory
def get_cache_dir(config: Optional[ShieldConfig] = None) -> Path:
    """Get the cache directory for RagShield data."""
    if config and config.performance.cache_dir:
        cache_dir = Path(config.performance.cache_dir)
    else:
        # Default to ~/.ragshield
        cache_dir = Path.home() / ".ragshield"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
