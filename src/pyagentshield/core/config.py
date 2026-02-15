"""Configuration management for AgentShield."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    provider: Literal["local", "openai", "mlx"] = "local"
    model: str = "all-MiniLM-L6-v2"

    # OpenAI-specific settings
    openai_model: str = "text-embedding-3-small"

    # OpenAI-compatible endpoint overrides
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_headers: Optional[Dict[str, str]] = None
    dimensions: Optional[int] = None

    # MLX-specific settings (Apple Silicon)
    mlx_cache_dir: Optional[str] = None  # Cache for converted MLX models
    mlx_convert_from_hf: bool = True  # Auto-convert from HuggingFace


class FinetunedCleanerConfig(BaseModel):
    """Configuration for finetuned cleaner."""

    # Model source (one of these is required when using finetuned)
    model_id: Optional[str] = None  # HuggingFace Hub ID
    model_path: Optional[str] = None  # Local path

    # LoRA settings
    base_model: str = "microsoft/phi-2"
    use_lora: bool = True

    # Inference settings
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.1
    torch_dtype: Optional[str] = None  # "float16", "bfloat16", "float32"


class HybridCleanerConfig(BaseModel):
    """Configuration for hybrid cleaner."""

    # Methods to combine (in order)
    methods: List[str] = Field(
        default_factory=lambda: ["heuristic", "finetuned"],
        description="Cleaning methods to combine: 'heuristic', 'finetuned', 'llm'"
    )

    # Combination mode
    mode: Literal[
        "sequential",
        "parallel_vote",
        "fallback",
        "best_drift",
        "least_drift"
    ] = "sequential"

    # Voting settings (for parallel_vote mode)
    vote_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Optional weights for each method
    weights: Optional[Dict[str, float]] = None


class CleaningConfig(BaseModel):
    """Configuration for text cleaning."""

    # Primary cleaning method
    method: Literal["heuristic", "llm", "finetuned", "hybrid"] = "heuristic"

    # LLM cleaner settings (OpenAI API)
    llm_model: str = "gpt-3.5-turbo"

    # OpenAI-compatible endpoint overrides (for LLM cleaner)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_headers: Optional[Dict[str, str]] = None

    # Finetuned cleaner settings
    finetuned: FinetunedCleanerConfig = Field(default_factory=FinetunedCleanerConfig)

    # Hybrid cleaner settings
    hybrid: HybridCleanerConfig = Field(default_factory=HybridCleanerConfig)


class ZEDDConfig(BaseModel):
    """Configuration for ZEDD detector."""

    # Threshold for drift detection (None = auto-calibrate)
    threshold: Optional[float] = None

    # DEPRECATED: Use cleaning.method instead
    # Kept for backward compatibility
    cleaning_method: Literal["heuristic", "llm", "finetuned", "hybrid"] = "heuristic"

    # DEPRECATED: Use cleaning.llm_model instead
    llm_model: str = "gpt-3.5-turbo"


class BehaviorConfig(BaseModel):
    """Configuration for detection behavior."""

    # Action to take when injection is detected
    on_detect: Literal["block", "warn", "flag", "filter"] = "flag"

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


class TelemetryConfig(BaseModel):
    """Configuration for telemetry to AgentShield Cloud."""

    enabled: bool = True
    api_key: Optional[str] = None
    endpoint: str = "https://api.agentshield.dev/v1/telemetry"
    project: Optional[str] = None
    environment: Optional[str] = None
    flush_interval: int = 30
    batch_size: int = 50


class ShieldConfig(BaseSettings):
    """
    Main configuration for AgentShield.

    Configuration is loaded from (in order of precedence):
    1. Explicit values passed to constructor
    2. Environment variables (AGENTSHIELD_* prefix)
    3. YAML config file (if specified)
    4. Default values

    Example YAML config:
        embeddings:
          provider: local
          model: all-MiniLM-L6-v2

        cleaning:
          method: hybrid
          hybrid:
            methods:
              - heuristic
              - finetuned
            mode: sequential
          finetuned:
            model_id: agentshield/cleaner-phi2-lora
            load_in_4bit: true

        zedd:
          threshold: null  # auto-calibrate

        behavior:
          on_detect: flag
          confidence_threshold: 0.5

    Environment variables:
        AGENTSHIELD_CLEANING__METHOD=hybrid
        AGENTSHIELD_CLEANING__FINETUNED__MODEL_ID=agentshield/cleaner-phi2-lora
        AGENTSHIELD_ZEDD__THRESHOLD=0.2
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTSHIELD_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Sub-configurations
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    zedd: ZEDDConfig = Field(default_factory=ZEDDConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    def model_post_init(self, __context: Any) -> None:
        """Handle backward compatibility after initialization."""
        # Sync cleaning.method with zedd.cleaning_method for backward compat
        # If zedd.cleaning_method was explicitly set (not default), use it
        if self.zedd.cleaning_method != "heuristic" and self.cleaning.method == "heuristic":
            self.cleaning.method = self.zedd.cleaning_method
        # Always keep them in sync
        self.zedd.cleaning_method = self.cleaning.method
        self.zedd.llm_model = self.cleaning.llm_model

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
            env_path = os.environ.get("AGENTSHIELD_CONFIG_PATH")
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
    """Get the cache directory for AgentShield data."""
    if config and config.performance.cache_dir:
        cache_dir = Path(config.performance.cache_dir)
    else:
        # Default to ~/.agentshield
        cache_dir = Path.home() / ".agentshield"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
