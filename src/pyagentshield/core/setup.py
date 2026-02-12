"""Model setup and readiness checks for AgentShield."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pyagentshield.core.config import ShieldConfig
from pyagentshield.core.exceptions import SetupError

logger = logging.getLogger(__name__)


@dataclass
class SetupResult:
    """Result of a setup operation."""

    success: bool
    model_name: str
    model_path: Optional[str] = None
    dimensions: Optional[int] = None
    download_time_ms: Optional[float] = None
    validation_time_ms: Optional[float] = None
    message: str = ""
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dimensions": self.dimensions,
            "download_time_ms": self.download_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "message": self.message,
            "skipped": self.skipped,
        }


def is_model_cached(model_name: Optional[str] = None, config: Optional[Union[ShieldConfig, Dict[str, Any], str, Path]] = None) -> bool:
    """
    Check if an embedding model is already downloaded/cached.

    This is a lightweight check that does NOT load the model.

    Args:
        model_name: Model name to check. If None, uses config or default.
        config: Optional configuration to determine model name.

    Returns:
        True if the model is cached locally and ready to use.
    """
    if model_name is None:
        loaded_config = ShieldConfig.load(config)
        model_name = loaded_config.embeddings.model

    # Local path — check if directory exists with model files
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        # Check for typical sentence-transformers model files
        has_config = (model_path / "config.json").exists()
        return has_config

    # HuggingFace Hub model — use try_to_load_from_cache
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_name, "config.json")
        # Returns None if not cached, a path string if cached,
        # or _CACHED_NO_EXIST sentinel if explicitly marked missing
        return isinstance(result, str)
    except Exception:
        # huggingface_hub not installed or other error
        return False


def setup(
    config: Optional[Union[ShieldConfig, Dict[str, Any], str, Path]] = None,
    model_name: Optional[str] = None,
) -> SetupResult:
    """
    Download and validate the embedding model.

    Call this during application startup (e.g., Dockerfile build step,
    FastAPI lifespan, Django AppConfig.ready) to ensure the model is
    ready before serving requests.

    Args:
        config: Optional configuration (same types as AgentShield.__init__).
        model_name: Override the model name from config.

    Returns:
        SetupResult with details about the setup.

    Raises:
        SetupError: If setup fails.
    """
    loaded_config = ShieldConfig.load(config)

    if model_name:
        loaded_config.embeddings.model = model_name

    effective_model = loaded_config.embeddings.model
    provider = loaded_config.embeddings.provider

    # Skip for non-local providers
    if provider not in ("local",):
        return SetupResult(
            success=True,
            model_name=effective_model,
            message=f"Setup not required for provider '{provider}' — models are accessed via API.",
            skipped=True,
        )

    logger.info(f"Setting up model: {effective_model}")

    # Step 1: Download/load the model
    download_start = time.monotonic()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(effective_model)
    except ImportError as e:
        raise SetupError(
            "sentence-transformers is required for local embeddings. "
            "Install with: pip install sentence-transformers"
        ) from e
    except Exception as e:
        raise SetupError(f"Failed to download/load model '{effective_model}': {e}") from e

    download_ms = (time.monotonic() - download_start) * 1000

    # Step 2: Validate with a test encode
    validation_start = time.monotonic()
    try:
        test_embedding = model.encode("AgentShield setup validation", convert_to_numpy=True, show_progress_bar=False)
        dimensions = len(test_embedding)
    except Exception as e:
        raise SetupError(f"Model loaded but validation failed: {e}") from e

    validation_ms = (time.monotonic() - validation_start) * 1000

    # Determine model path
    model_path = None
    local_path = Path(effective_model)
    if local_path.exists():
        model_path = str(local_path.resolve())
    else:
        # Try to find the cached path
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(effective_model, "config.json")
            if isinstance(cached, str):
                model_path = str(Path(cached).parent)
        except Exception:
            pass

    return SetupResult(
        success=True,
        model_name=effective_model,
        model_path=model_path,
        dimensions=dimensions,
        download_time_ms=round(download_ms, 1),
        validation_time_ms=round(validation_ms, 1),
        message=f"Model '{effective_model}' ready ({dimensions}d embeddings).",
    )
