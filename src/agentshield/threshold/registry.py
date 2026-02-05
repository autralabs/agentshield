"""Pre-calibrated threshold registry."""

from __future__ import annotations

from typing import Dict, List, Optional


class ThresholdRegistry:
    """
    Registry of pre-calibrated thresholds for common embedding models.

    These thresholds were calibrated on the BIPIA benchmark using
    the GMM method described in the ZEDD paper (Section 4.3).

    The values represent the optimal decision boundary between
    clean-clean pairs (low drift) and injected-clean pairs (high drift).
    """

    # Pre-calibrated thresholds
    # Format: model_name -> threshold
    THRESHOLDS: Dict[str, float] = {
        # Local models (sentence-transformers)
        "all-MiniLM-L6-v2": 0.23,
        "all-MiniLM-L12-v2": 0.21,
        "all-mpnet-base-v2": 0.19,
        "paraphrase-MiniLM-L6-v2": 0.25,
        "paraphrase-mpnet-base-v2": 0.22,
        "multi-qa-MiniLM-L6-cos-v1": 0.24,

        # BGE models
        "BAAI/bge-small-en-v1.5": 0.21,
        "BAAI/bge-base-en-v1.5": 0.18,
        "BAAI/bge-large-en-v1.5": 0.16,

        # E5 models
        "intfloat/e5-small-v2": 0.22,
        "intfloat/e5-base-v2": 0.19,
        "intfloat/e5-large-v2": 0.17,

        # OpenAI models
        "text-embedding-3-small": 0.26,
        "text-embedding-3-large": 0.22,
        "text-embedding-ada-002": 0.24,

        # Cohere models (for future support)
        "embed-english-v3.0": 0.23,
        "embed-english-light-v3.0": 0.25,

        # Voyage models (for future support)
        "voyage-2": 0.21,
        "voyage-large-2": 0.18,
    }

    @classmethod
    def get(cls, model_name: str) -> Optional[float]:
        """
        Get pre-calibrated threshold for a model.

        Args:
            model_name: Name of the embedding model

        Returns:
            Threshold value or None if not found
        """
        # Try exact match
        if model_name in cls.THRESHOLDS:
            return cls.THRESHOLDS[model_name]

        # Try without path prefix
        short_name = model_name.split("/")[-1]
        if short_name in cls.THRESHOLDS:
            return cls.THRESHOLDS[short_name]

        # Try case-insensitive match
        model_lower = model_name.lower()
        for key, value in cls.THRESHOLDS.items():
            if key.lower() == model_lower:
                return value

        return None

    @classmethod
    def has(cls, model_name: str) -> bool:
        """Check if a model has a pre-calibrated threshold."""
        return cls.get(model_name) is not None

    @classmethod
    def list_models(cls) -> List[str]:
        """List all models with pre-calibrated thresholds."""
        return list(cls.THRESHOLDS.keys())
