"""Shared test fixtures for AgentShield."""

from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from pyagentshield.core.config import ShieldConfig


class MockEmbeddingProvider:
    """Mock embedding provider that returns deterministic embeddings.

    Uses a simple hash-based approach: different texts get different embeddings,
    identical texts get identical embeddings. This lets us test drift detection
    without downloading real models.
    """

    def __init__(self, model_name: str = "mock-model", dimensions: int = 384):
        self._model_name = model_name
        self._dimensions = dimensions
        self._provider_type = "local"
        self._overrides: Dict[str, NDArray[np.floating]] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def set_embedding(self, text: str, embedding: NDArray[np.floating]) -> None:
        """Set a specific embedding for a text (for controlled test scenarios)."""
        self._overrides[text] = embedding

    def encode(self, text: str) -> NDArray[np.floating]:
        if text in self._overrides:
            return self._overrides[text]
        # Generate deterministic embedding from text hash
        rng = np.random.RandomState(hash(text) % (2**31))
        emb = rng.randn(self._dimensions).astype(np.float32)
        # Normalize to unit vector
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def encode_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> NDArray[np.floating]:
        return np.vstack([self.encode(t) for t in texts])


class MockTextCleaner:
    """Mock text cleaner with controllable behavior."""

    def __init__(self, method: str = "mock"):
        self._method = method
        self._overrides: Dict[str, str] = {}

    @property
    def method(self) -> str:
        return self._method

    def set_cleaned(self, original: str, cleaned: str) -> None:
        """Set a specific cleaned output for a given input."""
        self._overrides[original] = cleaned

    def clean(self, text: str) -> str:
        if text in self._overrides:
            return self._overrides[text]
        # Default: return unchanged (simulates clean text with no drift)
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]


@pytest.fixture
def mock_embedder():
    """Provide a mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def mock_cleaner():
    """Provide a mock text cleaner."""
    return MockTextCleaner()


@pytest.fixture
def default_config():
    """Provide a default ShieldConfig."""
    return ShieldConfig()


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache = tmp_path / "pyagentshield_cache"
    cache.mkdir()
    return cache
