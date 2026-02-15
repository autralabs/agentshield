"""OpenAI embedding provider."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

from pyagentshield.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider:
    """
    Embedding provider using OpenAI's embedding API.

    Requires the 'openai' extra: pip install pyagentshield[openai]

    Supported models:
    - text-embedding-3-small: Fast, cost-effective (default)
    - text-embedding-3-large: Higher quality
    - text-embedding-ada-002: Legacy model

    Note: Requires OPENAI_API_KEY environment variable or explicit api_key.
    """

    # Model dimensions
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        cache_embeddings: bool = True,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        dimensions: Optional[int] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the OpenAI embedding provider.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            cache_embeddings: Whether to cache embeddings in memory
            base_url: Custom API base URL for OpenAI-compatible endpoints
                (e.g. OpenRouter, Together, Ollama, vLLM)
            default_headers: Extra HTTP headers sent with every request
            dimensions: Explicit embedding dimensions. If not provided, uses
                static lookup, disk cache, or discovers from first API response.
            cache_dir: Directory for dimension cache persistence.
                Defaults to ``~/.agentshield``.
        """
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._cache_embeddings = cache_embeddings
        self._base_url = base_url
        self._default_headers = default_headers
        self._provider_type = "openai"
        self._cache_dir = cache_dir or Path.home() / ".agentshield"

        if not self._api_key:
            raise EmbeddingError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Dimension resolution: explicit > static dict > disk cache > provisional
        self._dim_lock = threading.Lock()
        if dimensions is not None:
            self._dimensions = dimensions
            self._dimensions_confirmed = True
        else:
            static = self.DIMENSIONS.get(model_name)
            if static is not None:
                self._dimensions = static
                self._dimensions_confirmed = True
            else:
                # Try disk cache
                cached_dims = self._load_cached_dimensions()
                if cached_dims is not None:
                    self._dimensions = cached_dims
                    self._dimensions_confirmed = True
                    logger.debug(
                        f"Loaded cached dimensions {cached_dims} for {model_name}"
                    )
                else:
                    # Will be discovered on first encode() call
                    self._dimensions = 1536  # provisional fallback
                    self._dimensions_confirmed = False
                    logger.info(
                        f"Unknown model {model_name}, dimensions will be "
                        f"discovered on first embedding call."
                    )

        # Lazy load client
        self._client: Any = None

        # In-memory cache
        self._cache: Dict[str, NDArray[np.floating]] = {}

    # ------------------------------------------------------------------
    # Dimension disk cache
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_host(base_url: Optional[str]) -> str:
        """Extract host identifier for dimension cache keying."""
        if not base_url:
            return "openai.com"
        parsed = urlparse(base_url)
        host = parsed.hostname
        if not host:
            return "openai.com"
        port = parsed.port
        if port is None:
            return host
        # Strip default ports only
        if (parsed.scheme == "https" and port == 443) or (
            parsed.scheme == "http" and port == 80
        ):
            return host
        return f"{host}:{port}"

    def _dim_cache_key(self) -> str:
        """Cache key for dimension persistence."""
        host = self._extract_host(self._base_url)
        return f"openai::{self._model_name}::{host}"

    def _load_cached_dimensions(self) -> Optional[int]:
        """Load dimensions from disk cache."""
        cache_file = self._cache_dir / "dimensions_cache.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file) as f:
                data = json.load(f)
            return data.get(self._dim_cache_key())
        except Exception:
            return None

    def _save_cached_dimensions(self, dims: int) -> None:
        """Save dimensions to disk cache."""
        cache_file = self._cache_dir / "dimensions_cache.json"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            data: Dict[str, int] = {}
            if cache_file.exists():
                with open(cache_file) as f:
                    data = json.load(f)
            data[self._dim_cache_key()] = dims
            tmp = cache_file.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.replace(cache_file)
        except Exception:
            logger.debug("Failed to save dimensions cache", exc_info=True)

    # ------------------------------------------------------------------
    # Dimension discovery
    # ------------------------------------------------------------------

    def _confirm_dimensions(self, embedding_len: int) -> None:
        """Confirm dimensions from an actual API response (thread-safe)."""
        if self._dimensions_confirmed:
            return
        with self._dim_lock:
            if self._dimensions_confirmed:
                return
            self._dimensions = embedding_len
            self._dimensions_confirmed = True
            logger.info(
                f"Discovered {self._dimensions} dimensions for {self._model_name}"
            )
            self._save_cached_dimensions(embedding_len)

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenAI embeddings. "
                    "Install with: pip install pyagentshield[openai]"
                ) from e

            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            self._client = OpenAI(**kwargs)

        return self._client

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def encode(self, text: str) -> NDArray[np.floating]:
        """
        Encode a single text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if self._cache_embeddings:
            key = self._cache_key(text)
            if key in self._cache:
                return self._cache[key]

        # Handle empty text
        if not text or not text.strip():
            return np.zeros(self.dimensions, dtype=np.float32)

        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self._model_name,
                input=text,
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e

        # Discover dimensions from first real response
        self._confirm_dimensions(len(embedding))

        # Cache result
        if self._cache_embeddings:
            self._cache[key] = embedding

        return embedding

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> NDArray[np.floating]:
        """
        Encode multiple texts efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for API calls

        Returns:
            Embedding matrix of shape (len(texts), dimensions)
        """
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        # Separate empty and non-empty texts
        empty_indices: List[int] = []
        texts_to_encode: List[Tuple[int, str]] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                empty_indices.append(i)
            elif self._cache_embeddings:
                cache_key = self._cache_key(text)
                if cache_key in self._cache:
                    # Will be filled from cache below
                    pass
                else:
                    texts_to_encode.append((i, text))
            else:
                texts_to_encode.append((i, text))

        # Build results dict for non-empty cached texts
        results: Dict[int, NDArray[np.floating]] = {}
        if self._cache_embeddings:
            for i, text in enumerate(texts):
                if text and text.strip():
                    cache_key = self._cache_key(text)
                    if cache_key in self._cache:
                        results[i] = self._cache[cache_key]

        # Encode uncached texts in batches
        if texts_to_encode:
            client = self._get_client()

            for batch_start in range(0, len(texts_to_encode), batch_size):
                batch = texts_to_encode[batch_start:batch_start + batch_size]
                indices, batch_texts = zip(*batch)

                try:
                    response = client.embeddings.create(
                        model=self._model_name,
                        input=list(batch_texts),
                    )
                except Exception as e:
                    raise EmbeddingError(f"OpenAI embedding batch failed: {e}") from e

                for idx, data in zip(indices, response.data):
                    emb = np.array(data.embedding, dtype=np.float32)
                    results[idx] = emb

                    # Discover dimensions from first real embedding
                    self._confirm_dimensions(len(emb))

                    if self._cache_embeddings:
                        self._cache[self._cache_key(texts[idx])] = emb

        # Now fill empty texts with zeros using confirmed dimensions
        for i in empty_indices:
            results[i] = np.zeros(self.dimensions, dtype=np.float32)

        # Stack in order
        ordered = [results[i] for i in range(len(texts))]
        return np.vstack(ordered)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
