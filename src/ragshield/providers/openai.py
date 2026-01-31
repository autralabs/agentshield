"""OpenAI embedding provider."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ragshield.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider:
    """
    Embedding provider using OpenAI's embedding API.

    Requires the 'openai' extra: pip install ragshield[openai]

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
    ):
        """
        Initialize the OpenAI embedding provider.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            cache_embeddings: Whether to cache embeddings in memory
        """
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._cache_embeddings = cache_embeddings

        if not self._api_key:
            raise EmbeddingError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Get dimensions
        self._dimensions = self.DIMENSIONS.get(model_name)
        if self._dimensions is None:
            logger.warning(
                f"Unknown model {model_name}, assuming 1536 dimensions. "
                f"Known models: {list(self.DIMENSIONS.keys())}"
            )
            self._dimensions = 1536

        # Lazy load client
        self._client: Any = None

        # In-memory cache
        self._cache: Dict[str, NDArray[np.floating]] = {}

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is required for OpenAI embeddings. "
                    "Install with: pip install ragshield[openai]"
                ) from e

            self._client = OpenAI(api_key=self._api_key)

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

        # Check which texts are already cached
        results: List[Optional[NDArray[np.floating]]] = [None] * len(texts)
        texts_to_encode: List[Tuple[int, str]] = []

        if self._cache_embeddings:
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    results[i] = np.zeros(self.dimensions, dtype=np.float32)
                    continue

                key = self._cache_key(text)
                if key in self._cache:
                    results[i] = self._cache[key]
                else:
                    texts_to_encode.append((i, text))
        else:
            texts_to_encode = [
                (i, t) for i, t in enumerate(texts)
                if t and t.strip()
            ]
            # Set zeros for empty texts
            for i, t in enumerate(texts):
                if not t or not t.strip():
                    results[i] = np.zeros(self.dimensions, dtype=np.float32)

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

                # Process results
                for idx, data in zip(indices, response.data):
                    emb = np.array(data.embedding, dtype=np.float32)
                    results[idx] = emb

                    if self._cache_embeddings:
                        text = texts[idx]
                        self._cache[self._cache_key(text)] = emb

        # Stack results
        return np.vstack(results)  # type: ignore

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
