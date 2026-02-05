"""Base embedding provider protocol."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    Embedding providers convert text into vector representations.
    AgentShield uses these embeddings to compute semantic drift.

    Implementations should:
    - Be thread-safe for concurrent use
    - Cache embeddings when possible for efficiency
    - Handle empty/whitespace-only strings gracefully
    """

    @property
    def model_name(self) -> str:
        """
        Get the model name/identifier.

        Used for threshold lookup and caching.
        """
        ...

    @property
    def dimensions(self) -> int:
        """
        Get the embedding dimensions.

        Returns:
            Number of dimensions in the embedding vectors
        """
        ...

    def encode(self, text: str) -> NDArray[np.floating]:
        """
        Encode a single text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as numpy array of shape (dimensions,)
        """
        ...

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> NDArray[np.floating]:
        """
        Encode multiple texts to embedding vectors.

        More efficient than calling encode() multiple times.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing

        Returns:
            Embedding matrix as numpy array of shape (len(texts), dimensions)
        """
        ...
