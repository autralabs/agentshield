"""Local embedding provider using sentence-transformers."""

from __future__ import annotations

import hashlib
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider:
    """
    Embedding provider using sentence-transformers.

    Uses local models for embedding generation - no API calls required.
    Supports caching to avoid redundant computation.

    Recommended models:
    - all-MiniLM-L6-v2: Fast, good quality (default)
    - all-mpnet-base-v2: Slower, better quality
    - paraphrase-MiniLM-L6-v2: Good for paraphrase detection
    - BAAI/bge-small-en-v1.5: State-of-the-art small model
    - BAAI/bge-base-en-v1.5: State-of-the-art base model
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the local embedding provider.

        Args:
            model_name: Name of the sentence-transformers model
            cache_embeddings: Whether to cache embeddings in memory
            device: Device to use ("cpu", "cuda", etc.). None for auto-detect.
        """
        self._model_name = model_name
        self._cache_embeddings = cache_embeddings
        self._device = device

        # Lazy load model
        self._model: Any = None
        self._dimensions: Optional[int] = None

        # In-memory cache
        self._cache: Dict[str, NDArray[np.floating]] = {}

    def _load_model(self) -> None:
        """Load the sentence-transformers model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

        # Warn if model is not cached (will trigger a download)
        from pyagentshield.core.setup import is_model_cached
        if not is_model_cached(self._model_name):
            warnings.warn(
                f"Embedding model '{self._model_name}' is not cached and will be "
                f"downloaded (~80MB). This may cause slow startup in production. "
                f"Run 'agentshield setup' or call pyagentshield.setup() during deployment "
                f"to pre-download the model.",
                stacklevel=2,
            )

        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimensions: {self._dimensions}")

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        if self._dimensions is None:
            self._load_model()
        return self._dimensions  # type: ignore

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

        # Ensure model is loaded
        self._load_model()

        # Handle empty/whitespace text
        if not text or not text.strip():
            embedding = np.zeros(self.dimensions, dtype=np.float32)
        else:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        # Ensure correct dtype
        embedding = np.asarray(embedding, dtype=np.float32)

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
            batch_size: Batch size for processing

        Returns:
            Embedding matrix of shape (len(texts), dimensions)
        """
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        # Ensure model is loaded
        self._load_model()

        # Check which texts are already cached
        results: List[Optional[NDArray[np.floating]]] = [None] * len(texts)
        texts_to_encode: List[Tuple[int, str]] = []

        if self._cache_embeddings:
            for i, text in enumerate(texts):
                key = self._cache_key(text)
                if key in self._cache:
                    results[i] = self._cache[key]
                else:
                    texts_to_encode.append((i, text))
        else:
            texts_to_encode = list(enumerate(texts))

        # Encode uncached texts
        if texts_to_encode:
            indices, uncached_texts = zip(*texts_to_encode)

            # Handle empty strings
            processed_texts = [t if t and t.strip() else " " for t in uncached_texts]

            embeddings = self._model.encode(
                list(processed_texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            embeddings = np.asarray(embeddings, dtype=np.float32)

            # Assign results and update cache
            for idx, (orig_idx, text) in enumerate(texts_to_encode):
                emb = embeddings[idx]

                # Zero out embeddings for truly empty texts
                if not text or not text.strip():
                    emb = np.zeros(self.dimensions, dtype=np.float32)

                results[orig_idx] = emb

                if self._cache_embeddings:
                    self._cache[self._cache_key(text)] = emb

        # Stack results
        return np.vstack(results)  # type: ignore

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
