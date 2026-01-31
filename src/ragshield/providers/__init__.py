"""Embedding providers for RagShield."""

from ragshield.providers.base import EmbeddingProvider
from ragshield.providers.local import LocalEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
]

# Conditional import for OpenAI
try:
    from ragshield.providers.openai import OpenAIEmbeddingProvider
    __all__.append("OpenAIEmbeddingProvider")
except ImportError:
    pass
