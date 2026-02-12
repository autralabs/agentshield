"""Embedding providers for AgentShield."""

from pyagentshield.providers.base import EmbeddingProvider
from pyagentshield.providers.local import LocalEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
]

# Conditional import for OpenAI
try:
    from pyagentshield.providers.openai import OpenAIEmbeddingProvider
    __all__.append("OpenAIEmbeddingProvider")
except ImportError:
    pass

# Conditional import for MLX (Apple Silicon)
try:
    from pyagentshield.providers.mlx import MLXEmbeddingProvider
    __all__.append("MLXEmbeddingProvider")
except ImportError:
    pass
