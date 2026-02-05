"""Embedding providers for AgentShield."""

from agentshield.providers.base import EmbeddingProvider
from agentshield.providers.local import LocalEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
]

# Conditional import for OpenAI
try:
    from agentshield.providers.openai import OpenAIEmbeddingProvider
    __all__.append("OpenAIEmbeddingProvider")
except ImportError:
    pass

# Conditional import for MLX (Apple Silicon)
try:
    from agentshield.providers.mlx import MLXEmbeddingProvider
    __all__.append("MLXEmbeddingProvider")
except ImportError:
    pass
