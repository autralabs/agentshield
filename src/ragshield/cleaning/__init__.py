"""Text cleaning utilities for RagShield."""

from ragshield.cleaning.base import TextCleaner
from ragshield.cleaning.heuristic import HeuristicCleaner
from ragshield.cleaning.hybrid import HybridCleaner, HybridMode, create_hybrid_cleaner

__all__ = [
    "TextCleaner",
    "HeuristicCleaner",
    "HybridCleaner",
    "HybridMode",
    "create_hybrid_cleaner",
]

# Conditional import for LLM cleaner (requires openai)
try:
    from ragshield.cleaning.llm import LLMCleaner
    __all__.append("LLMCleaner")
except ImportError:
    pass

# Conditional import for finetuned cleaner (requires transformers)
try:
    from ragshield.cleaning.finetuned import FinetunedCleaner
    __all__.append("FinetunedCleaner")
except ImportError:
    pass
