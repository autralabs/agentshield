"""Text cleaning utilities for AgentShield."""

from pyagentshield.cleaning.base import TextCleaner
from pyagentshield.cleaning.heuristic import HeuristicCleaner
from pyagentshield.cleaning.hybrid import HybridCleaner, HybridMode, create_hybrid_cleaner

__all__ = [
    "TextCleaner",
    "HeuristicCleaner",
    "HybridCleaner",
    "HybridMode",
    "create_hybrid_cleaner",
]

# Conditional import for LLM cleaner (requires openai)
try:
    from pyagentshield.cleaning.llm import LLMCleaner
    __all__.append("LLMCleaner")
except ImportError:
    pass

# Conditional import for finetuned cleaner (requires transformers)
try:
    from pyagentshield.cleaning.finetuned import FinetunedCleaner
    __all__.append("FinetunedCleaner")
except ImportError:
    pass
