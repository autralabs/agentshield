"""Framework integrations for RagShield."""

from typing import List

__all__: List[str] = []

# Conditional imports
try:
    from ragshield.integrations.langchain import ShieldRunnable
    __all__.append("ShieldRunnable")
except ImportError:
    pass
