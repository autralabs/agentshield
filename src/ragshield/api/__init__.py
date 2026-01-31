"""Public API functions for RagShield."""

from ragshield.api.scan import scan
from ragshield.api.decorator import shield

__all__ = [
    "scan",
    "shield",
]
