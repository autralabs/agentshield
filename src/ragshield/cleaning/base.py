"""Base text cleaner interface."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class TextCleaner(Protocol):
    """
    Protocol for text cleaning implementations.

    Text cleaners remove potential injection content from text,
    producing a "clean" version for comparison with the original.
    The semantic drift between original and cleaned versions
    indicates the presence of injected content.
    """

    @property
    def method(self) -> str:
        """
        Get the cleaning method name.

        Returns:
            Method identifier (e.g., "heuristic", "llm")
        """
        ...

    def clean(self, text: str) -> str:
        """
        Clean text by removing potential injection content.

        Args:
            text: Original text that may contain injections

        Returns:
            Cleaned text with injection attempts removed
        """
        ...

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts.

        Default implementation calls clean() for each text.
        Subclasses may override for batch efficiency.

        Args:
            texts: List of texts to clean

        Returns:
            List of cleaned texts
        """
        ...
