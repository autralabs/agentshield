"""Custom exceptions for RagShield."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ragshield.core.results import ScanResult


class RagShieldError(Exception):
    """Base exception for all RagShield errors."""

    pass


class PromptInjectionDetected(RagShieldError):
    """Raised when a prompt injection is detected and on_detect='block'."""

    def __init__(
        self,
        message: str,
        results: Optional[List[ScanResult]] = None,
    ):
        super().__init__(message)
        self.results = results or []

    def __str__(self) -> str:
        base = super().__str__()
        if self.results:
            details = [
                f"  - confidence={r.confidence:.2f}: {r.details.summary}"
                for r in self.results
            ]
            return f"{base}\n" + "\n".join(details)
        return base


class CalibrationError(RagShieldError):
    """Raised when threshold calibration fails."""

    pass


class ConfigurationError(RagShieldError):
    """Raised when configuration is invalid."""

    pass


class EmbeddingError(RagShieldError):
    """Raised when embedding generation fails."""

    pass


class CleaningError(RagShieldError):
    """Raised when text cleaning fails."""

    pass
