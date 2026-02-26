"""Custom exceptions for AgentShield."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pyagentshield.core.results import ScanResult


class AgentShieldError(Exception):
    """Base exception for all AgentShield errors."""

    pass


class PromptInjectionDetected(AgentShieldError):
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


class CalibrationError(AgentShieldError):
    """Raised when threshold calibration fails."""

    pass


class ConfigurationError(AgentShieldError):
    """Raised when configuration is invalid."""

    pass


class EmbeddingError(AgentShieldError):
    """Raised when embedding generation fails."""

    pass


class CleaningError(AgentShieldError):
    """Raised when text cleaning fails."""

    pass


class SetupError(AgentShieldError):
    """Raised when model setup or readiness check fails."""

    pass


class ThresholdUnavailableError(AgentShieldError):
    """Raised when cloud_only mode finds no rule and fail_open=False."""

    pass
