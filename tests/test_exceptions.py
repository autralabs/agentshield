"""Tests for custom exceptions."""

from agentshield.core.exceptions import (
    AgentShieldError,
    CalibrationError,
    CleaningError,
    ConfigurationError,
    EmbeddingError,
    PromptInjectionDetected,
)
from agentshield.core.results import ScanDetails, ScanResult


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(PromptInjectionDetected, AgentShieldError)
        assert issubclass(CalibrationError, AgentShieldError)
        assert issubclass(ConfigurationError, AgentShieldError)
        assert issubclass(EmbeddingError, AgentShieldError)
        assert issubclass(CleaningError, AgentShieldError)

    def test_base_inherits_from_exception(self):
        assert issubclass(AgentShieldError, Exception)


class TestPromptInjectionDetected:
    def test_basic_message(self):
        exc = PromptInjectionDetected("blocked")
        assert str(exc) == "blocked"
        assert exc.results == []

    def test_with_results(self):
        results = [
            ScanResult.suspicious(
                confidence=0.9,
                summary="High drift",
            )
        ]
        exc = PromptInjectionDetected("blocked", results=results)
        assert len(exc.results) == 1
        assert "High drift" in str(exc)

    def test_catchable_as_base(self):
        try:
            raise PromptInjectionDetected("test")
        except AgentShieldError:
            pass  # Should be caught
