"""Core components for AgentShield."""

from agentshield.core.config import ShieldConfig
from agentshield.core.results import ScanResult, DetectionSignal, ScanDetails
from agentshield.core.exceptions import (
    AgentShieldError,
    PromptInjectionDetected,
    CalibrationError,
    ConfigurationError,
)
from agentshield.core.shield import AgentShield

__all__ = [
    "ShieldConfig",
    "ScanResult",
    "DetectionSignal",
    "ScanDetails",
    "AgentShield",
    "AgentShieldError",
    "PromptInjectionDetected",
    "CalibrationError",
    "ConfigurationError",
]
