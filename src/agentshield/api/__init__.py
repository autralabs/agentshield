"""Public API functions for AgentShield."""

from agentshield.api.scan import scan
from agentshield.api.decorator import shield

__all__ = [
    "scan",
    "shield",
]
