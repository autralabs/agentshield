"""Public API functions for AgentShield."""

from pyagentshield.api.scan import scan
from pyagentshield.api.decorator import shield

__all__ = [
    "scan",
    "shield",
]
