"""
AgentShield - Prompt injection detection for Agents.

Uses ZEDD (Zero-Shot Embedding Drift Detection) to identify malicious
content in retrieved documents before they reach the LLM context window.

Basic usage:
    >>> from agentshield import scan
    >>> result = scan("some document text")
    >>> if result.is_suspicious:
    ...     print(f"Detected: {result.details.summary}")

Decorator usage:
    >>> from agentshield import shield
    >>> @shield(on_detect="warn")
    ... def process_docs(query: str, docs: list[str]) -> str:
    ...     return llm.invoke(build_prompt(query, docs))

LangChain integration:
    >>> from agentshield.integrations.langchain import ShieldRunnable
    >>> chain = retriever | ShieldRunnable() | prompt | llm
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from agentshield.core.config import ShieldConfig
from agentshield.core.results import ScanResult, DetectionSignal, ScanDetails
from agentshield.core.exceptions import (
    AgentShieldError,
    PromptInjectionDetected,
    CalibrationError,
    ConfigurationError,
)
from agentshield.core.shield import AgentShield
from agentshield.api.scan import scan
from agentshield.api.decorator import shield

__version__ = "0.1.0"

__all__ = [
    # Main class
    "AgentShield",
    # API functions
    "scan",
    "shield",
    # Config and results
    "ShieldConfig",
    "ScanResult",
    "DetectionSignal",
    "ScanDetails",
    # Exceptions
    "AgentShieldError",
    "PromptInjectionDetected",
    "CalibrationError",
    "ConfigurationError",
]
