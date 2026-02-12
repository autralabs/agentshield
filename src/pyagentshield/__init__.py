"""
AgentShield - Prompt injection detection for Agents.

Uses ZEDD (Zero-Shot Embedding Drift Detection) to identify malicious
content in retrieved documents before they reach the LLM context window.

Basic usage:
    >>> from pyagentshield import scan
    >>> result = scan("some document text")
    >>> if result.is_suspicious:
    ...     print(f"Detected: {result.details.summary}")

Decorator usage:
    >>> from pyagentshield import shield
    >>> @shield(on_detect="warn")
    ... def process_docs(query: str, docs: list[str]) -> str:
    ...     return llm.invoke(build_prompt(query, docs))

LangChain integration:
    >>> from pyagentshield.integrations.langchain import ShieldRunnable
    >>> chain = retriever | ShieldRunnable() | prompt | llm
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from pyagentshield.core.config import ShieldConfig
from pyagentshield.core.results import ScanResult, DetectionSignal, ScanDetails
from pyagentshield.core.exceptions import (
    AgentShieldError,
    PromptInjectionDetected,
    CalibrationError,
    ConfigurationError,
    SetupError,
)
from pyagentshield.core.shield import AgentShield
from pyagentshield.core.setup import setup, is_model_cached, SetupResult
from pyagentshield.api.scan import scan
from pyagentshield.api.decorator import shield

__version__ = "0.1.1"

__all__ = [
    # Main class
    "AgentShield",
    # API functions
    "scan",
    "shield",
    # Setup
    "setup",
    "is_model_cached",
    "SetupResult",
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
    "SetupError",
]
