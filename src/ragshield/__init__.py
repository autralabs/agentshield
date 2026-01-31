"""
RagShield - Prompt injection detection for RAG pipelines.

Uses ZEDD (Zero-Shot Embedding Drift Detection) to identify malicious
content in retrieved documents before they reach the LLM context window.

Basic usage:
    >>> from ragshield import scan
    >>> result = scan("some document text")
    >>> if result.is_suspicious:
    ...     print(f"Detected: {result.details.summary}")

Decorator usage:
    >>> from ragshield import shield
    >>> @shield(on_detect="warn")
    ... def process_docs(query: str, docs: list[str]) -> str:
    ...     return llm.invoke(build_prompt(query, docs))

LangChain integration:
    >>> from ragshield.integrations.langchain import ShieldRunnable
    >>> chain = retriever | ShieldRunnable() | prompt | llm
"""

from ragshield.core.config import ShieldConfig
from ragshield.core.results import ScanResult, DetectionSignal, ScanDetails
from ragshield.core.exceptions import (
    RagShieldError,
    PromptInjectionDetected,
    CalibrationError,
    ConfigurationError,
)
from ragshield.core.shield import RagShield
from ragshield.api.scan import scan
from ragshield.api.decorator import shield

__version__ = "0.1.0"

__all__ = [
    # Main class
    "RagShield",
    # API functions
    "scan",
    "shield",
    # Config and results
    "ShieldConfig",
    "ScanResult",
    "DetectionSignal",
    "ScanDetails",
    # Exceptions
    "RagShieldError",
    "PromptInjectionDetected",
    "CalibrationError",
    "ConfigurationError",
]
