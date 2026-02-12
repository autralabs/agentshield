"""Base detector protocol and context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, runtime_checkable

from pyagentshield.core.results import DetectionSignal

if TYPE_CHECKING:
    from pyagentshield.core.config import ShieldConfig


@dataclass
class DetectionContext:
    """
    Context passed to detectors during detection.

    Provides shared state and configuration to avoid redundant computation.
    """

    config: ShieldConfig

    # Pre-computed cleaned text (optional optimization)
    cleaned_text: Optional[str] = None

    # Pre-computed cleaned texts for batch operations
    cleaned_texts: Optional[List[str]] = None

    # Additional context (extensible)
    extra: Dict = field(default_factory=dict)


@runtime_checkable
class BaseDetector(Protocol):
    """
    Protocol that all detectors must implement.

    Detectors analyze text and produce a DetectionSignal indicating
    the likelihood of prompt injection.

    The protocol is designed for pluggability - new detection methods
    (e.g., pattern matching, ML classifiers) can be added by implementing
    this interface.
    """

    name: str

    def detect(self, text: str, context: DetectionContext) -> DetectionSignal:
        """
        Analyze text and return a detection signal.

        Args:
            text: The text to analyze
            context: Detection context with config and optional pre-computed data

        Returns:
            DetectionSignal with score (0-1), confidence, and metadata
        """
        ...

    def detect_batch(
        self,
        texts: List[str],
        context: DetectionContext,
    ) -> List[DetectionSignal]:
        """
        Batch detection for efficiency.

        Default implementation just calls detect() for each text,
        but detectors can override for more efficient batch processing.

        Args:
            texts: List of texts to analyze
            context: Detection context

        Returns:
            List of DetectionSignals, one per input text
        """
        ...
