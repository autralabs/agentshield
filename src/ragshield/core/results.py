"""Result classes for RagShield scan operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectionSignal:
    """
    Output from a single detector.

    Attributes:
        score: Detection score from 0.0 (safe) to 1.0 (malicious)
        confidence: How confident the detector is in this score
        metadata: Detector-specific details (e.g., drift value, matched patterns)
    """

    score: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp values to valid ranges
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata,
        }


@dataclass
class ScanDetails:
    """
    Human-readable explanation of the scan result.

    Attributes:
        summary: Brief description of the result
        risk_factors: List of reasons why the content was flagged
        drift_score: Raw drift value from ZEDD (if applicable)
        threshold: Threshold used for detection
    """

    summary: str
    risk_factors: List[str] = field(default_factory=list)
    drift_score: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "summary": self.summary,
            "risk_factors": self.risk_factors,
            "drift_score": round(self.drift_score, 6) if self.drift_score is not None else None,
            "threshold": round(self.threshold, 6) if self.threshold is not None else None,
        }


@dataclass
class ScanResult:
    """
    Result of scanning a single text for prompt injection.

    Attributes:
        is_suspicious: Whether the content is flagged as suspicious
        confidence: Overall confidence in the detection (0.0 to 1.0)
        signals: Per-detector signals (keyed by detector name)
        details: Human-readable explanation
        original_text: The scanned text (optional, for debugging)
        cleaned_text: The cleaned version (optional, for debugging)
    """

    is_suspicious: bool
    confidence: float
    signals: Dict[str, DetectionSignal] = field(default_factory=dict)
    details: ScanDetails = field(default_factory=lambda: ScanDetails(summary="No analysis performed"))
    original_text: Optional[str] = None
    cleaned_text: Optional[str] = None

    def __post_init__(self) -> None:
        # Clamp confidence to valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

    def __bool__(self) -> bool:
        """Allow `if result:` to check if suspicious."""
        return self.is_suspicious

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "is_suspicious": self.is_suspicious,
            "confidence": round(self.confidence, 4),
            "signals": {name: sig.to_dict() for name, sig in self.signals.items()},
            "details": self.details.to_dict(),
        }

    @classmethod
    def safe(cls, confidence: float = 0.0) -> ScanResult:
        """Create a safe (non-suspicious) result."""
        return cls(
            is_suspicious=False,
            confidence=confidence,
            details=ScanDetails(summary="No injection detected"),
        )

    @classmethod
    def suspicious(
        cls,
        confidence: float,
        summary: str,
        risk_factors: Optional[List[str]] = None,
        drift_score: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> ScanResult:
        """Create a suspicious result with details."""
        return cls(
            is_suspicious=True,
            confidence=confidence,
            details=ScanDetails(
                summary=summary,
                risk_factors=risk_factors or [],
                drift_score=drift_score,
                threshold=threshold,
            ),
        )
