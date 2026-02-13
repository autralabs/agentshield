"""Telemetry event definitions for AgentShield Cloud."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ScanEvent:
    """
    Telemetry event emitted after each scan.

    Privacy: No document text is ever included — only metrics and config metadata.
    """

    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: str = "scan"

    # SDK info
    sdk_version: str = ""
    session_id: str = ""

    # Scan metrics
    is_suspicious: bool = False
    confidence: float = 0.0
    drift_score: Optional[float] = None
    threshold: Optional[float] = None

    # Config snapshot
    embedding_model: str = ""
    cleaning_method: str = ""
    on_detect: str = ""

    # User-supplied context
    project: Optional[str] = None
    environment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "sdk_version": self.sdk_version,
            "session_id": self.session_id,
            "is_suspicious": self.is_suspicious,
            "confidence": round(self.confidence, 4),
            "drift_score": round(self.drift_score, 6) if self.drift_score is not None else None,
            "threshold": round(self.threshold, 6) if self.threshold is not None else None,
            "embedding_model": self.embedding_model,
            "cleaning_method": self.cleaning_method,
            "on_detect": self.on_detect,
            "project": self.project,
            "environment": self.environment,
        }
