"""ThresholdDecision dataclass for provenance-rich threshold resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Canonical source tokens — keep in sync with migration 005 DB CHECK constraint
# and backend/src/thresholds/report_handler.py VALID_EFFECTIVE_SOURCES
CANONICAL_SOURCES = frozenset({
    "local_pinned",    # explicit constructor/env/yaml pin, overrides everything
    "local_cache",     # from user's custom calibration cache (~/.agentshield/thresholds/)
    "finetuned",       # from calibration.json in a finetuned cleaner model directory
    "registry",        # from pre-calibrated ThresholdRegistry (model-name keyed)
    "auto_calibrated", # SDK ran calibration and computed threshold at runtime
    "cloud_manual",    # from a threshold_rule set in the dashboard
    "local_failopen",  # cloud_only mode, no rule found, fail_open=True, fell back to local
})

# Canonical mode tokens — keep in sync with migration 005 and report_handler
CANONICAL_MODES = frozenset({
    "local_only",
    "local_prefer",
    "cloud_prefer",
    "cloud_only",
    "observe",
})


@dataclass
class ThresholdDecision:
    """
    Result of a threshold resolution, with full provenance.

    Tracks which value was chosen, from where, under which mode, and any
    cloud candidate for observe-mode diff reporting.
    """

    value: float
    # Must be one of CANONICAL_SOURCES
    source: str
    # Must be one of CANONICAL_MODES
    mode: str
    fingerprint: str
    cloud_rule_id: Optional[str] = None
    cloud_rule_version: Optional[int] = None
    # Cloud candidate value — populated in observe mode for diff reporting;
    # also set when cloud_manual is the effective source.
    cloud_threshold: Optional[float] = None
