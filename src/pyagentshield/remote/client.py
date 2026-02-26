"""CloudThresholdClient — fetches and reports threshold sync data.

Uses stdlib only (urllib, json, threading, queue, time). No new dependencies.
Cache key is (fingerprint, environment_key) — not fingerprint alone — to
prevent bleed across environments.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CloudRule:
    id: str
    pipeline_fingerprint: str
    threshold_value: float
    version: int


@dataclass
class CloudResolution:
    resolution_mode: str
    fail_open_to_local: bool
    ttl_seconds: int
    matched_rule: Optional[CloudRule]
    project_settings: Dict[str, Any] = field(default_factory=dict)


class CloudThresholdClient:
    """
    Fetches cloud threshold resolution and reports SDK observations.

    Thread-safe. A single daemon thread drains the report queue.
    Resolve results are cached per (fingerprint, environment_key) with a TTL.
    """

    def __init__(
        self,
        api_key: str,
        resolve_endpoint: str,
        report_endpoint: str,
        ttl: int = 300,
        report_debounce_seconds: int = 60,
        report_enabled: bool = True,
        timeout_seconds: float = 3.0,
    ) -> None:
        self._api_key = api_key
        self._resolve_endpoint = resolve_endpoint
        self._report_endpoint = report_endpoint
        self._report_enabled = report_enabled
        self._report_debounce_seconds = report_debounce_seconds
        self._timeout_seconds = timeout_seconds
        self._default_ttl = ttl

        # Guards both _cache and _report_cache; all read/write must hold this lock.
        self._lock = threading.Lock()

        # Resolve cache: (fingerprint, env_key) → (CloudResolution, expires_monotonic)
        self._cache: Dict[Tuple[str, str], Tuple[CloudResolution, float]] = {}

        # Report dedup cache: (fingerprint, env_key) → (last_monotonic, last_source, last_value_4dp)
        # last_value_4dp is round(..., 4) — suppresses float jitter
        self._report_cache: Dict[Tuple[str, str], Tuple[float, Optional[str], Optional[float]]] = {}

        # Bounded queue caps memory under high cardinality; occasional drops are acceptable.
        self._report_queue: queue.Queue = queue.Queue(maxsize=512)
        self._report_thread = threading.Thread(
            target=self._drain_report_queue, daemon=True, name="agentshield-report"
        )
        self._report_thread.start()

    # ------------------------------------------------------------------ #
    # Resolve                                                              #
    # ------------------------------------------------------------------ #

    def get_resolution(
        self,
        fingerprint: str,
        environment: Optional[str] = None,
    ) -> Optional[CloudResolution]:
        """
        Cached GET /v1/thresholds/resolve. Returns None on any error.

        Cache key: (fingerprint, env_key) where env_key = environment or '*'.
        HTTP call is performed outside the lock to avoid blocking other threads.
        """
        env_key = environment or "*"
        cache_key = (fingerprint, env_key)
        now = time.monotonic()

        # 1. Check cache under lock
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                resolution, expires_at = entry
                if now < expires_at:
                    return resolution
                # Expired — remove
                del self._cache[cache_key]

        # 2. Perform HTTP GET outside lock
        resolution = self._do_resolve(fingerprint, environment)

        # 3. Store result under lock (even None results are not cached — let them retry)
        if resolution is not None:
            ttl = resolution.ttl_seconds or self._default_ttl
            with self._lock:
                self._cache[cache_key] = (resolution, now + ttl)

        return resolution

    def _do_resolve(
        self,
        fingerprint: str,
        environment: Optional[str],
    ) -> Optional[CloudResolution]:
        """Execute the HTTP GET. Returns None on any error."""
        try:
            params = f"fingerprint={urllib.parse.quote(fingerprint, safe='')}"
            if environment:
                params += f"&environment={urllib.parse.quote(environment, safe='')}"
            url = f"{self._resolve_endpoint}?{params}"

            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "User-Agent": "pyagentshield-threshold-sync",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            matched_rule: Optional[CloudRule] = None
            raw_rule = body.get("matched_rule")
            if raw_rule:
                matched_rule = CloudRule(
                    id=raw_rule["id"],
                    pipeline_fingerprint=raw_rule["pipeline_fingerprint"],
                    threshold_value=float(raw_rule["threshold_value"]),
                    version=int(raw_rule["version"]),
                )

            return CloudResolution(
                resolution_mode=body.get("resolution_mode", "local_only"),
                fail_open_to_local=bool(body.get("fail_open_to_local", True)),
                ttl_seconds=int(body.get("ttl_seconds", self._default_ttl)),
                matched_rule=matched_rule,
                project_settings=body.get("project_settings") or {},
            )

        except Exception:
            logger.debug("Failed to resolve cloud threshold for %s", fingerprint, exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #

    def report_if_due(
        self,
        fingerprint: str,
        environment: Optional[str],
        observation: dict,
    ) -> None:
        """
        Debounced, non-blocking POST /v1/thresholds/report. Never raises.

        Reports immediately if effective_source or effective_threshold (quantized
        to 4dp) changed, OR if the debounce window elapsed.
        Enqueues to bounded queue; daemon thread sends sequentially.
        """
        if not self._report_enabled:
            return

        env_key = environment or "*"
        cache_key = (fingerprint, env_key)
        now = time.monotonic()

        obs_source = observation.get("effective_source")
        raw_val = observation.get("effective_threshold")
        # Quantize to 4dp to suppress float jitter (0.24000001 vs 0.24)
        obs_value_q: Optional[float] = round(raw_val, 4) if raw_val is not None else None

        should_report = False
        with self._lock:
            last = self._report_cache.get(cache_key)
            # last = (last_monotonic, last_source, last_value_quantized_4dp)
            decision_changed = (
                last is None
                or last[1] != obs_source
                or last[2] != obs_value_q
            )
            time_elapsed = last is None or (now - last[0]) >= self._report_debounce_seconds
            if decision_changed or time_elapsed:
                self._report_cache[cache_key] = (now, obs_source, obs_value_q)
                should_report = True

        if should_report:
            try:
                self._report_queue.put_nowait(observation)
            except queue.Full:
                pass  # Bounded queue; occasional drops are acceptable

    def _drain_report_queue(self) -> None:
        """Single daemon thread that serializes all report HTTP calls."""
        while True:
            try:
                observation = self._report_queue.get()
                self._do_report(observation)
            except Exception:
                logger.debug("Report drain error", exc_info=True)

    def _do_report(self, observation: dict) -> None:
        """Execute the HTTP POST. Errors are silently logged."""
        try:
            payload = json.dumps(observation).encode("utf-8")
            req = urllib.request.Request(
                self._report_endpoint,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "pyagentshield-threshold-sync",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout_seconds):
                pass
            logger.debug(
                "Reported threshold observation: fp=%s source=%s",
                observation.get("pipeline_fingerprint"),
                observation.get("effective_source"),
            )
        except Exception:
            logger.debug("Failed to report threshold observation", exc_info=True)


# urllib.parse is part of stdlib — import here to avoid top-level issues in older Pythons
import urllib.parse  # noqa: E402
