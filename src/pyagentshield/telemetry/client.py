"""Telemetry client for sending scan events to AgentShield Cloud."""

from __future__ import annotations

import atexit
import gzip
import json
import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.request import Request, urlopen
from urllib.error import URLError

if TYPE_CHECKING:
    from pyagentshield.core.config import TelemetryConfig
    from pyagentshield.telemetry.events import ScanEvent

logger = logging.getLogger(__name__)


class TelemetryClient:
    """
    Buffered telemetry client that sends gzipped batches to AgentShield Cloud.

    - Events are buffered in memory and flushed periodically or when batch_size is reached.
    - A daemon thread handles periodic flushing so the main thread is never blocked.
    - All errors are silently caught (DEBUG log) — telemetry never crashes scans.
    - shutdown() performs a final flush and is registered via atexit.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.agentshield.dev/v1/telemetry",
        flush_interval: int = 30,
        batch_size: int = 50,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._flush_interval = flush_interval
        self._batch_size = batch_size

        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._shutdown = False

        # Start daemon flush thread
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="agentshield-telemetry"
        )
        self._flush_thread.start()

        # Register shutdown handler
        atexit.register(self.shutdown)

    def record(self, event: ScanEvent) -> None:
        """Append an event to the buffer. Auto-flushes at batch_size."""
        try:
            with self._lock:
                self._buffer.append(event.to_dict())
                should_flush = len(self._buffer) >= self._batch_size
            if should_flush:
                self._flush()
        except Exception:
            logger.debug("Failed to record telemetry event", exc_info=True)

    def _flush_loop(self) -> None:
        """Periodically flush the buffer (runs in daemon thread)."""
        while not self._shutdown:
            self._flush_event = threading.Event()
            self._flush_event.wait(timeout=self._flush_interval)
            if not self._shutdown:
                self._flush()

    def _flush(self) -> None:
        """Swap the buffer and POST gzipped JSON payload."""
        try:
            with self._lock:
                if not self._buffer:
                    return
                batch = self._buffer
                self._buffer = []

            payload = json.dumps({"events": batch}).encode("utf-8")
            compressed = gzip.compress(payload)

            req = Request(
                self._endpoint,
                data=compressed,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                    "Authorization": f"Bearer {self._api_key}",
                    "User-Agent": "pyagentshield-telemetry",
                },
            )
            urlopen(req, timeout=5)
            logger.debug("Flushed %d telemetry events", len(batch))
        except (URLError, OSError):
            logger.debug("Failed to flush telemetry events", exc_info=True)
        except Exception:
            logger.debug("Unexpected telemetry flush error", exc_info=True)

    def shutdown(self) -> None:
        """Final flush on interpreter exit."""
        self._shutdown = True
        self._flush()


class NoOpTelemetryClient:
    """No-op client used when telemetry is disabled or no API key is set."""

    def record(self, event: ScanEvent) -> None:  # noqa: ARG002
        pass

    def shutdown(self) -> None:
        pass


def create_telemetry_client(
    config: TelemetryConfig,
) -> Union[TelemetryClient, NoOpTelemetryClient]:
    """
    Factory: returns a real client if telemetry is enabled and an API key is set,
    otherwise returns NoOpTelemetryClient (zero overhead).
    """
    if config.enabled and config.api_key:
        return TelemetryClient(
            api_key=config.api_key,
            endpoint=config.endpoint,
            flush_interval=config.flush_interval,
            batch_size=config.batch_size,
        )
    return NoOpTelemetryClient()
