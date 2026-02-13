"""Tests for telemetry SDK."""

from __future__ import annotations

import gzip
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from pyagentshield.core.config import ShieldConfig, TelemetryConfig
from pyagentshield.telemetry.events import ScanEvent
from pyagentshield.telemetry.client import (
    NoOpTelemetryClient,
    TelemetryClient,
    create_telemetry_client,
)


# ---------------------------------------------------------------------------
# ScanEvent tests
# ---------------------------------------------------------------------------

class TestScanEvent:
    def test_defaults(self):
        event = ScanEvent()
        assert event.event_type == "scan"
        assert event.event_id  # UUID string
        assert event.timestamp  # ISO 8601

    def test_to_dict_schema(self):
        event = ScanEvent(
            sdk_version="0.1.1",
            session_id="abc123",
            is_suspicious=True,
            confidence=0.85,
            drift_score=0.1234,
            threshold=0.0800,
            embedding_model="all-MiniLM-L6-v2",
            cleaning_method="heuristic",
            on_detect="flag",
            project="my-project",
            environment="production",
        )
        d = event.to_dict()

        assert d["event_type"] == "scan"
        assert d["sdk_version"] == "0.1.1"
        assert d["is_suspicious"] is True
        assert d["confidence"] == 0.85
        assert d["drift_score"] == 0.1234
        assert d["threshold"] == 0.08
        assert d["embedding_model"] == "all-MiniLM-L6-v2"
        assert d["project"] == "my-project"
        assert d["environment"] == "production"
        # Serializable to JSON
        json.dumps(d)

    def test_to_dict_none_scores(self):
        event = ScanEvent()
        d = event.to_dict()
        assert d["drift_score"] is None
        assert d["threshold"] is None


# ---------------------------------------------------------------------------
# NoOpTelemetryClient tests
# ---------------------------------------------------------------------------

class TestNoOpTelemetryClient:
    def test_record_is_noop(self):
        client = NoOpTelemetryClient()
        event = ScanEvent()
        client.record(event)  # should not raise

    def test_shutdown_is_noop(self):
        client = NoOpTelemetryClient()
        client.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# TelemetryClient tests
# ---------------------------------------------------------------------------

class TestTelemetryClient:
    def test_buffer_accumulates(self):
        client = TelemetryClient(api_key="test-key", batch_size=100, flush_interval=9999)
        try:
            for _ in range(5):
                client.record(ScanEvent())
            with client._lock:
                assert len(client._buffer) == 5
        finally:
            client._shutdown = True

    @patch("pyagentshield.telemetry.client.urlopen")
    def test_flush_at_batch_size(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        client = TelemetryClient(api_key="test-key", batch_size=3, flush_interval=9999)
        try:
            for _ in range(3):
                client.record(ScanEvent())
            # After hitting batch_size, buffer should be flushed
            mock_urlopen.assert_called()
            with client._lock:
                assert len(client._buffer) == 0
        finally:
            client._shutdown = True

    @patch("pyagentshield.telemetry.client.urlopen")
    def test_gzipped_payload(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        client = TelemetryClient(api_key="test-key", batch_size=1, flush_interval=9999)
        try:
            client.record(ScanEvent(sdk_version="0.1.1"))
            # Inspect the request
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert request.get_header("Content-encoding") == "gzip"
            assert request.get_header("Authorization") == "Bearer test-key"

            # Decompress and verify payload
            decompressed = gzip.decompress(request.data)
            payload = json.loads(decompressed)
            assert "events" in payload
            assert len(payload["events"]) == 1
            assert payload["events"][0]["sdk_version"] == "0.1.1"
        finally:
            client._shutdown = True

    @patch("pyagentshield.telemetry.client.urlopen")
    def test_flush_error_does_not_raise(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("network error")
        client = TelemetryClient(api_key="test-key", batch_size=1, flush_interval=9999)
        try:
            # Should not raise even though urlopen fails
            client.record(ScanEvent())
        finally:
            client._shutdown = True

    @patch("pyagentshield.telemetry.client.urlopen")
    def test_shutdown_flushes(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        client = TelemetryClient(api_key="test-key", batch_size=100, flush_interval=9999)
        client.record(ScanEvent())
        client.record(ScanEvent())
        client.shutdown()
        mock_urlopen.assert_called()


# ---------------------------------------------------------------------------
# create_telemetry_client factory tests
# ---------------------------------------------------------------------------

class TestCreateTelemetryClient:
    def test_no_key_returns_noop(self):
        config = TelemetryConfig(enabled=True, api_key=None)
        client = create_telemetry_client(config)
        assert isinstance(client, NoOpTelemetryClient)

    def test_disabled_returns_noop(self):
        config = TelemetryConfig(enabled=False, api_key="some-key")
        client = create_telemetry_client(config)
        assert isinstance(client, NoOpTelemetryClient)

    def test_empty_key_returns_noop(self):
        config = TelemetryConfig(enabled=True, api_key="")
        client = create_telemetry_client(config)
        assert isinstance(client, NoOpTelemetryClient)

    def test_with_key_returns_real_client(self):
        config = TelemetryConfig(enabled=True, api_key="ask_live_key123")
        client = create_telemetry_client(config)
        try:
            assert isinstance(client, TelemetryClient)
        finally:
            client._shutdown = True


# ---------------------------------------------------------------------------
# TelemetryConfig env var loading
# ---------------------------------------------------------------------------

class TestTelemetryConfigEnvVars:
    def test_api_key_from_env(self):
        env = {"AGENTSHIELD_TELEMETRY__API_KEY": "ask_from_env"}
        with patch.dict(os.environ, env, clear=False):
            config = ShieldConfig()
            assert config.telemetry.api_key == "ask_from_env"

    def test_endpoint_from_env(self):
        env = {"AGENTSHIELD_TELEMETRY__ENDPOINT": "https://custom.endpoint/v1"}
        with patch.dict(os.environ, env, clear=False):
            config = ShieldConfig()
            assert config.telemetry.endpoint == "https://custom.endpoint/v1"

    def test_defaults(self):
        config = TelemetryConfig()
        assert config.enabled is True
        assert config.api_key is None
        assert config.flush_interval == 30
        assert config.batch_size == 50


# ---------------------------------------------------------------------------
# Shield integration test
# ---------------------------------------------------------------------------

class TestShieldTelemetryIntegration:
    @patch("pyagentshield.telemetry.client.create_telemetry_client")
    def test_shield_creates_telemetry_client(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        from pyagentshield.core.shield import AgentShield
        shield = AgentShield()

        mock_create.assert_called_once_with(shield.config.telemetry)

    @patch("pyagentshield.telemetry.client.create_telemetry_client")
    def test_scan_records_telemetry_event(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        from pyagentshield.core.shield import AgentShield

        shield = AgentShield()

        # Mock the detector to avoid loading models
        mock_signal = MagicMock()
        mock_signal.score = 0.1
        mock_signal.metadata = {"drift": 0.05, "threshold": 0.15, "cleaning_method": "heuristic"}
        shield._detector = MagicMock()
        shield._detector.detect.return_value = mock_signal

        shield.scan("test text")

        mock_client.record.assert_called_once()
        event = mock_client.record.call_args[0][0]
        assert event.is_suspicious is False
        assert event.drift_score == 0.05
        assert event.threshold == 0.15
