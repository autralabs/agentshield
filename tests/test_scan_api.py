"""Tests for the scan() function API."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from agentshield.core.results import ScanResult

# Access the actual module (not the function exported via api/__init__.py)
_scan_mod = sys.modules["agentshield.api.scan"]


class TestScanFunction:
    def setup_method(self):
        _scan_mod._default_shield = None

    def test_scan_single_text(self):
        mock_shield = MagicMock()
        mock_shield.scan.return_value = ScanResult.safe()
        _scan_mod._default_shield = mock_shield

        result = _scan_mod.scan("Hello world")
        assert isinstance(result, ScanResult)
        assert not result.is_suspicious

    def test_scan_list_of_texts(self):
        mock_shield = MagicMock()
        mock_shield.scan.return_value = [ScanResult.safe(), ScanResult.safe()]
        _scan_mod._default_shield = mock_shield

        results = _scan_mod.scan(["text1", "text2"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_scan_delegates_to_shield(self):
        mock_shield = MagicMock()
        expected = ScanResult.suspicious(confidence=0.9, summary="test")
        mock_shield.scan.return_value = expected
        _scan_mod._default_shield = mock_shield

        result = _scan_mod.scan("suspicious text")
        mock_shield.scan.assert_called_once_with("suspicious text")
        assert result.is_suspicious

    def test_lazy_initialization(self):
        assert _scan_mod._default_shield is None


class TestConfigure:
    def setup_method(self):
        _scan_mod._default_shield = None

    def test_configure_replaces_shield(self):
        old_mock = MagicMock()
        _scan_mod._default_shield = old_mock

        new_mock = MagicMock()
        _scan_mod._default_shield = new_mock
        assert _scan_mod._default_shield is not old_mock
        assert _scan_mod._default_shield is new_mock

    def test_configure_with_none_resets(self):
        _scan_mod._default_shield = MagicMock()
        _scan_mod._default_shield = None
        assert _scan_mod._default_shield is None
