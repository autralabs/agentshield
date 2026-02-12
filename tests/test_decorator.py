"""Tests for the @shield() decorator."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from pyagentshield.api.decorator import _extract_texts_from_value, shield
from pyagentshield.core.exceptions import PromptInjectionDetected
from pyagentshield.core.results import DetectionSignal, ScanResult


class TestExtractTexts:
    def test_string_value(self):
        assert _extract_texts_from_value("hello") == ["hello"]

    def test_list_of_strings(self):
        assert _extract_texts_from_value(["a", "b"]) == ["a", "b"]

    def test_nested_list(self):
        assert _extract_texts_from_value([["a", "b"], "c"]) == ["a", "b", "c"]

    def test_dict_values(self):
        texts = _extract_texts_from_value({"key": "value", "key2": "value2"})
        assert "value" in texts
        assert "value2" in texts

    def test_langchain_document(self):
        doc = MagicMock()
        doc.page_content = "test content"
        texts = _extract_texts_from_value(doc)
        assert texts == ["test content"]

    def test_llamaindex_node(self):
        node = MagicMock(spec=[])  # no page_content
        node.text = "node text"
        texts = _extract_texts_from_value(node)
        assert texts == ["node text"]

    def test_non_text_value(self):
        assert _extract_texts_from_value(42) == []
        assert _extract_texts_from_value(None) == []


class TestShieldDecorator:
    def _mock_shield(self, is_suspicious=False, confidence=0.1):
        """Create a mock AgentShield that returns controlled results."""
        mock = MagicMock()
        result = ScanResult(
            is_suspicious=is_suspicious,
            confidence=confidence,
            signals={"zedd": DetectionSignal(score=confidence, confidence=confidence)},
        )
        mock.scan.return_value = [result]
        return mock

    @patch("pyagentshield.core.shield.AgentShield")
    def test_clean_input_passes_through(self, MockShieldClass):
        mock_instance = self._mock_shield(is_suspicious=False)
        MockShieldClass.return_value = mock_instance

        @shield(on_detect="block")
        def my_func(text: str) -> str:
            return f"processed: {text}"

        result = my_func("safe text")
        assert result == "processed: safe text"

    @patch("pyagentshield.core.shield.AgentShield")
    def test_block_mode_raises(self, MockShieldClass):
        mock_instance = self._mock_shield(is_suspicious=True, confidence=0.9)
        MockShieldClass.return_value = mock_instance

        @shield(on_detect="block")
        def my_func(text: str) -> str:
            return "should not reach"

        with pytest.raises(PromptInjectionDetected):
            my_func("malicious text")

    @patch("pyagentshield.core.shield.AgentShield")
    def test_warn_mode_logs_and_continues(self, MockShieldClass, caplog):
        mock_instance = self._mock_shield(is_suspicious=True, confidence=0.9)
        MockShieldClass.return_value = mock_instance

        @shield(on_detect="warn")
        def my_func(text: str) -> str:
            return "continued"

        with caplog.at_level(logging.WARNING):
            result = my_func("suspicious text")
        assert result == "continued"
        assert "injection detected" in caplog.text.lower() or "Prompt injection" in caplog.text

    @patch("pyagentshield.core.shield.AgentShield")
    def test_flag_mode_passes_through(self, MockShieldClass):
        mock_instance = self._mock_shield(is_suspicious=True, confidence=0.9)
        MockShieldClass.return_value = mock_instance

        @shield(on_detect="flag")
        def my_func(text: str) -> str:
            return "passed"

        result = my_func("flagged text")
        assert result == "passed"

    @patch("pyagentshield.core.shield.AgentShield")
    def test_confidence_threshold_filtering(self, MockShieldClass):
        """Low-confidence detections should not trigger block."""
        mock_instance = self._mock_shield(is_suspicious=True, confidence=0.3)
        MockShieldClass.return_value = mock_instance

        @shield(on_detect="block", confidence_threshold=0.5)
        def my_func(text: str) -> str:
            return "passed"

        # Should NOT raise because confidence 0.3 < threshold 0.5
        result = my_func("borderline text")
        assert result == "passed"

    @patch("pyagentshield.core.shield.AgentShield")
    def test_scan_args_selective(self, MockShieldClass):
        """Only specified args should be scanned."""
        mock_instance = self._mock_shield(is_suspicious=False)
        MockShieldClass.return_value = mock_instance

        @shield(on_detect="block", scan_args=["docs"])
        def my_func(query: str, docs: list) -> str:
            return "ok"

        my_func("query", ["doc1", "doc2"])
        # Check that scan was called with only docs content
        call_args = mock_instance.scan.call_args
        scanned_texts = call_args[0][0]
        assert "query" not in scanned_texts
        assert "doc1" in scanned_texts

    @patch("pyagentshield.core.shield.AgentShield")
    def test_no_text_args_skips_scan(self, MockShieldClass):
        mock_instance = self._mock_shield()
        MockShieldClass.return_value = mock_instance

        @shield(on_detect="block")
        def my_func(count: int) -> int:
            return count * 2

        result = my_func(5)
        assert result == 10
        mock_instance.scan.assert_not_called()
