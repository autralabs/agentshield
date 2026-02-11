"""Tests for LangChain integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentshield.core.exceptions import PromptInjectionDetected
from agentshield.core.results import DetectionSignal, ScanResult
from agentshield.integrations.langchain import ShieldRunnable


def _make_safe_result():
    return ScanResult(
        is_suspicious=False,
        confidence=0.05,
        signals={"zedd": DetectionSignal(score=0.05, confidence=0.05)},
    )


def _make_suspicious_result():
    return ScanResult.suspicious(
        confidence=0.9,
        summary="Injection detected",
        drift_score=0.45,
        threshold=0.2,
    )


class TestShieldRunnableNormalization:
    @patch("agentshield.core.shield.AgentShield")
    def test_string_input(self, MockShield):
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [_make_safe_result()]
        MockShield.return_value = mock_instance

        runnable = ShieldRunnable()
        result = runnable.invoke("Hello world")
        assert result == "Hello world"

    @patch("agentshield.core.shield.AgentShield")
    def test_list_str_input(self, MockShield):
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [_make_safe_result(), _make_safe_result()]
        MockShield.return_value = mock_instance

        runnable = ShieldRunnable()
        result = runnable.invoke(["text1", "text2"])
        assert result == ["text1", "text2"]

    @patch("agentshield.core.shield.AgentShield")
    def test_dict_with_context_key(self, MockShield):
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [_make_safe_result()]
        MockShield.return_value = mock_instance

        runnable = ShieldRunnable()
        input_data = {"context": ["some doc"], "query": "question"}
        result = runnable.invoke(input_data)
        assert isinstance(result, dict)
        assert "context" in result

    @patch("agentshield.core.shield.AgentShield")
    def test_empty_list(self, MockShield):
        mock_instance = MagicMock()
        MockShield.return_value = mock_instance

        runnable = ShieldRunnable()
        result = runnable.invoke([])
        assert result == []

    @patch("agentshield.core.shield.AgentShield")
    def test_document_objects(self, MockShield):
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [_make_safe_result()]
        MockShield.return_value = mock_instance

        doc = MagicMock()
        doc.page_content = "test content"
        doc.metadata = {}

        runnable = ShieldRunnable()
        result = runnable.invoke([doc])
        assert len(result) == 1


class TestShieldRunnableModes:
    @patch("agentshield.core.shield.AgentShield")
    def test_block_raises_exception(self, MockShield):
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [_make_suspicious_result()]
        MockShield.return_value = mock_instance

        runnable = ShieldRunnable(on_detect="block")
        with pytest.raises(PromptInjectionDetected):
            runnable.invoke(["malicious doc"])

    @patch("agentshield.core.shield.AgentShield")
    def test_filter_removes_suspicious(self, MockShield):
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [
            _make_safe_result(),
            _make_suspicious_result(),
            _make_safe_result(),
        ]
        MockShield.return_value = mock_instance

        runnable = ShieldRunnable(on_detect="filter")
        result = runnable.invoke(["safe1", "malicious", "safe2"])
        assert len(result) == 2
        assert "malicious" not in result

    @patch("agentshield.core.shield.AgentShield")
    def test_flag_adds_metadata(self, MockShield):
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [_make_suspicious_result()]
        MockShield.return_value = mock_instance

        doc = MagicMock()
        doc.page_content = "test"
        doc.metadata = {}

        runnable = ShieldRunnable(on_detect="flag")
        result = runnable.invoke([doc])
        assert "_agentshield" in doc.metadata

    @patch("agentshield.core.shield.AgentShield")
    def test_warn_passes_through(self, MockShield):
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [_make_suspicious_result()]
        MockShield.return_value = mock_instance

        runnable = ShieldRunnable(on_detect="warn")
        result = runnable.invoke(["suspicious text"])
        assert len(result) == 1

    @patch("agentshield.core.shield.AgentShield")
    def test_confidence_threshold(self, MockShield):
        """Low-confidence results should not trigger actions."""
        low_confidence = ScanResult(
            is_suspicious=True,
            confidence=0.3,
            signals={"zedd": DetectionSignal(score=0.3, confidence=0.3)},
        )
        mock_instance = MagicMock()
        mock_instance.scan.return_value = [low_confidence]
        MockShield.return_value = mock_instance

        runnable = ShieldRunnable(on_detect="block", confidence_threshold=0.5)
        # Should NOT raise because confidence 0.3 < threshold 0.5
        result = runnable.invoke(["borderline text"])
        assert len(result) == 1
