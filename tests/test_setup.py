"""Tests for the setup and readiness system."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentshield.core.exceptions import AgentShieldError, SetupError
from agentshield.core.setup import SetupResult, is_model_cached, setup


class TestSetupError:
    def test_inherits_from_base(self):
        assert issubclass(SetupError, AgentShieldError)

    def test_message(self):
        err = SetupError("model not found")
        assert str(err) == "model not found"


class TestSetupResult:
    def test_to_dict(self):
        result = SetupResult(
            success=True,
            model_name="all-MiniLM-L6-v2",
            model_path="/cache/models/all-MiniLM-L6-v2",
            dimensions=384,
            download_time_ms=150.3,
            validation_time_ms=42.1,
            message="Ready",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["model_name"] == "all-MiniLM-L6-v2"
        assert d["dimensions"] == 384
        assert d["download_time_ms"] == 150.3
        assert d["skipped"] is False

    def test_to_dict_skipped(self):
        result = SetupResult(
            success=True,
            model_name="text-embedding-3-small",
            message="Not required for openai",
            skipped=True,
        )
        d = result.to_dict()
        assert d["skipped"] is True
        assert d["dimensions"] is None


class TestIsModelCached:
    def test_local_path_with_config_json(self, tmp_path):
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        assert is_model_cached(str(model_dir)) is True

    def test_local_path_without_config_json(self, tmp_path):
        model_dir = tmp_path / "empty-model"
        model_dir.mkdir()
        assert is_model_cached(str(model_dir)) is False

    def test_nonexistent_path(self):
        assert is_model_cached("/nonexistent/model/path") is False

    def test_accepts_config_parameter(self):
        """is_model_cached accepts config to determine model name."""
        # With a non-existent model and no hub, should return False
        result = is_model_cached(config={"embeddings": {"model": "/nonexistent"}})
        assert result is False

    def test_huggingface_hub_not_installed(self):
        """Falls back gracefully if huggingface_hub is missing."""
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            # Non-local-path model with no hub = False
            result = is_model_cached("some-hf-model-name")
            assert result is False


class TestSetup:
    def test_non_local_provider_skips(self):
        result = setup(config={"embeddings": {"provider": "openai"}})
        assert result.success is True
        assert result.skipped is True
        assert "not required" in result.message.lower()

    @patch("sentence_transformers.SentenceTransformer")
    def test_successful_setup(self, MockST):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
        MockST.return_value = mock_model

        result = setup(config={"embeddings": {"model": "test-model"}})
        assert result.success is True
        assert result.model_name == "test-model"
        assert result.dimensions == 384
        assert result.download_time_ms is not None
        assert result.validation_time_ms is not None
        assert result.skipped is False

    @patch("sentence_transformers.SentenceTransformer", side_effect=Exception("download failed"))
    def test_download_failure_raises_setup_error(self, MockST):
        with pytest.raises(SetupError, match="Failed to download"):
            setup(config={"embeddings": {"model": "bad-model"}})

    def test_model_name_override(self):
        result = setup(
            config={"embeddings": {"provider": "openai"}},
            model_name="overridden-model",
        )
        assert result.model_name == "overridden-model"


class TestLocalProviderWarning:
    @patch("agentshield.core.setup.is_model_cached", return_value=False)
    def test_warns_when_not_cached(self, mock_cached):
        """LocalEmbeddingProvider should warn when model is not cached."""
        from agentshield.providers.local import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(model_name="uncached-model")

        with pytest.warns(UserWarning, match="not cached"):
            # _load_model will fail at SentenceTransformer import/load,
            # but the warning should fire first
            try:
                provider._load_model()
            except Exception:
                pass

    @patch("agentshield.core.setup.is_model_cached", return_value=True)
    def test_no_warning_when_cached(self, mock_cached):
        """No warning when model is already cached."""
        import warnings
        from agentshield.providers.local import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(model_name="cached-model")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                provider._load_model()
            except UserWarning:
                pytest.fail("Should not warn when model is cached")
            except Exception:
                # Expected — SentenceTransformer won't actually load
                pass
