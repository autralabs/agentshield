"""Tests for OpenAI-compatible adapter (base_url/default_headers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from pyagentshield.core.config import ShieldConfig


class TestOpenAIProviderBaseUrl:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_client_receives_base_url_and_headers(self):
        from pyagentshield.providers.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://example.com"},
        )

        with patch.dict("sys.modules", {"openai": MagicMock()}):
            import sys

            mock_openai = sys.modules["openai"].OpenAI
            provider._client = None
            provider._get_client()
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                default_headers={"HTTP-Referer": "https://example.com"},
            )


class TestLLMCleanerBaseUrl:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_client_receives_base_url_and_headers(self):
        from pyagentshield.cleaning.llm import LLMCleaner

        cleaner = LLMCleaner(
            model="gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://example.com"},
        )

        with patch.dict("sys.modules", {"openai": MagicMock()}):
            import sys

            mock_openai = sys.modules["openai"].OpenAI
            cleaner._client = None
            cleaner._get_client()
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                default_headers={"HTTP-Referer": "https://example.com"},
            )


class TestConfigAndWiring:
    def test_config_parses_openai_compatible_fields(self):
        config = ShieldConfig.from_dict({
            "embeddings": {
                "provider": "openai",
                "api_key": "emb-key",
                "base_url": "https://openrouter.ai/api/v1",
                "default_headers": {"X-Test": "1"},
            },
            "cleaning": {
                "method": "llm",
                "api_key": "clean-key",
                "base_url": "https://together.xyz/v1",
                "default_headers": {"X-Test": "2"},
            },
        })
        assert config.embeddings.base_url == "https://openrouter.ai/api/v1"
        assert config.cleaning.base_url == "https://together.xyz/v1"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"})
    def test_hybrid_llm_receives_openai_compatible_fields(self):
        from pyagentshield.cleaning.hybrid import create_hybrid_cleaner

        cleaner = create_hybrid_cleaner(
            methods=["heuristic", "llm"],
            mode="sequential",
            llm_model="gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://example.com"},
        )
        llm = next(c for c in cleaner.cleaners if c.method == "llm")
        assert llm._base_url == "https://openrouter.ai/api/v1"
        assert llm._default_headers == {"HTTP-Referer": "https://example.com"}

    def test_shield_passes_cache_embeddings_to_openai_provider(self):
        from pyagentshield.core.shield import AgentShield

        shield = AgentShield(config={
            "embeddings": {"provider": "openai", "api_key": "test-key"},
            "performance": {"cache_embeddings": False},
            "telemetry": {"enabled": False},
        })
        assert shield.embedding_provider._cache_embeddings is False

    def test_config_parses_dimensions(self):
        config = ShieldConfig.from_dict({
            "embeddings": {
                "provider": "openai",
                "dimensions": 768,
            }
        })
        assert config.embeddings.dimensions == 768


class TestCapabilityDiscovery:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_explicit_dimensions_used(self):
        from pyagentshield.providers.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            model_name="custom-model",
            dimensions=768,
        )
        assert provider.dimensions == 768
        assert provider._dimensions_confirmed is True

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_unknown_model_provisional_dimensions(self, tmp_path):
        from pyagentshield.providers.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            model_name="unknown-provider-model",
            cache_dir=tmp_path,
        )
        assert provider.dimensions == 1536
        assert provider._dimensions_confirmed is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_dimensions_discovered_from_first_encode(self, tmp_path):
        from pyagentshield.providers.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            model_name="custom-model-512",
            cache_dir=tmp_path,
        )

        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = list(np.random.randn(512).astype(float))
        mock_response.data = [mock_data]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client

        emb = provider.encode("test text")
        assert len(emb) == 512
        assert provider._dimensions_confirmed is True
        assert provider.dimensions == 512

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_batch_empty_and_real_no_crash(self, tmp_path):
        from pyagentshield.providers.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            model_name="unknown-512-model",
            cache_dir=tmp_path,
        )

        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = list(np.random.randn(512).astype(float))
        mock_response.data = [mock_data]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client

        result = provider.encode_batch(["", "hello"])
        assert result.shape == (2, 512)
        assert np.allclose(result[0], 0.0)
        assert provider._dimensions_confirmed is True
        assert provider.dimensions == 512

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_dimensions_saved_and_loaded(self, tmp_path):
        from pyagentshield.providers.openai import OpenAIEmbeddingProvider

        p1 = OpenAIEmbeddingProvider(
            model_name="disk-cache-model",
            cache_dir=tmp_path,
        )
        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = list(np.random.randn(384).astype(float))
        mock_response.data = [mock_data]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        p1._client = mock_client
        p1.encode("discover dims")
        assert p1.dimensions == 384

        p2 = OpenAIEmbeddingProvider(
            model_name="disk-cache-model",
            cache_dir=tmp_path,
        )
        assert p2._dimensions_confirmed is True
        assert p2.dimensions == 384

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_different_ports_different_dim_cache_keys(self):
        from pyagentshield.providers.openai import OpenAIEmbeddingProvider

        p1 = OpenAIEmbeddingProvider(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434/v1",
        )
        p2 = OpenAIEmbeddingProvider(
            model_name="nomic-embed-text",
            base_url="http://localhost:8000/v1",
        )
        assert p1._dim_cache_key() != p2._dim_cache_key()
