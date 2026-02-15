"""Tests for OpenAI-compatible adapter (base_url/default_headers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
