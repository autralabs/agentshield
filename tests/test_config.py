"""Tests for configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agentshield.core.config import (
    BehaviorConfig,
    CleaningConfig,
    EmbeddingConfig,
    LoggingConfig,
    PerformanceConfig,
    ShieldConfig,
    ZEDDConfig,
    get_cache_dir,
)


class TestEmbeddingConfig:
    def test_defaults(self):
        config = EmbeddingConfig()
        assert config.provider == "local"
        assert config.model == "all-MiniLM-L6-v2"
        assert config.openai_model == "text-embedding-3-small"

    def test_custom_values(self):
        config = EmbeddingConfig(provider="openai", model="custom-model")
        assert config.provider == "openai"
        assert config.model == "custom-model"


class TestCleaningConfig:
    def test_defaults(self):
        config = CleaningConfig()
        assert config.method == "heuristic"
        assert config.llm_model == "gpt-3.5-turbo"

    def test_hybrid_defaults(self):
        config = CleaningConfig()
        assert config.hybrid.methods == ["heuristic", "finetuned"]
        assert config.hybrid.mode == "sequential"


class TestBehaviorConfig:
    def test_defaults(self):
        config = BehaviorConfig()
        assert config.on_detect == "flag"
        assert config.confidence_threshold == 0.5

    def test_confidence_bounds(self):
        config = BehaviorConfig(confidence_threshold=0.8)
        assert config.confidence_threshold == 0.8


class TestShieldConfig:
    def test_defaults(self):
        config = ShieldConfig()
        assert config.embeddings.provider == "local"
        assert config.cleaning.method == "heuristic"
        assert config.behavior.on_detect == "flag"

    def test_from_dict(self):
        config = ShieldConfig.from_dict({
            "embeddings": {"provider": "openai"},
            "cleaning": {"method": "llm"},
        })
        assert config.embeddings.provider == "openai"
        assert config.cleaning.method == "llm"

    def test_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump({
            "embeddings": {"provider": "local", "model": "test-model"},
            "behavior": {"on_detect": "block"},
        }))

        config = ShieldConfig.from_yaml(yaml_path)
        assert config.embeddings.model == "test-model"
        assert config.behavior.on_detect == "block"

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ShieldConfig.from_yaml("/nonexistent/path.yaml")

    def test_load_none_returns_defaults(self):
        config = ShieldConfig.load(None)
        assert config.embeddings.provider == "local"

    def test_load_passthrough_config(self):
        original = ShieldConfig()
        loaded = ShieldConfig.load(original)
        assert loaded is original

    def test_load_dict(self):
        config = ShieldConfig.load({"behavior": {"on_detect": "warn"}})
        assert config.behavior.on_detect == "warn"

    def test_load_yaml_path(self, tmp_path):
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump({"behavior": {"on_detect": "filter"}}))

        config = ShieldConfig.load(str(yaml_path))
        assert config.behavior.on_detect == "filter"

    def test_load_invalid_type(self):
        with pytest.raises(TypeError):
            ShieldConfig.load(12345)

    def test_to_dict(self):
        config = ShieldConfig()
        d = config.to_dict()
        assert "embeddings" in d
        assert "cleaning" in d
        assert "zedd" in d
        assert "behavior" in d

    def test_to_yaml(self, tmp_path):
        config = ShieldConfig()
        output_path = tmp_path / "output.yaml"
        config.to_yaml(output_path)
        assert output_path.exists()

        # Verify it round-trips
        loaded = ShieldConfig.from_yaml(output_path)
        assert loaded.embeddings.provider == config.embeddings.provider

    def test_backward_compat_zedd_cleaning_method(self):
        """If zedd.cleaning_method is set but cleaning.method isn't, sync them."""
        config = ShieldConfig.from_dict({
            "zedd": {"cleaning_method": "llm"},
        })
        assert config.cleaning.method == "llm"

    def test_load_from_env_var_config_path(self, tmp_path):
        yaml_path = tmp_path / "env_config.yaml"
        yaml_path.write_text(yaml.dump({"behavior": {"on_detect": "block"}}))

        with patch.dict(os.environ, {"AGENTSHIELD_CONFIG_PATH": str(yaml_path)}):
            config = ShieldConfig.load(None)
            assert config.behavior.on_detect == "block"


class TestGetCacheDir:
    def test_default_cache_dir(self):
        cache_dir = get_cache_dir()
        assert cache_dir == Path.home() / ".agentshield"

    def test_custom_cache_dir(self, tmp_path):
        custom_dir = str(tmp_path / "custom_cache")
        config = ShieldConfig.from_dict({
            "performance": {"cache_dir": custom_dir}
        })
        cache_dir = get_cache_dir(config)
        assert cache_dir == Path(custom_dir)
        assert cache_dir.exists()
