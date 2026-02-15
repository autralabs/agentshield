"""Tests for threshold registry and manager."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from pyagentshield.threshold.fingerprint import create_pipeline_fingerprint
from pyagentshield.threshold.registry import ThresholdRegistry
from pyagentshield.threshold.manager import ThresholdManager


class TestThresholdRegistry:
    def test_known_model_exact_match(self):
        threshold = ThresholdRegistry.get("all-MiniLM-L6-v2")
        assert threshold == 0.23

    def test_known_model_with_prefix(self):
        threshold = ThresholdRegistry.get("BAAI/bge-small-en-v1.5")
        assert threshold == 0.21

    def test_short_name_lookup(self):
        """Input with prefix should match key without prefix.

        The short_name logic strips the input prefix, so 'org/model'
        finds 'model' in the registry. Note: looking up 'model' does NOT
        find 'org/model' — the reverse is not implemented.
        """
        # all-MiniLM-L6-v2 is stored without prefix, so input with prefix finds it
        threshold = ThresholdRegistry.get("sentence-transformers/all-MiniLM-L6-v2")
        assert threshold == 0.23

    def test_case_insensitive_match(self):
        threshold = ThresholdRegistry.get("ALL-MINILM-L6-V2")
        assert threshold == 0.23

    def test_unknown_model_returns_none(self):
        assert ThresholdRegistry.get("nonexistent-model") is None

    def test_has_known_model(self):
        assert ThresholdRegistry.has("all-MiniLM-L6-v2")

    def test_has_unknown_model(self):
        assert not ThresholdRegistry.has("nonexistent-model")

    def test_list_models(self):
        models = ThresholdRegistry.list_models()
        assert len(models) > 0
        assert "all-MiniLM-L6-v2" in models

    def test_openai_models_present(self):
        assert ThresholdRegistry.get("text-embedding-3-small") is not None
        assert ThresholdRegistry.get("text-embedding-3-large") is not None


class TestThresholdManager:
    def test_explicit_default_takes_precedence(self, tmp_cache_dir):
        manager = ThresholdManager(
            cache_dir=tmp_cache_dir,
            default_threshold=0.15,
        )
        # Even for a known model, explicit default wins
        threshold = manager.get_threshold("all-MiniLM-L6-v2", auto_calibrate=False)
        assert threshold == 0.15

    def test_registry_lookup(self, tmp_cache_dir):
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        threshold = manager.get_threshold("all-MiniLM-L6-v2", auto_calibrate=False)
        assert threshold == 0.23

    def test_custom_threshold_over_registry(self, tmp_cache_dir):
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        manager.set_threshold("all-MiniLM-L6-v2", 0.30, save=False)
        threshold = manager.get_threshold("all-MiniLM-L6-v2", auto_calibrate=False)
        assert threshold == 0.30

    def test_save_and_load_custom_thresholds(self, tmp_cache_dir):
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        manager.set_threshold("my-custom-model", 0.42, save=True)

        # Create new manager pointing to same cache
        manager2 = ThresholdManager(cache_dir=tmp_cache_dir)
        threshold = manager2.get_threshold("my-custom-model", auto_calibrate=False)
        assert threshold == 0.42

    def test_unknown_model_no_calibrate_raises(self, tmp_cache_dir):
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        with pytest.raises(ValueError, match="No threshold found"):
            manager.get_threshold("unknown-model-xyz", auto_calibrate=False)

    def test_unknown_model_auto_calibrate_fallback(self, tmp_cache_dir):
        """Without provider/cleaner, auto-calibrate falls back to 0.20."""
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        threshold = manager.get_threshold(
            "unknown-model-xyz", auto_calibrate=True
        )
        assert threshold == 0.20

    def test_has_threshold(self, tmp_cache_dir):
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        assert manager.has_threshold("all-MiniLM-L6-v2")
        assert not manager.has_threshold("nonexistent-model")

    def test_has_threshold_with_default(self, tmp_cache_dir):
        manager = ThresholdManager(cache_dir=tmp_cache_dir, default_threshold=0.1)
        # With explicit default, has_threshold returns True for anything
        assert manager.has_threshold("anything")

    def test_has_threshold_respects_pipeline_rules(self, tmp_cache_dir):
        from tests.conftest import MockEmbeddingProvider, MockTextCleaner

        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        provider = MockEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        heuristic_cleaner = MockTextCleaner(method="heuristic")
        llm_cleaner = MockTextCleaner(method="llm")

        # Heuristic path can use registry
        assert manager.has_threshold(
            "all-MiniLM-L6-v2",
            embedding_provider=provider,
            text_cleaner=heuristic_cleaner,
        )
        # Non-heuristic path cannot rely on heuristic-only registry
        assert not manager.has_threshold(
            "all-MiniLM-L6-v2",
            embedding_provider=provider,
            text_cleaner=llm_cleaner,
        )

    def test_list_thresholds(self, tmp_cache_dir):
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        manager.set_threshold("custom-model", 0.5, save=False)
        all_thresholds = manager.list_thresholds()
        assert "all-MiniLM-L6-v2" in all_thresholds  # from registry
        assert "custom-model" in all_thresholds  # custom

    def test_clear_custom_thresholds(self, tmp_cache_dir):
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        manager.set_threshold("test-model", 0.3, save=True)
        manager.clear_custom_thresholds()
        assert not manager.has_threshold("test-model")

    def test_finetuned_calibration_json(self, tmp_path, tmp_cache_dir):
        """Should load threshold from model directory's calibration.json."""
        model_dir = tmp_path / "my-finetuned-model"
        model_dir.mkdir()
        cal_file = model_dir / "calibration.json"
        cal_file.write_text(json.dumps({"threshold": 0.0083}))

        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        threshold = manager.get_threshold(
            str(model_dir), auto_calibrate=False
        )
        assert threshold == pytest.approx(0.0083)

    # ---- Fingerprint-based keying tests (v0.1.3) ----

    def test_fingerprint_lookup(self, tmp_cache_dir, mock_embedder, mock_cleaner):
        """Threshold stored with fingerprint key is retrieved via fingerprint."""
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        fp = create_pipeline_fingerprint("local", "mock-model", "mock")
        manager.set_threshold(fp, 0.35, save=True)

        threshold = manager.get_threshold(
            "mock-model",
            embedding_provider=mock_embedder,
            text_cleaner=mock_cleaner,
            auto_calibrate=False,
        )
        assert threshold == 0.35

    def test_same_model_different_cleaners(self, tmp_cache_dir):
        """Same model + different cleaners = different thresholds."""
        manager = ThresholdManager(cache_dir=tmp_cache_dir)

        fp_heuristic = create_pipeline_fingerprint("local", "my-model", "heuristic")
        fp_llm = create_pipeline_fingerprint("local", "my-model", "llm", "gpt-4o-mini")

        manager.set_threshold(fp_heuristic, 0.20, save=False)
        manager.set_threshold(fp_llm, 0.30, save=False)

        assert manager._custom_thresholds[fp_heuristic] == 0.20
        assert manager._custom_thresholds[fp_llm] == 0.30
        assert fp_heuristic != fp_llm

    def test_same_model_different_base_urls(self, tmp_cache_dir):
        """Same model on different providers = different fingerprints."""
        fp1 = create_pipeline_fingerprint("openai.com", "text-embedding-3-small", "heuristic")
        fp2 = create_pipeline_fingerprint("openrouter.ai", "text-embedding-3-small", "heuristic")

        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        manager.set_threshold(fp1, 0.26, save=False)
        manager.set_threshold(fp2, 0.28, save=False)

        assert manager._custom_thresholds[fp1] == 0.26
        assert manager._custom_thresholds[fp2] == 0.28

    def test_backward_compat_model_name_only(self, tmp_cache_dir):
        """get_threshold(model_name='all-MiniLM-L6-v2') still works via registry."""
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        threshold = manager.get_threshold("all-MiniLM-L6-v2", auto_calibrate=False)
        assert threshold == 0.23

    def test_cache_migration_from_old_format(self, tmp_cache_dir):
        """Old cache format is migrated: version marker added, keys preserved."""
        cache_file = tmp_cache_dir / "custom_thresholds.json"
        cache_file.write_text(json.dumps({"my-model": 0.42}))

        manager = ThresholdManager(cache_dir=tmp_cache_dir)

        # Old model-name key should be preserved as-is
        assert "my-model" in manager._custom_thresholds
        assert manager._custom_thresholds["my-model"] == 0.42
        # Model-name fallback path should find it
        assert manager.get_threshold("my-model", auto_calibrate=False) == 0.42

    def test_legacy_custom_key_not_used_for_non_heuristic_pipeline(self, tmp_cache_dir):
        """Legacy model-name custom thresholds should not apply to LLM pipelines."""
        from tests.conftest import MockEmbeddingProvider, MockTextCleaner

        cache_file = tmp_cache_dir / "custom_thresholds.json"
        cache_file.write_text(json.dumps({"all-MiniLM-L6-v2": 0.99}))
        manager = ThresholdManager(cache_dir=tmp_cache_dir)

        provider = MockEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        llm_cleaner = MockTextCleaner(method="llm")

        with pytest.raises(ValueError, match="No threshold found"):
            manager.get_threshold(
                "all-MiniLM-L6-v2",
                embedding_provider=provider,
                text_cleaner=llm_cleaner,
                auto_calibrate=False,
            )

    def test_legacy_custom_skip_logs_reason(self, tmp_cache_dir, caplog):
        """When legacy key is skipped for non-heuristic pipeline, reason is logged."""
        from tests.conftest import MockEmbeddingProvider, MockTextCleaner

        cache_file = tmp_cache_dir / "custom_thresholds.json"
        cache_file.write_text(json.dumps({"all-MiniLM-L6-v2": 0.99}))
        manager = ThresholdManager(cache_dir=tmp_cache_dir)

        provider = MockEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        llm_cleaner = MockTextCleaner(method="llm")

        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError, match="No threshold found"):
                manager.get_threshold(
                    "all-MiniLM-L6-v2",
                    embedding_provider=provider,
                    text_cleaner=llm_cleaner,
                    auto_calibrate=False,
                )

        assert "Ignoring legacy model-name custom threshold for non-heuristic pipeline" in caplog.text

    def test_migration_creates_backup(self, tmp_cache_dir):
        """Migration creates a .backup file with the original content."""
        cache_file = tmp_cache_dir / "custom_thresholds.json"
        original = {"my-model": 0.42}
        cache_file.write_text(json.dumps(original))

        ThresholdManager(cache_dir=tmp_cache_dir)

        backup = cache_file.with_suffix(".json.backup")
        assert backup.exists()
        assert json.loads(backup.read_text()) == original

    def test_fingerprint_with_registry_fallback(self, tmp_cache_dir, mock_cleaner):
        """When no fingerprint match, heuristic pipelines can use registry fallback."""
        from tests.conftest import MockEmbeddingProvider, MockTextCleaner

        provider = MockEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        heuristic_cleaner = MockTextCleaner(method="heuristic")
        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        threshold = manager.get_threshold(
            "all-MiniLM-L6-v2",
            embedding_provider=provider,
            text_cleaner=heuristic_cleaner,
            auto_calibrate=False,
        )
        # Should fall through fingerprint miss → registry hit
        assert threshold == 0.23

    def test_non_heuristic_pipeline_skips_registry_fallback(self, tmp_cache_dir):
        """Registry values are heuristic-calibrated and must not apply to LLM cleaners."""
        from tests.conftest import MockEmbeddingProvider, MockTextCleaner

        provider = MockEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        llm_cleaner = MockTextCleaner(method="llm")
        manager = ThresholdManager(cache_dir=tmp_cache_dir)

        with pytest.raises(ValueError, match="No threshold found"):
            manager.get_threshold(
                "all-MiniLM-L6-v2",
                embedding_provider=provider,
                text_cleaner=llm_cleaner,
                auto_calibrate=False,
            )

    def test_v2_cache_not_re_migrated(self, tmp_cache_dir):
        """A v2 cache file is loaded as-is, not re-migrated."""
        fp = create_pipeline_fingerprint("local", "test-model", "heuristic")
        cache_file = tmp_cache_dir / "custom_thresholds.json"
        cache_file.write_text(json.dumps({"_version": 2, fp: 0.55}))

        manager = ThresholdManager(cache_dir=tmp_cache_dir)
        assert manager._custom_thresholds[fp] == 0.55
        # No backup should be created
        assert not cache_file.with_suffix(".json.backup").exists()
