"""Tests for threshold registry and manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
