"""Threshold management for AgentShield."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from agentshield.threshold.registry import ThresholdRegistry

if TYPE_CHECKING:
    from agentshield.providers.base import EmbeddingProvider
    from agentshield.cleaning.base import TextCleaner

logger = logging.getLogger(__name__)


class ThresholdManager:
    """
    Manages detection thresholds for different embedding models.

    Threshold resolution order:
    1. Explicit default (if set in config)
    2. Custom (user-calibrated, stored in cache)
    3. Pre-calibrated registry
    4. Auto-calibrate on first use

    Usage:
        manager = ThresholdManager(cache_dir=Path.home() / ".agentshield")
        threshold = manager.get_threshold("all-MiniLM-L6-v2")
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_threshold: Optional[float] = None,
    ):
        """
        Initialize the threshold manager.

        Args:
            cache_dir: Directory for caching calibrated thresholds
            default_threshold: Override threshold for all models (if set)
        """
        self.cache_dir = cache_dir or Path.home() / ".agentshield" / "thresholds"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.default_threshold = default_threshold
        self._custom_thresholds: Dict[str, float] = {}

        # Load cached thresholds
        self._load_cached_thresholds()

    def _load_cached_thresholds(self) -> None:
        """Load all cached thresholds from disk."""
        cache_file = self.cache_dir / "custom_thresholds.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self._custom_thresholds = json.load(f)
                logger.debug(f"Loaded {len(self._custom_thresholds)} cached thresholds")
            except Exception as e:
                logger.warning(f"Failed to load cached thresholds: {e}")

    def _save_cached_thresholds(self) -> None:
        """Save custom thresholds to disk."""
        cache_file = self.cache_dir / "custom_thresholds.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self._custom_thresholds, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cached thresholds: {e}")

    def get_threshold(
        self,
        model_name: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        text_cleaner: Optional[TextCleaner] = None,
        auto_calibrate: bool = True,
        model_path: Optional[str] = None,
    ) -> float:
        """
        Get threshold for a model.

        Args:
            model_name: Name of the embedding model
            embedding_provider: Provider for auto-calibration (if needed)
            text_cleaner: Cleaner for auto-calibration (if needed)
            auto_calibrate: Whether to auto-calibrate if no threshold found
            model_path: Local path to finetuned model (for loading calibration.json)

        Returns:
            Threshold value

        Raises:
            ValueError: If no threshold found and auto_calibrate=False
        """
        # 1. Check explicit default
        if self.default_threshold is not None:
            return self.default_threshold

        # 2. Check custom (user-calibrated)
        if model_name in self._custom_thresholds:
            return self._custom_thresholds[model_name]

        # 3. Check for calibration.json in finetuned model directory
        finetuned_threshold = self._load_finetuned_calibration(model_name, model_path)
        if finetuned_threshold is not None:
            return finetuned_threshold

        # 4. Check registry
        registry_threshold = ThresholdRegistry.get(model_name)
        if registry_threshold is not None:
            return registry_threshold

        # 5. Auto-calibrate
        if auto_calibrate:
            if embedding_provider is None or text_cleaner is None:
                logger.warning(
                    f"No threshold for {model_name} and cannot auto-calibrate "
                    f"(missing embedding_provider or text_cleaner). Using default 0.20."
                )
                return 0.20

            logger.info(f"Auto-calibrating threshold for {model_name}...")
            threshold = self._auto_calibrate(model_name, embedding_provider, text_cleaner)
            return threshold

        raise ValueError(
            f"No threshold found for model '{model_name}'. "
            f"Run calibration with: agentshield calibrate --model {model_name}"
        )

    def _load_finetuned_calibration(
        self,
        model_name: str,
        model_path: Optional[str] = None,
    ) -> Optional[float]:
        """
        Load threshold from a finetuned model's calibration.json.

        The calibration.json is generated during finetuning and contains
        the GMM-calibrated threshold specific to that model.

        Args:
            model_name: HuggingFace Hub model ID or local path
            model_path: Explicit local path (overrides model_name)

        Returns:
            Threshold value or None if not found
        """
        paths_to_check = []

        # Check explicit model_path first
        if model_path:
            paths_to_check.append(Path(model_path))

        # Check if model_name is a local path
        if model_name and "/" in model_name:
            # Could be HuggingFace Hub ID, check local first
            local_path = Path(model_name)
            if local_path.exists():
                paths_to_check.append(local_path)

            # Check HuggingFace cache
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            # HF cache uses -- as separator
            safe_name = model_name.replace("/", "--")
            hf_model_path = hf_cache / f"models--{safe_name}"
            if hf_model_path.exists():
                # Find the snapshot directory
                snapshots = hf_model_path / "snapshots"
                if snapshots.exists():
                    # Get most recent snapshot
                    snapshot_dirs = list(snapshots.iterdir())
                    if snapshot_dirs:
                        paths_to_check.append(max(snapshot_dirs, key=lambda p: p.stat().st_mtime))

        # Check each path for calibration.json
        for path in paths_to_check:
            calibration_file = path / "calibration.json"
            if calibration_file.exists():
                try:
                    with open(calibration_file) as f:
                        data = json.load(f)
                    threshold = data.get("threshold")
                    if threshold is not None:
                        logger.info(
                            f"Loaded calibrated threshold {threshold:.4f} "
                            f"from {calibration_file}"
                        )
                        return float(threshold)
                except Exception as e:
                    logger.warning(f"Failed to load calibration from {calibration_file}: {e}")

        return None

    def _auto_calibrate(
        self,
        model_name: str,
        embedding_provider: EmbeddingProvider,
        text_cleaner: TextCleaner,
    ) -> float:
        """Auto-calibrate threshold for a model."""
        from agentshield.threshold.calibrator import ThresholdCalibrator

        calibrator = ThresholdCalibrator(
            embedding_provider=embedding_provider,
            text_cleaner=text_cleaner,
        )

        threshold = calibrator.calibrate()

        # Save to cache
        self.set_threshold(model_name, threshold, save=True)

        return threshold

    def set_threshold(
        self,
        model_name: str,
        threshold: float,
        save: bool = True,
    ) -> None:
        """
        Set a custom threshold for a model.

        Args:
            model_name: Name of the embedding model
            threshold: Threshold value
            save: Whether to persist to disk
        """
        self._custom_thresholds[model_name] = threshold

        if save:
            self._save_cached_thresholds()

        logger.info(f"Set threshold for {model_name}: {threshold:.6f}")

    def has_threshold(self, model_name: str) -> bool:
        """Check if a threshold exists for a model."""
        if self.default_threshold is not None:
            return True
        if model_name in self._custom_thresholds:
            return True
        return ThresholdRegistry.has(model_name)

    def list_thresholds(self) -> Dict[str, float]:
        """List all available thresholds."""
        thresholds = dict(ThresholdRegistry.THRESHOLDS)
        thresholds.update(self._custom_thresholds)
        return thresholds

    def clear_custom_thresholds(self) -> None:
        """Clear all custom thresholds."""
        self._custom_thresholds.clear()
        cache_file = self.cache_dir / "custom_thresholds.json"
        if cache_file.exists():
            cache_file.unlink()
