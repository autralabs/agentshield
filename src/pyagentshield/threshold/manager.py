"""Threshold management for AgentShield."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from pyagentshield.threshold.fingerprint import (
    create_pipeline_fingerprint,
    extract_host,
)
from pyagentshield.threshold.registry import ThresholdRegistry

if TYPE_CHECKING:
    from pyagentshield.providers.base import EmbeddingProvider
    from pyagentshield.cleaning.base import TextCleaner

logger = logging.getLogger(__name__)

# Cache format version — triggers migration from older formats.
_CACHE_VERSION = 2


class ThresholdManager:
    """
    Manages detection thresholds for different embedding models.

    Threshold resolution order:
    1. Explicit default (if set in config)
    2. Custom (user-calibrated, stored in cache) — fingerprint key first, then model-name key
    3. Finetuned calibration.json in model directory
    4. Pre-calibrated registry (model-name only — registry was calibrated with heuristic cleaner)
    5. Auto-calibrate on first use

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
        self._lock = threading.Lock()
        # Per-scan thread-local override set by resolve_with_mode()
        # so get_threshold() returns the cloud-resolved value transparently.
        self._thread_local = threading.local()

        # Load cached thresholds (migrating if needed)
        self._load_cached_thresholds()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_cached_thresholds(self) -> None:
        """Load all cached thresholds from disk, migrating old format if needed."""
        cache_file = self.cache_dir / "custom_thresholds.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                if data.get("_version") != _CACHE_VERSION:
                    data = self._migrate_legacy_cache(data, cache_file)
                # Strip internal keys
                self._custom_thresholds = {
                    k: v for k, v in data.items() if not k.startswith("_")
                }
                logger.debug(f"Loaded {len(self._custom_thresholds)} cached thresholds")
            except Exception as e:
                logger.warning(f"Failed to load cached thresholds: {e}")

    def _migrate_legacy_cache(
        self, data: Dict, cache_file: Path
    ) -> Dict:
        """Migrate v1 cache to v2 by adding a version marker.

        Old format::

            {"all-MiniLM-L6-v2": 0.23}

        New format — keeps old model-name keys intact so they remain
        reachable via the legacy model-name fallback path (step 2b) for
        backward-compatible calls (no cleaner provided) and heuristic
        pipelines. We cannot know which provider/cleaner was used when the
        entry was created, so renaming keys would risk making them
        unreachable at runtime::

            {"_version": 2, "all-MiniLM-L6-v2": 0.23}

        A ``.backup`` copy of the original file is created before overwriting.
        """
        backup = cache_file.with_suffix(".json.backup")
        shutil.copy2(cache_file, backup)
        logger.info(f"Backed up legacy threshold cache to {backup}")

        migrated: Dict = {"_version": _CACHE_VERSION}
        for key, value in data.items():
            if key.startswith("_"):
                continue
            # Keep original model-name key as-is — the legacy model-name
            # fallback in get_threshold (step 2b) can still use it for
            # backward-compatible calls and heuristic pipelines.
            migrated[key] = value

        self._save_data(migrated)
        return migrated

    def _save_cached_thresholds(self) -> None:
        """Save custom thresholds to disk (thread-safe)."""
        data: Dict = {"_version": _CACHE_VERSION}
        data.update(self._custom_thresholds)
        self._save_data(data)

    def _save_data(self, data: Dict) -> None:
        """Atomic-ish write of threshold data to disk."""
        cache_file = self.cache_dir / "custom_thresholds.json"
        tmp_file = cache_file.with_suffix(".json.tmp")
        try:
            with self._lock:
                with open(tmp_file, "w") as f:
                    json.dump(data, f, indent=2)
                tmp_file.replace(cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cached thresholds: {e}")
            if tmp_file.exists():
                tmp_file.unlink()

    # ------------------------------------------------------------------
    # Fingerprint helpers
    # ------------------------------------------------------------------

    def _build_fingerprint(
        self,
        model_name: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        text_cleaner: Optional[TextCleaner] = None,
        provider_host: Optional[str] = None,
    ) -> Optional[str]:
        """Build a fingerprint if enough information is available."""
        if text_cleaner is None:
            return None

        host = provider_host
        if host is None:
            base_url = getattr(embedding_provider, "_base_url", None)
            provider_type = getattr(embedding_provider, "_provider_type", None)
            if provider_type is None:
                # Infer from class name
                cls_name = type(embedding_provider).__name__.lower() if embedding_provider else ""
                if "openai" in cls_name:
                    provider_type = "openai"
                elif "mlx" in cls_name:
                    provider_type = "mlx"
                elif "local" in cls_name:
                    provider_type = "local"
                elif cls_name:
                    # Preserve provider identity for unknown provider classes
                    # instead of collapsing to "local", which can cause key collisions.
                    provider_type = f"class:{cls_name}"
                    logger.debug(
                        "Unknown embedding provider type for %s; using class-derived host key %s",
                        type(embedding_provider).__name__,
                        provider_type,
                    )
                else:
                    provider_type = "local"
            host = extract_host(base_url, provider_type)

        cleaning_model: Optional[str] = None
        cleaning_host: Optional[str] = None
        method = text_cleaner.method

        # Build cleaner identities for methods that materially change drift
        # distributions (LLM backend/model and finetuned model source).
        cleaner_identities = []

        def _append_cleaner_identity(cleaner: TextCleaner) -> None:
            c_method = cleaner.method
            if c_method == "llm":
                default_host = extract_host(None, "openai")
                m = getattr(cleaner, "_model", None) or "unknown"
                bu = getattr(cleaner, "_base_url", None)
                h = extract_host(bu, "openai") if bu else default_host
                cleaner_identities.append(f"{m}@{h}")
            elif c_method == "finetuned":
                model_path = getattr(cleaner, "_model_path", None)
                model_id = getattr(cleaner, "_model_id", None)
                source = str(model_path) if model_path else model_id or "unknown"
                cleaner_identities.append(f"finetuned:{source}")

        if method in {"llm", "finetuned"}:
            _append_cleaner_identity(text_cleaner)
        else:
            nested = getattr(text_cleaner, "cleaners", None)
            if nested:
                for cleaner in nested:
                    _append_cleaner_identity(cleaner)

        if cleaner_identities:
            cleaning_model = "+".join(cleaner_identities)

        return create_pipeline_fingerprint(
            provider_host=host,
            embedding_model=model_name,
            cleaning_method=method,
            cleaning_model=cleaning_model,
            cleaning_host=cleaning_host,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_threshold(
        self,
        model_name: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        text_cleaner: Optional[TextCleaner] = None,
        provider_host: Optional[str] = None,
        auto_calibrate: bool = True,
        model_path: Optional[str] = None,
    ) -> float:
        """
        Get threshold for a pipeline configuration.

        When ``text_cleaner`` (and optionally ``provider_host``) are provided,
        a fingerprint key is used for lookup/storage.  When only ``model_name``
        is given, the legacy model-name-only path is used for backward
        compatibility.

        Args:
            model_name: Name of the embedding model
            embedding_provider: Provider for auto-calibration (if needed)
            text_cleaner: Cleaner for auto-calibration and fingerprinting
            provider_host: Explicit provider host for fingerprinting
            auto_calibrate: Whether to auto-calibrate if no threshold found
            model_path: Local path to finetuned model (for loading calibration.json)

        Returns:
            Threshold value

        Raises:
            ValueError: If no threshold found and auto_calibrate=False
        """
        # 0. Per-scan thread-local override (set by resolve_with_mode before detect())
        _tl = getattr(self, "_thread_local", None)
        override = getattr(_tl, "_scan_threshold_override", None) if _tl is not None else None
        if override is not None:
            return override

        # 1. Check explicit default
        if self.default_threshold is not None:
            return self.default_threshold

        # Build fingerprint if we can
        fingerprint = self._build_fingerprint(
            model_name, embedding_provider, text_cleaner, provider_host
        )

        # 2. Check custom cache — fingerprint key first
        if fingerprint and fingerprint in self._custom_thresholds:
            return self._custom_thresholds[fingerprint]

        # 2b. Legacy fallback: model-name-only custom cache key.
        # Restricted to backward-compatible calls / heuristic-equivalent
        # pipelines so old thresholds don't leak across cleaner types.
        if model_name in self._custom_thresholds:
            if self._legacy_custom_fallback_allowed(text_cleaner):
                return self._custom_thresholds[model_name]
            logger.info(
                "Ignoring legacy model-name custom threshold for non-heuristic pipeline: %s",
                model_name,
            )

        # 3. Check for calibration.json in finetuned model directory
        finetuned_threshold = self._load_finetuned_calibration(model_name, model_path)
        if finetuned_threshold is not None:
            return finetuned_threshold

        # 4. Check registry (heuristic pipelines only — registry is heuristic-calibrated)
        if self._registry_fallback_allowed(text_cleaner):
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
            threshold = self._auto_calibrate(
                model_name, embedding_provider, text_cleaner, provider_host
            )
            return threshold

        raise ValueError(
            f"No threshold found for model '{model_name}'. "
            f"Run calibration with: agentshield calibrate --model {model_name}"
        )

    @staticmethod
    def _registry_fallback_allowed(text_cleaner: Optional[TextCleaner]) -> bool:
        """Allow model-name registry fallback only for heuristic-equivalent pipelines."""
        if text_cleaner is None:
            return True

        method = text_cleaner.method
        if method == "heuristic":
            return True

        nested = getattr(text_cleaner, "cleaners", None)
        if not nested:
            return False

        # Hybrid cleaner is heuristic-equivalent only if all nested cleaners are heuristic.
        return all(getattr(cleaner, "method", None) == "heuristic" for cleaner in nested)

    @staticmethod
    def _legacy_custom_fallback_allowed(text_cleaner: Optional[TextCleaner]) -> bool:
        """Allow legacy model-name custom-key fallback under the same safety rules as registry fallback."""
        return ThresholdManager._registry_fallback_allowed(text_cleaner)

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
        provider_host: Optional[str] = None,
    ) -> float:
        """Auto-calibrate threshold for a model."""
        from pyagentshield.threshold.calibrator import ThresholdCalibrator

        calibrator = ThresholdCalibrator(
            embedding_provider=embedding_provider,
            text_cleaner=text_cleaner,
        )

        threshold = calibrator.calibrate()

        # Save with fingerprint key
        fingerprint = self._build_fingerprint(
            model_name, embedding_provider, text_cleaner, provider_host
        )
        key = fingerprint or model_name
        self.set_threshold(key, threshold, save=True)

        return threshold

    def set_threshold(
        self,
        model_name: str,
        threshold: float,
        save: bool = True,
    ) -> None:
        """
        Set a custom threshold.

        The ``model_name`` argument can be either a plain model name
        (legacy) or a full fingerprint string.

        Args:
            model_name: Model name or pipeline fingerprint
            threshold: Threshold value
            save: Whether to persist to disk
        """
        self._custom_thresholds[model_name] = threshold

        if save:
            self._save_cached_thresholds()

        logger.info(f"Set threshold for {model_name}: {threshold:.6f}")

    def has_threshold(
        self,
        model_name: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        text_cleaner: Optional[TextCleaner] = None,
        provider_host: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        """Check if a threshold exists for a model/pipeline without auto-calibrating."""
        if self.default_threshold is not None:
            return True

        fingerprint = self._build_fingerprint(
            model_name, embedding_provider, text_cleaner, provider_host
        )
        if fingerprint and fingerprint in self._custom_thresholds:
            return True

        if (
            model_name in self._custom_thresholds
            and self._legacy_custom_fallback_allowed(text_cleaner)
        ):
            return True

        if self._load_finetuned_calibration(model_name, model_path) is not None:
            return True

        if self._registry_fallback_allowed(text_cleaner):
            return ThresholdRegistry.has(model_name)

        return False

    def list_thresholds(self) -> Dict[str, float]:
        """List all available thresholds."""
        thresholds = dict(ThresholdRegistry.THRESHOLDS)
        thresholds.update(self._custom_thresholds)
        return thresholds

    # ------------------------------------------------------------------
    # Cloud-aware threshold resolution
    # ------------------------------------------------------------------

    def _resolve_local_with_source(
        self,
        fingerprint: Optional[str],
        model_name: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        text_cleaner: Optional[TextCleaner] = None,
        provider_host: Optional[str] = None,
        allow_auto_calibrate: bool = True,
        model_path: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str], bool]:
        """
        Resolve the local threshold chain and return (value, source, is_auto_calibrated).

        source is one of the canonical tokens:
          'local_pinned' | 'local_cache' | 'finetuned' | 'registry' | 'auto_calibrated'

        Returns (None, None, False) when nothing found and allow_auto_calibrate=False.
        """
        # 1. Explicit pin — always wins
        if self.default_threshold is not None:
            return self.default_threshold, "local_pinned", False

        # 2. Custom cache — fingerprint key first
        if fingerprint and fingerprint in self._custom_thresholds:
            return self._custom_thresholds[fingerprint], "local_cache", False

        # 2b. Legacy model-name cache key (heuristic-equivalent pipelines only)
        if model_name in self._custom_thresholds:
            if self._legacy_custom_fallback_allowed(text_cleaner):
                return self._custom_thresholds[model_name], "local_cache", False

        # 3. Finetuned model's calibration.json
        finetuned_threshold = self._load_finetuned_calibration(model_name, model_path)
        if finetuned_threshold is not None:
            return finetuned_threshold, "finetuned", False

        # 4. Pre-calibrated registry (heuristic-equivalent pipelines only)
        if self._registry_fallback_allowed(text_cleaner):
            registry_threshold = ThresholdRegistry.get(model_name)
            if registry_threshold is not None:
                return registry_threshold, "registry", False

        # 5. Auto-calibrate (optional)
        if allow_auto_calibrate:
            if embedding_provider is not None and text_cleaner is not None:
                try:
                    threshold = self._auto_calibrate(
                        model_name, embedding_provider, text_cleaner, provider_host
                    )
                    return threshold, "auto_calibrated", True
                except Exception as e:
                    logger.warning("Auto-calibration failed: %s. Using fallback 0.20.", e)
            else:
                logger.warning(
                    "No threshold for %s, cannot auto-calibrate (missing provider/cleaner). "
                    "Using default 0.20.",
                    model_name,
                )
            return 0.20, "auto_calibrated", True

        return None, None, False

    def resolve_with_mode(
        self,
        fingerprint: str,
        cloud_client: Optional[object] = None,
        environment: Optional[str] = None,
        cloud_resolution: Optional[object] = None,
        model_name: str = "",
        embedding_provider: Optional[EmbeddingProvider] = None,
        text_cleaner: Optional[TextCleaner] = None,
        provider_host: Optional[str] = None,
        auto_calibrate: bool = True,
        model_path: Optional[str] = None,
    ) -> "ThresholdDecision":
        """
        Resolve threshold with cloud awareness and return a ThresholdDecision.

        If cloud_resolution is provided it is used directly (no HTTP call).
        If cloud_client is provided and cloud_resolution is None, fetches the
        cloud resolution first. This lets shield fetch once and reuse the result
        for both settings application and mode resolution.

        The resolved decision.value is stored in a thread-local so the
        detector's get_threshold() call picks it up transparently in the
        same scan. Call _clear_threshold_override() after detect() returns.

        Args:
            fingerprint: Full pipeline fingerprint (already computed by shield)
            cloud_client: Optional CloudThresholdClient; used only if cloud_resolution is None
            environment: User-supplied environment string (nullable)
            cloud_resolution: Pre-fetched CloudResolution (skips HTTP when provided)
            model_name / embedding_provider / text_cleaner / ...: passed to local chain
        """
        from pyagentshield.threshold.decision import ThresholdDecision

        # --- Fetch cloud resolution (if not already provided) ---
        if cloud_resolution is None and cloud_client is not None:
            cloud_resolution = cloud_client.get_resolution(fingerprint, environment)

        mode: str = (
            cloud_resolution.resolution_mode if cloud_resolution else "local_only"
        )
        fail_open: bool = (
            cloud_resolution.fail_open_to_local if cloud_resolution else True
        )
        cloud_rule = cloud_resolution.matched_rule if cloud_resolution else None
        cloud_threshold_value: Optional[float] = (
            cloud_rule.threshold_value if cloud_rule else None
        )
        cloud_rule_id: Optional[str] = cloud_rule.id if cloud_rule else None
        cloud_rule_version: Optional[int] = cloud_rule.version if cloud_rule else None

        # --- Apply mode logic ---
        decision: ThresholdDecision

        if mode == "local_only":
            value, source, _ = self._resolve_local_with_source(
                fingerprint, model_name, embedding_provider, text_cleaner,
                provider_host, allow_auto_calibrate=auto_calibrate, model_path=model_path,
            )
            if value is None:
                value, source = 0.20, "auto_calibrated"
            decision = ThresholdDecision(
                value=value, source=source, mode=mode, fingerprint=fingerprint
            )

        elif mode == "local_prefer":
            # First try non-auto sources
            value, source, is_auto = self._resolve_local_with_source(
                fingerprint, model_name, embedding_provider, text_cleaner,
                provider_host, allow_auto_calibrate=False, model_path=model_path,
            )
            if value is not None:
                # Strong local signal — use it
                decision = ThresholdDecision(
                    value=value, source=source, mode=mode, fingerprint=fingerprint,
                    cloud_threshold=cloud_threshold_value,
                )
            elif cloud_threshold_value is not None:
                # Only auto-calibrate would give a local value; cloud has a rule → use cloud
                decision = ThresholdDecision(
                    value=cloud_threshold_value, source="cloud_manual", mode=mode,
                    fingerprint=fingerprint, cloud_rule_id=cloud_rule_id,
                    cloud_rule_version=cloud_rule_version, cloud_threshold=cloud_threshold_value,
                )
            else:
                # No cloud rule → fall back to auto-calibrate
                value, source, _ = self._resolve_local_with_source(
                    fingerprint, model_name, embedding_provider, text_cleaner,
                    provider_host, allow_auto_calibrate=True, model_path=model_path,
                )
                if value is None:
                    value, source = 0.20, "auto_calibrated"
                decision = ThresholdDecision(
                    value=value, source=source, mode=mode, fingerprint=fingerprint
                )

        elif mode == "cloud_prefer":
            if cloud_threshold_value is not None:
                decision = ThresholdDecision(
                    value=cloud_threshold_value, source="cloud_manual", mode=mode,
                    fingerprint=fingerprint, cloud_rule_id=cloud_rule_id,
                    cloud_rule_version=cloud_rule_version, cloud_threshold=cloud_threshold_value,
                )
            else:
                value, source, _ = self._resolve_local_with_source(
                    fingerprint, model_name, embedding_provider, text_cleaner,
                    provider_host, allow_auto_calibrate=auto_calibrate, model_path=model_path,
                )
                if value is None:
                    value, source = 0.20, "auto_calibrated"
                decision = ThresholdDecision(
                    value=value, source=source, mode=mode, fingerprint=fingerprint
                )

        elif mode == "cloud_only":
            if cloud_threshold_value is not None:
                decision = ThresholdDecision(
                    value=cloud_threshold_value, source="cloud_manual", mode=mode,
                    fingerprint=fingerprint, cloud_rule_id=cloud_rule_id,
                    cloud_rule_version=cloud_rule_version, cloud_threshold=cloud_threshold_value,
                )
            elif fail_open:
                value, source, _ = self._resolve_local_with_source(
                    fingerprint, model_name, embedding_provider, text_cleaner,
                    provider_host, allow_auto_calibrate=auto_calibrate, model_path=model_path,
                )
                if value is None:
                    value, source = 0.20, "auto_calibrated"
                decision = ThresholdDecision(
                    value=value, source="local_failopen", mode=mode, fingerprint=fingerprint
                )
            else:
                from pyagentshield.core.exceptions import ThresholdUnavailableError
                raise ThresholdUnavailableError(
                    f"No cloud threshold rule found for '{fingerprint}' "
                    "and fail_open=False. Set fail_open=True or add a cloud rule."
                )

        elif mode == "observe":
            # Use local chain; never apply cloud value; expose cloud_threshold for diff
            value, source, _ = self._resolve_local_with_source(
                fingerprint, model_name, embedding_provider, text_cleaner,
                provider_host, allow_auto_calibrate=auto_calibrate, model_path=model_path,
            )
            if value is None:
                value, source = 0.20, "auto_calibrated"
            decision = ThresholdDecision(
                value=value, source=source, mode=mode, fingerprint=fingerprint,
                cloud_threshold=cloud_threshold_value,
                cloud_rule_id=cloud_rule_id,
                cloud_rule_version=cloud_rule_version,
            )

        else:
            # Unknown mode — safe fallback to local_only
            logger.warning("Unknown threshold_resolution_mode '%s'; falling back to local_only", mode)
            value, source, _ = self._resolve_local_with_source(
                fingerprint, model_name, embedding_provider, text_cleaner,
                provider_host, allow_auto_calibrate=auto_calibrate, model_path=model_path,
            )
            if value is None:
                value, source = 0.20, "auto_calibrated"
            decision = ThresholdDecision(
                value=value, source=source, mode="local_only", fingerprint=fingerprint
            )

        # Store as thread-local so get_threshold() transparently returns decision.value
        # within the same scan. Shield must call _clear_threshold_override() after detect().
        _tl = getattr(self, "_thread_local", None)
        if _tl is not None:
            _tl._scan_threshold_override = decision.value
        return decision

    def _clear_threshold_override(self) -> None:
        """Clear the per-scan thread-local threshold override. Call after detect() returns."""
        _tl = getattr(self, "_thread_local", None)
        if _tl is not None:
            _tl._scan_threshold_override = None

    def clear_custom_thresholds(self) -> None:
        """Clear all custom thresholds."""
        self._custom_thresholds.clear()
        cache_file = self.cache_dir / "custom_thresholds.json"
        if cache_file.exists():
            cache_file.unlink()
