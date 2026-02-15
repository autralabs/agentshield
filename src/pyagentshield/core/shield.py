"""Main AgentShield class - orchestrates detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pyagentshield.core.config import ShieldConfig, get_cache_dir
from pyagentshield.core.results import ScanResult, DetectionSignal, ScanDetails

if TYPE_CHECKING:
    from pyagentshield.detectors.base import BaseDetector
    from pyagentshield.providers.base import EmbeddingProvider
    from pyagentshield.cleaning.base import TextCleaner
    from pyagentshield.threshold.manager import ThresholdManager

logger = logging.getLogger(__name__)


class AgentShield:
    """
    Main entry point for AgentShield functionality.

    Orchestrates the detection pipeline:
    1. Clean text (heuristic, finetuned, LLM, or hybrid)
    2. Generate embeddings
    3. Run ZEDD detection
    4. Return results

    Usage:
        >>> shield = AgentShield()
        >>> result = shield.scan("some document text")
        >>> if result.is_suspicious:
        ...     print(f"Detected: {result.details.summary}")

    With custom config:
        >>> shield = AgentShield(config={"zedd": {"threshold": 0.2}})
        >>> shield = AgentShield(config="path/to/config.yaml")

    With finetuned cleaner:
        >>> shield = AgentShield(config={
        ...     "cleaning": {
        ...         "method": "finetuned",
        ...         "finetuned": {"model_id": "pyagentshield/cleaner-phi2-lora"}
        ...     }
        ... })

    With hybrid cleaner:
        >>> shield = AgentShield(config={
        ...     "cleaning": {
        ...         "method": "hybrid",
        ...         "hybrid": {
        ...             "methods": ["heuristic", "finetuned"],
        ...             "mode": "sequential"
        ...         }
        ...     }
        ... })
    """

    def __init__(
        self,
        config: Optional[Union[ShieldConfig, Dict[str, Any], str, Path]] = None,
    ):
        """
        Initialize AgentShield.

        Args:
            config: Configuration (ShieldConfig, dict, path to YAML, or None for defaults)
        """
        self.config = ShieldConfig.load(config)
        self._setup_logging()

        # Telemetry client (eager — just an if-check when disabled)
        from pyagentshield.telemetry.client import create_telemetry_client
        self._telemetry = create_telemetry_client(self.config.telemetry)
        self._session_id = __import__("uuid").uuid4().hex

        # Lazy-initialized components
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._text_cleaner: Optional[TextCleaner] = None
        self._threshold_manager: Optional[ThresholdManager] = None
        self._detector: Optional[BaseDetector] = None

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        level = getattr(logging, self.config.logging.level)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            if self.config.logging.format == "text"
            else '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
        )

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = self._create_embedding_provider()
        return self._embedding_provider

    @property
    def text_cleaner(self) -> TextCleaner:
        """Get or create text cleaner."""
        if self._text_cleaner is None:
            self._text_cleaner = self._create_text_cleaner()
        return self._text_cleaner

    @property
    def threshold_manager(self) -> ThresholdManager:
        """Get or create threshold manager."""
        if self._threshold_manager is None:
            self._threshold_manager = self._create_threshold_manager()
        return self._threshold_manager

    @property
    def detector(self) -> BaseDetector:
        """Get or create ZEDD detector."""
        if self._detector is None:
            self._detector = self._create_detector()
        return self._detector

    def _create_embedding_provider(self) -> EmbeddingProvider:
        """Create embedding provider based on config."""
        if self.config.embeddings.provider == "local":
            from pyagentshield.providers.local import LocalEmbeddingProvider

            return LocalEmbeddingProvider(
                model_name=self.config.embeddings.model,
                cache_embeddings=self.config.performance.cache_embeddings,
            )
        elif self.config.embeddings.provider == "openai":
            from pyagentshield.providers.openai import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(
                model_name=self.config.embeddings.openai_model,
                api_key=self.config.embeddings.api_key,
                cache_embeddings=self.config.performance.cache_embeddings,
                base_url=self.config.embeddings.base_url,
                default_headers=self.config.embeddings.default_headers,
            )
        elif self.config.embeddings.provider == "mlx":
            from pyagentshield.providers.mlx import MLXEmbeddingProvider

            return MLXEmbeddingProvider(
                model_name=self.config.embeddings.model,
                cache_embeddings=self.config.performance.cache_embeddings,
                cache_dir=self.config.embeddings.mlx_cache_dir,
                convert_from_hf=self.config.embeddings.mlx_convert_from_hf,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embeddings.provider}")

    def _create_text_cleaner(self) -> TextCleaner:
        """Create text cleaner based on config."""
        method = self.config.cleaning.method

        if method == "heuristic":
            from pyagentshield.cleaning.heuristic import HeuristicCleaner
            return HeuristicCleaner()

        elif method == "llm":
            from pyagentshield.cleaning.llm import LLMCleaner
            return LLMCleaner(
                model=self.config.cleaning.llm_model,
                api_key=self.config.cleaning.api_key,
                base_url=self.config.cleaning.base_url,
                default_headers=self.config.cleaning.default_headers,
            )

        elif method == "finetuned":
            from pyagentshield.cleaning.finetuned import FinetunedCleaner

            ft_config = self.config.cleaning.finetuned
            return FinetunedCleaner(
                model_id=ft_config.model_id,
                model_path=ft_config.model_path,
                base_model=ft_config.base_model,
                use_lora=ft_config.use_lora,
                device=ft_config.device,
                load_in_4bit=ft_config.load_in_4bit,
                load_in_8bit=ft_config.load_in_8bit,
                max_new_tokens=ft_config.max_new_tokens,
                temperature=ft_config.temperature,
                torch_dtype=ft_config.torch_dtype,
            )

        elif method == "hybrid":
            from pyagentshield.cleaning.hybrid import create_hybrid_cleaner

            hybrid_config = self.config.cleaning.hybrid
            ft_config = self.config.cleaning.finetuned

            return create_hybrid_cleaner(
                methods=hybrid_config.methods,
                mode=hybrid_config.mode,
                vote_threshold=hybrid_config.vote_threshold,
                weights=hybrid_config.weights,
                # Pass finetuned config for any finetuned cleaners in hybrid
                model_id=ft_config.model_id,
                model_path=ft_config.model_path,
                base_model=ft_config.base_model,
                use_lora=ft_config.use_lora,
                device=ft_config.device,
                load_in_4bit=ft_config.load_in_4bit,
                load_in_8bit=ft_config.load_in_8bit,
                max_new_tokens=ft_config.max_new_tokens,
                temperature=ft_config.temperature,
                # Pass LLM config for any LLM cleaners in hybrid
                llm_model=self.config.cleaning.llm_model,
                api_key=self.config.cleaning.api_key,
                base_url=self.config.cleaning.base_url,
                default_headers=self.config.cleaning.default_headers,
            )

        else:
            raise ValueError(f"Unknown cleaning method: {method}")

    def _create_threshold_manager(self) -> ThresholdManager:
        """Create threshold manager."""
        from pyagentshield.threshold.manager import ThresholdManager

        return ThresholdManager(
            cache_dir=get_cache_dir(self.config),
            default_threshold=self.config.zedd.threshold,
        )

    def _create_detector(self) -> BaseDetector:
        """Create ZEDD detector."""
        from pyagentshield.detectors.zedd import ZEDDDetector

        return ZEDDDetector(
            embedding_provider=self.embedding_provider,
            text_cleaner=self.text_cleaner,
            threshold_manager=self.threshold_manager,
        )

    def scan(self, text: Union[str, List[str]]) -> Union[ScanResult, List[ScanResult]]:
        """
        Scan text for prompt injections.

        Args:
            text: Single text string or list of texts to scan

        Returns:
            ScanResult for single text, or list of ScanResults for multiple texts
        """
        if isinstance(text, str):
            return self._scan_single(text)
        return [self._scan_single(t) for t in text]

    def _scan_single(self, text: str) -> ScanResult:
        """Scan a single text."""
        from pyagentshield.detectors.base import DetectionContext

        # Create detection context
        context = DetectionContext(
            config=self.config,
            cleaned_text=None,  # Will be computed by detector if needed
        )

        # Run detection
        signal = self.detector.detect(text, context)

        # Build result
        is_suspicious = signal.score >= self.config.behavior.confidence_threshold

        # Build details
        drift_score = signal.metadata.get("drift")
        threshold = signal.metadata.get("threshold")
        cleaning_method = signal.metadata.get("cleaning_method", self.config.cleaning.method)

        if is_suspicious:
            summary = f"Potential injection detected (drift={drift_score:.4f}, threshold={threshold:.4f})"
            risk_factors = ["Embedding drift exceeds threshold"]
        else:
            summary = "No injection detected"
            risk_factors = []

        details = ScanDetails(
            summary=summary,
            risk_factors=risk_factors,
            drift_score=drift_score,
            threshold=threshold,
            cleaning_method=cleaning_method,
        )

        result = ScanResult(
            is_suspicious=is_suspicious,
            confidence=signal.score,
            signals={"zedd": signal},
            details=details,
        )

        # Record telemetry event (no-op if disabled)
        try:
            from pyagentshield.telemetry.events import ScanEvent
            import pyagentshield

            event = ScanEvent(
                sdk_version=pyagentshield.__version__,
                session_id=self._session_id,
                is_suspicious=is_suspicious,
                confidence=signal.score,
                drift_score=drift_score,
                threshold=threshold,
                embedding_model=self.config.embeddings.model,
                cleaning_method=cleaning_method,
                on_detect=self.config.behavior.on_detect,
                project=self.config.telemetry.project,
                environment=self.config.telemetry.environment,
            )
            self._telemetry.record(event)
        except Exception:
            logger.debug("Failed to record telemetry", exc_info=True)

        return result

    def calibrate(
        self,
        corpus: Optional[List[str]] = None,
        save: bool = True,
    ) -> float:
        """
        Calibrate detection threshold for the current embedding model.

        Args:
            corpus: Optional list of clean texts to use for calibration.
                   If None, uses built-in calibration dataset.
            save: Whether to save the calibrated threshold to cache.

        Returns:
            The calibrated threshold value.
        """
        from pyagentshield.threshold.calibrator import ThresholdCalibrator

        calibrator = ThresholdCalibrator(
            embedding_provider=self.embedding_provider,
            text_cleaner=self.text_cleaner,
        )

        threshold = calibrator.calibrate(corpus)

        if save:
            self.threshold_manager.set_threshold(
                self.embedding_provider.model_name,
                threshold,
                save=True,
            )

        return threshold

    def get_cleaner_info(self) -> Dict[str, Any]:
        """
        Get information about the current cleaner configuration.

        Returns:
            Dictionary with cleaner details
        """
        cleaner = self.text_cleaner
        return {
            "method": cleaner.method,
            "config": {
                "primary_method": self.config.cleaning.method,
                "llm_model": self.config.cleaning.llm_model,
                "finetuned": self.config.cleaning.finetuned.model_dump(),
                "hybrid": self.config.cleaning.hybrid.model_dump(),
            },
        }
