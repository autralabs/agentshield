"""
ZEDD (Zero-Shot Embedding Drift Detection) implementation.

This module implements the ZEDD paper: arXiv:2601.12359v1
"Zero-Shot Embedding Drift Detection: A Lightweight Defense
Against Prompt Injections in LLMs"

The core insight is that adversarial prompts subtly shift the semantic
representation of inputs in the embedding space, even when the surface
text appears clean. By measuring this "drift" between original and
cleaned versions, we can detect injection attempts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from numpy.typing import NDArray

from pyagentshield.core.results import DetectionSignal
from pyagentshield.detectors.base import BaseDetector, DetectionContext

if TYPE_CHECKING:
    from pyagentshield.cleaning.base import TextCleaner
    from pyagentshield.providers.base import EmbeddingProvider
    from pyagentshield.threshold.manager import ThresholdManager

logger = logging.getLogger(__name__)


class ZEDDDetector:
    """
    Zero-Shot Embedding Drift Detection.

    Implements the ZEDD algorithm from arXiv:2601.12359v1.

    Algorithm overview:
    1. For input text x, generate embedding f(x)
    2. Clean x to remove potential injections, producing x'
    3. Generate embedding f(x')
    4. Compute drift: Drift(x, x') = 1 - cos_sim(f(x), f(x'))
    5. If drift > threshold, flag as suspicious

    The key insight is that injected content creates semantic drift
    when removed, while clean content produces minimal drift.
    """

    name: str = "zedd"

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        text_cleaner: TextCleaner,
        threshold_manager: ThresholdManager,
    ):
        """
        Initialize the ZEDD detector.

        Args:
            embedding_provider: Provider for generating embeddings
            text_cleaner: Cleaner for producing cleaned versions
            threshold_manager: Manager for threshold lookup/calibration
        """
        self.embeddings = embedding_provider
        self.cleaner = text_cleaner
        self.thresholds = threshold_manager

        # Cache for embeddings (keyed by text hash)
        self._embedding_cache: Dict[int, NDArray[np.floating]] = {}

    def detect(self, text: str, context: DetectionContext) -> DetectionSignal:
        """
        Detect prompt injection by measuring embedding drift.

        Implements the core ZEDD algorithm:
        1. Get cleaned version of text
        2. Compute embeddings for both versions
        3. Calculate cosine drift
        4. Compare against calibrated threshold
        5. Return normalized score

        Args:
            text: Original text to analyze
            context: Detection context with config and optional pre-computed data

        Returns:
            DetectionSignal with score, confidence, and metadata
        """
        # Step 1: Get or compute cleaned version
        cleaned = context.cleaned_text
        if cleaned is None:
            cleaned = self.cleaner.clean(text)

        # Step 2: Compute embeddings
        emb_original = self._get_embedding(text)
        emb_cleaned = self._get_embedding(cleaned)

        # Step 3: Compute drift (Equation 1 from paper)
        drift = self._compute_cosine_drift(emb_original, emb_cleaned)

        # Step 4: Get calibrated threshold
        threshold = self.thresholds.get_threshold(
            self.embeddings.model_name,
            embedding_provider=self.embeddings,
            text_cleaner=self.cleaner,
        )

        # Step 5: Convert drift to normalized score (0-1)
        score = self._drift_to_score(drift, threshold)

        # Step 6: Compute confidence
        confidence = self._compute_confidence(drift, threshold)

        return DetectionSignal(
            score=score,
            confidence=confidence,
            metadata={
                "drift": round(float(drift), 6),
                "threshold": round(float(threshold), 6),
                "model": self.embeddings.model_name,
                "cleaning_method": self.cleaner.method,
            },
        )

    def detect_batch(
        self,
        texts: List[str],
        context: DetectionContext,
    ) -> List[DetectionSignal]:
        """
        Batch detection for efficiency.

        Processes multiple texts in batches to minimize embedding calls.

        Args:
            texts: List of texts to analyze
            context: Detection context

        Returns:
            List of DetectionSignals, one per input
        """
        if not texts:
            return []

        # Get or compute cleaned versions
        if context.cleaned_texts:
            cleaned_texts = context.cleaned_texts
        else:
            cleaned_texts = self.cleaner.clean_batch(texts)

        # Batch embed all texts
        all_texts = texts + cleaned_texts
        all_embeddings = self.embeddings.encode_batch(all_texts)

        n = len(texts)
        emb_originals = all_embeddings[:n]
        emb_cleaned = all_embeddings[n:]

        # Get threshold
        threshold = self.thresholds.get_threshold(
            self.embeddings.model_name,
            embedding_provider=self.embeddings,
            text_cleaner=self.cleaner,
        )

        # Compute drift and scores for each pair
        signals = []
        for i in range(n):
            drift = self._compute_cosine_drift(emb_originals[i], emb_cleaned[i])
            score = self._drift_to_score(drift, threshold)
            confidence = self._compute_confidence(drift, threshold)

            signals.append(DetectionSignal(
                score=score,
                confidence=confidence,
                metadata={
                    "drift": round(float(drift), 6),
                    "threshold": round(float(threshold), 6),
                    "model": self.embeddings.model_name,
                    "cleaning_method": self.cleaner.method,
                },
            ))

        return signals

    def _compute_cosine_drift(
        self,
        emb_a: NDArray[np.floating],
        emb_b: NDArray[np.floating],
    ) -> float:
        """
        Compute cosine drift between two embeddings.

        Implements Equation (1) from the ZEDD paper:

            Drift(x, x') = 1 - (f(x) · f(x')) / (||f(x)|| · ||f(x')||)

        Where:
            - f(x) is the embedding of the original text
            - f(x') is the embedding of the cleaned text
            - · denotes dot product
            - ||·|| denotes L2 norm

        The drift score interpretation:
            - 0.0: Identical embeddings (no drift)
            - 1.0: Orthogonal embeddings (maximum meaningful drift)
            - 2.0: Opposite embeddings (rare in practice)

        For prompt injection detection:
            - Low drift (< threshold): Cleaning removed little content → likely clean
            - High drift (>= threshold): Cleaning removed significant content → likely injected

        Args:
            emb_a: Embedding of original text, shape (dimensions,)
            emb_b: Embedding of cleaned text, shape (dimensions,)

        Returns:
            Drift score in range [0, 2]
        """
        # Compute L2 norms
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        # Handle zero vectors (empty or whitespace-only text)
        if norm_a == 0 or norm_b == 0:
            # If one is zero and other is not, maximum uncertainty
            if norm_a != norm_b:
                return 1.0
            # Both zero: no drift
            return 0.0

        # Normalize to unit vectors
        unit_a = emb_a / norm_a
        unit_b = emb_b / norm_b

        # Cosine similarity = dot product of unit vectors
        cosine_sim = float(np.dot(unit_a, unit_b))

        # Clamp to valid range (handle floating point errors)
        cosine_sim = max(-1.0, min(1.0, cosine_sim))

        # Drift = 1 - similarity
        drift = 1.0 - cosine_sim

        return drift

    def _drift_to_score(self, drift: float, threshold: float) -> float:
        """
        Convert raw drift to normalized 0-1 score.

        Uses a sigmoid-like mapping centered at the threshold:
            - drift = 0 → score ≈ 0 (very safe)
            - drift = threshold → score = 0.5 (borderline)
            - drift = 2*threshold → score ≈ 1 (very suspicious)

        The steepness is calibrated so that:
            - 95% of score range is within [0, 2*threshold]
            - Score changes most rapidly near threshold

        Args:
            drift: Raw drift value
            threshold: Calibrated threshold

        Returns:
            Normalized score in [0, 1]
        """
        if threshold <= 0:
            # Edge case: invalid threshold
            return 1.0 if drift > 0 else 0.0

        # Ratio of drift to threshold
        ratio = drift / threshold

        # Sigmoid mapping: score = 1 / (1 + exp(-k*(ratio - 1)))
        # k=4 gives good sensitivity around threshold
        k = 4.0
        score = 1.0 / (1.0 + np.exp(-k * (ratio - 1.0)))

        return float(np.clip(score, 0.0, 1.0))

    def _compute_confidence(self, drift: float, threshold: float) -> float:
        """
        Compute confidence based on distance from threshold.

        Confidence interpretation:
            - High confidence: Drift is far from threshold (clear decision)
            - Low confidence: Drift is near threshold (borderline case)

        The formula:
            confidence = 1 - exp(-2 * |drift - threshold| / threshold)

        This gives:
            - At threshold (drift = threshold): confidence ≈ 0
            - Far from threshold: confidence → 1

        Args:
            drift: Raw drift value
            threshold: Calibrated threshold

        Returns:
            Confidence in [0, 1]
        """
        if threshold <= 0:
            return 1.0

        # Relative distance from threshold
        relative_distance = abs(drift - threshold) / threshold

        # Exponential confidence curve
        confidence = 1.0 - np.exp(-2.0 * relative_distance)

        return float(np.clip(confidence, 0.0, 1.0))

    def _get_embedding(self, text: str) -> NDArray[np.floating]:
        """Get embedding with caching."""
        cache_key = hash(text)

        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.embeddings.encode(text)

        return self._embedding_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
