"""Hybrid text cleaner combining multiple cleaning strategies."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from pyagentshield.core.exceptions import CleaningError

if TYPE_CHECKING:
    from pyagentshield.cleaning.base import TextCleaner

logger = logging.getLogger(__name__)


class HybridMode(str, Enum):
    """
    Modes for combining multiple cleaners.

    SEQUENTIAL: Run cleaners in order, each on the output of the previous.
                Best for: heuristic -> finetuned (pre-filter then refine)

    PARALLEL_VOTE: Run all cleaners independently, vote on changes.
                   Best for: multiple methods to achieve consensus.

    FALLBACK: Try primary cleaner, fall back to secondary on error.
              Best for: finetuned with heuristic backup.

    BEST_DRIFT: Run all cleaners, use the one producing maximum drift.
                Best for: detecting injections (higher drift = more aggressive cleaning)

    LEAST_DRIFT: Run all cleaners, use the one producing minimum drift.
                 Best for: preserving content (gentler cleaning)
    """

    SEQUENTIAL = "sequential"
    PARALLEL_VOTE = "parallel_vote"
    FALLBACK = "fallback"
    BEST_DRIFT = "best_drift"
    LEAST_DRIFT = "least_drift"


class HybridCleaner:
    """
    Combines multiple text cleaning strategies.

    This cleaner implements the Strategy pattern, allowing flexible
    combination of different cleaning approaches:

    - Heuristic (fast, free, regex-based)
    - Finetuned (accurate, local model)
    - LLM (most accurate, API-based)

    Example configurations:

    1. Sequential (heuristic pre-filter, then finetuned):
        HybridCleaner(
            cleaners=[heuristic, finetuned],
            mode=HybridMode.SEQUENTIAL
        )

    2. Fallback (finetuned with heuristic backup):
        HybridCleaner(
            cleaners=[finetuned, heuristic],
            mode=HybridMode.FALLBACK
        )

    3. Consensus (agree on changes):
        HybridCleaner(
            cleaners=[heuristic, finetuned],
            mode=HybridMode.PARALLEL_VOTE,
            vote_threshold=0.5
        )
    """

    def __init__(
        self,
        cleaners: Sequence[TextCleaner],
        mode: HybridMode = HybridMode.SEQUENTIAL,
        vote_threshold: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the hybrid cleaner.

        Args:
            cleaners: List of cleaner instances to combine.
            mode: How to combine cleaner outputs.
            vote_threshold: For PARALLEL_VOTE, minimum agreement ratio.
            weights: Optional weights for each cleaner by method name.
                    Default is equal weight for all.
        """
        if not cleaners:
            raise CleaningError("HybridCleaner requires at least one cleaner")

        self._cleaners = list(cleaners)
        self._mode = mode
        self._vote_threshold = vote_threshold
        self._weights = weights or {}

        # Build method name
        method_names = [c.method for c in self._cleaners]
        self._method = f"hybrid[{mode.value}]({'+'.join(method_names)})"

    @property
    def method(self) -> str:
        """Get the cleaning method name."""
        return self._method

    @property
    def cleaners(self) -> List[TextCleaner]:
        """Get the list of cleaners."""
        return self._cleaners

    @property
    def mode(self) -> HybridMode:
        """Get the combination mode."""
        return self._mode

    def clean(self, text: str) -> str:
        """
        Clean text using the configured strategy.

        Args:
            text: Original text that may contain injections

        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return text

        if self._mode == HybridMode.SEQUENTIAL:
            return self._clean_sequential(text)
        elif self._mode == HybridMode.PARALLEL_VOTE:
            return self._clean_parallel_vote(text)
        elif self._mode == HybridMode.FALLBACK:
            return self._clean_fallback(text)
        elif self._mode == HybridMode.BEST_DRIFT:
            return self._clean_best_drift(text)
        elif self._mode == HybridMode.LEAST_DRIFT:
            return self._clean_least_drift(text)
        else:
            raise CleaningError(f"Unknown hybrid mode: {self._mode}")

    def _clean_sequential(self, text: str) -> str:
        """
        Run cleaners sequentially, each on output of previous.

        Flow: text -> cleaner1 -> result1 -> cleaner2 -> result2 -> ...
        """
        result = text
        for cleaner in self._cleaners:
            try:
                result = cleaner.clean(result)
            except Exception as e:
                logger.warning(
                    f"Cleaner {cleaner.method} failed in sequential mode: {e}"
                )
                # Continue with current result
        return result

    def _clean_fallback(self, text: str) -> str:
        """
        Try cleaners in order until one succeeds.

        Flow: Try cleaner1, if fails try cleaner2, ...
        """
        last_error: Optional[Exception] = None

        for cleaner in self._cleaners:
            try:
                return cleaner.clean(text)
            except Exception as e:
                logger.warning(f"Cleaner {cleaner.method} failed, trying fallback: {e}")
                last_error = e

        # All failed, raise the last error
        if last_error:
            raise CleaningError(f"All cleaners failed. Last error: {last_error}")
        return text

    def _clean_parallel_vote(self, text: str) -> str:
        """
        Run all cleaners in parallel and vote on the result.

        Each cleaner votes on whether content should be kept.
        Content is kept if vote ratio >= threshold.
        """
        results: List[str] = []
        weights: List[float] = []

        for cleaner in self._cleaners:
            try:
                result = cleaner.clean(text)
                results.append(result)
                weights.append(self._weights.get(cleaner.method, 1.0))
            except Exception as e:
                logger.warning(f"Cleaner {cleaner.method} failed in parallel mode: {e}")

        if not results:
            logger.warning("All cleaners failed in parallel mode, returning original")
            return text

        # Simple voting: use the result that appears most often
        # weighted by cleaner weights
        result_votes: Dict[str, float] = {}
        for result, weight in zip(results, weights):
            if result in result_votes:
                result_votes[result] += weight
            else:
                result_votes[result] = weight

        # Find result with highest weight
        best_result = max(result_votes.keys(), key=lambda r: result_votes[r])
        total_weight = sum(weights)
        vote_ratio = result_votes[best_result] / total_weight if total_weight > 0 else 0

        if vote_ratio >= self._vote_threshold:
            return best_result

        # Below threshold: return the result with most cleaning (shortest)
        # This is more conservative (removes more potential injections)
        return min(results, key=len)

    def _clean_best_drift(self, text: str) -> str:
        """
        Run all cleaners and return the one with maximum drift from original.

        Best for detection: higher drift = more aggressive cleaning
        = more potential injection content removed.
        """
        return self._clean_by_drift(text, maximize=True)

    def _clean_least_drift(self, text: str) -> str:
        """
        Run all cleaners and return the one with minimum drift from original.

        Best for content preservation: lower drift = gentler cleaning.
        """
        return self._clean_by_drift(text, maximize=False)

    def _clean_by_drift(self, text: str, maximize: bool) -> str:
        """Run all cleaners and select by drift magnitude."""
        results: List[tuple[str, float]] = []

        for cleaner in self._cleaners:
            try:
                result = cleaner.clean(text)
                drift = self._compute_simple_drift(text, result)
                results.append((result, drift))
            except Exception as e:
                logger.warning(f"Cleaner {cleaner.method} failed: {e}")

        if not results:
            logger.warning("All cleaners failed, returning original")
            return text

        # Sort by drift
        results.sort(key=lambda x: x[1], reverse=maximize)
        return results[0][0]

    @staticmethod
    def _compute_simple_drift(original: str, cleaned: str) -> float:
        """
        Compute a simple text-based drift measure.

        Uses character-level Jaccard distance as a proxy for semantic drift.
        For actual semantic drift, use embedding-based comparison.
        """
        if not original:
            return 0.0 if not cleaned else 1.0

        # Normalize
        orig_chars = set(original.lower())
        clean_chars = set(cleaned.lower())

        # Jaccard distance
        intersection = len(orig_chars & clean_chars)
        union = len(orig_chars | clean_chars)

        if union == 0:
            return 0.0

        similarity = intersection / union
        drift = 1.0 - similarity

        return drift

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean multiple texts."""
        return [self.clean(text) for text in texts]


def create_hybrid_cleaner(
    methods: List[str],
    mode: str = "sequential",
    **kwargs: Any,
) -> HybridCleaner:
    """
    Factory function to create a HybridCleaner from method names.

    Args:
        methods: List of method names ("heuristic", "finetuned", "llm")
        mode: Combination mode ("sequential", "fallback", etc.)
        **kwargs: Additional arguments passed to cleaner constructors

    Returns:
        Configured HybridCleaner instance

    Example:
        cleaner = create_hybrid_cleaner(
            methods=["heuristic", "finetuned"],
            mode="sequential",
            model_id="pyagentshield/cleaner-phi2-lora",
        )
    """
    from pyagentshield.cleaning.heuristic import HeuristicCleaner

    cleaners: List[TextCleaner] = []

    for method in methods:
        if method == "heuristic":
            cleaners.append(HeuristicCleaner())

        elif method == "finetuned":
            from pyagentshield.cleaning.finetuned import FinetunedCleaner

            finetuned_kwargs = {
                k: v for k, v in kwargs.items()
                if k in [
                    "model_id", "model_path", "base_model", "use_lora",
                    "device", "load_in_4bit", "load_in_8bit",
                    "max_new_tokens", "temperature",
                ]
            }
            cleaners.append(FinetunedCleaner(**finetuned_kwargs))

        elif method == "llm":
            from pyagentshield.cleaning.llm import LLMCleaner

            llm_kwargs = {
                k: v for k, v in kwargs.items()
                if k in [
                    "model", "api_key", "temperature",
                    "base_url", "default_headers",
                ]
            }
            # Rename 'model' to avoid conflict with finetuned
            if "llm_model" in kwargs:
                llm_kwargs["model"] = kwargs["llm_model"]
            # Allow llm-prefixed overrides for when both embed and clean
            # base_urls differ
            if "llm_base_url" in kwargs:
                llm_kwargs["base_url"] = kwargs["llm_base_url"]
            if "llm_api_key" in kwargs:
                llm_kwargs["api_key"] = kwargs["llm_api_key"]
            if "llm_default_headers" in kwargs:
                llm_kwargs["default_headers"] = kwargs["llm_default_headers"]
            cleaners.append(LLMCleaner(**llm_kwargs))

        else:
            raise CleaningError(f"Unknown cleaning method: {method}")

    return HybridCleaner(
        cleaners=cleaners,
        mode=HybridMode(mode),
        vote_threshold=kwargs.get("vote_threshold", 0.5),
        weights=kwargs.get("weights"),
    )
