"""Tests for the hybrid text cleaner."""

from __future__ import annotations

import pytest

from agentshield.cleaning.hybrid import HybridCleaner, HybridMode
from agentshield.core.exceptions import CleaningError


class FakeCleaner:
    """Simple fake cleaner for testing."""

    def __init__(self, method: str, transform=None, should_fail=False):
        self._method = method
        self._transform = transform or (lambda x: x)
        self._should_fail = should_fail

    @property
    def method(self) -> str:
        return self._method

    def clean(self, text: str) -> str:
        if self._should_fail:
            raise CleaningError(f"{self._method} failed")
        return self._transform(text)

    def clean_batch(self, texts):
        return [self.clean(t) for t in texts]


class TestHybridCleanerInit:
    def test_empty_cleaners_raises(self):
        with pytest.raises(CleaningError):
            HybridCleaner(cleaners=[])

    def test_method_name_format(self):
        c1 = FakeCleaner("heuristic")
        c2 = FakeCleaner("llm")
        hybrid = HybridCleaner(cleaners=[c1, c2], mode=HybridMode.SEQUENTIAL)
        assert "sequential" in hybrid.method
        assert "heuristic" in hybrid.method
        assert "llm" in hybrid.method


class TestSequentialMode:
    def test_chains_cleaners(self):
        c1 = FakeCleaner("first", lambda x: x.replace("bad", ""))
        c2 = FakeCleaner("second", lambda x: x.replace("evil", ""))
        hybrid = HybridCleaner(cleaners=[c1, c2], mode=HybridMode.SEQUENTIAL)

        result = hybrid.clean("bad and evil text")
        assert "bad" not in result
        assert "evil" not in result
        assert "text" in result

    def test_order_matters(self):
        c1 = FakeCleaner("upper", lambda x: x.upper())
        c2 = FakeCleaner("strip", lambda x: x.strip())
        hybrid = HybridCleaner(cleaners=[c1, c2], mode=HybridMode.SEQUENTIAL)

        result = hybrid.clean("  hello  ")
        assert result == "HELLO"

    def test_failure_continues(self):
        c1 = FakeCleaner("failing", should_fail=True)
        c2 = FakeCleaner("working", lambda x: x + " cleaned")
        hybrid = HybridCleaner(cleaners=[c1, c2], mode=HybridMode.SEQUENTIAL)

        result = hybrid.clean("text")
        assert "cleaned" in result


class TestFallbackMode:
    def test_uses_first_success(self):
        c1 = FakeCleaner("primary", lambda x: "primary: " + x)
        c2 = FakeCleaner("backup", lambda x: "backup: " + x)
        hybrid = HybridCleaner(cleaners=[c1, c2], mode=HybridMode.FALLBACK)

        result = hybrid.clean("text")
        assert result.startswith("primary:")

    def test_falls_back_on_failure(self):
        c1 = FakeCleaner("failing", should_fail=True)
        c2 = FakeCleaner("backup", lambda x: "backup: " + x)
        hybrid = HybridCleaner(cleaners=[c1, c2], mode=HybridMode.FALLBACK)

        result = hybrid.clean("text")
        assert result.startswith("backup:")

    def test_all_fail_raises(self):
        c1 = FakeCleaner("fail1", should_fail=True)
        c2 = FakeCleaner("fail2", should_fail=True)
        hybrid = HybridCleaner(cleaners=[c1, c2], mode=HybridMode.FALLBACK)

        with pytest.raises(CleaningError, match="All cleaners failed"):
            hybrid.clean("text")


class TestParallelVoteMode:
    def test_majority_wins(self):
        c1 = FakeCleaner("a", lambda x: "cleaned")
        c2 = FakeCleaner("b", lambda x: "cleaned")
        c3 = FakeCleaner("c", lambda x: "different")
        hybrid = HybridCleaner(
            cleaners=[c1, c2, c3], mode=HybridMode.PARALLEL_VOTE
        )

        result = hybrid.clean("text")
        assert result == "cleaned"

    def test_all_same(self):
        c1 = FakeCleaner("a", lambda x: "same")
        c2 = FakeCleaner("b", lambda x: "same")
        hybrid = HybridCleaner(
            cleaners=[c1, c2], mode=HybridMode.PARALLEL_VOTE
        )

        result = hybrid.clean("text")
        assert result == "same"


class TestDriftModes:
    def test_best_drift_picks_most_changed(self):
        c1 = FakeCleaner("gentle", lambda x: x)  # No change
        c2 = FakeCleaner("aggressive", lambda x: "")  # Total removal
        hybrid = HybridCleaner(
            cleaners=[c1, c2], mode=HybridMode.BEST_DRIFT
        )

        result = hybrid.clean("some text with content")
        assert result == ""  # aggressive cleaner wins

    def test_least_drift_picks_least_changed(self):
        c1 = FakeCleaner("gentle", lambda x: x)  # No change
        c2 = FakeCleaner("aggressive", lambda x: "")  # Total removal
        hybrid = HybridCleaner(
            cleaners=[c1, c2], mode=HybridMode.LEAST_DRIFT
        )

        result = hybrid.clean("some text")
        assert result == "some text"  # gentle cleaner wins


class TestEdgeCases:
    def test_empty_string(self):
        c1 = FakeCleaner("test")
        hybrid = HybridCleaner(cleaners=[c1], mode=HybridMode.SEQUENTIAL)
        assert hybrid.clean("") == ""

    def test_whitespace_only(self):
        c1 = FakeCleaner("test")
        hybrid = HybridCleaner(cleaners=[c1], mode=HybridMode.SEQUENTIAL)
        assert hybrid.clean("   ") == "   "

    def test_batch_clean(self):
        c1 = FakeCleaner("test", lambda x: x.upper())
        hybrid = HybridCleaner(cleaners=[c1], mode=HybridMode.SEQUENTIAL)
        results = hybrid.clean_batch(["hello", "world"])
        assert results == ["HELLO", "WORLD"]

    def test_simple_drift_computation(self):
        drift = HybridCleaner._compute_simple_drift("hello", "hello")
        assert drift == 0.0

        drift = HybridCleaner._compute_simple_drift("abc", "xyz")
        assert drift > 0.0

        drift = HybridCleaner._compute_simple_drift("", "")
        assert drift == 0.0
