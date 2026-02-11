"""Tests for result dataclasses."""

from agentshield.core.results import DetectionSignal, ScanDetails, ScanResult


class TestDetectionSignal:
    def test_basic_creation(self):
        signal = DetectionSignal(score=0.8, confidence=0.9)
        assert signal.score == 0.8
        assert signal.confidence == 0.9
        assert signal.metadata == {}

    def test_score_clamped_to_0_1(self):
        signal = DetectionSignal(score=1.5, confidence=-0.3)
        assert signal.score == 1.0
        assert signal.confidence == 0.0

    def test_metadata(self):
        signal = DetectionSignal(
            score=0.5, confidence=0.7, metadata={"drift": 0.12, "model": "test"}
        )
        assert signal.metadata["drift"] == 0.12

    def test_to_dict(self):
        signal = DetectionSignal(score=0.8123, confidence=0.9456)
        d = signal.to_dict()
        assert d["score"] == 0.8123
        assert d["confidence"] == 0.9456
        assert "metadata" in d


class TestScanDetails:
    def test_basic_creation(self):
        details = ScanDetails(summary="No injection detected")
        assert details.summary == "No injection detected"
        assert details.risk_factors == []
        assert details.drift_score is None

    def test_with_all_fields(self):
        details = ScanDetails(
            summary="Injection detected",
            risk_factors=["High drift"],
            drift_score=0.45,
            threshold=0.20,
            cleaning_method="heuristic",
        )
        assert details.drift_score == 0.45
        assert details.cleaning_method == "heuristic"

    def test_to_dict(self):
        details = ScanDetails(
            summary="test", drift_score=0.123456789, threshold=0.2
        )
        d = details.to_dict()
        assert d["drift_score"] == 0.123457  # rounded to 6 places
        assert d["summary"] == "test"


class TestScanResult:
    def test_basic_creation(self):
        result = ScanResult(is_suspicious=False, confidence=0.1)
        assert not result.is_suspicious
        assert result.confidence == 0.1

    def test_bool_returns_is_suspicious(self):
        safe = ScanResult(is_suspicious=False, confidence=0.0)
        suspicious = ScanResult(is_suspicious=True, confidence=0.9)
        assert not bool(safe)
        assert bool(suspicious)

    def test_confidence_clamped(self):
        result = ScanResult(is_suspicious=False, confidence=2.0)
        assert result.confidence == 1.0

    def test_safe_factory(self):
        result = ScanResult.safe(confidence=0.05)
        assert not result.is_suspicious
        assert result.confidence == 0.05

    def test_suspicious_factory(self):
        result = ScanResult.suspicious(
            confidence=0.9,
            summary="Potential injection",
            risk_factors=["High drift"],
            drift_score=0.45,
            threshold=0.20,
        )
        assert result.is_suspicious
        assert result.confidence == 0.9
        assert result.details.drift_score == 0.45

    def test_to_dict(self):
        result = ScanResult(
            is_suspicious=True,
            confidence=0.85,
            signals={"zedd": DetectionSignal(score=0.85, confidence=0.9)},
        )
        d = result.to_dict()
        assert d["is_suspicious"] is True
        assert d["confidence"] == 0.85
        assert "zedd" in d["signals"]
