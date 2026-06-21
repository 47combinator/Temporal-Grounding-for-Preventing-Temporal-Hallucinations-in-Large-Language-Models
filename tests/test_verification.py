"""
Tests for the verification and hallucination detection modules.

These tests verify the temporal verification logic, consistency checking,
corruption strategies, and hallucination detection using mock data
(no Neo4j connection required).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verification.verification_result import (
    TemporalClaim,
    TemporalFact,
    VerificationResult,
    VerificationStatus,
)
from src.verification.consistency_checker import ConsistencyChecker
from src.hallucination.detector import HallucinationDetector


class TestVerificationResult:
    """Tests for the data classes."""

    def test_temporal_claim_creation(self) -> None:
        """Test creating a TemporalClaim."""
        claim = TemporalClaim(
            subject="United States",
            predicate="Make statement",
            object="Russia",
            timestamp="2014-03-15",
        )
        assert claim.subject == "United States"
        assert claim.timestamp == "2014-03-15"
        assert claim.source_sentence == ""

    def test_temporal_fact_creation(self) -> None:
        """Test creating a TemporalFact."""
        fact = TemporalFact(
            subject="United States",
            predicate="Make statement",
            object="Russia",
            timestamp="2014-03-15",
            event_id="icews14-train-42",
        )
        assert fact.event_id == "icews14-train-42"

    def test_verification_result_creation(self) -> None:
        """Test creating a VerificationResult."""
        claim = TemporalClaim("A", "rel", "B", "2014-01-01")
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.SUPPORTED,
            confidence=1.0,
            evidence=[],
            explanation="Fully supported.",
            checks={},
        )
        assert result.status == VerificationStatus.SUPPORTED
        assert result.confidence == 1.0


class TestConsistencyChecker:
    """Tests for the ConsistencyChecker class."""

    @pytest.fixture
    def checker(self) -> ConsistencyChecker:
        """Create a ConsistencyChecker with a mock graph store."""
        mock_store = MagicMock()
        return ConsistencyChecker(mock_store)

    def test_check_ordering_a_before_b(self, checker: ConsistencyChecker) -> None:
        """Test ordering check when A is before B."""
        fact_a = TemporalFact("A", "rel", "B", "2014-01-01")
        fact_b = TemporalFact("A", "rel", "C", "2014-06-15")

        result = checker.check_ordering(fact_a, fact_b)

        assert result["a_before_b"] is True
        assert result["b_before_a"] is False
        assert result["simultaneous"] is False
        assert result["distance_days"] > 0

    def test_check_ordering_simultaneous(self, checker: ConsistencyChecker) -> None:
        """Test ordering check when events are on the same day."""
        fact_a = TemporalFact("A", "rel", "B", "2014-03-15")
        fact_b = TemporalFact("C", "rel", "D", "2014-03-15")

        result = checker.check_ordering(fact_a, fact_b)

        assert result["simultaneous"] is True
        assert result["distance_days"] == 0

    def test_chain_consistency_valid(self, checker: ConsistencyChecker) -> None:
        """Test chain consistency with a valid chronological chain."""
        events = [
            TemporalFact("A", "rel", "B", "2014-01-01"),
            TemporalFact("A", "rel", "C", "2014-03-15"),
            TemporalFact("A", "rel", "D", "2014-07-20"),
        ]

        result = checker.check_chain_consistency(events)

        assert result["consistent"] is True
        assert len(result["violations"]) == 0

    def test_chain_consistency_invalid(self, checker: ConsistencyChecker) -> None:
        """Test chain consistency with a violation."""
        events = [
            TemporalFact("A", "rel", "B", "2014-07-20"),
            TemporalFact("A", "rel", "C", "2014-03-15"),
            TemporalFact("A", "rel", "D", "2014-01-01"),
        ]

        result = checker.check_chain_consistency(events)

        assert result["consistent"] is False
        assert len(result["violations"]) > 0

    def test_compute_tcs_all_supported(self, checker: ConsistencyChecker) -> None:
        """Test TCS when all claims are supported."""
        claim = TemporalClaim("A", "rel", "B", "2014-01-01")
        results = [
            VerificationResult(
                claim=claim,
                status=VerificationStatus.SUPPORTED,
                confidence=1.0,
                evidence=[],
                explanation="",
                checks={},
            )
            for _ in range(5)
        ]
        tcs = checker.compute_temporal_consistency_score(results)
        assert tcs == 1.0

    def test_compute_tcs_none_supported(self, checker: ConsistencyChecker) -> None:
        """Test TCS when no claims are supported."""
        claim = TemporalClaim("A", "rel", "B", "2014-01-01")
        results = [
            VerificationResult(
                claim=claim,
                status=VerificationStatus.UNSUPPORTED,
                confidence=0.3,
                evidence=[],
                explanation="",
                checks={},
            )
            for _ in range(3)
        ]
        tcs = checker.compute_temporal_consistency_score(results)
        assert tcs == 0.0

    def test_compute_tcs_empty(self, checker: ConsistencyChecker) -> None:
        """Test TCS with an empty list."""
        tcs = checker.compute_temporal_consistency_score([])
        assert tcs == 0.0


class TestHallucinationDetector:
    """Tests for the HallucinationDetector class."""

    @pytest.fixture
    def detector(self) -> HallucinationDetector:
        """Create a HallucinationDetector."""
        return HallucinationDetector()

    def _make_result(
        self, status: VerificationStatus, confidence: float = 1.0
    ) -> VerificationResult:
        """Helper to create a VerificationResult."""
        claim = TemporalClaim("A", "rel", "B", "2014-01-01")
        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=[],
            explanation="",
            checks={},
        )

    def test_detect_clean(self, detector: HallucinationDetector) -> None:
        """Test detection when all claims are supported."""
        results = [
            self._make_result(VerificationStatus.SUPPORTED)
            for _ in range(3)
        ]
        report = detector.detect(results)
        assert report["classification"] == "clean"

    def test_detect_hallucination(self, detector: HallucinationDetector) -> None:
        """Test detection when a claim is temporally impossible."""
        results = [
            self._make_result(VerificationStatus.SUPPORTED),
            self._make_result(VerificationStatus.TEMPORALLY_IMPOSSIBLE, 0.0),
        ]
        report = detector.detect(results)
        assert report["classification"] == "contains_hallucination"
        assert len(report["hallucinated_claims"]) >= 1

    def test_detect_unverifiable(self, detector: HallucinationDetector) -> None:
        """Test detection when claims are unsupported."""
        results = [
            self._make_result(VerificationStatus.UNSUPPORTED, 0.3),
            self._make_result(VerificationStatus.SUPPORTED),
        ]
        report = detector.detect(results)
        assert report["classification"] in (
            "contains_hallucination", "contains_unverifiable"
        )

    def test_hallucination_score_zero(self, detector: HallucinationDetector) -> None:
        """Test hallucination score when everything is supported."""
        results = [
            self._make_result(VerificationStatus.SUPPORTED)
            for _ in range(5)
        ]
        score = detector.compute_hallucination_score(results)
        assert score == 0.0

    def test_hallucination_score_one(self, detector: HallucinationDetector) -> None:
        """Test hallucination score when everything is unsupported."""
        results = [
            self._make_result(VerificationStatus.UNSUPPORTED, 0.3)
            for _ in range(5)
        ]
        score = detector.compute_hallucination_score(results)
        assert score == 1.0
