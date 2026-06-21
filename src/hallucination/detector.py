"""
Hallucination detector.

HallucinationDetector consumes a batch of :class:`VerificationResult` objects
produced by the verification pipeline and classifies the overall LLM response
as clean, containing hallucinations, or containing unverifiable claims.  It
also provides per-claim classification with hallucination type tagging and
computes an aggregate hallucination score.
"""

from __future__ import annotations

from typing import Any

from src.verification.verification_result import (
    VerificationResult,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Hallucination-type labels
# ---------------------------------------------------------------------------

_HALLUCINATION_TYPE_MAP: dict[VerificationStatus, str | None] = {
    VerificationStatus.SUPPORTED: None,
    VerificationStatus.PARTIALLY_SUPPORTED: None,
    VerificationStatus.UNSUPPORTED: "factual_hallucination",
    VerificationStatus.TEMPORALLY_IMPOSSIBLE: "temporal_impossibility",
    VerificationStatus.CANNOT_VERIFY: None,
}


class HallucinationDetector:
    """Classify LLM outputs based on temporal verification results.

    The detector does not require a knowledge-graph connection; it operates
    purely on the :class:`VerificationResult` objects emitted by the
    upstream verification pipeline.
    """

    def __init__(self) -> None:
        """Initialise the detector (no external dependencies required)."""

    # ------------------------------------------------------------------
    # Single-claim classification
    # ------------------------------------------------------------------

    def classify_claim(self, result: VerificationResult) -> dict[str, Any]:
        """Classify a single verification result as hallucinated or not.

        A claim is considered a hallucination if its status is
        ``UNSUPPORTED`` or ``TEMPORALLY_IMPOSSIBLE``.

        Parameters
        ----------
        result : VerificationResult
            The verification outcome for one claim.

        Returns
        -------
        dict
            ``{"is_hallucination": bool, "hallucination_type": str | None,
            "confidence": float}``
        """
        is_hallucination = result.status in (
            VerificationStatus.UNSUPPORTED,
            VerificationStatus.TEMPORALLY_IMPOSSIBLE,
        )

        hallucination_type = _HALLUCINATION_TYPE_MAP.get(result.status)

        return {
            "is_hallucination": is_hallucination,
            "hallucination_type": hallucination_type,
            "confidence": result.confidence,
        }

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect(
        self, verification_results: list[VerificationResult]
    ) -> dict[str, Any]:
        """Classify an entire LLM response based on its verification results.

        Parameters
        ----------
        verification_results : list[VerificationResult]
            All verification results for the claims in the response.

        Returns
        -------
        dict
            ``{"classification": str, "hallucinated_claims": list[VerificationResult],
            "tcs": float, "per_claim": list[dict]}``

            ``classification`` is one of:

            * ``"clean"`` -- all claims are SUPPORTED or PARTIALLY_SUPPORTED.
            * ``"contains_hallucination"`` -- at least one claim is UNSUPPORTED
              or TEMPORALLY_IMPOSSIBLE.
            * ``"contains_unverifiable"`` -- no hallucinations detected but at
              least one claim is CANNOT_VERIFY.
        """
        per_claim: list[dict[str, Any]] = []
        hallucinated_claims: list[VerificationResult] = []

        has_hallucination = False
        has_unverifiable = False

        for result in verification_results:
            claim_classification = self.classify_claim(result)
            per_claim.append(
                {
                    "claim_subject": result.claim.subject,
                    "claim_predicate": result.claim.predicate,
                    "claim_object": result.claim.object,
                    "claim_timestamp": result.claim.timestamp,
                    "status": result.status.value,
                    **claim_classification,
                }
            )

            if claim_classification["is_hallucination"]:
                hallucinated_claims.append(result)
                has_hallucination = True

            if result.status == VerificationStatus.CANNOT_VERIFY:
                has_unverifiable = True

        if has_hallucination:
            classification = "contains_hallucination"
        elif has_unverifiable:
            classification = "contains_unverifiable"
        else:
            classification = "clean"

        tcs = self.compute_hallucination_score(verification_results)

        return {
            "classification": classification,
            "hallucinated_claims": hallucinated_claims,
            "tcs": 1.0 - tcs,  # TCS = 1 - hallucination_score
            "per_claim": per_claim,
        }

    # ------------------------------------------------------------------
    # Aggregate score
    # ------------------------------------------------------------------

    def compute_hallucination_score(
        self, results: list[VerificationResult]
    ) -> float:
        """Compute an aggregate hallucination score for a set of results.

        ``0.0`` means fully supported (no hallucination).
        ``1.0`` means complete hallucination.

        The score is the fraction of claims that are UNSUPPORTED,
        TEMPORALLY_IMPOSSIBLE, or CANNOT_VERIFY.

        Parameters
        ----------
        results : list[VerificationResult]
            Verification results for a batch of claims.

        Returns
        -------
        float
            A score in ``[0.0, 1.0]``.  Returns ``0.0`` if the list is
            empty.
        """
        if not results:
            return 0.0

        unsupported_count = sum(
            1
            for r in results
            if r.status
            in (
                VerificationStatus.UNSUPPORTED,
                VerificationStatus.TEMPORALLY_IMPOSSIBLE,
                VerificationStatus.CANNOT_VERIFY,
            )
        )

        return unsupported_count / len(results)
