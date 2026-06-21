"""
Human-readable explanation generator for verification results.

ExplanationGenerator transforms structured :class:`VerificationResult`
objects into clear, readable text that helps users understand *why* a
temporal claim was classified as supported, unsupported, or hallucinated.
Each :class:`VerificationStatus` has a dedicated template enriched with
specific evidence details (timestamps, distances, entity names).
"""

from __future__ import annotations

from typing import Any

from src.verification.verification_result import (
    VerificationResult,
    VerificationStatus,
)


class ExplanationGenerator:
    """Generate human-readable explanations for verification outcomes.

    The generator is stateless and does not require external dependencies.
    """

    def __init__(self) -> None:
        """Initialise the explanation generator."""

    # ------------------------------------------------------------------
    # Single-result explanation
    # ------------------------------------------------------------------

    def explain(self, result: VerificationResult) -> str:
        """Generate a human-readable explanation for a verification result.

        Parameters
        ----------
        result : VerificationResult
            The verification outcome to explain.

        Returns
        -------
        str
            A multi-sentence explanation referencing the claim, the
            verification status, and any supporting evidence.
        """
        claim = result.claim
        status = result.status
        checks = result.checks
        evidence = result.evidence

        # ----- SUPPORTED -----
        if status == VerificationStatus.SUPPORTED:
            return self._explain_supported(claim, evidence)

        # ----- PARTIALLY_SUPPORTED -----
        if status == VerificationStatus.PARTIALLY_SUPPORTED:
            return self._explain_partially_supported(claim, checks, evidence)

        # ----- UNSUPPORTED -----
        if status == VerificationStatus.UNSUPPORTED:
            return self._explain_unsupported(claim, checks)

        # ----- TEMPORALLY_IMPOSSIBLE -----
        if status == VerificationStatus.TEMPORALLY_IMPOSSIBLE:
            return self._explain_temporally_impossible(claim, checks)

        # ----- CANNOT_VERIFY -----
        if status == VerificationStatus.CANNOT_VERIFY:
            return self._explain_cannot_verify(claim)

        # Fallback
        return (
            f"Claim: '{claim.subject}' {claim.predicate} '{claim.object}' "
            f"on {claim.timestamp}. Status: {status.value}."
        )

    # ------------------------------------------------------------------
    # Batch explanation
    # ------------------------------------------------------------------

    def explain_batch(self, results: list[VerificationResult]) -> list[str]:
        """Generate explanations for a batch of verification results.

        Parameters
        ----------
        results : list[VerificationResult]
            The verification outcomes to explain.

        Returns
        -------
        list[str]
            One explanation string per result, in the same order.
        """
        return [self.explain(r) for r in results]

    # ------------------------------------------------------------------
    # Evidence summary
    # ------------------------------------------------------------------

    def format_evidence_summary(
        self, results: list[VerificationResult]
    ) -> str:
        """Create a summary report of all verification results.

        The summary includes a header with aggregate statistics followed
        by a per-claim breakdown.

        Parameters
        ----------
        results : list[VerificationResult]
            All verification results to summarise.

        Returns
        -------
        str
            A formatted multi-line summary string.
        """
        if not results:
            return "No verification results to summarise."

        # Aggregate counts.
        counts: dict[str, int] = {}
        for r in results:
            key = r.status.value
            counts[key] = counts.get(key, 0) + 1

        total = len(results)
        supported = counts.get(VerificationStatus.SUPPORTED.value, 0)
        partial = counts.get(VerificationStatus.PARTIALLY_SUPPORTED.value, 0)
        unsupported = counts.get(VerificationStatus.UNSUPPORTED.value, 0)
        impossible = counts.get(
            VerificationStatus.TEMPORALLY_IMPOSSIBLE.value, 0
        )
        cannot_verify = counts.get(VerificationStatus.CANNOT_VERIFY.value, 0)

        tcs = (supported + partial) / total if total > 0 else 0.0

        lines: list[str] = [
            "=" * 60,
            "TEMPORAL VERIFICATION SUMMARY",
            "=" * 60,
            f"Total claims verified: {total}",
            f"  Supported:              {supported}",
            f"  Partially supported:    {partial}",
            f"  Unsupported:            {unsupported}",
            f"  Temporally impossible:  {impossible}",
            f"  Cannot verify:          {cannot_verify}",
            f"Temporal Consistency Score (TCS): {tcs:.2%}",
            "-" * 60,
            "PER-CLAIM DETAILS:",
            "-" * 60,
        ]

        for idx, r in enumerate(results, start=1):
            claim = r.claim
            lines.append(
                f"[{idx}] ({r.status.value.upper()}) "
                f"'{claim.subject}' {claim.predicate} '{claim.object}' "
                f"on {claim.timestamp}"
            )
            explanation = self.explain(r)
            # Indent explanation lines.
            for exp_line in explanation.split("\n"):
                lines.append(f"    {exp_line}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal template methods
    # ------------------------------------------------------------------

    @staticmethod
    def _explain_supported(
        claim: Any, evidence: list[Any]
    ) -> str:
        """Build explanation for a SUPPORTED claim.

        Parameters
        ----------
        claim : TemporalClaim
            The original claim.
        evidence : list[TemporalFact]
            Supporting facts from the KG.

        Returns
        -------
        str
            Human-readable explanation.
        """
        evidence_ts = [e.timestamp for e in evidence if e.timestamp]
        ts_detail = (
            f" Evidence timestamps: {', '.join(evidence_ts)}."
            if evidence_ts
            else ""
        )
        return (
            f"SUPPORTED: The claim that '{claim.subject}' {claim.predicate} "
            f"'{claim.object}' on {claim.timestamp} is fully corroborated "
            f"by the knowledge graph.{ts_detail}"
        )

    @staticmethod
    def _explain_partially_supported(
        claim: Any, checks: dict[str, Any], evidence: list[Any]
    ) -> str:
        """Build explanation for a PARTIALLY_SUPPORTED claim.

        Parameters
        ----------
        claim : TemporalClaim
            The original claim.
        checks : dict
            Sub-check results.
        evidence : list[TemporalFact]
            Related facts from the KG.

        Returns
        -------
        str
            Human-readable explanation.
        """
        ts_check = checks.get("timestamp", {})
        actual_ts = ts_check.get("actual_timestamps", [])
        distance = ts_check.get("temporal_distance_days")

        actual_str = ", ".join(actual_ts) if actual_ts else "unknown"
        distance_detail = (
            f" The closest recorded date is {distance} day(s) away."
            if distance is not None
            else ""
        )

        return (
            f"PARTIALLY SUPPORTED: The relation between '{claim.subject}' "
            f"and '{claim.object}' via '{claim.predicate}' exists in the "
            f"knowledge graph, but the claimed date ({claim.timestamp}) "
            f"does not exactly match the recorded date(s): "
            f"{actual_str}.{distance_detail}"
        )

    @staticmethod
    def _explain_unsupported(
        claim: Any, checks: dict[str, Any]
    ) -> str:
        """Build explanation for an UNSUPPORTED claim.

        Parameters
        ----------
        claim : TemporalClaim
            The original claim.
        checks : dict
            Sub-check results.

        Returns
        -------
        str
            Human-readable explanation.
        """
        rel_check = checks.get("relation", {})
        ts_check = checks.get("timestamp", {})

        if rel_check and not rel_check.get("exists"):
            return (
                f"UNSUPPORTED: No evidence was found in the knowledge graph "
                f"for the relation '{claim.predicate}' between "
                f"'{claim.subject}' and '{claim.object}'. This claim "
                f"appears to be a factual hallucination."
            )

        actual_ts = ts_check.get("actual_timestamps", [])
        actual_str = ", ".join(actual_ts) if actual_ts else "unknown"
        return (
            f"UNSUPPORTED: The relation '{claim.predicate}' between "
            f"'{claim.subject}' and '{claim.object}' exists, but the "
            f"claimed date ({claim.timestamp}) significantly differs "
            f"from the recorded date(s): {actual_str}. This is likely a "
            f"temporal hallucination."
        )

    @staticmethod
    def _explain_temporally_impossible(
        claim: Any, checks: dict[str, Any]
    ) -> str:
        """Build explanation for a TEMPORALLY_IMPOSSIBLE claim.

        Parameters
        ----------
        claim : TemporalClaim
            The original claim.
        checks : dict
            Sub-check results.

        Returns
        -------
        str
            Human-readable explanation.
        """
        ordering_check = checks.get("ordering", {})
        actual_order = ordering_check.get("actual_order", "unknown")

        return (
            f"TEMPORALLY IMPOSSIBLE: The claimed temporal placement of "
            f"'{claim.subject}' {claim.predicate} '{claim.object}' on "
            f"{claim.timestamp} violates known temporal constraints. "
            f"The actual ordering is '{actual_order}'. This event could "
            f"not have occurred at the stated time."
        )

    @staticmethod
    def _explain_cannot_verify(claim: Any) -> str:
        """Build explanation for a CANNOT_VERIFY claim.

        Parameters
        ----------
        claim : TemporalClaim
            The original claim.

        Returns
        -------
        str
            Human-readable explanation.
        """
        return (
            f"CANNOT VERIFY: The entity '{claim.subject}' was not found "
            f"in the knowledge graph. There is insufficient evidence to "
            f"either support or refute the claim that '{claim.subject}' "
            f"{claim.predicate} '{claim.object}' on {claim.timestamp}."
        )
