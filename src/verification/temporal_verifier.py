"""
Core temporal verification engine.

TemporalVerifier orchestrates a multi-stage verification pipeline that checks
LLM-generated temporal claims against the knowledge graph.  Each claim passes
through four stages -- entity verification, relation verification, timestamp
verification, and ordering verification -- and the aggregated evidence is
condensed into a single :class:`VerificationResult`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from config.settings import TIMESTAMP_TOLERANCE_DAYS
from src.verification.verification_result import (
    TemporalClaim,
    TemporalFact,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence scores keyed by verification status
# ---------------------------------------------------------------------------

_STATUS_CONFIDENCE: dict[VerificationStatus, float] = {
    VerificationStatus.SUPPORTED: 1.0,
    VerificationStatus.PARTIALLY_SUPPORTED: 0.7,
    VerificationStatus.UNSUPPORTED: 0.3,
    VerificationStatus.TEMPORALLY_IMPOSSIBLE: 0.0,
    VerificationStatus.CANNOT_VERIFY: 0.0,
}


def _parse_date(date_str: str) -> date | None:
    """Attempt to parse an ISO-8601 date string into a ``date`` object.

    Parameters
    ----------
    date_str : str
        A date string in ``YYYY-MM-DD`` format.

    Returns
    -------
    date or None
        The parsed date, or ``None`` if parsing fails.
    """
    try:
        parts = date_str.strip().split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


class TemporalVerifier:
    """Multi-stage verification engine for temporal claims.

    The verifier depends on a *graph_store* -- an instance of
    :class:`src.knowledge_graph.graph_store.TemporalKGStore` -- which exposes
    the following methods used by this class:

    * ``query_entity_events(entity_name, limit=...)`` -- find events for an entity.
    * ``verify_entity_relation(subject, predicate, obj)`` -- check a triple at any time.
    * ``verify_fact(subject, predicate, obj, timestamp)`` -- exact fact lookup.

    Parameters
    ----------
    graph_store : TemporalKGStore
        A TemporalKGStore instance that provides read access to the temporal
        knowledge graph.
    """

    def __init__(self, graph_store: Any) -> None:
        """Initialise the verifier with a knowledge-graph store.

        Parameters
        ----------
        graph_store : TemporalKGStore
            The backing graph store.
        """
        self._store = graph_store

    # ------------------------------------------------------------------
    # Stage 1: Entity verification
    # ------------------------------------------------------------------

    def verify_entity(self, entity_name: str) -> dict[str, Any]:
        """Check whether an entity exists in the knowledge graph.

        Uses ``query_entity_events`` with ``limit=1`` as a lightweight
        existence check.

        Parameters
        ----------
        entity_name : str
            The entity name to look up.

        Returns
        -------
        dict
            ``{"exists": bool, "name": str}``
        """
        try:
            results = self._store.query_entity_events(entity_name, limit=1)
            exists = len(results) > 0
        except Exception:
            logger.exception("Error querying entity '%s'", entity_name)
            exists = False

        return {"exists": exists, "name": entity_name}

    # ------------------------------------------------------------------
    # Stage 2: Relation verification
    # ------------------------------------------------------------------

    def verify_relation(
        self, subject: str, predicate: str, obj: str
    ) -> dict[str, Any]:
        """Check whether a (subject, predicate, object) triple exists at any timestamp.

        Delegates to ``TemporalKGStore.verify_entity_relation``.

        Parameters
        ----------
        subject : str
            The subject (actor) entity.
        predicate : str
            The relation or event type.
        obj : str
            The object (target) entity.

        Returns
        -------
        dict
            ``{"exists": bool, "found_predicates": list[str],
            "timestamps": list[str]}``
        """
        try:
            result = self._store.verify_entity_relation(subject, predicate, obj)
        except Exception:
            logger.exception(
                "Error querying relation ('%s', '%s', '%s')",
                subject,
                predicate,
                obj,
            )
            return {"exists": False, "found_predicates": [], "timestamps": []}

        exists: bool = result.get("exists", False)
        timestamps: list[str] = result.get("timestamps", [])
        found_predicates = [predicate] if exists else []

        return {
            "exists": exists,
            "found_predicates": found_predicates,
            "timestamps": timestamps,
        }

    # ------------------------------------------------------------------
    # Stage 3: Timestamp verification
    # ------------------------------------------------------------------

    def verify_timestamp(
        self, subject: str, predicate: str, obj: str, claimed_timestamp: str
    ) -> dict[str, Any]:
        """Check whether the claimed timestamp matches KG evidence.

        The match can be ``exact`` (identical date), ``range`` (within the
        configured tolerance window), ``proximity`` (the claim falls near a
        recorded timestamp), or ``mismatch`` (no temporal correspondence).

        Parameters
        ----------
        subject : str
            The subject entity.
        predicate : str
            The relation or event type.
        obj : str
            The object entity.
        claimed_timestamp : str
            The timestamp asserted by the claim (ISO-8601).

        Returns
        -------
        dict
            ``{"match_type": str, "actual_timestamps": list[str],
            "temporal_distance_days": int | None}``
        """
        relation_check = self.verify_relation(subject, predicate, obj)
        actual_timestamps: list[str] = relation_check.get("timestamps", [])

        if not actual_timestamps:
            return {
                "match_type": "mismatch",
                "actual_timestamps": [],
                "temporal_distance_days": None,
            }

        claimed_date = _parse_date(claimed_timestamp)
        if claimed_date is None:
            return {
                "match_type": "mismatch",
                "actual_timestamps": actual_timestamps,
                "temporal_distance_days": None,
            }

        min_distance: int | None = None
        for ts_str in actual_timestamps:
            actual_date = _parse_date(ts_str)
            if actual_date is None:
                continue
            distance = abs((claimed_date - actual_date).days)
            if min_distance is None or distance < min_distance:
                min_distance = distance

        if min_distance is None:
            return {
                "match_type": "mismatch",
                "actual_timestamps": actual_timestamps,
                "temporal_distance_days": None,
            }

        tolerance = TIMESTAMP_TOLERANCE_DAYS

        if min_distance == 0:
            match_type = "exact"
        elif min_distance <= tolerance:
            match_type = "range"
        elif min_distance <= tolerance * 3 + 30:
            # Within a loose proximity window -- facts that are *close-ish*.
            match_type = "proximity"
        else:
            match_type = "mismatch"

        return {
            "match_type": match_type,
            "actual_timestamps": actual_timestamps,
            "temporal_distance_days": min_distance,
        }

    # ------------------------------------------------------------------
    # Stage 4: Ordering verification
    # ------------------------------------------------------------------

    def verify_ordering(
        self,
        event_a: dict[str, str],
        event_b: dict[str, str],
        claimed_order: str,
    ) -> dict[str, Any]:
        """Verify the temporal ordering between two events.

        Parameters
        ----------
        event_a : dict
            Dict with at least a ``"timestamp"`` key (ISO-8601).
        event_b : dict
            Dict with at least a ``"timestamp"`` key (ISO-8601).
        claimed_order : str
            ``"before"`` if the claim states A happened before B, or
            ``"after"`` if A happened after B.

        Returns
        -------
        dict
            ``{"valid": bool, "actual_order": str,
            "timestamp_a": str, "timestamp_b": str}``
        """
        ts_a_str = event_a.get("timestamp", "")
        ts_b_str = event_b.get("timestamp", "")

        date_a = _parse_date(ts_a_str)
        date_b = _parse_date(ts_b_str)

        if date_a is None or date_b is None:
            return {
                "valid": False,
                "actual_order": "unknown",
                "timestamp_a": ts_a_str,
                "timestamp_b": ts_b_str,
            }

        if date_a < date_b:
            actual_order = "before"
        elif date_a > date_b:
            actual_order = "after"
        else:
            actual_order = "simultaneous"

        valid = (
            (claimed_order == "before" and actual_order == "before")
            or (claimed_order == "after" and actual_order == "after")
            or actual_order == "simultaneous"
        )

        return {
            "valid": valid,
            "actual_order": actual_order,
            "timestamp_a": ts_a_str,
            "timestamp_b": ts_b_str,
        }

    # ------------------------------------------------------------------
    # Aggregate verification
    # ------------------------------------------------------------------

    def verify_claim(self, claim: TemporalClaim) -> VerificationResult:
        """Run all verification stages on a single temporal claim.

        The aggregation logic follows these rules in order:

        1. If the subject entity is not found in the KG -> ``CANNOT_VERIFY``.
        2. If the entity exists but no matching relation is found -> ``UNSUPPORTED``.
        3. If the relation exists but the timestamp does not match:
           - ``proximity`` match -> ``PARTIALLY_SUPPORTED``
           - ``range`` match -> ``PARTIALLY_SUPPORTED``
           - ``mismatch`` -> ``UNSUPPORTED``
        4. If entity, relation, and timestamp all match -> ``SUPPORTED``.

        Parameters
        ----------
        claim : TemporalClaim
            The claim to verify.

        Returns
        -------
        VerificationResult
            A fully populated verification result.
        """
        checks: dict[str, Any] = {}

        # Stage 1 -- entity
        entity_check = self.verify_entity(claim.subject)
        checks["entity"] = entity_check

        if not entity_check["exists"]:
            status = VerificationStatus.CANNOT_VERIFY
            return VerificationResult(
                claim=claim,
                status=status,
                confidence=_STATUS_CONFIDENCE[status],
                evidence=[],
                explanation=(
                    f"Entity '{claim.subject}' was not found in the "
                    f"knowledge graph. The claim cannot be verified."
                ),
                checks=checks,
            )

        # Stage 2 -- relation
        relation_check = self.verify_relation(
            claim.subject, claim.predicate, claim.object
        )
        checks["relation"] = relation_check

        if not relation_check["exists"]:
            status = VerificationStatus.UNSUPPORTED
            return VerificationResult(
                claim=claim,
                status=status,
                confidence=_STATUS_CONFIDENCE[status],
                evidence=[],
                explanation=(
                    f"No relation '{claim.predicate}' between "
                    f"'{claim.subject}' and '{claim.object}' was found "
                    f"in the knowledge graph."
                ),
                checks=checks,
            )

        # Build evidence list from the relation results
        evidence = self._build_evidence(claim, relation_check)

        # Stage 3 -- timestamp
        timestamp_check = self.verify_timestamp(
            claim.subject, claim.predicate, claim.object, claim.timestamp
        )
        checks["timestamp"] = timestamp_check

        match_type = timestamp_check["match_type"]

        if match_type == "exact":
            status = VerificationStatus.SUPPORTED
        elif match_type in ("range", "proximity"):
            status = VerificationStatus.PARTIALLY_SUPPORTED
        else:
            status = VerificationStatus.UNSUPPORTED

        confidence = _STATUS_CONFIDENCE[status]

        explanation = self._build_explanation(claim, checks, status)

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence,
            explanation=explanation,
            checks=checks,
        )

    def verify_claims(
        self, claims: list[TemporalClaim]
    ) -> list[VerificationResult]:
        """Verify a batch of temporal claims.

        Parameters
        ----------
        claims : list[TemporalClaim]
            The claims to verify.

        Returns
        -------
        list[VerificationResult]
            One result per claim, in the same order.
        """
        results: list[VerificationResult] = []
        for claim in claims:
            result = self.verify_claim(claim)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence(
        claim: TemporalClaim, relation_check: dict[str, Any]
    ) -> list[TemporalFact]:
        """Convert raw relation-check data into a list of TemporalFact objects.

        Parameters
        ----------
        claim : TemporalClaim
            The original claim (used for subject/object context).
        relation_check : dict
            The output of :meth:`verify_relation`.

        Returns
        -------
        list[TemporalFact]
            Reconstructed evidence facts.
        """
        facts: list[TemporalFact] = []
        predicates = relation_check.get("found_predicates", [])
        timestamps = relation_check.get("timestamps", [])

        predicate = predicates[0] if predicates else claim.predicate
        for ts in timestamps:
            facts.append(
                TemporalFact(
                    subject=claim.subject,
                    predicate=predicate,
                    object=claim.object,
                    timestamp=ts,
                )
            )
        return facts

    @staticmethod
    def _build_explanation(
        claim: TemporalClaim,
        checks: dict[str, Any],
        status: VerificationStatus,
    ) -> str:
        """Construct a human-readable explanation from check details.

        Parameters
        ----------
        claim : TemporalClaim
            The original claim.
        checks : dict
            Results of each sub-check.
        status : VerificationStatus
            The aggregated verification status.

        Returns
        -------
        str
            A concise explanation.
        """
        ts_check = checks.get("timestamp", {})
        actual = ts_check.get("actual_timestamps", [])
        distance = ts_check.get("temporal_distance_days")

        if status == VerificationStatus.SUPPORTED:
            return (
                f"The claim that '{claim.subject}' {claim.predicate} "
                f"'{claim.object}' on {claim.timestamp} is fully supported "
                f"by the knowledge graph."
            )

        if status == VerificationStatus.PARTIALLY_SUPPORTED:
            actual_str = ", ".join(actual) if actual else "unknown"
            distance_str = (
                f" ({distance} day(s) difference)" if distance is not None else ""
            )
            return (
                f"The relation between '{claim.subject}' and "
                f"'{claim.object}' exists, but the claimed timestamp "
                f"({claim.timestamp}) differs from recorded "
                f"timestamp(s): {actual_str}{distance_str}."
            )

        if status == VerificationStatus.UNSUPPORTED:
            rel_check = checks.get("relation", {})
            if rel_check.get("exists"):
                actual_str = ", ".join(actual) if actual else "unknown"
                return (
                    f"The relation exists but the claimed timestamp "
                    f"({claim.timestamp}) does not match any recorded "
                    f"timestamp(s): {actual_str}."
                )
            return (
                f"No evidence found for the relation '{claim.predicate}' "
                f"between '{claim.subject}' and '{claim.object}'."
            )

        return f"Verification status: {status.value}."
