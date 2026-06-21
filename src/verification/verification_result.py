"""
Data classes for temporal verification outcomes.

Defines the canonical representations used throughout the verification
pipeline: claims extracted from LLM output, facts stored in the knowledge
graph, and the result of verifying a claim against the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationStatus(Enum):
    """Possible outcomes of verifying a temporal claim against the KG.

    Members
    -------
    SUPPORTED
        The claim is fully corroborated by at least one matching fact.
    PARTIALLY_SUPPORTED
        The entity and relation exist but the timestamp is close (within
        the configured tolerance) rather than an exact match.
    UNSUPPORTED
        The claim contradicts the evidence in the KG.
    TEMPORALLY_IMPOSSIBLE
        The claimed temporal ordering violates logical constraints (e.g.
        an event placed before the birth of its actor).
    CANNOT_VERIFY
        Insufficient evidence in the KG to support or refute the claim.
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    TEMPORALLY_IMPOSSIBLE = "temporally_impossible"
    CANNOT_VERIFY = "cannot_verify"


@dataclass
class TemporalClaim:
    """A temporal assertion extracted from LLM-generated text.

    Attributes
    ----------
    subject : str
        The entity performing or associated with the action.
    predicate : str
        The relation or action described.
    object : str
        The target entity or value.
    timestamp : str
        The claimed date or time-point (ISO-8601 string, e.g. ``2014-03-15``).
    source_sentence : str
        The verbatim sentence from which the claim was extracted.
    """

    subject: str
    predicate: str
    object: str
    timestamp: str
    source_sentence: str = ""


@dataclass
class TemporalFact:
    """A single temporal fact stored in the knowledge graph.

    Attributes
    ----------
    subject : str
        The actor or source entity.
    predicate : str
        The relation or event type.
    object : str
        The target entity.
    timestamp : str
        The recorded date or time-point (ISO-8601 string).
    event_id : str
        The unique identifier of the reified Event node in Neo4j.
    """

    subject: str
    predicate: str
    object: str
    timestamp: str
    event_id: str = ""


@dataclass
class VerificationResult:
    """The outcome of verifying a single :class:`TemporalClaim`.

    Attributes
    ----------
    claim : TemporalClaim
        The original claim being verified.
    status : VerificationStatus
        The verification verdict.
    confidence : float
        A confidence score in ``[0.0, 1.0]``.
    evidence : list[TemporalFact]
        Facts retrieved from the KG that informed the decision.
    explanation : str
        A human-readable explanation of the verdict.
    checks : dict[str, Any]
        Individual results from each sub-check (entity, relation, timestamp,
        ordering) so downstream consumers can inspect details.
    """

    claim: TemporalClaim
    status: VerificationStatus
    confidence: float
    evidence: list[TemporalFact] = field(default_factory=list)
    explanation: str = ""
    checks: dict[str, Any] = field(default_factory=dict)
