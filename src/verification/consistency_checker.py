"""
Temporal consistency checker.

ConsistencyChecker provides utilities for validating temporal ordering and
multi-hop consistency across sets of temporal facts and verification results.
It operates on parsed timestamps to detect ordering violations, temporal
overlaps, and to compute the aggregate Temporal Consistency Score (TCS).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from src.verification.verification_result import (
    TemporalFact,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


def _parse_date(date_str: str) -> date | None:
    """Parse an ISO-8601 date string (``YYYY-MM-DD``) into a ``date`` object.

    Parameters
    ----------
    date_str : str
        The date string to parse.

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


class ConsistencyChecker:
    """Check temporal ordering and multi-hop consistency of facts.

    The checker works with :class:`TemporalFact` objects and can detect:

    * Pair-wise ordering violations.
    * Temporal overlap between range-based facts.
    * Chain-level chronological inconsistencies.
    * Aggregate Temporal Consistency Score (TCS) over a batch of
      :class:`VerificationResult` objects.

    Parameters
    ----------
    graph_store : object
        A TemporalKGStore (or compatible) instance.  Reserved for future use
        where the checker may need to query the graph for additional context
        (e.g. predecessor events).
    """

    def __init__(self, graph_store: Any) -> None:
        """Initialise with a backing knowledge-graph store.

        Parameters
        ----------
        graph_store : object
            The graph store instance (may be ``None`` for pure in-memory
            consistency checks that do not require graph access).
        """
        self._store = graph_store

    # ------------------------------------------------------------------
    # Pair-wise ordering
    # ------------------------------------------------------------------

    def check_ordering(
        self, event_a: TemporalFact, event_b: TemporalFact
    ) -> dict[str, Any]:
        """Compare the timestamps of two events.

        Parameters
        ----------
        event_a : TemporalFact
            The first event.
        event_b : TemporalFact
            The second event.

        Returns
        -------
        dict
            ``{"a_before_b": bool, "b_before_a": bool,
            "simultaneous": bool, "distance_days": int}``

            If either timestamp cannot be parsed the result defaults to
            ``simultaneous=False``, ``a_before_b=False``, ``b_before_a=False``
            and ``distance_days=0``.
        """
        date_a = _parse_date(event_a.timestamp)
        date_b = _parse_date(event_b.timestamp)

        if date_a is None or date_b is None:
            logger.warning(
                "Unparseable timestamp(s): a='%s', b='%s'",
                event_a.timestamp,
                event_b.timestamp,
            )
            return {
                "a_before_b": False,
                "b_before_a": False,
                "simultaneous": False,
                "distance_days": 0,
            }

        distance = abs((date_a - date_b).days)

        return {
            "a_before_b": date_a < date_b,
            "b_before_a": date_b < date_a,
            "simultaneous": date_a == date_b,
            "distance_days": distance,
        }

    # ------------------------------------------------------------------
    # Range overlap
    # ------------------------------------------------------------------

    def check_overlap(
        self, event_a: TemporalFact, event_b: TemporalFact
    ) -> dict[str, Any]:
        """Check temporal overlap for range-based facts.

        Range-based facts are expected to carry attributes ``valid_from``
        and ``valid_to`` as part of their timestamp (encoded as
        ``"YYYY-MM-DD/YYYY-MM-DD"`` or two separate fields).  If these
        are not available the method falls back to treating the single
        ``timestamp`` as a point event (zero-length range).

        Parameters
        ----------
        event_a : TemporalFact
            The first fact (possibly range-based).
        event_b : TemporalFact
            The second fact (possibly range-based).

        Returns
        -------
        dict
            ``{"overlaps": bool, "overlap_start": str | None,
            "overlap_end": str | None}``
        """
        start_a, end_a = self._extract_range(event_a.timestamp)
        start_b, end_b = self._extract_range(event_b.timestamp)

        if any(d is None for d in (start_a, end_a, start_b, end_b)):
            return {
                "overlaps": False,
                "overlap_start": None,
                "overlap_end": None,
            }

        # Compute the intersection of [start_a, end_a] and [start_b, end_b].
        overlap_start = max(start_a, start_b)  # type: ignore[arg-type]
        overlap_end = min(end_a, end_b)  # type: ignore[arg-type]

        if overlap_start <= overlap_end:  # type: ignore[operator]
            return {
                "overlaps": True,
                "overlap_start": overlap_start.isoformat(),  # type: ignore[union-attr]
                "overlap_end": overlap_end.isoformat(),  # type: ignore[union-attr]
            }

        return {"overlaps": False, "overlap_start": None, "overlap_end": None}

    # ------------------------------------------------------------------
    # Chain consistency
    # ------------------------------------------------------------------

    def check_chain_consistency(
        self, events: list[TemporalFact]
    ) -> dict[str, Any]:
        """Verify that a chain of events is in chronological order.

        Parameters
        ----------
        events : list[TemporalFact]
            An ordered list of facts that are expected to be chronological.

        Returns
        -------
        dict
            ``{"consistent": bool,
            "violations": list[tuple[int, int]]}``

            ``violations`` contains ``(i, j)`` pairs where ``events[i]`` has
            a timestamp later than ``events[j]`` (``j = i + 1``).
        """
        violations: list[tuple[int, int]] = []

        for i in range(len(events) - 1):
            date_i = _parse_date(events[i].timestamp)
            date_j = _parse_date(events[i + 1].timestamp)

            if date_i is None or date_j is None:
                # Cannot determine order -- treat as a violation.
                violations.append((i, i + 1))
                continue

            if date_i > date_j:
                violations.append((i, i + 1))

        return {
            "consistent": len(violations) == 0,
            "violations": violations,
        }

    # ------------------------------------------------------------------
    # Aggregate TCS
    # ------------------------------------------------------------------

    def compute_temporal_consistency_score(
        self, results: list[VerificationResult]
    ) -> float:
        """Compute the Temporal Consistency Score (TCS).

        TCS = (number of SUPPORTED + PARTIALLY_SUPPORTED) / total claims.

        Parameters
        ----------
        results : list[VerificationResult]
            Verification results for a batch of claims.

        Returns
        -------
        float
            A score in ``[0.0, 1.0]``.  Returns ``0.0`` when the input
            list is empty.
        """
        if not results:
            return 0.0

        supported_count = sum(
            1
            for r in results
            if r.status
            in (
                VerificationStatus.SUPPORTED,
                VerificationStatus.PARTIALLY_SUPPORTED,
            )
        )

        return supported_count / len(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_range(timestamp_str: str) -> tuple[date | None, date | None]:
        """Extract a ``(start, end)`` date pair from a timestamp string.

        If the string contains a ``/`` it is treated as an interval
        (``"YYYY-MM-DD/YYYY-MM-DD"``).  Otherwise the single date is
        used for both start and end (point event).

        Parameters
        ----------
        timestamp_str : str
            The raw timestamp string.

        Returns
        -------
        tuple[date | None, date | None]
            Parsed start and end dates, or ``(None, None)`` on failure.
        """
        if "/" in timestamp_str:
            parts = timestamp_str.split("/")
            if len(parts) == 2:
                return _parse_date(parts[0]), _parse_date(parts[1])
            return None, None

        d = _parse_date(timestamp_str)
        return d, d
