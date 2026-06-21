"""
Temporal fact retriever for the Temporal Grounding and Verification Framework.

Retrieves relevant temporal facts from the Neo4j knowledge graph by parsing
natural-language questions, selecting an appropriate Cypher template, and
executing parameterised queries against the graph store.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import RETRIEVAL_TOP_K
from src.retrieval.query_parser import QueryParser
from src.retrieval import query_templates as qt

logger = logging.getLogger(__name__)


class TemporalRetriever:
    """Retrieve temporal facts from Neo4j based on natural-language questions.

    Orchestrates query parsing, Cypher template selection, parameter binding,
    and fact retrieval, returning ranked lists of temporal quadruples.
    """

    # Map query types to their Cypher templates
    _TEMPLATE_MAP: dict[str, str] = {
        "entity_at_time": qt.ENTITY_AT_TIME,
        "entity_pair": qt.ENTITY_PAIR,
        "events_before": qt.EVENTS_BEFORE,
        "events_after": qt.EVENTS_AFTER,
        "entity_all_events": qt.ENTITY_ALL_EVENTS,
        "predicate_search": qt.PREDICATE_SEARCH,
        "general": qt.ENTITY_ALL_EVENTS,
    }

    def __init__(
        self,
        graph_store: Any,
        query_parser: QueryParser | None = None,
    ) -> None:
        """Initialise the temporal retriever.

        Args:
            graph_store: A graph store instance that exposes a ``query(cypher,
                parameters)`` method returning a list of record dicts.  This
                is typically a ``TemporalKGStore`` but any compatible object
                will work (duck typing).
            query_parser: Optional pre-configured QueryParser.  If not
                provided, a default parser with no known entity list is
                created.
        """
        self.graph_store = graph_store
        self.parser = query_parser or QueryParser()

    # ------------------------------------------------------------------
    # Template selection
    # ------------------------------------------------------------------

    def _select_template(self, parsed_query: dict[str, Any]) -> str:
        """Choose the appropriate Cypher template based on the parsed query.

        Selection logic:
            - Uses the ``query_type`` field from the parsed query to look up
              the matching template.
            - Falls back to ``ENTITY_ALL_EVENTS`` for unknown query types.

        Args:
            parsed_query: The dict returned by ``QueryParser.parse()``.

        Returns:
            The Cypher query template string.
        """
        query_type = parsed_query.get("query_type", "general")
        template = self._TEMPLATE_MAP.get(query_type)
        if template is None:
            logger.warning(
                "Unknown query type '%s'; falling back to ENTITY_ALL_EVENTS.",
                query_type,
            )
            template = qt.ENTITY_ALL_EVENTS
        return template

    # ------------------------------------------------------------------
    # Parameter building
    # ------------------------------------------------------------------

    def _build_params(
        self, parsed_query: dict[str, Any], top_k: int = RETRIEVAL_TOP_K
    ) -> dict[str, Any]:
        """Build the parameter dict for the selected Cypher template.

        Maps extracted entities and temporal info to the ``$param``
        placeholders used in the Cypher templates.

        Args:
            parsed_query: The dict returned by ``QueryParser.parse()``.
            top_k: Maximum number of results to return.

        Returns:
            Dictionary of query parameters.
        """
        entities = parsed_query.get("entities", [])
        date_range = parsed_query.get("date_range")
        query_type = parsed_query.get("query_type", "general")
        temporal = parsed_query.get("temporal", {})

        params: dict[str, Any] = {"limit": top_k}

        # Entity parameters
        if query_type == "entity_pair" and len(entities) >= 2:
            params["entity_a"] = entities[0]
            params["entity_b"] = entities[1]
        elif entities:
            params["entity_name"] = entities[0]

        # Date range parameters
        if date_range:
            params["start_date"] = date_range[0]
            params["end_date"] = date_range[1]
        else:
            params["start_date"] = None
            params["end_date"] = None

        # Reference date for before/after queries
        if query_type == "events_before" and date_range:
            params["reference_date"] = date_range[1]
        elif query_type == "events_after" and date_range:
            params["reference_date"] = date_range[0]
        elif query_type in ("events_before", "events_after"):
            # Try to extract a reference from years
            years = temporal.get("years", [])
            if years:
                if query_type == "events_before":
                    params["reference_date"] = f"{years[0]}-01-01"
                else:
                    params["reference_date"] = f"{years[0]}-12-31"

        return params

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, question: str, top_k: int = RETRIEVAL_TOP_K
    ) -> list[dict[str, Any]]:
        """Retrieve temporal facts relevant to a natural-language question.

        Pipeline:
            1. Parse the question (entities, temporal expressions).
            2. Select the Cypher query template.
            3. Build parameters.
            4. Execute the query against the graph store.
            5. Attach relevance scores and return ranked facts.

        Args:
            question: The natural-language temporal question.
            top_k: Maximum number of facts to return.

        Returns:
            List of fact dicts, each containing: subject, predicate, object,
            timestamp, event_id, relevance_score.  Ordered by relevance
            (higher is better).
        """
        parsed = self.parser.parse(question)
        logger.info(
            "Parsed query: type=%s, entities=%s, date_range=%s",
            parsed["query_type"],
            parsed["entities"],
            parsed["date_range"],
        )

        template = self._select_template(parsed)
        params = self._build_params(parsed, top_k)

        try:
            records = self.graph_store.query(template, params)
        except Exception:
            logger.exception("Neo4j query failed for question: %s", question)
            return []

        facts = self._records_to_facts(records, parsed)
        return facts

    def retrieve_for_entities(
        self,
        entities: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Retrieve temporal facts by entity names and optional date range.

        Bypasses question parsing and directly queries the graph store.

        Args:
            entities: List of entity names to query.
            start_date: Optional start date in ISO format (YYYY-MM-DD).
            end_date: Optional end date in ISO format (YYYY-MM-DD).
            top_k: Maximum number of facts to return.

        Returns:
            List of fact dicts with subject, predicate, object, timestamp,
            event_id, and relevance_score.
        """
        all_facts: list[dict[str, Any]] = []

        if len(entities) >= 2:
            params: dict[str, Any] = {
                "entity_a": entities[0],
                "entity_b": entities[1],
                "start_date": start_date,
                "end_date": end_date,
                "limit": top_k,
            }
            try:
                records = self.graph_store.query(qt.ENTITY_PAIR, params)
                all_facts.extend(self._records_to_facts(records))
            except Exception:
                logger.exception(
                    "Neo4j query failed for entity pair: %s", entities[:2]
                )
        else:
            for entity in entities:
                if start_date and end_date:
                    template = qt.ENTITY_AT_TIME
                    params = {
                        "entity_name": entity,
                        "start_date": start_date,
                        "end_date": end_date,
                        "limit": top_k,
                    }
                else:
                    template = qt.ENTITY_ALL_EVENTS
                    params = {"entity_name": entity, "limit": top_k}

                try:
                    records = self.graph_store.query(template, params)
                    all_facts.extend(self._records_to_facts(records))
                except Exception:
                    logger.exception(
                        "Neo4j query failed for entity: %s", entity
                    )

        return all_facts[:top_k]

    def retrieve_for_verification(
        self,
        subject: str,
        predicate: str,
        obj: str,
        timestamp: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve facts that match or are close to a specific claim.

        Used during the verification stage to check whether a claimed
        temporal quadruple exists in the knowledge graph.  Performs an
        exact-match query first; if no results are found, falls back to
        a broader entity-based search.

        Args:
            subject: The subject entity name.
            predicate: The predicate / relation.
            obj: The object entity name.
            timestamp: Optional timestamp in ISO format (YYYY-MM-DD).

        Returns:
            List of matching or related fact dicts.
        """
        # Exact match attempt
        params: dict[str, Any] = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "timestamp": timestamp,
            "limit": RETRIEVAL_TOP_K,
        }
        try:
            records = self.graph_store.query(qt.FACT_VERIFICATION, params)
            if records:
                facts = self._records_to_facts(records)
                for fact in facts:
                    fact["relevance_score"] = 1.0
                return facts
        except Exception:
            logger.exception("Exact verification query failed.")

        # Broader fallback: find any events between the two entities
        fallback_params: dict[str, Any] = {
            "entity_a": subject,
            "entity_b": obj,
            "start_date": None,
            "end_date": None,
            "limit": RETRIEVAL_TOP_K,
        }
        try:
            records = self.graph_store.query(qt.ENTITY_PAIR, fallback_params)
            facts = self._records_to_facts(records)
            # Score based on predicate and timestamp similarity
            for fact in facts:
                score = 0.0
                if fact.get("predicate") == predicate:
                    score += 0.5
                if timestamp and fact.get("timestamp") == timestamp:
                    score += 0.5
                elif timestamp and fact.get("timestamp", "")[:4] == timestamp[:4]:
                    score += 0.2
                fact["relevance_score"] = score
            facts.sort(key=lambda f: f["relevance_score"], reverse=True)
            return facts
        except Exception:
            logger.exception("Fallback verification query failed.")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _records_to_facts(
        records: list[dict[str, Any]],
        parsed_query: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert raw Neo4j records into standardised fact dicts.

        Args:
            records: List of record dicts from the graph store.
            parsed_query: Optional parsed query for relevance scoring.

        Returns:
            List of fact dicts with keys: subject, predicate, object,
            timestamp, event_id, relevance_score.
        """
        facts: list[dict[str, Any]] = []
        for record in records:
            fact: dict[str, Any] = {
                "subject": record.get("subject", ""),
                "predicate": record.get("predicate", ""),
                "object": record.get("object", ""),
                "timestamp": record.get("timestamp", ""),
                "event_id": record.get("event_id", ""),
                "relevance_score": 1.0,
            }
            if parsed_query:
                fact["relevance_score"] = _compute_relevance(
                    fact, parsed_query
                )
            facts.append(fact)

        # Sort by relevance (descending), then by timestamp (ascending)
        facts.sort(
            key=lambda f: (-f["relevance_score"], f.get("timestamp", ""))
        )
        return facts


def _compute_relevance(
    fact: dict[str, Any], parsed_query: dict[str, Any]
) -> float:
    """Compute a simple relevance score for a retrieved fact.

    Scoring heuristic:
        - +0.5 if the fact's subject or object matches a query entity.
        - +0.3 if the fact's timestamp falls within the query date range.
        - +0.2 baseline for any returned result.

    Args:
        fact: A standardised fact dict.
        parsed_query: The parsed query dict.

    Returns:
        A float relevance score in [0.0, 1.0].
    """
    score = 0.2  # Baseline for being a returned result
    entities = {e.lower() for e in parsed_query.get("entities", [])}
    date_range = parsed_query.get("date_range")

    # Entity match
    subj_lower = fact.get("subject", "").lower()
    obj_lower = fact.get("object", "").lower()
    if subj_lower in entities or obj_lower in entities:
        score += 0.5

    # Temporal match
    if date_range and fact.get("timestamp"):
        ts = fact["timestamp"]
        if date_range[0] <= ts <= date_range[1]:
            score += 0.3

    return min(score, 1.0)
