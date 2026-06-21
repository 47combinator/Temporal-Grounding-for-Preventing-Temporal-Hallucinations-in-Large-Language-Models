"""
Neo4j temporal knowledge graph store for the Temporal Grounding and Verification Framework.

Provides the ``TemporalKGStore`` class -- the primary interface for inserting
and querying temporal facts stored in Neo4j using the Event Node Reification
pattern:

    (Entity)-[:ACTOR]->(Event)-[:TARGET]->(Entity)

All Cypher queries use parameterised variables ($param syntax) to prevent
injection and allow query-plan caching.
"""

import logging
from typing import Any

from neo4j import GraphDatabase, Driver  # type: ignore[import-untyped]

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_BATCH_SIZE
from src.knowledge_graph.schema import Neo4jSchema

logger = logging.getLogger(__name__)


class TemporalKGStore:
    """High-level Neo4j interface for temporal knowledge graph operations.

    Supports bulk insertion of event-reified facts, temporal and
    predicate-based querying, and fact verification.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        """Create a driver and verify connectivity to Neo4j.

        Args:
            uri: Bolt URI for the Neo4j instance. Defaults to
                ``NEO4J_URI`` from settings.
            user: Database user. Defaults to ``NEO4J_USER``.
            password: Database password. Defaults to ``NEO4J_PASSWORD``.
        """
        self._uri: str = uri or NEO4J_URI
        self._user: str = user or NEO4J_USER
        self._password: str = password or NEO4J_PASSWORD

        self._driver: Driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", self._uri)

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()
        logger.info("Neo4j driver closed.")

    def __enter__(self) -> "TemporalKGStore":
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context manager and close the driver."""
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def setup_schema(self) -> None:
        """Create or verify the database schema (constraints and indexes).

        Delegates to ``Neo4jSchema.setup()``.  Safe to call repeatedly.
        """
        schema = Neo4jSchema(self._driver)
        schema.setup()

    # ------------------------------------------------------------------
    # Bulk insertion
    # ------------------------------------------------------------------

    def batch_insert_events(self, events: list[dict[str, Any]]) -> int:
        """Bulk-insert temporal facts as Event-reified sub-graphs.

        Each event dictionary must contain:

        - ``event_id`` (str): Unique identifier for the event.
        - ``actor`` (str): Name of the subject / source entity.
        - ``target`` (str): Name of the object / target entity.
        - ``predicate`` (str): The relation / action label.
        - ``timestamp`` (str): ISO date string (``YYYY-MM-DD``).
        - ``source`` (str): Provenance tag (e.g. ``"icews14"``).

        Optional keys:

        - ``valid_from`` (str): Start of validity interval.
        - ``valid_to`` (str): End of validity interval.

        The Cypher uses MERGE for Entity nodes (to avoid duplicates)
        and CREATE for Event nodes (each event is unique by event_id).
        Timestamps are converted to native Neo4j ``date()`` values.

        Args:
            events: List of event dictionaries to insert.

        Returns:
            Total number of Event nodes created.
        """
        if not events:
            return 0

        cypher = """
        UNWIND $batch AS evt
        MERGE (actor:Entity {name: evt.actor})
        MERGE (target:Entity {name: evt.target})
        CREATE (e:Event {
            event_id:   evt.event_id,
            predicate:  evt.predicate,
            timestamp:  date(evt.timestamp),
            source:     evt.source
        })
        WITH e, actor, target, evt
        CALL {
            WITH e, evt
            WITH e, evt
            WHERE evt.valid_from IS NOT NULL
            SET e.valid_from = date(evt.valid_from)
        }
        CALL {
            WITH e, evt
            WITH e, evt
            WHERE evt.valid_to IS NOT NULL
            SET e.valid_to = date(evt.valid_to)
        }
        CREATE (actor)-[:ACTOR]->(e)
        CREATE (e)-[:TARGET]->(target)
        """

        total_created = 0
        num_batches = (len(events) + NEO4J_BATCH_SIZE - 1) // NEO4J_BATCH_SIZE

        for batch_idx in range(num_batches):
            start = batch_idx * NEO4J_BATCH_SIZE
            end = start + NEO4J_BATCH_SIZE
            batch = events[start:end]

            with self._driver.session() as session:
                result = session.run(cypher, batch=batch)
                summary = result.consume()
                created = summary.counters.nodes_created
                total_created += created

            logger.debug(
                "Batch %d/%d: inserted %d events (%d nodes created).",
                batch_idx + 1,
                num_batches,
                len(batch),
                created,
            )

        logger.info(
            "Bulk insert complete: %d events processed, %d total nodes created.",
            len(events),
            total_created,
        )
        return total_created

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_entity_events(
        self,
        entity_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find events involving an entity (as actor OR target).

        Args:
            entity_name: Name of the entity to search for.
            start_date: Optional ISO date lower bound (inclusive).
            end_date: Optional ISO date upper bound (inclusive).
            limit: Maximum number of results.

        Returns:
            List of event dictionaries with actor, predicate, target,
            and timestamp fields.
        """
        conditions = ["(e_ent.name = $entity_name)"]
        params: dict[str, Any] = {"entity_name": entity_name, "limit": limit}

        if start_date is not None:
            conditions.append("ev.timestamp >= date($start_date)")
            params["start_date"] = start_date
        if end_date is not None:
            conditions.append("ev.timestamp <= date($end_date)")
            params["end_date"] = end_date

        where_clause = " AND ".join(conditions)

        cypher = f"""
        MATCH (actor:Entity)-[:ACTOR]->(ev:Event)-[:TARGET]->(target:Entity)
        WHERE (actor.name = $entity_name OR target.name = $entity_name)
        {"AND ev.timestamp >= date($start_date)" if start_date else ""}
        {"AND ev.timestamp <= date($end_date)" if end_date else ""}
        RETURN actor.name AS actor, ev.predicate AS predicate,
               target.name AS target, toString(ev.timestamp) AS timestamp,
               ev.event_id AS event_id, ev.source AS source
        ORDER BY ev.timestamp
        LIMIT $limit
        """

        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def query_events_between_entities(
        self,
        actor: str,
        target: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find events between a specific actor-target pair.

        Args:
            actor: Name of the source entity.
            target: Name of the target entity.
            start_date: Optional ISO date lower bound (inclusive).
            end_date: Optional ISO date upper bound (inclusive).

        Returns:
            List of event dictionaries.
        """
        params: dict[str, Any] = {"actor": actor, "target": target}
        date_filters: list[str] = []

        if start_date is not None:
            date_filters.append("AND ev.timestamp >= date($start_date)")
            params["start_date"] = start_date
        if end_date is not None:
            date_filters.append("AND ev.timestamp <= date($end_date)")
            params["end_date"] = end_date

        date_clause = " ".join(date_filters)

        cypher = f"""
        MATCH (a:Entity {{name: $actor}})-[:ACTOR]->(ev:Event)-[:TARGET]->(t:Entity {{name: $target}})
        WHERE true {date_clause}
        RETURN a.name AS actor, ev.predicate AS predicate,
               t.name AS target, toString(ev.timestamp) AS timestamp,
               ev.event_id AS event_id, ev.source AS source
        ORDER BY ev.timestamp
        """

        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def query_events_by_predicate(
        self,
        predicate: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find events with a specific predicate (relation).

        Args:
            predicate: The relation / action label to search for.
            start_date: Optional ISO date lower bound (inclusive).
            end_date: Optional ISO date upper bound (inclusive).
            limit: Maximum number of results.

        Returns:
            List of event dictionaries.
        """
        params: dict[str, Any] = {"predicate": predicate, "limit": limit}
        date_filters: list[str] = []

        if start_date is not None:
            date_filters.append("AND ev.timestamp >= date($start_date)")
            params["start_date"] = start_date
        if end_date is not None:
            date_filters.append("AND ev.timestamp <= date($end_date)")
            params["end_date"] = end_date

        date_clause = " ".join(date_filters)

        cypher = f"""
        MATCH (actor:Entity)-[:ACTOR]->(ev:Event {{predicate: $predicate}})-[:TARGET]->(target:Entity)
        WHERE true {date_clause}
        RETURN actor.name AS actor, ev.predicate AS predicate,
               target.name AS target, toString(ev.timestamp) AS timestamp,
               ev.event_id AS event_id, ev.source AS source
        ORDER BY ev.timestamp
        LIMIT $limit
        """

        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Fact verification
    # ------------------------------------------------------------------

    def verify_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        timestamp: str,
    ) -> dict[str, Any]:
        """Check whether an exact temporal fact exists in the graph.

        Args:
            subject: Actor entity name.
            predicate: Relation label.
            obj: Target entity name.
            timestamp: ISO date string to match exactly.

        Returns:
            Dictionary with ``exists`` (bool) and ``matching_facts``
            (list of matching event records).
        """
        cypher = """
        MATCH (a:Entity {name: $subject})-[:ACTOR]->(ev:Event {predicate: $predicate})-[:TARGET]->(t:Entity {name: $obj})
        WHERE ev.timestamp = date($timestamp)
        RETURN a.name AS actor, ev.predicate AS predicate,
               t.name AS target, toString(ev.timestamp) AS timestamp,
               ev.event_id AS event_id, ev.source AS source
        """
        params = {
            "subject": subject,
            "predicate": predicate,
            "obj": obj,
            "timestamp": timestamp,
        }

        with self._driver.session() as session:
            result = session.run(cypher, **params)
            records = [dict(r) for r in result]

        return {"exists": len(records) > 0, "matching_facts": records}

    def verify_entity_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> dict[str, Any]:
        """Check whether a subject-predicate-object triple exists at any timestamp.

        Args:
            subject: Actor entity name.
            predicate: Relation label.
            obj: Target entity name.

        Returns:
            Dictionary with ``exists`` (bool) and ``timestamps`` (sorted
            list of ISO date strings where the triple holds).
        """
        cypher = """
        MATCH (a:Entity {name: $subject})-[:ACTOR]->(ev:Event {predicate: $predicate})-[:TARGET]->(t:Entity {name: $obj})
        RETURN DISTINCT toString(ev.timestamp) AS timestamp
        ORDER BY ev.timestamp
        """
        params = {"subject": subject, "predicate": predicate, "obj": obj}

        with self._driver.session() as session:
            result = session.run(cypher, **params)
            timestamps = [record["timestamp"] for record in result]

        return {"exists": len(timestamps) > 0, "timestamps": timestamps}

    # ------------------------------------------------------------------
    # Temporal neighbourhood
    # ------------------------------------------------------------------

    def get_temporal_neighbors(
        self,
        entity: str,
        timestamp: str,
        window_days: int = 30,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find events temporally close to a reference timestamp.

        Returns events involving the entity within +/- ``window_days``
        of the given *timestamp*, ordered by temporal proximity.

        Args:
            entity: Name of the entity.
            timestamp: Reference ISO date string.
            window_days: Half-width of the temporal window in days.
            limit: Maximum number of results.

        Returns:
            List of event dictionaries ordered by proximity to the
            reference timestamp.
        """
        cypher = """
        MATCH (actor:Entity)-[:ACTOR]->(ev:Event)-[:TARGET]->(target:Entity)
        WHERE (actor.name = $entity OR target.name = $entity)
          AND ev.timestamp >= date($timestamp) - duration({days: $window_days})
          AND ev.timestamp <= date($timestamp) + duration({days: $window_days})
        RETURN actor.name AS actor, ev.predicate AS predicate,
               target.name AS target, toString(ev.timestamp) AS timestamp,
               ev.event_id AS event_id, ev.source AS source,
               abs(duration.between(date($timestamp), ev.timestamp).days) AS distance_days
        ORDER BY distance_days ASC, ev.timestamp
        LIMIT $limit
        """
        params = {
            "entity": entity,
            "timestamp": timestamp,
            "window_days": window_days,
            "limit": limit,
        }

        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def get_entity_count(self) -> int:
        """Return the total number of Entity nodes in the graph.

        Returns:
            Integer count of Entity nodes.
        """
        cypher = "MATCH (e:Entity) RETURN count(e) AS cnt"
        with self._driver.session() as session:
            result = session.run(cypher)
            record = result.single()
            return record["cnt"] if record else 0

    def get_event_count(self) -> int:
        """Return the total number of Event nodes in the graph.

        Returns:
            Integer count of Event nodes.
        """
        cypher = "MATCH (ev:Event) RETURN count(ev) AS cnt"
        with self._driver.session() as session:
            result = session.run(cypher)
            record = result.single()
            return record["cnt"] if record else 0

    def get_all_entity_names(self) -> list[str]:
        """Return a sorted list of all entity names in the graph.

        Returns:
            Sorted list of entity name strings.
        """
        cypher = "MATCH (e:Entity) RETURN e.name AS name ORDER BY e.name"
        with self._driver.session() as session:
            result = session.run(cypher)
            return [record["name"] for record in result]

    def get_all_predicates(self) -> list[str]:
        """Return a sorted list of all distinct predicates in the graph.

        Returns:
            Sorted list of predicate strings.
        """
        cypher = (
            "MATCH (ev:Event) "
            "RETURN DISTINCT ev.predicate AS predicate "
            "ORDER BY predicate"
        )
        with self._driver.session() as session:
            result = session.run(cypher)
            return [record["predicate"] for record in result]
