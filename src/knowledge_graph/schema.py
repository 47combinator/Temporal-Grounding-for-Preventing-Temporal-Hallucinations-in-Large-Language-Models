"""
Neo4j schema management for the Temporal Grounding and Verification Framework.

Defines the database constraints and indexes that support the Event Node
Reification model:

    (Entity)-[:ACTOR]->(Event)-[:TARGET]->(Entity)

The Event node carries: event_id, predicate, timestamp (native Neo4j date),
source, and optional valid_from / valid_to fields for interval-scoped facts.
"""

import logging
from typing import Any

from neo4j import Driver  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class Neo4jSchema:
    """Manages Neo4j schema elements (constraints and indexes).

    All DDL statements use ``IF NOT EXISTS`` so calling ``setup()``
    multiple times is safe and idempotent.
    """

    def __init__(self, driver: Driver) -> None:
        """Initialise the schema manager.

        Args:
            driver: An active ``neo4j.Driver`` instance.
        """
        self._driver: Driver = driver

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def create_constraints(self) -> None:
        """Create uniqueness constraints on core node properties.

        Creates the following constraints (idempotent):
        - ``Entity.name`` must be unique.
        - ``Event.event_id`` must be unique.
        """
        constraints: list[str] = [
            (
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT event_id_unique IF NOT EXISTS "
                "FOR (ev:Event) REQUIRE ev.event_id IS UNIQUE"
            ),
        ]

        with self._driver.session() as session:
            for cypher in constraints:
                session.run(cypher)
                logger.info("Executed constraint: %s", cypher.split("IF NOT EXISTS")[0].strip())

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    def create_indexes(self) -> None:
        """Create indexes to accelerate temporal and predicate queries.

        Creates the following indexes (idempotent):
        - Range index on ``Event.timestamp`` for date-range filtering.
        - Range index on ``Event.predicate`` for predicate look-ups.
        - Composite range index on ``(Event.timestamp, Event.predicate)``
          for combined temporal-predicate queries.
        """
        indexes: list[str] = [
            (
                "CREATE INDEX event_timestamp_idx IF NOT EXISTS "
                "FOR (ev:Event) ON (ev.timestamp)"
            ),
            (
                "CREATE INDEX event_predicate_idx IF NOT EXISTS "
                "FOR (ev:Event) ON (ev.predicate)"
            ),
            (
                "CREATE INDEX event_ts_pred_idx IF NOT EXISTS "
                "FOR (ev:Event) ON (ev.timestamp, ev.predicate)"
            ),
        ]

        with self._driver.session() as session:
            for cypher in indexes:
                session.run(cypher)
                logger.info("Executed index: %s", cypher.split("IF NOT EXISTS")[0].strip())

    # ------------------------------------------------------------------
    # Composite operations
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Run full schema setup: constraints first, then indexes.

        Safe to call repeatedly -- all statements use ``IF NOT EXISTS``.
        """
        logger.info("Setting up Neo4j schema...")
        self.create_constraints()
        self.create_indexes()
        logger.info("Neo4j schema setup complete.")

    def clear_database(self) -> None:
        """Delete all nodes and relationships in the database.

        This is a destructive operation intended for testing and
        development resets.  Uses ``CALL {} IN TRANSACTIONS`` to avoid
        out-of-memory errors on large graphs.
        """
        logger.warning("Clearing all data from the Neo4j database.")

        # Use batched deletion to handle large datasets without
        # exhausting transaction memory.
        cypher = (
            "MATCH (n) "
            "CALL { WITH n DETACH DELETE n } "
            "IN TRANSACTIONS OF 10000 ROWS"
        )

        with self._driver.session() as session:
            result: Any = session.run(cypher)
            summary = result.consume()
            logger.info(
                "Database cleared. Nodes deleted: %d",
                summary.counters.nodes_deleted,
            )
