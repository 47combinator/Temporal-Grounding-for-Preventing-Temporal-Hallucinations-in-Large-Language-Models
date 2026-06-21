"""
Knowledge graph builder for the Temporal Grounding and Verification Framework.

Orchestrates the loading of temporal datasets (ICEWS14, CronQuestions) into
the Neo4j graph store via ``TemporalKGStore``.  Handles event ID generation,
progress reporting, and multi-dataset builds.
"""

import logging
from typing import Any

from tqdm import tqdm

from src.data_loader.icews_loader import ICEWS14Loader
from src.data_loader.cronquestions_loader import CronQuestionsLoader
from src.knowledge_graph.graph_store import TemporalKGStore
from config.settings import NEO4J_BATCH_SIZE

logger = logging.getLogger(__name__)

# Recognised ICEWS14 splits in canonical order.
_ICEWS_SPLITS: list[str] = ["train", "valid", "test"]


class GraphBuilder:
    """Orchestrates the construction of the temporal knowledge graph.

    Reads facts from dataset loaders, transforms them into the
    event-reified format expected by ``TemporalKGStore``, and bulk-inserts
    them into Neo4j.
    """

    def __init__(self, store: TemporalKGStore) -> None:
        """Initialise the builder with a graph store.

        Args:
            store: An active ``TemporalKGStore`` instance connected to
                Neo4j.
        """
        self._store: TemporalKGStore = store

    # ------------------------------------------------------------------
    # ICEWS14
    # ------------------------------------------------------------------

    def load_icews14(
        self,
        loader: ICEWS14Loader,
        splits: list[str] | None = None,
    ) -> int:
        """Load ICEWS14 facts into the knowledge graph.

        Each fact is converted into an event-reified sub-graph with a
        unique ``event_id`` of the form ``icews14-<split>-<index>``.

        Args:
            loader: An initialised ``ICEWS14Loader`` pointing at the
                downloaded ICEWS14 data.
            splits: List of splits to load (e.g. ``["train"]``).
                Defaults to all splits: train, valid, test.

        Returns:
            Total number of nodes created across all splits.
        """
        if splits is None:
            splits = list(_ICEWS_SPLITS)

        total_created = 0

        for split in splits:
            logger.info("Loading ICEWS14 '%s' split...", split)
            facts = loader.load_split(split)

            events: list[dict[str, Any]] = []
            for idx, fact in enumerate(
                tqdm(facts, desc=f"Preparing icews14-{split}", unit="fact")
            ):
                events.append(
                    {
                        "event_id": f"icews14-{split}-{idx}",
                        "actor": fact["subject"],
                        "target": fact["object"],
                        "predicate": fact["relation"],
                        "timestamp": fact["timestamp"],
                        "source": "icews14",
                    }
                )

            logger.info(
                "Inserting %d events from ICEWS14 '%s' split...",
                len(events),
                split,
            )
            created = self._store.batch_insert_events(events)
            total_created += created
            logger.info(
                "ICEWS14 '%s': %d nodes created.", split, created
            )

        logger.info(
            "ICEWS14 loading complete. Total nodes created: %d", total_created
        )
        return total_created

    # ------------------------------------------------------------------
    # CronQuestions KG
    # ------------------------------------------------------------------

    def load_cronquestions_kg(self, loader: CronQuestionsLoader) -> int:
        """Load CronQuestions temporal KG facts into the graph.

        Each fact is converted into an event-reified sub-graph with a
        unique ``event_id`` of the form ``cronq-<index>``.  For facts
        with explicit start/end times, the Event node receives
        ``valid_from`` and ``valid_to`` properties in addition to the
        primary ``timestamp`` (set to ``start_time`` when available).

        Args:
            loader: An initialised ``CronQuestionsLoader`` pointing at
                the extracted CronQuestions data.

        Returns:
            Total number of nodes created.
        """
        logger.info("Loading CronQuestions KG...")
        kg_facts = loader.load_kg()

        events: list[dict[str, Any]] = []
        for idx, fact in enumerate(
            tqdm(kg_facts, desc="Preparing cronq", unit="fact")
        ):
            # Determine the primary timestamp: prefer start_time, fall
            # back to end_time, or skip if neither is available.
            primary_ts = fact.get("start_time") or fact.get("end_time")
            if not primary_ts:
                logger.debug(
                    "Skipping fact cronq-%d: no valid timestamp.", idx
                )
                continue

            event: dict[str, Any] = {
                "event_id": f"cronq-{idx}",
                "actor": fact["subject"],
                "target": fact["object"],
                "predicate": fact["relation"],
                "timestamp": primary_ts,
                "source": "cronquestions",
            }

            # Attach interval bounds when available.
            if fact.get("start_time"):
                event["valid_from"] = fact["start_time"]
            if fact.get("end_time"):
                event["valid_to"] = fact["end_time"]

            events.append(event)

        logger.info("Inserting %d events from CronQuestions KG...", len(events))
        created = self._store.batch_insert_events(events)
        logger.info("CronQuestions KG: %d nodes created.", created)
        return created

    # ------------------------------------------------------------------
    # Full build
    # ------------------------------------------------------------------

    def build(
        self,
        icews_loader: ICEWS14Loader | None = None,
        cronq_loader: CronQuestionsLoader | None = None,
    ) -> dict[str, Any]:
        """Run a full knowledge graph build.

        Executes the following steps in order:

        1. Set up the Neo4j schema (constraints and indexes).
        2. Load ICEWS14 data (if a loader is provided).
        3. Load CronQuestions KG data (if a loader is provided).

        Args:
            icews_loader: Optional ``ICEWS14Loader`` for ICEWS14 data.
            cronq_loader: Optional ``CronQuestionsLoader`` for
                CronQuestions data.

        Returns:
            Summary dictionary with counts of entities and events
            created.
        """
        logger.info("Starting full knowledge graph build...")

        # Step 1: Schema
        self._store.setup_schema()

        # Step 2: ICEWS14
        icews_nodes = 0
        if icews_loader is not None:
            icews_nodes = self.load_icews14(icews_loader)

        # Step 3: CronQuestions
        cronq_nodes = 0
        if cronq_loader is not None:
            cronq_nodes = self.load_cronquestions_kg(cronq_loader)

        # Summary
        summary: dict[str, Any] = {
            "icews14_nodes_created": icews_nodes,
            "cronquestions_nodes_created": cronq_nodes,
            "total_entities": self._store.get_entity_count(),
            "total_events": self._store.get_event_count(),
        }

        logger.info("Knowledge graph build complete. Summary: %s", summary)
        return summary
