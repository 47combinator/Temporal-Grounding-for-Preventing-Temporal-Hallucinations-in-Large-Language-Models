"""
Benchmark generator for temporal hallucination detection.

BenchmarkGenerator creates a balanced dataset of *positive* (genuine) and
*negative* (corrupted) temporal facts suitable for evaluating the accuracy
of the verification pipeline.  Negative examples are produced by applying
controlled corruptions (timestamp, entity, relation, ordering) to real
facts drawn from the knowledge graph.
"""

from __future__ import annotations

import copy
import json
import logging
import random as _random_module
from pathlib import Path
from typing import Any

from src.hallucination.corruption_strategies import (
    corrupt_entity,
    corrupt_relation,
    corrupt_temporal_order,
    corrupt_timestamp,
)

logger = logging.getLogger(__name__)


class BenchmarkGenerator:
    """Create the Temporal Hallucination Benchmark dataset.

    The generator loads all facts from the backing knowledge-graph store,
    then produces corrupted variants to serve as negative examples.  An
    accidental-positive filter ensures that no corruption coincidentally
    recreates a genuine fact.

    The graph store must expose:

    * ``query_entity_events(entity_name, limit=...)`` -- for iterating facts.
    * ``get_all_entity_names()`` -- all entity names.
    * ``get_all_predicates()`` -- all predicate labels.

    Parameters
    ----------
    graph_store : object
        A TemporalKGStore (or compatible) instance.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, graph_store: Any, seed: int = 42) -> None:
        """Initialise the generator with a KG store and random seed.

        Parameters
        ----------
        graph_store : object
            Graph store providing entity/predicate/event access.
        seed : int
            Random seed.
        """
        self._store = graph_store
        self._seed = seed
        self._rng = _random_module.Random(seed)

    # ------------------------------------------------------------------
    # Fact loading
    # ------------------------------------------------------------------

    def _load_facts(self) -> list[dict[str, Any]]:
        """Load all facts from the KG as positive examples.

        If the graph store has a ``get_all_facts()`` method, it is used
        directly.  Otherwise the loader falls back to iterating over all
        entity names and collecting events via ``query_entity_events``,
        de-duplicating by ``event_id``.

        Each returned fact is a dict with at least ``subject``, ``predicate``,
        ``object``, and ``timestamp`` keys.

        Returns
        -------
        list[dict]
            All facts currently stored in the knowledge graph.
        """
        # Prefer a dedicated bulk-export method if available.
        if hasattr(self._store, "get_all_facts"):
            try:
                facts = self._store.get_all_facts()
                logger.info("Loaded %d facts via get_all_facts().", len(facts))
                return facts
            except Exception:
                logger.exception("get_all_facts() failed; falling back to entity scan.")

        # Fallback: iterate over all entities and gather events.
        try:
            entity_names = self._store.get_all_entity_names()
        except Exception:
            logger.exception("Failed to retrieve entity names from the graph store.")
            return []

        seen_ids: set[str] = set()
        facts: list[dict[str, Any]] = []

        for name in entity_names:
            try:
                events = self._store.query_entity_events(name, limit=10_000)
            except Exception:
                logger.warning("Failed to query events for entity '%s'.", name)
                continue

            for evt in events:
                eid = evt.get("event_id", "")
                if eid and eid in seen_ids:
                    continue
                seen_ids.add(eid)

                facts.append(
                    {
                        "subject": evt.get("actor", ""),
                        "predicate": evt.get("predicate", ""),
                        "object": evt.get("target", ""),
                        "timestamp": evt.get("timestamp", ""),
                        "event_id": eid,
                    }
                )

        logger.info("Loaded %d facts from the knowledge graph (entity scan).", len(facts))
        return facts

    # ------------------------------------------------------------------
    # Accidental-positive guard
    # ------------------------------------------------------------------

    @staticmethod
    def _fact_key(fact: dict[str, Any]) -> tuple[str, str, str, str]:
        """Return a hashable tuple representation of a fact.

        Parameters
        ----------
        fact : dict
            A temporal fact.

        Returns
        -------
        tuple[str, str, str, str]
            ``(subject, predicate, object, timestamp)`` tuple.
        """
        return (
            fact.get("subject", ""),
            fact.get("predicate", ""),
            fact.get("object", ""),
            fact.get("timestamp", ""),
        )

    def _is_accidental_positive(
        self, corrupted: dict[str, Any], all_facts_set: set[tuple[str, str, str, str]]
    ) -> bool:
        """Check whether a corruption accidentally reproduces a real fact.

        Parameters
        ----------
        corrupted : dict
            A corrupted fact dict.
        all_facts_set : set[tuple[str, str, str, str]]
            Set of ``(subject, predicate, object, timestamp)`` tuples for
            all genuine facts.

        Returns
        -------
        bool
            ``True`` if the corrupted fact matches a genuine fact.
        """
        return self._fact_key(corrupted) in all_facts_set

    # ------------------------------------------------------------------
    # Per-strategy generators
    # ------------------------------------------------------------------

    def generate_timestamp_corruptions(
        self,
        facts: list[dict[str, Any]],
        all_timestamps: list[str],
    ) -> list[dict[str, Any]]:
        """Generate timestamp-corrupted versions of all facts.

        Parameters
        ----------
        facts : list[dict]
            The positive facts.
        all_timestamps : list[str]
            Pool of all known timestamps.

        Returns
        -------
        list[dict]
            Corrupted facts with ``corruption_type='timestamp'``.
        """
        corrupted: list[dict[str, Any]] = []
        for fact in facts:
            c = corrupt_timestamp(fact, all_timestamps, self._rng)
            corrupted.append(c)
        return corrupted

    def generate_entity_corruptions(
        self,
        facts: list[dict[str, Any]],
        all_entities: list[str],
    ) -> list[dict[str, Any]]:
        """Generate entity-corrupted versions of all facts.

        Parameters
        ----------
        facts : list[dict]
            The positive facts.
        all_entities : list[str]
            Pool of all known entity names.

        Returns
        -------
        list[dict]
            Corrupted facts with ``corruption_type='entity'``.
        """
        corrupted: list[dict[str, Any]] = []
        for fact in facts:
            c = corrupt_entity(fact, all_entities, self._rng)
            corrupted.append(c)
        return corrupted

    def generate_relation_corruptions(
        self,
        facts: list[dict[str, Any]],
        all_relations: list[str],
    ) -> list[dict[str, Any]]:
        """Generate relation-corrupted versions of all facts.

        Parameters
        ----------
        facts : list[dict]
            The positive facts.
        all_relations : list[str]
            Pool of all known relation names.

        Returns
        -------
        list[dict]
            Corrupted facts with ``corruption_type='relation'``.
        """
        corrupted: list[dict[str, Any]] = []
        for fact in facts:
            c = corrupt_relation(fact, all_relations, self._rng)
            corrupted.append(c)
        return corrupted

    def generate_order_corruptions(
        self, facts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate ordering corruptions by swapping timestamps within entity groups.

        Pairs of facts that share a common subject entity have their
        timestamps swapped.  Only pairs with *different* timestamps are
        considered.

        Parameters
        ----------
        facts : list[dict]
            The positive facts.

        Returns
        -------
        list[dict]
            Corrupted facts with ``corruption_type='ordering'``.
        """
        # Group facts by subject entity.
        entity_groups: dict[str, list[dict[str, Any]]] = {}
        for fact in facts:
            key = fact.get("subject", "")
            entity_groups.setdefault(key, []).append(fact)

        corrupted: list[dict[str, Any]] = []
        for _entity, group in entity_groups.items():
            if len(group) < 2:
                continue

            # Sample pairs within the group (cap to avoid combinatorial explosion).
            indices = list(range(len(group)))
            self._rng.shuffle(indices)
            pairs_count = min(len(group) // 2, len(group))
            for k in range(0, pairs_count - 1, 2):
                fa = group[indices[k]]
                fb = group[indices[k + 1]]
                if fa.get("timestamp") == fb.get("timestamp"):
                    continue
                ca, cb = corrupt_temporal_order(fa, fb)
                corrupted.extend([ca, cb])

        return corrupted

    # ------------------------------------------------------------------
    # Full benchmark generation
    # ------------------------------------------------------------------

    def generate(
        self, positive_facts: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Generate the complete Temporal Hallucination Benchmark.

        Parameters
        ----------
        positive_facts : list[dict] or None
            Pre-loaded positive facts.  If ``None`` the generator will
            load them from the graph store.

        Returns
        -------
        dict
            ``{"positives": [...], "negatives": [...], "statistics": {...}}``
        """
        if positive_facts is None:
            positive_facts = self._load_facts()

        if not positive_facts:
            logger.warning("No positive facts available for benchmark generation.")
            return {
                "positives": [],
                "negatives": [],
                "statistics": {
                    "num_positives": 0,
                    "num_negatives": 0,
                    "num_timestamp_corruptions": 0,
                    "num_entity_corruptions": 0,
                    "num_relation_corruptions": 0,
                    "num_order_corruptions": 0,
                    "num_accidental_positives_removed": 0,
                },
            }

        # Build pools.
        all_timestamps = list(
            {f.get("timestamp", "") for f in positive_facts if f.get("timestamp")}
        )
        all_entities = list(
            {f.get("subject", "") for f in positive_facts if f.get("subject")}
            | {f.get("object", "") for f in positive_facts if f.get("object")}
        )
        all_relations = list(
            {f.get("predicate", "") for f in positive_facts if f.get("predicate")}
        )

        # Generate corruptions.
        ts_corruptions = self.generate_timestamp_corruptions(
            positive_facts, all_timestamps
        )
        ent_corruptions = self.generate_entity_corruptions(
            positive_facts, all_entities
        )
        rel_corruptions = self.generate_relation_corruptions(
            positive_facts, all_relations
        )
        ord_corruptions = self.generate_order_corruptions(positive_facts)

        all_negatives = (
            ts_corruptions + ent_corruptions + rel_corruptions + ord_corruptions
        )

        # Filter accidental positives.
        all_facts_set: set[tuple[str, str, str, str]] = {
            self._fact_key(f) for f in positive_facts
        }
        filtered_negatives: list[dict[str, Any]] = []
        accidental_count = 0
        for neg in all_negatives:
            if self._is_accidental_positive(neg, all_facts_set):
                accidental_count += 1
            else:
                filtered_negatives.append(neg)

        # Label for downstream use.
        for fact in positive_facts:
            fact["label"] = 1
        for neg in filtered_negatives:
            neg["label"] = 0

        statistics = {
            "num_positives": len(positive_facts),
            "num_negatives": len(filtered_negatives),
            "num_timestamp_corruptions": len(ts_corruptions),
            "num_entity_corruptions": len(ent_corruptions),
            "num_relation_corruptions": len(rel_corruptions),
            "num_order_corruptions": len(ord_corruptions),
            "num_accidental_positives_removed": accidental_count,
        }

        logger.info(
            "Benchmark generated: %d positives, %d negatives "
            "(%d accidental positives removed).",
            statistics["num_positives"],
            statistics["num_negatives"],
            statistics["num_accidental_positives_removed"],
        )

        return {
            "positives": positive_facts,
            "negatives": filtered_negatives,
            "statistics": statistics,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, benchmark: dict[str, Any], output_path: Path) -> None:
        """Save the benchmark dataset to a JSON file.

        Parameters
        ----------
        benchmark : dict
            The benchmark dict returned by :meth:`generate`.
        output_path : Path
            Destination file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(benchmark, fh, indent=2, default=str)

        logger.info("Benchmark saved to %s", output_path)

    @staticmethod
    def load(input_path: Path) -> dict[str, Any]:
        """Load a benchmark dataset from a JSON file.

        Parameters
        ----------
        input_path : Path
            Source file path.

        Returns
        -------
        dict
            The benchmark dict with ``positives``, ``negatives``, and
            ``statistics`` keys.
        """
        input_path = Path(input_path)
        with open(input_path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)

        logger.info("Benchmark loaded from %s", input_path)
        return data
