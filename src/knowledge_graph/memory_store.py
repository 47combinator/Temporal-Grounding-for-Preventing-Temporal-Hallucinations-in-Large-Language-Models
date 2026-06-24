"""
In-memory temporal knowledge graph store using NetworkX.

This module provides the same interface as TemporalKGStore but stores
everything in memory using NetworkX. No external database required.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)


class InMemoryKGStore:
    """
    In-memory temporal knowledge graph using NetworkX.

    Provides the same query interface as TemporalKGStore so it can be
    used as a drop-in replacement when Neo4j is not available.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._entities: dict[str, dict] = {}  # entity_id -> {name, ...}
        self._events: dict[str, dict] = {}    # event_id -> {predicate, timestamp, ...}
        self._entity_name_index: dict[str, str] = {}  # lower(name) -> entity_id

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def add_entity(self, entity_id: str, name: str, **attrs: Any) -> None:
        """Add an entity node."""
        self._entities[entity_id] = {"name": name, "entity_id": entity_id, **attrs}
        self._entity_name_index[name.lower()] = entity_id
        self.graph.add_node(f"entity:{entity_id}", type="entity", name=name, **attrs)

    def add_event(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        timestamp: str,
        source: str = "icews14",
    ) -> str:
        """Add a temporal event (fact) linking subject -> event -> object."""
        event_id = str(uuid.uuid4())[:12]
        event_data = {
            "event_id": event_id,
            "predicate": predicate,
            "timestamp": timestamp,
            "source": source,
            "subject_id": subject_id,
            "object_id": object_id,
        }
        self._events[event_id] = event_data

        event_node = f"event:{event_id}"
        subj_node = f"entity:{subject_id}"
        obj_node = f"entity:{object_id}"

        self.graph.add_node(event_node, type="event", **event_data)

        # Ensure entity nodes exist
        if not self.graph.has_node(subj_node):
            name = self._entities.get(subject_id, {}).get("name", subject_id)
            self.graph.add_node(subj_node, type="entity", name=name)
        if not self.graph.has_node(obj_node):
            name = self._entities.get(object_id, {}).get("name", object_id)
            self.graph.add_node(obj_node, type="entity", name=name)

        self.graph.add_edge(subj_node, event_node, relation="ACTOR")
        self.graph.add_edge(event_node, obj_node, relation="TARGET")

        return event_id

    def bulk_load_facts(self, facts: list[dict]) -> int:
        """Load a batch of facts. Each fact: {subject_id, predicate, object_id, timestamp}."""
        count = 0
        for fact in facts:
            self.add_event(
                subject_id=str(fact["subject_id"]),
                predicate=fact["predicate"],
                object_id=str(fact["object_id"]),
                timestamp=fact["timestamp"],
                source=fact.get("source", "icews14"),
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Queries (same interface as TemporalKGStore)
    # ------------------------------------------------------------------

    def entity_exists(self, entity_name: str) -> bool:
        """Check if an entity exists by name (case-insensitive)."""
        return entity_name.lower() in self._entity_name_index

    def get_entity_id(self, entity_name: str) -> Optional[str]:
        """Get entity ID from name."""
        return self._entity_name_index.get(entity_name.lower())

    def find_entities_fuzzy(self, name: str, limit: int = 5) -> list[dict]:
        """Find entities whose name contains the given substring."""
        name_lower = name.lower()
        results = []
        for eid, edata in self._entities.items():
            if name_lower in edata["name"].lower():
                results.append(edata)
                if len(results) >= limit:
                    break
        return results

    def get_events_by_entity(
        self, entity_name: str, limit: int = 50
    ) -> list[dict]:
        """Get all events involving an entity (as subject or object)."""
        entity_id = self.get_entity_id(entity_name)
        if not entity_id:
            return []

        entity_node = f"entity:{entity_id}"
        results = []

        # Events where entity is ACTOR (subject)
        if self.graph.has_node(entity_node):
            for _, event_node in self.graph.out_edges(entity_node):
                if self.graph.nodes[event_node].get("type") == "event":
                    event_data = dict(self.graph.nodes[event_node])
                    # Find target
                    for _, target_node in self.graph.out_edges(event_node):
                        if self.graph.nodes[target_node].get("type") == "entity":
                            event_data["subject"] = entity_name
                            event_data["object"] = self.graph.nodes[target_node].get("name", "")
                            results.append(event_data)
                            break

            # Events where entity is TARGET (object)
            for source_node, _ in self.graph.in_edges(entity_node):
                if self.graph.nodes[source_node].get("type") == "event":
                    event_data = dict(self.graph.nodes[source_node])
                    # Find subject
                    for actor_node, _ in self.graph.in_edges(source_node):
                        if self.graph.nodes[actor_node].get("type") == "entity":
                            event_data["subject"] = self.graph.nodes[actor_node].get("name", "")
                            event_data["object"] = entity_name
                            results.append(event_data)
                            break

        return results[:limit]

    def get_events_by_entity_and_time(
        self,
        entity_name: str,
        start_date: str,
        end_date: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get events involving entity within a time range."""
        all_events = self.get_events_by_entity(entity_name, limit=9999)
        filtered = []
        for ev in all_events:
            ts = ev.get("timestamp", "")
            if start_date <= ts <= end_date:
                filtered.append(ev)
        return filtered[:limit]

    def get_events_between_entities(
        self,
        entity1_name: str,
        entity2_name: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get all events between two entities."""
        id1 = self.get_entity_id(entity1_name)
        id2 = self.get_entity_id(entity2_name)
        if not id1 or not id2:
            return []

        results = []
        for eid, edata in self._events.items():
            if (edata["subject_id"] == id1 and edata["object_id"] == id2) or \
               (edata["subject_id"] == id2 and edata["object_id"] == id1):
                # Attach readable names
                subj_name = self._entities.get(edata["subject_id"], {}).get("name", "")
                obj_name = self._entities.get(edata["object_id"], {}).get("name", "")
                result = {**edata, "subject": subj_name, "object": obj_name}
                results.append(result)

        return results[:limit]

    def get_events_between_entities_in_range(
        self,
        entity1_name: str,
        entity2_name: str,
        start_date: str,
        end_date: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get events between two entities within a time range."""
        all_events = self.get_events_between_entities(entity1_name, entity2_name, limit=9999)
        filtered = [
            ev for ev in all_events
            if start_date <= ev.get("timestamp", "") <= end_date
        ]
        return filtered[:limit]

    def verify_fact(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        timestamp: str,
    ) -> dict:
        """
        Verify a specific temporal fact against the knowledge graph.

        Returns a dict with verification details:
        - entity_exists: bool
        - relation_exists: bool
        - timestamp_match: bool
        - matching_facts: list of supporting facts
        """
        subj_id = self.get_entity_id(subject_name)
        obj_id = self.get_entity_id(object_name)

        result = {
            "subject_exists": subj_id is not None,
            "object_exists": obj_id is not None,
            "relation_exists": False,
            "timestamp_match": False,
            "exact_match": False,
            "matching_facts": [],
            "closest_timestamps": [],
        }

        if not subj_id or not obj_id:
            return result

        # Find matching events
        for eid, edata in self._events.items():
            if edata["subject_id"] == subj_id and edata["object_id"] == obj_id:
                if edata["predicate"].lower() == predicate.lower():
                    result["relation_exists"] = True
                    result["closest_timestamps"].append(edata["timestamp"])
                    if edata["timestamp"] == timestamp:
                        result["timestamp_match"] = True
                        result["exact_match"] = True
                        result["matching_facts"].append({
                            "subject": subject_name,
                            "predicate": predicate,
                            "object": object_name,
                            "timestamp": edata["timestamp"],
                        })

        return result

    def get_all_entity_names(self) -> list[str]:
        """Return all entity names."""
        return [e["name"] for e in self._entities.values()]

    def get_all_relations(self) -> list[str]:
        """Return all unique relation/predicate names."""
        return list({e["predicate"] for e in self._events.values()})

    def get_all_timestamps(self) -> list[str]:
        """Return all unique timestamps."""
        return sorted({e["timestamp"] for e in self._events.values()})

    def get_statistics(self) -> dict:
        """Return graph statistics."""
        return {
            "num_entities": len(self._entities),
            "num_events": len(self._events),
            "num_relations": len(self.get_all_relations()),
            "num_timestamps": len(self.get_all_timestamps()),
        }

    # ------------------------------------------------------------------
    # Persistence (save/load to disk)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the graph to a JSON file for reloading later."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": self._entities,
            "events": self._events,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved graph to %s (%d entities, %d events)",
                     path, len(self._entities), len(self._events))

    def load(self, path: Path) -> None:
        """Load a previously saved graph from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._entities = data["entities"]
        self._events = data["events"]

        # Rebuild indexes
        self._entity_name_index = {}
        for eid, edata in self._entities.items():
            self._entity_name_index[edata["name"].lower()] = eid

        # Rebuild NetworkX graph
        self.graph = nx.DiGraph()
        for eid, edata in self._entities.items():
            self.graph.add_node(f"entity:{eid}", type="entity", **edata)
        for evid, evdata in self._events.items():
            event_node = f"event:{evid}"
            subj_node = f"entity:{evdata['subject_id']}"
            obj_node = f"entity:{evdata['object_id']}"
            self.graph.add_node(event_node, type="event", **evdata)
            self.graph.add_edge(subj_node, event_node, relation="ACTOR")
            self.graph.add_edge(event_node, obj_node, relation="TARGET")

        logger.info("Loaded graph from %s (%d entities, %d events)",
                     path, len(self._entities), len(self._events))

    def close(self) -> None:
        """No-op for compatibility with TemporalKGStore."""
        pass
