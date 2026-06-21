"""
Script to set up the Neo4j database schema.

Run this once after installing Neo4j to create the required constraints
and indexes for the temporal knowledge graph.

Usage:
    python -m scripts.setup_neo4j
"""

import sys
from pathlib import Path

# Ensure project root is on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from src.knowledge_graph.graph_store import TemporalKGStore


def main() -> None:
    """Connect to Neo4j and create the schema (constraints and indexes)."""
    print(f"Connecting to Neo4j at {NEO4J_URI} ...")

    try:
        with TemporalKGStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as store:
            print("Connected successfully.")
            print("Setting up schema (constraints and indexes) ...")
            store.setup_schema()
            print("Schema setup complete.")

            entity_count = store.get_entity_count()
            event_count = store.get_event_count()
            print(f"Current database state: {entity_count} entities, {event_count} events.")
    except Exception as exc:
        print(f"Error: {exc}")
        print(
            "Make sure Neo4j is running and the credentials in .env are correct."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
