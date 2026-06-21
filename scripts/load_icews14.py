"""
Script to download ICEWS14 and load it into Neo4j.

Downloads the dataset from Hugging Face (if not already present), parses it,
and bulk-loads all temporal facts into the Neo4j graph database.

Usage:
    python -m scripts.load_icews14
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ICEWS14_DIR,
)
from src.data_loader.download import download_icews14
from src.data_loader.icews_loader import ICEWS14Loader
from src.knowledge_graph.graph_store import TemporalKGStore
from src.knowledge_graph.graph_builder import GraphBuilder


def main() -> None:
    """Download ICEWS14, parse it, and load into Neo4j."""

    # -- Step 1: Download if needed ----------------------------------------
    if not ICEWS14_DIR.exists() or not any(ICEWS14_DIR.iterdir()):
        print("ICEWS14 data not found. Downloading ...")
        download_icews14(ICEWS14_DIR)
    else:
        print(f"ICEWS14 data found at {ICEWS14_DIR}")

    # -- Step 2: Parse ------------------------------------------------------
    print("Parsing ICEWS14 dataset ...")
    loader = ICEWS14Loader(ICEWS14_DIR)
    stats = loader.get_statistics()

    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # -- Step 3: Load into Neo4j -------------------------------------------
    print(f"\nConnecting to Neo4j at {NEO4J_URI} ...")

    try:
        with TemporalKGStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as store:
            print("Connected.")
            builder = GraphBuilder(store)

            start_time = time.time()
            builder.build(icews_loader=loader)
            elapsed = time.time() - start_time

            entity_count = store.get_entity_count()
            event_count = store.get_event_count()

            print(f"\nLoading complete in {elapsed:.1f} seconds.")
            print(f"Graph contains: {entity_count} entities, {event_count} events.")

    except Exception as exc:
        print(f"Error: {exc}")
        print(
            "Make sure Neo4j is running and the credentials in .env are correct."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
