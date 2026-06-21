"""
Script to generate the Temporal Hallucination Benchmark.

Takes verified facts from the Neo4j knowledge graph, applies controlled
corruptions (timestamp, entity, relation, ordering), filters accidental
positives, and saves the benchmark as a JSON file.

Usage:
    python -m scripts.generate_benchmark
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BENCHMARK_DIR,
)
from src.knowledge_graph.graph_store import TemporalKGStore
from src.hallucination.benchmark_generator import BenchmarkGenerator


def main() -> None:
    """Generate the Temporal Hallucination Benchmark and save to disk."""

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARK_DIR / "temporal_hallucination_benchmark.json"
    stats_path = BENCHMARK_DIR / "benchmark_stats.json"

    print(f"Connecting to Neo4j at {NEO4J_URI} ...")

    try:
        with TemporalKGStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as store:
            print("Connected.")

            event_count = store.get_event_count()
            if event_count == 0:
                print(
                    "Error: No events in the graph. "
                    "Run 'python -m scripts.load_icews14' first."
                )
                sys.exit(1)

            print(f"Graph contains {event_count} events.")
            print("Generating benchmark ...")

            generator = BenchmarkGenerator(store)
            start_time = time.time()
            benchmark = generator.generate()
            elapsed = time.time() - start_time

            # Save benchmark
            generator.save(benchmark, output_path)

            # Save statistics separately
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(benchmark["statistics"], f, indent=2)

            stats = benchmark["statistics"]
            print(f"\nBenchmark generated in {elapsed:.1f} seconds.")
            print(f"Positive examples:  {stats.get('num_positives', 'N/A')}")
            print(f"Negative examples:  {stats.get('num_negatives', 'N/A')}")
            print(f"Total examples:     {stats.get('total', 'N/A')}")
            print(f"\nSaved to: {output_path}")
            print(f"Stats at: {stats_path}")

    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
