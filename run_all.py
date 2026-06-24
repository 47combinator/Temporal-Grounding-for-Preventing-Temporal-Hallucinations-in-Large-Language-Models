"""
Master runner: Downloads ICEWS14 dataset, builds the in-memory knowledge graph,
generates the temporal hallucination benchmark, and runs the full evaluation.

Usage:
    python run_all.py
"""

import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_DIR,
    ICEWS14_DIR,
    PROCESSED_DATA_DIR,
    BENCHMARK_DIR,
    RANDOM_SEED,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_all")


# ===================================================================
# STEP 1: Download and parse ICEWS14
# ===================================================================

def step1_load_data():
    """Download ICEWS14 if not present, then parse it."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading ICEWS14 dataset")
    logger.info("=" * 60)

    # Download if not already present
    if not (ICEWS14_DIR / "train.txt").exists():
        logger.info("ICEWS14 not found locally, downloading from Hugging Face...")
        from src.data_loader.download import download_icews14
        download_icews14()
    else:
        logger.info("ICEWS14 already downloaded at %s", ICEWS14_DIR)

    from src.data_loader.icews_loader import ICEWS14Loader

    loader = ICEWS14Loader(data_dir=str(ICEWS14_DIR))

    # load_all() returns dict[split_name -> list[fact_dict]]
    splits = loader.load_all()

    # Flatten all splits into one big list
    all_facts = []
    for split_name, split_facts in splits.items():
        logger.info("  %s split: %d facts", split_name, len(split_facts))
        all_facts.extend(split_facts)

    # Normalize key names: the loader uses "relation", our pipeline uses "predicate"
    for fact in all_facts:
        if "relation" in fact and "predicate" not in fact:
            fact["predicate"] = fact["relation"]

    stats = loader.get_statistics()
    logger.info("Total facts loaded: %d", len(all_facts))
    logger.info("  Entities:   %d", stats["num_entities"])
    logger.info("  Relations:  %d", stats["num_relations"])
    logger.info("  Timestamps: %d", stats["num_timestamps"])

    # Show sample
    logger.info("Sample facts:")
    for f in all_facts[:3]:
        logger.info("  %s | %s | %s | %s",
                     f["subject"], f["predicate"], f["object"], f["timestamp"])

    return loader, all_facts


# ===================================================================
# STEP 2: Build in-memory knowledge graph
# ===================================================================

def step2_build_graph(loader, facts):
    """Build the in-memory knowledge graph from ICEWS14 facts."""
    logger.info("=" * 60)
    logger.info("STEP 2: Building in-memory knowledge graph")
    logger.info("=" * 60)

    from src.knowledge_graph.memory_store import InMemoryKGStore

    store = InMemoryKGStore()

    # Add all entities -- loader.get_entities() returns a sorted list of names
    # We also need the id->name mapping from the internal cache
    loader._ensure_mappings()
    id2entity = loader._id2entity
    for eid, name in id2entity.items():
        store.add_entity(str(eid), name)
    logger.info("Added %d entities", len(id2entity))

    # Add all events
    batch = []
    for fact in facts:
        batch.append({
            "subject_id": str(fact["subject_id"]),
            "predicate": fact["predicate"],
            "object_id": str(fact["object_id"]),
            "timestamp": fact["timestamp"],
        })

    loaded = store.bulk_load_facts(batch)
    logger.info("Loaded %d events into graph", loaded)

    # Save to disk for reuse
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = PROCESSED_DATA_DIR / "knowledge_graph.json"
    store.save(graph_path)
    logger.info("Graph saved to %s", graph_path)

    stats = store.get_statistics()
    logger.info("Graph stats: %s", stats)

    return store


# ===================================================================
# STEP 3: Generate temporal hallucination benchmark
# ===================================================================

def step3_generate_benchmark(store, facts):
    """Generate the temporal hallucination benchmark using corruption."""
    logger.info("=" * 60)
    logger.info("STEP 3: Generating temporal hallucination benchmark")
    logger.info("=" * 60)

    from src.hallucination.corruption_strategies import (
        corrupt_timestamp,
        corrupt_entity,
        corrupt_relation,
        corrupt_temporal_order,
    )

    rng = random.Random(RANDOM_SEED)
    all_timestamps = store.get_all_timestamps()
    all_entities = store.get_all_entity_names()
    all_relations = store.get_all_relations()

    # Sample facts for benchmark (use 5000 for manageable size)
    sample_size = min(5000, len(facts))
    sampled_facts = rng.sample(facts, sample_size)

    benchmark = []
    corruption_counts = {"positive": 0, "timestamp": 0, "entity": 0, "relation": 0, "ordering": 0}

    for fact in sampled_facts:
        # Positive example (correct fact)
        benchmark.append({
            "subject": fact["subject"],
            "predicate": fact["predicate"],
            "object": fact["object"],
            "timestamp": fact["timestamp"],
            "label": "correct",
            "corruption_type": "none",
        })
        corruption_counts["positive"] += 1

        # Timestamp corruption
        corrupted = corrupt_timestamp(fact, all_timestamps, rng)
        benchmark.append({
            "subject": corrupted["subject"],
            "predicate": corrupted["predicate"],
            "object": corrupted["object"],
            "timestamp": corrupted["timestamp"],
            "label": "hallucinated",
            "corruption_type": "timestamp",
        })
        corruption_counts["timestamp"] += 1

        # Entity corruption
        corrupted = corrupt_entity(fact, all_entities, rng)
        benchmark.append({
            "subject": corrupted["subject"],
            "predicate": corrupted["predicate"],
            "object": corrupted["object"],
            "timestamp": corrupted["timestamp"],
            "label": "hallucinated",
            "corruption_type": "entity",
        })
        corruption_counts["entity"] += 1

        # Relation corruption
        corrupted = corrupt_relation(fact, all_relations, rng)
        benchmark.append({
            "subject": corrupted["subject"],
            "predicate": corrupted["predicate"],
            "object": corrupted["object"],
            "timestamp": corrupted["timestamp"],
            "label": "hallucinated",
            "corruption_type": "relation",
        })
        corruption_counts["relation"] += 1

    # Ordering corruption (pairs of facts with same subject)
    subject_groups = {}
    for fact in sampled_facts[:2000]:
        subj = fact["subject"]
        if subj not in subject_groups:
            subject_groups[subj] = []
        subject_groups[subj].append(fact)

    order_count = 0
    for subj, group in subject_groups.items():
        if len(group) >= 2 and order_count < 1000:
            pair = rng.sample(group, 2)
            if pair[0]["timestamp"] != pair[1]["timestamp"]:
                ca, cb = corrupt_temporal_order(pair[0], pair[1])
                benchmark.append({
                    "subject": ca["subject"],
                    "predicate": ca["predicate"],
                    "object": ca["object"],
                    "timestamp": ca["timestamp"],
                    "label": "hallucinated",
                    "corruption_type": "ordering",
                    "original_timestamp": pair[0]["timestamp"],
                })
                order_count += 1
                corruption_counts["ordering"] += 1

    # Save benchmark
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_path = BENCHMARK_DIR / "temporal_hallucination_benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    logger.info("Benchmark generated: %d total examples", len(benchmark))
    for ctype, count in corruption_counts.items():
        logger.info("  %s: %d", ctype, count)
    logger.info("Saved to %s", benchmark_path)

    return benchmark


# ===================================================================
# STEP 4: Run verification on benchmark (no LLM needed)
# ===================================================================

def step4_run_verification(store, benchmark):
    """Run the temporal verifier on the benchmark to evaluate detection accuracy."""
    logger.info("=" * 60)
    logger.info("STEP 4: Running temporal verification on benchmark")
    logger.info("=" * 60)

    results = {
        "total": 0,
        "correct_detections": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "by_type": {},
    }

    for item in benchmark:
        results["total"] += 1
        ctype = item["corruption_type"]
        if ctype not in results["by_type"]:
            results["by_type"][ctype] = {"total": 0, "correct": 0}
        results["by_type"][ctype]["total"] += 1

        # Verify against KG
        verification = store.verify_fact(
            subject_name=item["subject"],
            predicate=item["predicate"],
            object_name=item["object"],
            timestamp=item["timestamp"],
        )

        is_correct_fact = item["label"] == "correct"
        kg_says_valid = verification["exact_match"]

        if is_correct_fact and kg_says_valid:
            results["correct_detections"] += 1
            results["by_type"][ctype]["correct"] += 1
        elif is_correct_fact and not kg_says_valid:
            results["false_positives"] += 1
        elif not is_correct_fact and not kg_says_valid:
            results["correct_detections"] += 1
            results["by_type"][ctype]["correct"] += 1
        elif not is_correct_fact and kg_says_valid:
            results["false_negatives"] += 1

    accuracy = results["correct_detections"] / results["total"] if results["total"] > 0 else 0

    # Compute precision, recall, F1
    tp = results["correct_detections"] - results["by_type"].get("none", {}).get("correct", 0)
    fp = results["false_positives"]
    fn = results["false_negatives"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logger.info("=" * 60)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info("Total examples:       %d", results["total"])
    logger.info("Correct detections:   %d", results["correct_detections"])
    logger.info("False positives:      %d", results["false_positives"])
    logger.info("False negatives:      %d", results["false_negatives"])
    logger.info("Overall accuracy:     %.4f (%.1f%%)", accuracy, accuracy * 100)
    logger.info("Precision:            %.4f", precision)
    logger.info("Recall:               %.4f", recall)
    logger.info("F1 Score:             %.4f", f1)
    logger.info("")
    logger.info("Per-corruption breakdown:")
    for ctype, data in results["by_type"].items():
        type_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        logger.info("  %-12s  %d/%d = %.1f%%", ctype, data["correct"], data["total"], type_acc * 100)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "verification_results.json"
    results["accuracy"] = accuracy
    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = f1
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results


# ===================================================================
# STEP 5: Test LLM connection (LM Studio)
# ===================================================================

def step5_test_llm():
    """Test if LM Studio is reachable and the model responds."""
    logger.info("=" * 60)
    logger.info("STEP 5: Testing LM Studio connection")
    logger.info("=" * 60)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="lm-studio",
        )

        logger.info("Connecting to %s with model %s...", OLLAMA_BASE_URL, OLLAMA_MODEL)

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a temporal fact verification assistant. Answer concisely."
                },
                {
                    "role": "user",
                    "content": "Verify this claim: 'The United States made a diplomatic statement to Russia on March 15, 2014.' Is this temporally plausible? Answer in one sentence."
                }
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        answer = response.choices[0].message.content
        logger.info("LLM Response: %s", answer[:300])
        logger.info("LM Studio connection successful!")
        return True

    except Exception as e:
        logger.warning("LM Studio not reachable: %s", e)
        logger.warning("The verification pipeline (Steps 1-4) works without LLM.")
        logger.warning("Start LM Studio server to enable LLM-based reasoning.")
        return False


# ===================================================================
# STEP 6: Run sample pipeline queries with LLM grounding
# ===================================================================

def step6_sample_queries(store, llm_available: bool):
    """Run sample queries through the retrieval engine."""
    logger.info("=" * 60)
    logger.info("STEP 6: Running sample retrieval queries")
    logger.info("=" * 60)

    sample_queries = [
        ("Russia", None, None),
        ("United States", "2014-03-01", "2014-03-31"),
        ("China", "2014-06-01", "2014-06-30"),
    ]

    for entity, start, end in sample_queries:
        if start and end:
            events = store.get_events_by_entity_and_time(entity, start, end, limit=5)
            logger.info("Events for '%s' (%s to %s): %d found", entity, start, end, len(events))
        else:
            events = store.get_events_by_entity(entity, limit=5)
            logger.info("Events for '%s': %d found", entity, len(events))

        for ev in events[:3]:
            logger.info("  %s | %s | %s | %s",
                         ev.get("subject", "?"),
                         ev.get("predicate", "?"),
                         ev.get("object", "?"),
                         ev.get("timestamp", "?"))

    if llm_available:
        logger.info("")
        logger.info("Running LLM-grounded temporal Q&A...")

        from openai import OpenAI
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="lm-studio")

        # Retrieve facts about Russia in March 2014
        events = store.get_events_by_entity_and_time("Russia", "2014-03-01", "2014-03-31", limit=10)
        context = "\n".join([
            f"- {ev.get('subject', '?')} {ev.get('predicate', '?')} {ev.get('object', '?')} on {ev.get('timestamp', '?')}"
            for ev in events
        ])

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a temporal fact verification assistant. "
                        "Answer ONLY based on the provided temporal facts. "
                        "If the facts do not contain the answer, say so. "
                        "Do not use any outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on these temporal facts from the ICEWS14 knowledge graph:\n\n"
                        f"{context}\n\n"
                        f"Question: What diplomatic interactions did Russia have in March 2014? "
                        f"List the key events with dates."
                    ),
                },
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        answer = response.choices[0].message.content
        logger.info("Grounded LLM answer:\n%s", answer[:500])

        # Now verify the LLM answer by extracting any claims and checking them
        logger.info("")
        logger.info("Verifying LLM claims against knowledge graph...")

        verify_response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract temporal claims from the following text. "
                        "Return each claim as a JSON object with keys: "
                        "subject, predicate, object, timestamp. "
                        "Return a JSON array of claims. Only return the JSON, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": answer[:500],
                },
            ],
            temperature=0.0,
            max_tokens=LLM_MAX_TOKENS,
        )

        claims_text = verify_response.choices[0].message.content
        logger.info("Extracted claims: %s", claims_text[:300])

        # Try to parse and verify claims
        try:
            # Find JSON array in response
            start_idx = claims_text.find("[")
            end_idx = claims_text.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                claims = json.loads(claims_text[start_idx:end_idx])
                for claim in claims[:5]:
                    subj = claim.get("subject", "")
                    pred = claim.get("predicate", "")
                    obj = claim.get("object", "")
                    ts = claim.get("timestamp", "")
                    if subj and pred and obj and ts:
                        result = store.verify_fact(subj, pred, obj, ts)
                        status = "SUPPORTED" if result["exact_match"] else "UNSUPPORTED"
                        logger.info("  Claim: %s %s %s on %s -> %s",
                                     subj, pred, obj, ts, status)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not parse claims: %s", e)


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Run the complete pipeline."""
    start = time.time()

    logger.info("=" * 60)
    logger.info("TEMPORAL GROUNDING AND VERIFICATION FRAMEWORK")
    logger.info("Complete Pipeline Runner")
    logger.info("=" * 60)
    logger.info("Using LM Studio at %s", OLLAMA_BASE_URL)
    logger.info("Model: %s", OLLAMA_MODEL)
    logger.info("")

    # Step 1: Load data
    loader, facts = step1_load_data()

    # Step 2: Build graph
    store = step2_build_graph(loader, facts)

    # Step 3: Generate benchmark
    benchmark = step3_generate_benchmark(store, facts)

    # Step 4: Run verification
    results = step4_run_verification(store, benchmark)

    # Step 5: Test LLM
    llm_ok = step5_test_llm()

    # Step 6: Sample queries
    step6_sample_queries(store, llm_ok)

    elapsed = time.time() - start
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPLETE! Total time: %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Output files:")
    logger.info("  Knowledge graph: data/processed/knowledge_graph.json")
    logger.info("  Benchmark:       data/benchmarks/temporal_hallucination_benchmark.json")
    logger.info("  Results:         results/verification_results.json")

    return results


if __name__ == "__main__":
    main()
