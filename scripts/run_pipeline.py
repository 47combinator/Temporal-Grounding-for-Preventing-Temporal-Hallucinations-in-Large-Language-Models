"""
Script to run the full Temporal Verification Pipeline on a question.

Usage:
    python -m scripts.run_pipeline --question "Who made a statement to Russia in March 2014?"
    python -m scripts.run_pipeline --input-file questions.txt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.pipeline import TemporalVerificationPipeline


def format_result(result: dict) -> str:
    """Format a pipeline result for human-readable console output."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"QUESTION: {result['question']}")
    lines.append("-" * 70)

    # Retrieved facts
    facts = result.get("retrieved_facts", [])
    lines.append(f"\nRETRIEVED FACTS ({len(facts)}):")
    for i, fact in enumerate(facts[:10], 1):
        subj = fact.get("subject", "?")
        pred = fact.get("predicate", "?")
        obj = fact.get("object", "?")
        ts = fact.get("timestamp", "?")
        lines.append(f"  {i}. ({subj}, {pred}, {obj}, {ts})")

    # Answer
    lines.append(f"\nANSWER: {result.get('answer', 'N/A')}")

    # Verification
    tcs = result.get("tcs", None)
    if tcs is not None:
        lines.append(f"\nTEMPORAL CONSISTENCY SCORE: {tcs:.2f}")

    report = result.get("hallucination_report", {})
    classification = report.get("classification", "N/A")
    lines.append(f"CLASSIFICATION: {classification}")

    # Per-claim details
    explanations = result.get("explanations", [])
    if explanations:
        lines.append(f"\nVERIFICATION DETAILS ({len(explanations)}):")
        for i, explanation in enumerate(explanations, 1):
            lines.append(f"  {i}. {explanation}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Temporal Verification Pipeline."
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="A single question to process.",
    )
    parser.add_argument(
        "--input-file", type=str, default=None,
        help="Path to a text file with one question per line.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of temporal facts to retrieve (default: 10).",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Optional path to save results as JSON.",
    )
    args = parser.parse_args()

    if not args.question and not args.input_file:
        parser.print_help()
        print("\nError: Provide either --question or --input-file.")
        sys.exit(1)

    # Build questions list
    questions = []
    if args.question:
        questions.append(args.question)
    elif args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(questions)} question(s) ...")
    print("Initialising pipeline ...")

    try:
        pipeline = TemporalVerificationPipeline.from_config()
    except Exception as exc:
        print(f"Error initialising pipeline: {exc}")
        print(
            "Make sure Neo4j and Ollama are running. "
            "Run 'python -m scripts.load_icews14' to load data first."
        )
        sys.exit(1)

    results = []
    for question in questions:
        print(f"\nProcessing: {question}")
        result = pipeline.process(question, top_k=args.top_k)
        results.append(result)
        print(format_result(result))

    # Save JSON if requested
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serialisable objects
        serialisable = []
        for r in results:
            entry = {
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "tcs": r.get("tcs", None),
                "retrieved_facts": r.get("retrieved_facts", []),
                "explanations": r.get("explanations", []),
                "classification": r.get("hallucination_report", {}).get(
                    "classification", "N/A"
                ),
            }
            serialisable.append(entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
