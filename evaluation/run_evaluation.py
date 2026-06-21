"""
Evaluation runner for the Temporal Grounding and Verification Framework.

Orchestrates the full evaluation suite:
- Benchmark evaluation (temporal hallucination detection).
- CronQuestions evaluation (temporal QA accuracy across baselines).
- Ablation study (component contribution analysis).

Usage
-----
Run a specific experiment::

    python -m evaluation.run_evaluation --experiment benchmark
    python -m evaluation.run_evaluation --experiment cronquestions
    python -m evaluation.run_evaluation --experiment ablation
    python -m evaluation.run_evaluation --experiment all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from evaluation.baselines import BaselineRunner
from evaluation.metrics import (
    answer_accuracy,
    compute_precision_recall_f1,
    generate_classification_report,
    generate_confusion_matrix,
    hallucination_detection_accuracy,
    per_corruption_type_metrics,
    temporal_consistency_score,
)
from src.pipeline.pipeline import TemporalVerificationPipeline

logger = logging.getLogger(__name__)

# Ensure the results directory exists.
RESULTS_DIR = settings.EVAL_RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# Benchmark evaluation
# -----------------------------------------------------------------------


def run_benchmark_evaluation(
    pipeline: TemporalVerificationPipeline,
    benchmark_path: Path,
) -> dict[str, Any]:
    """Run hallucination detection evaluation on the synthetic benchmark.

    Loads the benchmark dataset, runs the pipeline's verification on each
    example, and computes detection metrics.

    Parameters
    ----------
    pipeline :
        A fully initialised ``TemporalVerificationPipeline``.
    benchmark_path :
        Path to the benchmark JSON file. Each entry must have keys
        ``"question"``, ``"is_hallucinated"`` (bool), ``"corruption_type"``,
        and ``"claim"`` (dict with subject/predicate/object/timestamp).

    Returns
    -------
    dict[str, Any]
        A dictionary containing overall accuracy, per-corruption metrics,
        classification report, confusion matrix, and detailed results.
    """
    logger.info("Loading benchmark from %s", benchmark_path)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    logger.info("Benchmark contains %d examples.", len(benchmark))

    y_true: list[bool] = []
    y_pred: list[bool] = []
    detailed_results: list[dict[str, Any]] = []

    for i, example in enumerate(benchmark):
        logger.info(
            "Processing benchmark example %d / %d", i + 1, len(benchmark)
        )
        question = example.get("question", "")
        ground_truth = bool(example.get("is_hallucinated", False))
        corruption_type = example.get("corruption_type", "none")

        try:
            result = pipeline.process(question)
            predicted = result["hallucination_report"].get(
                "is_hallucinated", False
            )
        except Exception:
            logger.exception(
                "Pipeline failed for benchmark example %d.", i
            )
            predicted = False

        y_true.append(ground_truth)
        y_pred.append(predicted)

        detailed_results.append(
            {
                "question": question,
                "corruption_type": corruption_type,
                "y_true": ground_truth,
                "y_pred": predicted,
            }
        )

    # Compute metrics.
    accuracy = hallucination_detection_accuracy(y_true, y_pred)
    prf = compute_precision_recall_f1(y_true, y_pred)
    per_type = per_corruption_type_metrics(detailed_results)
    report = generate_classification_report(y_true, y_pred)
    cm = generate_confusion_matrix(y_true, y_pred)

    summary = {
        "experiment": "benchmark",
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(benchmark),
        "accuracy": accuracy,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "per_corruption_type": per_type,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "detailed_results": detailed_results,
    }

    # Save results.
    output_path = RESULTS_DIR / "benchmark_results.json"
    _save_json(summary, output_path)
    logger.info("Benchmark results saved to %s", output_path)

    # Print summary.
    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 60)
    print(f"Examples:  {len(benchmark)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {prf['precision']:.4f}")
    print(f"Recall:    {prf['recall']:.4f}")
    print(f"F1 Score:  {prf['f1']:.4f}")
    print("\nPer-corruption type:")
    for ctype, cmetrics in per_type.items():
        print(
            f"  {ctype:20s}  acc={cmetrics['accuracy']:.4f}  "
            f"f1={cmetrics['f1']:.4f}  n={cmetrics['count']}"
        )
    print("\nClassification Report:")
    print(report)
    print("=" * 60)

    return summary


# -----------------------------------------------------------------------
# CronQuestions evaluation
# -----------------------------------------------------------------------


def run_cronquestions_evaluation(
    pipeline: TemporalVerificationPipeline,
    loader: Any,
    sample_size: int,
) -> dict[str, Any]:
    """Run comparative evaluation on sampled CronQuestions.

    Samples questions from the CronQuestions dataset, runs all four
    baselines, and computes comparative metrics.

    Parameters
    ----------
    pipeline :
        A fully initialised ``TemporalVerificationPipeline``.
    loader :
        A CronQuestions data loader with a ``load_questions()`` method
        returning a list of dicts with ``"question"`` and ``"answer"`` keys.
    sample_size :
        Number of questions to sample for evaluation.

    Returns
    -------
    dict[str, Any]
        Comparative results across all baselines.
    """
    logger.info("Loading CronQuestions dataset...")
    all_questions = loader.load_questions()
    logger.info("Loaded %d questions total.", len(all_questions))

    # Sample questions reproducibly.
    import random
    rng = random.Random(settings.RANDOM_SEED)
    if sample_size < len(all_questions):
        sampled = rng.sample(all_questions, sample_size)
    else:
        sampled = all_questions

    questions = [q["question"] for q in sampled]
    gold_answers = [q["answer"] for q in sampled]

    logger.info("Sampled %d questions for evaluation.", len(questions))

    # Run all baselines.
    baseline_runner = BaselineRunner(
        graph_store=pipeline.graph_store,
        reasoner=pipeline.reasoner,
    )
    all_results = baseline_runner.run_all(
        questions, pipeline=pipeline
    )

    # Compute metrics per baseline.
    comparison: dict[str, dict[str, float]] = {}
    for method_name, method_results in all_results.items():
        predictions = [r.get("answer", "") for r in method_results]
        acc = answer_accuracy(predictions, gold_answers)

        # Compute TCS where available.
        tcs_values: list[float] = []
        for r in method_results:
            vr = r.get("verification_results", [])
            if vr:
                tcs_values.append(temporal_consistency_score(vr))

        avg_tcs = (
            sum(tcs_values) / len(tcs_values) if tcs_values else 0.0
        )

        comparison[method_name] = {
            "answer_accuracy": acc,
            "avg_tcs": avg_tcs,
            "num_questions": len(questions),
        }

    summary = {
        "experiment": "cronquestions",
        "timestamp": datetime.now().isoformat(),
        "sample_size": len(questions),
        "comparison": comparison,
        "detailed_results": {
            method: [
                {
                    "question": r.get("question", ""),
                    "answer": r.get("answer", ""),
                    "tcs": r.get("tcs", None),
                }
                for r in results
            ]
            for method, results in all_results.items()
        },
    }

    # Save results.
    output_path = RESULTS_DIR / "cronquestions_results.json"
    _save_json(summary, output_path)
    logger.info("CronQuestions results saved to %s", output_path)

    # Print summary table.
    print("\n" + "=" * 60)
    print("CRONQUESTIONS EVALUATION RESULTS")
    print("=" * 60)
    print(f"Sample size: {len(questions)}\n")
    print(f"{'Method':<20s} {'Accuracy':>10s} {'Avg TCS':>10s}")
    print("-" * 42)
    for method, metrics in comparison.items():
        print(
            f"{method:<20s} "
            f"{metrics['answer_accuracy']:>10.4f} "
            f"{metrics['avg_tcs']:>10.4f}"
        )
    print("=" * 60)

    return summary


# -----------------------------------------------------------------------
# Ablation study
# -----------------------------------------------------------------------


def run_ablation_study(
    graph_store: Any,
    questions: list[str],
) -> dict[str, Any]:
    """Run an ablation study measuring the contribution of each component.

    Tests configurations with selected components disabled:
    - Full pipeline (all components).
    - Without temporal verification.
    - Without hallucination detection.
    - Without explanation generation.
    - Without retrieval (direct LLM).

    Parameters
    ----------
    graph_store :
        A ``TemporalKGStore`` instance.
    questions :
        List of evaluation questions (a small subset is typically used).

    Returns
    -------
    dict[str, Any]
        A dictionary keyed by ablation condition, each mapping to
        computed metrics.
    """
    from src.knowledge_graph.graph_store import TemporalKGStore
    from src.reasoning.llm_reasoner import LLMReasoner
    from src.reasoning.claim_extractor import ClaimExtractor
    from src.retrieval.query_parser import QueryParser
    from src.retrieval.temporal_retriever import TemporalRetriever
    from src.verification.temporal_verifier import TemporalVerifier
    from src.hallucination.detector import HallucinationDetector
    from src.explanation.explanation_generator import ExplanationGenerator

    logger.info("Starting ablation study with %d questions.", len(questions))

    # Build full set of components.
    entity_names = graph_store.get_all_entity_names()
    query_parser = QueryParser(entity_names=entity_names)
    retriever = TemporalRetriever(
        graph_store=graph_store,
        query_parser=query_parser,
        top_k=settings.RETRIEVAL_TOP_K,
    )
    reasoner = LLMReasoner(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
    claim_extractor = ClaimExtractor(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
    )
    verifier = TemporalVerifier(
        graph_store=graph_store,
        timestamp_tolerance_days=settings.TIMESTAMP_TOLERANCE_DAYS,
        partial_support_threshold=settings.PARTIAL_SUPPORT_THRESHOLD,
    )
    detector = HallucinationDetector()
    explainer = ExplanationGenerator(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
    )

    # Define ablation conditions.
    # Each condition is a dict of component overrides (None = disabled).
    class _NoOpVerifier:
        """Stub verifier that marks all claims as CANNOT_VERIFY."""

        def verify(self, claim: Any) -> Any:
            """Return a cannot-verify result for the given claim."""
            from src.verification.verification_result import (
                VerificationResult,
                VerificationStatus,
            )
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.CANNOT_VERIFY,
                confidence=0.0,
                explanation="Verification disabled for ablation.",
            )

    class _NoOpDetector:
        """Stub detector that never flags hallucinations."""

        def detect(self, results: list[Any]) -> dict[str, Any]:
            """Return a report indicating no hallucination."""
            return {"is_hallucinated": False, "hallucinated_claims": []}

    class _NoOpExplainer:
        """Stub explainer that returns empty explanations."""

        def explain(self, result: Any) -> str:
            """Return an empty explanation string."""
            return ""

    conditions: dict[str, dict[str, Any]] = {
        "full_pipeline": {
            "verifier": verifier,
            "detector": detector,
            "explainer": explainer,
        },
        "no_verification": {
            "verifier": _NoOpVerifier(),
            "detector": detector,
            "explainer": explainer,
        },
        "no_detection": {
            "verifier": verifier,
            "detector": _NoOpDetector(),
            "explainer": explainer,
        },
        "no_explanation": {
            "verifier": verifier,
            "detector": detector,
            "explainer": _NoOpExplainer(),
        },
    }

    ablation_results: dict[str, Any] = {}

    for condition_name, overrides in conditions.items():
        logger.info("Running ablation condition: %s", condition_name)

        pipeline = TemporalVerificationPipeline(
            graph_store=graph_store,
            retriever=retriever,
            reasoner=reasoner,
            claim_extractor=claim_extractor,
            verifier=overrides["verifier"],
            detector=overrides["detector"],
            explainer=overrides["explainer"],
        )

        batch_results = pipeline.process_batch(questions)

        # Compute aggregate TCS.
        tcs_values = [r.get("tcs", 0.0) for r in batch_results]
        avg_tcs = sum(tcs_values) / len(tcs_values) if tcs_values else 0.0

        # Count hallucination detections.
        halluc_count = sum(
            1
            for r in batch_results
            if r.get("hallucination_report", {}).get(
                "is_hallucinated", False
            )
        )

        ablation_results[condition_name] = {
            "avg_tcs": avg_tcs,
            "hallucinations_detected": halluc_count,
            "num_questions": len(questions),
        }

    summary = {
        "experiment": "ablation",
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "conditions": ablation_results,
    }

    # Save results.
    output_path = RESULTS_DIR / "ablation_results.json"
    _save_json(summary, output_path)
    logger.info("Ablation results saved to %s", output_path)

    # Print summary table.
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"Questions: {len(questions)}\n")
    print(
        f"{'Condition':<25s} {'Avg TCS':>10s} {'Halluc.':>10s}"
    )
    print("-" * 47)
    for cond, metrics in ablation_results.items():
        print(
            f"{cond:<25s} "
            f"{metrics['avg_tcs']:>10.4f} "
            f"{metrics['hallucinations_detected']:>10d}"
        )
    print("=" * 60)

    return summary


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------


def _save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary to a JSON file with pretty printing.

    Parameters
    ----------
    data :
        The data to serialise.
    path :
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------


def main() -> None:
    """Parse command-line arguments and run the requested experiments.

    Supported experiments:
    - ``benchmark``: Run hallucination detection on the synthetic benchmark.
    - ``cronquestions``: Run comparative evaluation on CronQuestions.
    - ``ablation``: Run the ablation study.
    - ``all``: Run every experiment.
    """
    parser = argparse.ArgumentParser(
        description="Run the Temporal Grounding and Verification evaluation suite."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["benchmark", "cronquestions", "ablation", "all"],
        default="all",
        help="Which experiment to run (default: all).",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=None,
        help=(
            "Path to the benchmark JSON file. "
            "Defaults to data/benchmarks/temporal_hallucination_benchmark.json."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Number of CronQuestions to sample. "
            f"Defaults to {settings.CRONQUESTIONS_SAMPLE_SIZE}."
        ),
    )
    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    experiment = args.experiment
    benchmark_path = (
        Path(args.benchmark_path)
        if args.benchmark_path
        else settings.BENCHMARK_DIR / "temporal_hallucination_benchmark.json"
    )
    sample_size = args.sample_size or settings.CRONQUESTIONS_SAMPLE_SIZE

    logger.info("Building pipeline from configuration...")
    pipeline = TemporalVerificationPipeline.from_config()

    all_summaries: dict[str, Any] = {}

    # --- Benchmark ---
    if experiment in ("benchmark", "all"):
        if benchmark_path.exists():
            summary = run_benchmark_evaluation(pipeline, benchmark_path)
            all_summaries["benchmark"] = summary
        else:
            logger.error(
                "Benchmark file not found: %s. "
                "Run 'python -m scripts.generate_benchmark' first.",
                benchmark_path,
            )

    # --- CronQuestions ---
    if experiment in ("cronquestions", "all"):
        try:
            from src.data_loader.cronquestions_loader import (
                CronQuestionsLoader,
            )
            loader = CronQuestionsLoader(
                data_dir=settings.CRONQUESTIONS_DIR
            )
            summary = run_cronquestions_evaluation(
                pipeline, loader, sample_size
            )
            all_summaries["cronquestions"] = summary
        except ImportError:
            logger.error(
                "CronQuestionsLoader not available. "
                "Ensure src/data_loader/cronquestions_loader.py exists."
            )
        except FileNotFoundError:
            logger.error(
                "CronQuestions data not found at %s.",
                settings.CRONQUESTIONS_DIR,
            )

    # --- Ablation ---
    if experiment in ("ablation", "all"):
        # Use a small set of questions for the ablation study.
        ablation_questions = _load_ablation_questions(benchmark_path)
        if ablation_questions:
            summary = run_ablation_study(
                pipeline.graph_store, ablation_questions
            )
            all_summaries["ablation"] = summary
        else:
            logger.error(
                "No questions available for the ablation study."
            )

    # Save combined results.
    if all_summaries:
        combined_path = RESULTS_DIR / "evaluation_summary.json"
        _save_json(all_summaries, combined_path)
        logger.info("Combined results saved to %s", combined_path)

    logger.info("Evaluation complete.")


def _load_ablation_questions(benchmark_path: Path) -> list[str]:
    """Load a small set of questions for the ablation study.

    Extracts questions from the benchmark file, or generates simple
    test questions if no benchmark is available.

    Parameters
    ----------
    benchmark_path :
        Path to the benchmark JSON file.

    Returns
    -------
    list[str]
        A list of question strings for the ablation study.
    """
    if benchmark_path.exists():
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        # Take a small sample for ablation (up to 50 questions).
        import random
        rng = random.Random(settings.RANDOM_SEED)
        sample = rng.sample(
            benchmark, min(50, len(benchmark))
        )
        return [ex.get("question", "") for ex in sample if ex.get("question")]

    logger.warning(
        "Benchmark file not found at %s. "
        "Using empty question list for ablation.",
        benchmark_path,
    )
    return []


if __name__ == "__main__":
    main()
