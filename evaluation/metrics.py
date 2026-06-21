"""
Evaluation metrics for the Temporal Grounding and Verification Framework.

Provides functions for computing hallucination detection accuracy,
temporal consistency scores, answer accuracy, precision/recall/F1,
per-corruption-type breakdowns, and scikit-learn classification reports.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.verification.verification_result import VerificationStatus


def hallucination_detection_accuracy(
    y_true: list[Any], y_pred: list[Any]
) -> float:
    """Compute the accuracy of hallucination detection.

    Parameters
    ----------
    y_true :
        Ground-truth labels (e.g. ``True``/``False`` or ``1``/``0``)
        indicating whether each example is a hallucination.
    y_pred :
        Predicted labels from the detector.

    Returns
    -------
    float
        The fraction of correct predictions in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If the input lists have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)}."
        )
    if not y_true:
        raise ValueError("Cannot compute accuracy on empty lists.")

    return float(accuracy_score(y_true, y_pred))


def temporal_consistency_score(
    verification_results: list[dict[str, Any]]
) -> float:
    """Compute the Temporal Consistency Score (TCS).

    TCS is defined as the ratio of temporally consistent claims
    (SUPPORTED or PARTIALLY_SUPPORTED) to the total number of
    verified claims.

    Parameters
    ----------
    verification_results :
        A list of verification result dictionaries, each containing
        a ``"status"`` key with a ``VerificationStatus`` value or its
        string representation.

    Returns
    -------
    float
        The TCS in ``[0.0, 1.0]``, or ``1.0`` when no results are
        provided.
    """
    if not verification_results:
        return 1.0

    consistent_statuses = {
        VerificationStatus.SUPPORTED.value,
        VerificationStatus.PARTIALLY_SUPPORTED.value,
        "supported",
        "partially_supported",
    }

    consistent_count = 0
    total = 0
    for vr in verification_results:
        status = vr.get("status")
        if isinstance(status, VerificationStatus):
            status = status.value
        if status is not None:
            total += 1
            if status in consistent_statuses:
                consistent_count += 1

    if total == 0:
        return 1.0

    return consistent_count / total


def answer_accuracy(
    predictions: list[str], gold_answers: list[str]
) -> float:
    """Compute exact-match accuracy between predicted and gold answers.

    Comparison is case-insensitive and strips leading/trailing whitespace.

    Parameters
    ----------
    predictions :
        The predicted answer strings.
    gold_answers :
        The expected (gold-standard) answer strings.

    Returns
    -------
    float
        The fraction of exact matches in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If the input lists have different lengths or are empty.
    """
    if len(predictions) != len(gold_answers):
        raise ValueError(
            f"Length mismatch: predictions has {len(predictions)} elements, "
            f"gold_answers has {len(gold_answers)}."
        )
    if not predictions:
        raise ValueError("Cannot compute accuracy on empty lists.")

    matches = sum(
        1
        for pred, gold in zip(predictions, gold_answers)
        if pred.strip().lower() == gold.strip().lower()
    )
    return matches / len(predictions)


def compute_precision_recall_f1(
    y_true: list[Any], y_pred: list[Any]
) -> dict[str, float]:
    """Compute precision, recall, and F1 score.

    Uses binary averaging. For multi-class problems the ``weighted``
    average is used automatically.

    Parameters
    ----------
    y_true :
        Ground-truth labels.
    y_pred :
        Predicted labels.

    Returns
    -------
    dict[str, float]
        A dictionary with keys ``"precision"``, ``"recall"``, ``"f1"``.

    Raises
    ------
    ValueError
        If the input lists have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)}."
        )
    if not y_true:
        raise ValueError(
            "Cannot compute metrics on empty lists."
        )

    unique_labels = set(y_true) | set(y_pred)
    average = "binary" if len(unique_labels) <= 2 else "weighted"

    return {
        "precision": float(
            precision_score(
                y_true, y_pred, average=average, zero_division=0.0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, average=average, zero_division=0.0
            )
        ),
        "f1": float(
            f1_score(
                y_true, y_pred, average=average, zero_division=0.0
            )
        ),
    }


def per_corruption_type_metrics(
    benchmark_results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Break down detection accuracy by corruption type.

    Expects each benchmark result dictionary to contain:
    - ``"corruption_type"``: one of ``"timestamp"``, ``"entity"``,
      ``"relation"``, ``"ordering"``, or ``"none"`` (for genuine facts).
    - ``"y_true"``: the ground-truth label (``True`` = hallucinated).
    - ``"y_pred"``: the predicted label.

    Parameters
    ----------
    benchmark_results :
        A list of per-example result dictionaries.

    Returns
    -------
    dict[str, dict[str, float]]
        A nested dictionary keyed by corruption type, each containing
        ``"accuracy"``, ``"precision"``, ``"recall"``, ``"f1"``, and
        ``"count"`` for that subset.
    """
    # Group results by corruption type.
    by_type: dict[str, list[dict[str, Any]]] = {}
    for result in benchmark_results:
        ctype = result.get("corruption_type", "unknown")
        by_type.setdefault(ctype, []).append(result)

    metrics: dict[str, dict[str, float]] = {}
    for ctype, results in sorted(by_type.items()):
        y_true = [r["y_true"] for r in results]
        y_pred = [r["y_pred"] for r in results]

        if len(set(y_true)) < 2:
            # Only one class present -- compute what we can.
            acc = float(accuracy_score(y_true, y_pred))
            prf = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            acc = float(accuracy_score(y_true, y_pred))
            prf = compute_precision_recall_f1(y_true, y_pred)

        metrics[ctype] = {
            "accuracy": acc,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "count": len(results),
        }

    return metrics


def generate_classification_report(
    y_true: list[Any],
    y_pred: list[Any],
    labels: Optional[list[Any]] = None,
) -> str:
    """Generate a text classification report using scikit-learn.

    Parameters
    ----------
    y_true :
        Ground-truth labels.
    y_pred :
        Predicted labels.
    labels :
        Optional list of label values to include in the report.
        If ``None``, all labels present in ``y_true`` and ``y_pred``
        are used.

    Returns
    -------
    str
        A formatted classification report string.
    """
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0.0,
    )


def generate_confusion_matrix(
    y_true: list[Any],
    y_pred: list[Any],
    labels: Optional[list[Any]] = None,
) -> np.ndarray:
    """Generate a confusion matrix using scikit-learn.

    Parameters
    ----------
    y_true :
        Ground-truth labels.
    y_pred :
        Predicted labels.
    labels :
        Optional ordered list of label values.

    Returns
    -------
    numpy.ndarray
        The confusion matrix of shape ``(n_classes, n_classes)``.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)
