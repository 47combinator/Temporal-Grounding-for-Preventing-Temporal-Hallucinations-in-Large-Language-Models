"""
Corruption strategies for synthetic temporal hallucination generation.

Each function takes a well-formed temporal fact (dict with keys ``subject``,
``predicate``, ``object``, ``timestamp``) and returns a corrupted copy.
Corrupted dicts carry additional metadata: ``corruption_type``,
``original_fact``, and ``corrupted_field``.
"""

from __future__ import annotations

import copy
import random as _random_module
from typing import Any


def corrupt_timestamp(
    fact: dict[str, Any],
    all_timestamps: list[str],
    rng: _random_module.Random,
) -> dict[str, Any]:
    """Replace the timestamp with a randomly chosen different timestamp.

    Parameters
    ----------
    fact : dict
        A temporal fact with at least ``subject``, ``predicate``, ``object``,
        and ``timestamp`` keys.
    all_timestamps : list[str]
        Pool of all known timestamps to draw from.
    rng : random.Random
        Seeded random number generator for reproducibility.

    Returns
    -------
    dict
        A corrupted copy of the fact with ``corruption_type='timestamp'``
        and ``corrupted_field='timestamp'``.
    """
    original = copy.deepcopy(fact)
    candidates = [ts for ts in all_timestamps if ts != fact["timestamp"]]
    if not candidates:
        # Edge case: only one timestamp exists in the entire pool.
        new_ts = fact["timestamp"]
    else:
        new_ts = rng.choice(candidates)

    corrupted = copy.deepcopy(fact)
    corrupted["timestamp"] = new_ts
    corrupted["corruption_type"] = "timestamp"
    corrupted["original_fact"] = original
    corrupted["corrupted_field"] = "timestamp"
    return corrupted


def corrupt_entity(
    fact: dict[str, Any],
    all_entities: list[str],
    rng: _random_module.Random,
    field: str = "random",
) -> dict[str, Any]:
    """Replace the subject or object with a randomly chosen different entity.

    Parameters
    ----------
    fact : dict
        A temporal fact.
    all_entities : list[str]
        Pool of all known entity names.
    rng : random.Random
        Seeded random number generator.
    field : str
        ``"subject"``, ``"object"``, or ``"random"`` (default).  When
        ``"random"`` the field to corrupt is chosen uniformly at random.

    Returns
    -------
    dict
        A corrupted copy with ``corruption_type='entity'`` and
        ``corrupted_field`` set to the field that was changed.
    """
    original = copy.deepcopy(fact)

    if field == "random":
        chosen_field: str = rng.choice(["subject", "object"])
    else:
        chosen_field = field

    current_value = fact[chosen_field]
    candidates = [e for e in all_entities if e != current_value]
    if not candidates:
        new_value = current_value
    else:
        new_value = rng.choice(candidates)

    corrupted = copy.deepcopy(fact)
    corrupted[chosen_field] = new_value
    corrupted["corruption_type"] = "entity"
    corrupted["original_fact"] = original
    corrupted["corrupted_field"] = chosen_field
    return corrupted


def corrupt_relation(
    fact: dict[str, Any],
    all_relations: list[str],
    rng: _random_module.Random,
) -> dict[str, Any]:
    """Replace the predicate with a randomly chosen different relation.

    Parameters
    ----------
    fact : dict
        A temporal fact.
    all_relations : list[str]
        Pool of all known relation names.
    rng : random.Random
        Seeded random number generator.

    Returns
    -------
    dict
        A corrupted copy with ``corruption_type='relation'`` and
        ``corrupted_field='predicate'``.
    """
    original = copy.deepcopy(fact)
    candidates = [r for r in all_relations if r != fact["predicate"]]
    if not candidates:
        new_rel = fact["predicate"]
    else:
        new_rel = rng.choice(candidates)

    corrupted = copy.deepcopy(fact)
    corrupted["predicate"] = new_rel
    corrupted["corruption_type"] = "relation"
    corrupted["original_fact"] = original
    corrupted["corrupted_field"] = "predicate"
    return corrupted


def corrupt_temporal_order(
    fact_a: dict[str, Any], fact_b: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Swap the timestamps between two facts.

    Parameters
    ----------
    fact_a : dict
        The first temporal fact.
    fact_b : dict
        The second temporal fact.

    Returns
    -------
    tuple[dict, dict]
        Both corrupted copies with ``corruption_type='ordering'``.
    """
    original_a = copy.deepcopy(fact_a)
    original_b = copy.deepcopy(fact_b)

    corrupted_a = copy.deepcopy(fact_a)
    corrupted_b = copy.deepcopy(fact_b)

    corrupted_a["timestamp"] = fact_b["timestamp"]
    corrupted_b["timestamp"] = fact_a["timestamp"]

    corrupted_a["corruption_type"] = "ordering"
    corrupted_a["original_fact"] = original_a
    corrupted_a["corrupted_field"] = "timestamp"

    corrupted_b["corruption_type"] = "ordering"
    corrupted_b["original_fact"] = original_b
    corrupted_b["corrupted_field"] = "timestamp"

    return corrupted_a, corrupted_b


def apply_corruption(
    fact: dict[str, Any],
    strategy: str,
    all_entities: list[str],
    all_relations: list[str],
    all_timestamps: list[str],
    rng: _random_module.Random,
) -> dict[str, Any]:
    """Dispatch to the appropriate corruption function.

    Parameters
    ----------
    fact : dict
        The fact to corrupt.
    strategy : str
        One of ``"timestamp"``, ``"entity"``, ``"relation"``.  The
        ``"ordering"`` strategy requires two facts and should be called
        directly via :func:`corrupt_temporal_order`.
    all_entities : list[str]
        Pool of entity names.
    all_relations : list[str]
        Pool of relation names.
    all_timestamps : list[str]
        Pool of timestamp strings.
    rng : random.Random
        Seeded random number generator.

    Returns
    -------
    dict
        The corrupted fact.

    Raises
    ------
    ValueError
        If *strategy* is not one of the recognised values.
    """
    if strategy == "timestamp":
        return corrupt_timestamp(fact, all_timestamps, rng)
    if strategy == "entity":
        return corrupt_entity(fact, all_entities, rng)
    if strategy == "relation":
        return corrupt_relation(fact, all_relations, rng)
    raise ValueError(
        f"Unknown corruption strategy '{strategy}'. "
        f"Expected one of: 'timestamp', 'entity', 'relation'."
    )
