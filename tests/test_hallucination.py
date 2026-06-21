"""
Tests for the corruption strategies used in benchmark generation.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hallucination.corruption_strategies import (
    corrupt_timestamp,
    corrupt_entity,
    corrupt_relation,
    corrupt_temporal_order,
)


@pytest.fixture
def sample_fact() -> dict:
    """A sample temporal fact for corruption."""
    return {
        "subject": "United States",
        "predicate": "Make statement",
        "object": "Russia",
        "timestamp": "2014-03-15",
    }


@pytest.fixture
def rng() -> random.Random:
    """A seeded random number generator for reproducibility."""
    return random.Random(42)


@pytest.fixture
def all_timestamps() -> list:
    """Sample timestamps for corruption."""
    return [f"2014-{m:02d}-15" for m in range(1, 13)]


@pytest.fixture
def all_entities() -> list:
    """Sample entities for corruption."""
    return ["United States", "Russia", "China", "Germany", "France", "India"]


@pytest.fixture
def all_relations() -> list:
    """Sample relations for corruption."""
    return ["Make statement", "Threaten", "Consult", "Praise", "Criticize"]


class TestTimestampCorruption:
    """Tests for timestamp corruption strategy."""

    def test_timestamp_changes(
        self, sample_fact: dict, all_timestamps: list, rng: random.Random
    ) -> None:
        """The corrupted timestamp must differ from the original."""
        corrupted = corrupt_timestamp(sample_fact, all_timestamps, rng)
        assert corrupted["timestamp"] != sample_fact["timestamp"]

    def test_other_fields_preserved(
        self, sample_fact: dict, all_timestamps: list, rng: random.Random
    ) -> None:
        """Subject, predicate, and object must remain unchanged."""
        corrupted = corrupt_timestamp(sample_fact, all_timestamps, rng)
        assert corrupted["subject"] == sample_fact["subject"]
        assert corrupted["predicate"] == sample_fact["predicate"]
        assert corrupted["object"] == sample_fact["object"]

    def test_corruption_type_label(
        self, sample_fact: dict, all_timestamps: list, rng: random.Random
    ) -> None:
        """The corruption type must be labelled correctly."""
        corrupted = corrupt_timestamp(sample_fact, all_timestamps, rng)
        assert corrupted["corruption_type"] == "timestamp"

    def test_original_fact_preserved(
        self, sample_fact: dict, all_timestamps: list, rng: random.Random
    ) -> None:
        """The original fact must be stored in the corrupted output."""
        corrupted = corrupt_timestamp(sample_fact, all_timestamps, rng)
        assert "original_fact" in corrupted


class TestEntityCorruption:
    """Tests for entity corruption strategy."""

    def test_entity_changes(
        self, sample_fact: dict, all_entities: list, rng: random.Random
    ) -> None:
        """At least one entity must differ from the original."""
        corrupted = corrupt_entity(sample_fact, all_entities, rng)
        changed = (
            corrupted["subject"] != sample_fact["subject"]
            or corrupted["object"] != sample_fact["object"]
        )
        assert changed

    def test_corruption_type_label(
        self, sample_fact: dict, all_entities: list, rng: random.Random
    ) -> None:
        """The corruption type must be labelled correctly."""
        corrupted = corrupt_entity(sample_fact, all_entities, rng)
        assert corrupted["corruption_type"] == "entity"

    def test_corrupted_field_specified(
        self, sample_fact: dict, all_entities: list, rng: random.Random
    ) -> None:
        """The corrupted field (subject or object) must be specified."""
        corrupted = corrupt_entity(sample_fact, all_entities, rng)
        assert corrupted["corrupted_field"] in ("subject", "object")


class TestRelationCorruption:
    """Tests for relation corruption strategy."""

    def test_relation_changes(
        self, sample_fact: dict, all_relations: list, rng: random.Random
    ) -> None:
        """The corrupted relation must differ from the original."""
        corrupted = corrupt_relation(sample_fact, all_relations, rng)
        assert corrupted["predicate"] != sample_fact["predicate"]

    def test_entities_preserved(
        self, sample_fact: dict, all_relations: list, rng: random.Random
    ) -> None:
        """Entities and timestamp must remain unchanged."""
        corrupted = corrupt_relation(sample_fact, all_relations, rng)
        assert corrupted["subject"] == sample_fact["subject"]
        assert corrupted["object"] == sample_fact["object"]
        assert corrupted["timestamp"] == sample_fact["timestamp"]

    def test_corruption_type_label(
        self, sample_fact: dict, all_relations: list, rng: random.Random
    ) -> None:
        """The corruption type must be labelled correctly."""
        corrupted = corrupt_relation(sample_fact, all_relations, rng)
        assert corrupted["corruption_type"] == "relation"


class TestOrderCorruption:
    """Tests for temporal order corruption strategy."""

    def test_timestamps_swapped(self) -> None:
        """Timestamps between the two facts must be swapped."""
        fact_a = {
            "subject": "A", "predicate": "rel", "object": "B",
            "timestamp": "2014-03-01",
        }
        fact_b = {
            "subject": "A", "predicate": "rel", "object": "C",
            "timestamp": "2014-07-15",
        }

        corrupted_a, corrupted_b = corrupt_temporal_order(fact_a, fact_b)

        assert corrupted_a["timestamp"] == "2014-07-15"
        assert corrupted_b["timestamp"] == "2014-03-01"

    def test_corruption_type_label(self) -> None:
        """Both corrupted facts must have ordering corruption type."""
        fact_a = {
            "subject": "A", "predicate": "rel", "object": "B",
            "timestamp": "2014-01-01",
        }
        fact_b = {
            "subject": "A", "predicate": "rel", "object": "C",
            "timestamp": "2014-12-31",
        }

        corrupted_a, corrupted_b = corrupt_temporal_order(fact_a, fact_b)

        assert corrupted_a["corruption_type"] == "ordering"
        assert corrupted_b["corruption_type"] == "ordering"
