"""
Tests for the data loading modules.

These tests verify that the ICEWS14 and CronQuestions parsers correctly
handle dataset files and produce the expected data structures.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader.icews_loader import ICEWS14Loader


class TestICEWS14Loader:
    """Tests for the ICEWS14Loader class."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory with sample ICEWS14 data files."""
        data_dir = tmp_path / "icews14"
        data_dir.mkdir()

        # Create entity2id.txt
        entity2id = data_dir / "entity2id.txt"
        entity2id.write_text(
            "United States\t0\n"
            "Russia\t1\n"
            "China\t2\n"
            "United Kingdom\t3\n"
        )

        # Create relation2id.txt
        relation2id = data_dir / "relation2id.txt"
        relation2id.write_text(
            "Make statement\t0\n"
            "Threaten\t1\n"
            "Consult\t2\n"
        )

        # Create train.txt (subject_id, relation_id, object_id, timestamp_id)
        train = data_dir / "train.txt"
        train.write_text(
            "0\t0\t1\t0\n"   # US, Make statement, Russia, 2014-01-01
            "1\t1\t2\t10\n"  # Russia, Threaten, China, 2014-01-11
            "0\t2\t3\t74\n"  # US, Consult, UK, 2014-03-16
        )

        # Create valid.txt
        valid = data_dir / "valid.txt"
        valid.write_text("2\t0\t0\t100\n")  # China, Make statement, US, 2014-04-11

        # Create test.txt
        test = data_dir / "test.txt"
        test.write_text("3\t1\t1\t200\n")  # UK, Threaten, Russia, 2014-07-20

        return data_dir

    def test_load_mappings(self, sample_data_dir: Path) -> None:
        """Test that entity and relation mappings load correctly."""
        loader = ICEWS14Loader(sample_data_dir)
        id2entity, id2relation = loader._load_mappings()

        assert id2entity[0] == "United States"
        assert id2entity[1] == "Russia"
        assert id2relation[0] == "Make statement"
        assert id2relation[1] == "Threaten"

    def test_timestamp_conversion(self, sample_data_dir: Path) -> None:
        """Test that timestamp IDs map to correct dates."""
        loader = ICEWS14Loader(sample_data_dir)

        assert loader._timestamp_id_to_date(0) == "2014-01-01"
        assert loader._timestamp_id_to_date(1) == "2014-01-02"
        assert loader._timestamp_id_to_date(364) == "2014-12-31"

    def test_load_split(self, sample_data_dir: Path) -> None:
        """Test loading a single split."""
        loader = ICEWS14Loader(sample_data_dir)
        train_facts = loader.load_split("train")

        assert len(train_facts) == 3
        assert train_facts[0]["subject"] == "United States"
        assert train_facts[0]["relation"] == "Make statement"
        assert train_facts[0]["object"] == "Russia"
        assert train_facts[0]["timestamp"] == "2014-01-01"

    def test_load_all(self, sample_data_dir: Path) -> None:
        """Test loading all splits."""
        loader = ICEWS14Loader(sample_data_dir)
        all_data = loader.load_all()

        assert "train" in all_data
        assert "valid" in all_data
        assert "test" in all_data
        assert len(all_data["train"]) == 3
        assert len(all_data["valid"]) == 1
        assert len(all_data["test"]) == 1

    def test_get_entities(self, sample_data_dir: Path) -> None:
        """Test extracting unique entity names."""
        loader = ICEWS14Loader(sample_data_dir)
        entities = loader.get_entities()

        assert "United States" in entities
        assert "Russia" in entities

    def test_get_relations(self, sample_data_dir: Path) -> None:
        """Test extracting unique relation names."""
        loader = ICEWS14Loader(sample_data_dir)
        relations = loader.get_relations()

        assert "Make statement" in relations
        assert "Threaten" in relations

    def test_get_statistics(self, sample_data_dir: Path) -> None:
        """Test statistics computation."""
        loader = ICEWS14Loader(sample_data_dir)
        stats = loader.get_statistics()

        assert stats["num_entities"] == 4
        assert stats["num_relations"] == 3
