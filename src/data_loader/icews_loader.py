"""
ICEWS14 dataset loader for the Temporal Grounding and Verification Framework.

Parses the tab-separated ICEWS14 files produced by ``download.py`` and
exposes the data as lists of typed dictionaries suitable for knowledge
graph construction.

The ICEWS14 dataset covers 365 days (2014-01-01 to 2014-12-31).  Each fact
is a quadruple (subject, relation, object, timestamp) encoded as integer
IDs in the raw files, with separate mapping files for decoding.
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from config.settings import ICEWS14_DIR, ICEWS14_START_DATE, ICEWS14_NUM_DAYS

logger = logging.getLogger(__name__)

# Pre-compute the base date once at module level.
_BASE_DATE: date = date.fromisoformat(ICEWS14_START_DATE)

# Recognised split names and their file names.
_SPLIT_FILES: dict[str, str] = {
    "train": "train.txt",
    "valid": "valid.txt",
    "test": "test.txt",
}


class ICEWS14Loader:
    """Loader for the ICEWS14 temporal knowledge graph dataset.

    Reads the tab-separated files produced by the download step and
    provides high-level accessors for facts, entities, relations, and
    dataset statistics.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialise the loader.

        Args:
            data_dir: Path to the directory containing the ICEWS14 files.
                Defaults to the configured ``ICEWS14_DIR``.
        """
        self.data_dir: Path = Path(data_dir) if data_dir is not None else ICEWS14_DIR

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"ICEWS14 data directory not found: {self.data_dir}. "
                "Run the download step first (python -m src.data_loader.download)."
            )

        # Lazily populated caches.
        self._id2entity: dict[int, str] | None = None
        self._id2relation: dict[int, str] | None = None

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _load_mappings(self) -> tuple[dict[int, str], dict[int, str]]:
        """Load entity and relation ID-to-name mappings from disk.

        Each mapping file has one entry per line in the format
        ``name<TAB>id``.

        Returns:
            A tuple ``(id2entity, id2relation)`` where each is a dict
            mapping integer IDs to human-readable names.

        Raises:
            FileNotFoundError: If the mapping files do not exist.
        """
        id2entity = self._read_mapping_file(self.data_dir / "entity2id.txt")
        id2relation = self._read_mapping_file(self.data_dir / "relation2id.txt")

        logger.info(
            "Loaded %d entity and %d relation mappings.",
            len(id2entity),
            len(id2relation),
        )
        return id2entity, id2relation

    @staticmethod
    def _read_mapping_file(path: Path) -> dict[int, str]:
        """Read a single ``name<TAB>id`` mapping file.

        Args:
            path: Path to the mapping file.

        Returns:
            Dictionary mapping integer IDs to name strings.
        """
        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found: {path}")

        mapping: dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit("\t", maxsplit=1)
                if len(parts) != 2:
                    logger.warning("Skipping malformed mapping line: %r", line)
                    continue
                name, id_str = parts
                mapping[int(id_str)] = name

        return mapping

    def _ensure_mappings(self) -> None:
        """Lazily load mappings if they have not been loaded yet."""
        if self._id2entity is None or self._id2relation is None:
            self._id2entity, self._id2relation = self._load_mappings()

    # ------------------------------------------------------------------
    # Timestamp conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp_id_to_date(tid: int) -> str:
        """Convert a timestamp ID (0-364) to an ISO date string.

        Day 0 corresponds to 2014-01-01; day 364 to 2014-12-31.

        Args:
            tid: Integer timestamp ID in the range [0, 364].

        Returns:
            Date string in ``YYYY-MM-DD`` format.

        Raises:
            ValueError: If *tid* is outside the valid range.
        """
        if not 0 <= tid < ICEWS14_NUM_DAYS:
            raise ValueError(
                f"Timestamp ID {tid} is out of range [0, {ICEWS14_NUM_DAYS - 1}]."
            )
        return (_BASE_DATE + timedelta(days=tid)).isoformat()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_split(self, split: str) -> list[dict[str, Any]]:
        """Load a single data split (train, valid, or test).

        Each line in the split file has the format
        ``subject_id<TAB>relation_id<TAB>object_id<TAB>timestamp_id``.
        The method resolves the integer IDs to human-readable names.

        Args:
            split: One of ``"train"``, ``"valid"``, or ``"test"``.

        Returns:
            List of fact dictionaries with keys:
            ``subject``, ``relation``, ``object``, ``timestamp``,
            ``subject_id``, ``relation_id``, ``object_id``, ``timestamp_id``.

        Raises:
            ValueError: If *split* is not a recognised split name.
            FileNotFoundError: If the split file is missing.
        """
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split '{split}'. Choose from {list(_SPLIT_FILES.keys())}."
            )

        self._ensure_mappings()
        assert self._id2entity is not None and self._id2relation is not None

        split_path = self.data_dir / _SPLIT_FILES[split]
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        facts: list[dict[str, Any]] = []
        with open(split_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 4:
                    logger.warning(
                        "Skipping malformed line %d in %s: %r",
                        line_no,
                        split_path.name,
                        line,
                    )
                    continue

                subj_id, rel_id, obj_id, ts_id = (int(p) for p in parts)

                facts.append(
                    {
                        "subject": self._id2entity.get(subj_id, f"ENTITY_{subj_id}"),
                        "relation": self._id2relation.get(rel_id, f"RELATION_{rel_id}"),
                        "object": self._id2entity.get(obj_id, f"ENTITY_{obj_id}"),
                        "timestamp": self._timestamp_id_to_date(ts_id),
                        "subject_id": subj_id,
                        "relation_id": rel_id,
                        "object_id": obj_id,
                        "timestamp_id": ts_id,
                    }
                )

        logger.info("Loaded %d facts from %s split.", len(facts), split)
        return facts

    def load_all(self) -> dict[str, list[dict[str, Any]]]:
        """Load all three splits (train, valid, test).

        Returns:
            Dictionary mapping split name to list of fact dictionaries.
        """
        return {split: self.load_split(split) for split in _SPLIT_FILES}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_entities(self) -> list[str]:
        """Return all unique entity names from the mapping file.

        Returns:
            Sorted list of entity name strings.
        """
        self._ensure_mappings()
        assert self._id2entity is not None
        return sorted(self._id2entity.values())

    def get_relations(self) -> list[str]:
        """Return all unique relation names from the mapping file.

        Returns:
            Sorted list of relation name strings.
        """
        self._ensure_mappings()
        assert self._id2relation is not None
        return sorted(self._id2relation.values())

    def get_statistics(self) -> dict[str, Any]:
        """Compute summary statistics for the ICEWS14 dataset.

        Returns:
            Dictionary with counts of entities, relations, timestamps,
            and the number of facts in each split.
        """
        self._ensure_mappings()
        assert self._id2entity is not None and self._id2relation is not None

        stats: dict[str, Any] = {
            "num_entities": len(self._id2entity),
            "num_relations": len(self._id2relation),
            "num_timestamps": ICEWS14_NUM_DAYS,
        }

        for split in _SPLIT_FILES:
            split_path = self.data_dir / _SPLIT_FILES[split]
            if split_path.exists():
                with open(split_path, "r", encoding="utf-8") as fh:
                    count = sum(1 for line in fh if line.strip())
                stats[f"num_facts_{split}"] = count
            else:
                stats[f"num_facts_{split}"] = 0

        stats["num_facts_total"] = sum(
            stats[f"num_facts_{s}"] for s in _SPLIT_FILES
        )

        return stats
