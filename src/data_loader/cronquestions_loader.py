"""
CronQuestions dataset loader for the Temporal Grounding and Verification Framework.

Parses the CronQuestions / CronKGQA dataset which provides:
- A temporal knowledge graph derived from Wikidata with time-scoped facts.
- A question-answering benchmark of temporal questions grounded in that KG.

The KG facts are quintuples: (subject, relation, object, start_time, end_time).
Questions include temporal constraints and are categorised by answer type.
"""

import json
import logging
from pathlib import Path
from typing import Any

from config.settings import CRONQUESTIONS_DIR

logger = logging.getLogger(__name__)

# Expected sub-directory layout within the CronQuestions data root.
_KG_SUBDIR = "full_kg"
_QUESTIONS_SUBDIR = "questions"
_ENTITY_MAP_FILE = "wd_id2entity_text.txt"


class CronQuestionsLoader:
    """Loader for the CronQuestions temporal QA dataset.

    Reads entity mappings, temporal knowledge graph facts, and
    question-answer pairs from the extracted CronQuestions data
    directory.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialise the loader.

        Args:
            data_dir: Path to the root of the extracted CronQuestions
                data.  Defaults to the configured ``CRONQUESTIONS_DIR``.
        """
        self.data_dir: Path = Path(data_dir) if data_dir is not None else CRONQUESTIONS_DIR

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"CronQuestions data directory not found: {self.data_dir}. "
                "Download and extract data_v2.zip first "
                "(see src.data_loader.download.download_cronquestions)."
            )

        # Lazily populated caches.
        self._entities: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Entity mapping
    # ------------------------------------------------------------------

    def load_entities(self) -> dict[str, str]:
        """Load the Wikidata entity ID to human-readable name mapping.

        The mapping file (``wd_id2entity_text.txt``) has one entry per
        line in tab-separated format: ``wikidata_id<TAB>entity_name``.

        Returns:
            Dictionary mapping Wikidata IDs (e.g. ``"Q42"``) to their
            textual labels (e.g. ``"Douglas Adams"``).  Returns an
            empty dict if the mapping file does not exist.
        """
        if self._entities is not None:
            return self._entities

        entity_path = self.data_dir / _ENTITY_MAP_FILE
        if not entity_path.exists():
            logger.warning(
                "Entity mapping file not found: %s. "
                "Entity names will not be resolved.",
                entity_path,
            )
            self._entities = {}
            return self._entities

        entities: dict[str, str] = {}
        with open(entity_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                if len(parts) == 2:
                    entities[parts[0]] = parts[1]

        self._entities = entities
        logger.info("Loaded %d entity mappings from %s.", len(entities), entity_path)
        return self._entities

    # ------------------------------------------------------------------
    # Knowledge graph
    # ------------------------------------------------------------------

    def load_kg(self) -> list[dict[str, Any]]:
        """Load the temporal knowledge graph facts.

        Reads all split files (``train.txt``, ``valid.txt``, ``test.txt``)
        from the ``full_kg/`` sub-directory.  Each line is a tab-separated
        quintuple: ``subject_id  relation_id  object_id  start_time  end_time``.

        Returns:
            List of fact dictionaries with keys:
            ``subject``, ``relation``, ``object``, ``start_time``, ``end_time``,
            ``subject_id``, ``relation_id``, ``object_id``.
        """
        kg_dir = self.data_dir / _KG_SUBDIR
        if not kg_dir.exists():
            raise FileNotFoundError(
                f"KG directory not found: {kg_dir}. "
                "Ensure the CronQuestions data is fully extracted."
            )

        entities = self.load_entities()
        facts: list[dict[str, Any]] = []

        for split_file in sorted(kg_dir.glob("*.txt")):
            count_before = len(facts)
            with open(split_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 5:
                        logger.warning(
                            "Skipping malformed KG line in %s: %r",
                            split_file.name,
                            line,
                        )
                        continue

                    subj_id, rel_id, obj_id = parts[0], parts[1], parts[2]
                    start_time = _normalise_time(parts[3])
                    end_time = _normalise_time(parts[4])

                    facts.append(
                        {
                            "subject": entities.get(subj_id, subj_id),
                            "relation": rel_id,
                            "object": entities.get(obj_id, obj_id),
                            "start_time": start_time,
                            "end_time": end_time,
                            "subject_id": subj_id,
                            "relation_id": rel_id,
                            "object_id": obj_id,
                        }
                    )

            logger.info(
                "Loaded %d facts from %s.",
                len(facts) - count_before,
                split_file.name,
            )

        logger.info("Total CronQuestions KG facts loaded: %d", len(facts))
        return facts

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------

    def load_questions(self, split: str = "test") -> list[dict[str, Any]]:
        """Load question-answer pairs for a given split.

        Reads the JSON file at ``questions/<split>.json``.  Each entry
        is expected to contain at least the fields ``question``,
        ``answer``, and ``type``.  The loader preserves all original
        fields and normalises the output keys.

        Args:
            split: The split to load (``"train"``, ``"valid"``, or
                ``"test"``).  Defaults to ``"test"``.

        Returns:
            List of QA dictionaries with keys:
            ``question``, ``answer``, ``answer_type``, ``entities``,
            ``timestamps``.
        """
        questions_dir = self.data_dir / _QUESTIONS_SUBDIR
        json_path = questions_dir / f"{split}.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"Questions file not found: {json_path}. "
                "Ensure the CronQuestions data is fully extracted."
            )

        with open(json_path, "r", encoding="utf-8") as fh:
            raw_data: list[dict[str, Any]] = json.load(fh)

        questions: list[dict[str, Any]] = []
        for entry in raw_data:
            questions.append(
                {
                    "question": entry.get("question", ""),
                    "answer": entry.get("answer", entry.get("answers", [])),
                    "answer_type": entry.get("type", entry.get("answer_type", "unknown")),
                    "entities": entry.get("entities", []),
                    "timestamps": entry.get("timestamps", entry.get("times", [])),
                }
            )

        logger.info(
            "Loaded %d questions from %s split.", len(questions), split
        )
        return questions

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Compute summary statistics for the CronQuestions dataset.

        Returns:
            Dictionary with counts of entities, KG facts, and questions
            per split.
        """
        entities = self.load_entities()
        stats: dict[str, Any] = {
            "num_entities": len(entities),
        }

        # KG facts count (without fully loading into memory).
        kg_dir = self.data_dir / _KG_SUBDIR
        if kg_dir.exists():
            total_kg = 0
            for split_file in sorted(kg_dir.glob("*.txt")):
                with open(split_file, "r", encoding="utf-8") as fh:
                    count = sum(1 for line in fh if line.strip())
                stats[f"num_kg_facts_{split_file.stem}"] = count
                total_kg += count
            stats["num_kg_facts_total"] = total_kg
        else:
            stats["num_kg_facts_total"] = 0

        # Question counts.
        questions_dir = self.data_dir / _QUESTIONS_SUBDIR
        if questions_dir.exists():
            total_q = 0
            for json_file in sorted(questions_dir.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                count = len(data) if isinstance(data, list) else 0
                stats[f"num_questions_{json_file.stem}"] = count
                total_q += count
            stats["num_questions_total"] = total_q
        else:
            stats["num_questions_total"] = 0

        return stats


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _normalise_time(raw: str) -> str:
    """Normalise a timestamp string from the CronQuestions KG.

    CronQuestions uses various sentinel values for unknown timestamps
    (e.g. ``"####-##-##"``, ``"-"``, empty strings).  This function
    returns the raw value if it looks like a valid date component, or
    an empty string otherwise.

    Args:
        raw: Raw timestamp string from the data file.

    Returns:
        Cleaned timestamp string, or ``""`` if the value is a
        placeholder for an unknown timestamp.
    """
    raw = raw.strip()
    if not raw or raw == "-" or "#" in raw:
        return ""
    return raw
