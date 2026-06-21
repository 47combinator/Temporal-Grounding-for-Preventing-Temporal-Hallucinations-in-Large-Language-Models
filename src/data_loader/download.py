"""
Dataset download utilities for the Temporal Grounding and Verification Framework.

Downloads and prepares the following datasets:
- ICEWS14: Integrated Crisis Early Warning System events for 2014,
  sourced from Hugging Face (linxy/ICEWS14).
- CronQuestions: temporal question-answering benchmark over Wikidata
  (from the CronKGQA GitHub repository).
"""

import logging
from pathlib import Path

from config.settings import ICEWS14_DIR, CRONQUESTIONS_DIR

logger = logging.getLogger(__name__)


def download_icews14(target_dir: Path | None = None) -> Path:
    """Download the ICEWS14 dataset from Hugging Face and save as text files.

    Uses the ``datasets`` library to fetch the ``linxy/ICEWS14`` dataset
    (``all`` configuration).  Produces the following files inside *target_dir*:

    - ``train.txt``, ``valid.txt``, ``test.txt`` -- tab-separated quadruples
      with columns: subject_id, relation_id, object_id, timestamp_id.
    - ``entity2id.txt`` -- mapping of entity name to integer ID.
    - ``relation2id.txt`` -- mapping of relation name to integer ID.

    Args:
        target_dir: Directory to write the files into. Defaults to the
            configured ``ICEWS14_DIR``.

    Returns:
        The resolved *target_dir* path.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to download ICEWS14. "
            "Install it with: pip install datasets"
        ) from exc

    if target_dir is None:
        target_dir = ICEWS14_DIR

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ICEWS14 from Hugging Face (linxy/ICEWS14)...")
    dataset = load_dataset("linxy/ICEWS14", "all")

    # ------------------------------------------------------------------
    # Write split files (train / valid / test)
    # ------------------------------------------------------------------
    split_name_map: dict[str, str] = {
        "train": "train.txt",
        "validation": "valid.txt",
        "test": "test.txt",
    }

    for hf_split, filename in split_name_map.items():
        if hf_split not in dataset:
            logger.warning("Split '%s' not found in dataset -- skipping.", hf_split)
            continue

        out_path = target_dir / filename
        split_data = dataset[hf_split]
        count = 0
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in split_data:
                line = "\t".join(
                    str(row[col])
                    for col in ("subject", "relation", "object", "timestamp")
                )
                fh.write(line + "\n")
                count += 1

        logger.info("Wrote %d facts to %s", count, out_path)

    # ------------------------------------------------------------------
    # Extract and write entity / relation mappings
    # ------------------------------------------------------------------
    _write_id_mapping(dataset, "subject", target_dir / "entity2id.txt")
    _write_id_mapping(dataset, "relation", target_dir / "relation2id.txt")

    logger.info("ICEWS14 download complete. Files saved to %s", target_dir)
    return target_dir


def _write_id_mapping(
    dataset,  # type: ignore[type-arg]  # HF DatasetDict
    feature_name: str,
    out_path: Path,
) -> None:
    """Extract a ClassLabel mapping from the dataset features and write it.

    The output format is one mapping per line: ``name<TAB>id``.

    Args:
        dataset: A Hugging Face ``DatasetDict`` containing the ICEWS14 splits.
        feature_name: The column whose ``ClassLabel`` feature to extract
            (e.g. ``"subject"`` or ``"relation"``).
        out_path: File to write the mapping to.
    """
    # Grab the feature spec from the first available split.
    first_split = next(iter(dataset.values()))
    feature = first_split.features[feature_name]

    # ClassLabel stores the names list directly.
    names: list[str] = feature.names

    with open(out_path, "w", encoding="utf-8") as fh:
        for idx, name in enumerate(names):
            fh.write(f"{name}\t{idx}\n")

    logger.info(
        "Wrote %d %s mappings to %s",
        len(names),
        feature_name,
        out_path,
    )


def download_cronquestions(target_dir: Path | None = None) -> Path:
    """Prepare the CronQuestions dataset directory.

    The CronQuestions dataset is part of the CronKGQA project hosted on
    GitHub.  The full data (``data_v2.zip``) must be obtained separately
    because GitHub does not reliably serve large binary files via raw
    download.

    Manual steps after running this function:

    1. Visit https://github.com/apoorvumang/CronKGQA
    2. Download ``data_v2.zip`` from the repository (check the README
       for the latest Google Drive / direct link).
    3. Extract the archive into the directory printed below.

    The expected directory layout after extraction::

        <target_dir>/
            wd_id2entity_text.txt
            full_kg/
                train.txt
                valid.txt
                test.txt
            questions/
                train.json  (or .pickle)
                valid.json
                test.json

    Args:
        target_dir: Directory to prepare. Defaults to the configured
            ``CRONQUESTIONS_DIR``.

    Returns:
        The resolved *target_dir* path.
    """
    if target_dir is None:
        target_dir = CRONQUESTIONS_DIR

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    instructions = (
        "\n"
        "=================================================================\n"
        "  CronQuestions dataset -- manual download required\n"
        "=================================================================\n"
        "\n"
        "  1. Visit https://github.com/apoorvumang/CronKGQA\n"
        "  2. Download 'data_v2.zip' (see the repo README for links).\n"
        f"  3. Extract into: {target_dir}\n"
        "\n"
        "  Expected layout after extraction:\n"
        f"    {target_dir / 'wd_id2entity_text.txt'}\n"
        f"    {target_dir / 'full_kg' / 'train.txt'}\n"
        f"    {target_dir / 'questions' / 'test.json'}\n"
        "\n"
        "=================================================================\n"
    )

    print(instructions)
    logger.info("CronQuestions directory created at %s", target_dir)
    return target_dir


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print("--- Downloading ICEWS14 ---")
    download_icews14()

    print("\n--- Preparing CronQuestions ---")
    download_cronquestions()
