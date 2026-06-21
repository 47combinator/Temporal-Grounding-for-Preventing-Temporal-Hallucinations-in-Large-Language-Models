"""
Centralised configuration for the Temporal Grounding and Verification Framework.

All configurable values (file paths, model parameters, database credentials,
verification thresholds) are defined here. Sensitive values are loaded from
environment variables via python-dotenv.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

ICEWS14_DIR = RAW_DATA_DIR / "icews14"
CRONQUESTIONS_DIR = RAW_DATA_DIR / "cronquestions"

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

NEO4J_BATCH_SIZE = 5000  # Records per UNWIND batch during bulk loading.

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
LLM_TEMPERATURE = 0.0  # Deterministic output for reproducibility.
LLM_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K = 10  # Number of temporal facts to retrieve per query.
RETRIEVAL_WINDOW_DAYS = 30  # Default temporal window for neighbour queries.

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

TIMESTAMP_TOLERANCE_DAYS = 0  # Exact match required for ICEWS14 daily data.
PARTIAL_SUPPORT_THRESHOLD = 0.5  # Minimum confidence for partial support.

# ---------------------------------------------------------------------------
# Benchmark generation
# ---------------------------------------------------------------------------

CORRUPTIONS_PER_FACT = 3  # Number of corrupted versions per positive fact.
RANDOM_SEED = 42  # For reproducibility of benchmark generation.

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

CRONQUESTIONS_SAMPLE_SIZE = 2000  # Number of questions to sample for evaluation.
EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# ---------------------------------------------------------------------------
# ICEWS14 dataset constants
# ---------------------------------------------------------------------------

ICEWS14_START_DATE = "2014-01-01"
ICEWS14_NUM_DAYS = 365
