"""
Pipeline package for the Temporal Grounding and Verification Framework.

Provides the end-to-end TemporalVerificationPipeline that orchestrates
retrieval, reasoning, claim extraction, verification, hallucination
detection, and explanation generation.
"""

from src.pipeline.pipeline import TemporalVerificationPipeline

__all__ = ["TemporalVerificationPipeline"]
