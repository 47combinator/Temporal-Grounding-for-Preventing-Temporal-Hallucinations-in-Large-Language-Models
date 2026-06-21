"""
Verification package for temporal claim verification.

Provides the core verification engine (TemporalVerifier), consistency
checking (ConsistencyChecker), and data classes that describe verification
outcomes (VerificationStatus, VerificationResult, TemporalClaim, TemporalFact).
"""

from src.verification.verification_result import (
    TemporalClaim,
    TemporalFact,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "VerificationStatus",
    "TemporalClaim",
    "TemporalFact",
    "VerificationResult",
]
