"""Formal verification interfaces and evidence records."""

from .quadratic import (
    OptionalVerificationDependencyError,
    QuadraticVerificationResult,
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
    verify_quadratic_roa,
)

__all__ = [
    "OptionalVerificationDependencyError",
    "QuadraticVerificationResult",
    "VerificationClaim",
    "VerificationObligation",
    "VerificationStatus",
    "verify_quadratic_roa",
]
