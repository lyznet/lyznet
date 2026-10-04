"""Formal verification interfaces and evidence records."""

from .quadratic import (
    OptionalVerificationDependencyError,
    QuadraticVerificationResult,
    VerificationObligation,
    VerificationStatus,
    verify_quadratic_roa,
)

__all__ = [
    "OptionalVerificationDependencyError",
    "QuadraticVerificationResult",
    "VerificationObligation",
    "VerificationStatus",
    "verify_quadratic_roa",
]
