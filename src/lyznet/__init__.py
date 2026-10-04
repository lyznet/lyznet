"""LyZNet public interface."""

from importlib.metadata import PackageNotFoundError, version

from .analysis import QuadraticLyapunov, linearize, quadratic_lyapunov
from .sets import Box, EuclideanBall, SublevelSet, WholeSpace
from .systems import ContinuousTimeSystem, ControlAffineSystem, System
from .verification import (
    OptionalVerificationDependencyError,
    QuadraticVerificationResult,
    VerificationObligation,
    VerificationStatus,
    verify_quadratic_roa,
)
from .workflows import (
    LyapunovSearchResult,
    LyapunovSearchTimings,
    VerifiedLyapunovCertificate,
    find_lyap,
    find_lyapunov,
)


try:
    __version__ = version("lyznet")
except PackageNotFoundError:
    __version__ = "uninstalled"


__all__ = [
    "Box",
    "ContinuousTimeSystem",
    "ControlAffineSystem",
    "EuclideanBall",
    "LyapunovSearchResult",
    "LyapunovSearchTimings",
    "OptionalVerificationDependencyError",
    "QuadraticLyapunov",
    "QuadraticVerificationResult",
    "SublevelSet",
    "System",
    "VerificationObligation",
    "VerificationStatus",
    "VerifiedLyapunovCertificate",
    "WholeSpace",
    "__version__",
    "find_lyap",
    "find_lyapunov",
    "linearize",
    "quadratic_lyapunov",
    "verify_quadratic_roa",
]
