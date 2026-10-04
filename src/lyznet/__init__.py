"""LyZNet public interface."""

from importlib.metadata import PackageNotFoundError, version

from .analysis import QuadraticLyapunov, linearize, quadratic_lyapunov
from .certificates import PolynomialLyapunov
from .equilibria import NumericalEquilibrium, solve_equilibrium
from .sets import Box, EuclideanBall, SublevelSet, WholeSpace
from .systems import ContinuousTimeSystem, ControlAffineSystem, System
from .verification import (
    GlobalLyapunovVerificationResult,
    OptionalVerificationDependencyError,
    QuadraticVerificationResult,
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
    verify_quadratic_roa,
    verify_global_lyapunov,
    verify_global_quadratic,
)
from .synthesis import (
    PolynomialLyapunovSynthesisResult,
    SynthesisStatus,
    synthesize_global_polynomial_lyapunov,
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
    "GlobalLyapunovVerificationResult",
    "LyapunovSearchResult",
    "LyapunovSearchTimings",
    "NumericalEquilibrium",
    "OptionalVerificationDependencyError",
    "PolynomialLyapunov",
    "PolynomialLyapunovSynthesisResult",
    "QuadraticLyapunov",
    "QuadraticVerificationResult",
    "SublevelSet",
    "SynthesisStatus",
    "System",
    "VerificationObligation",
    "VerificationClaim",
    "VerificationStatus",
    "VerifiedLyapunovCertificate",
    "WholeSpace",
    "__version__",
    "find_lyap",
    "find_lyapunov",
    "linearize",
    "quadratic_lyapunov",
    "solve_equilibrium",
    "synthesize_global_polynomial_lyapunov",
    "verify_global_lyapunov",
    "verify_global_quadratic",
    "verify_quadratic_roa",
]
