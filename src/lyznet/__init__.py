"""LyZNet public interface."""

from importlib.metadata import PackageNotFoundError, version

from .analysis import QuadraticLyapunov, linearize, quadratic_lyapunov
from .sets import Box, EuclideanBall, WholeSpace
from .systems import ContinuousTimeSystem, ControlAffineSystem, System


try:
    __version__ = version("lyznet")
except PackageNotFoundError:
    __version__ = "uninstalled"


__all__ = [
    "Box",
    "ContinuousTimeSystem",
    "ControlAffineSystem",
    "EuclideanBall",
    "QuadraticLyapunov",
    "System",
    "WholeSpace",
    "__version__",
    "linearize",
    "quadratic_lyapunov",
]
