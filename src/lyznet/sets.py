"""Basic numerical sets used to define systems."""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


def _finite_points(points, dimension):
    """Return real finite points whose final axis has ``dimension`` entries."""

    values = np.asarray(points)
    if values.ndim == 0 or values.shape[-1] != dimension:
        raise ValueError(
            "Expected points with final dimension {}, received shape {}."
            .format(dimension, values.shape)
        )
    if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
        raise ValueError("Points must contain finite real numbers.")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Points must contain finite real numbers.")
    return values


@dataclass(frozen=True)
class Box:
    """A closed axis-aligned box with finite floating-point bounds."""

    bounds: tuple[tuple[float, float], ...]

    def __init__(self, bounds):
        normalised = []
        for index, bound in enumerate(bounds):
            if len(bound) != 2:
                raise ValueError(
                    "Each box bound must contain a lower and upper value; "
                    "bound {} has length {}.".format(index, len(bound))
                )
            lower, upper = float(bound[0]), float(bound[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError("Box bounds must be finite.")
            if lower >= upper:
                raise ValueError(
                    "Box lower bounds must be strictly less than upper bounds."
                )
            if not np.isfinite(upper - lower):
                raise ValueError("Box bound widths must be finite.")
            normalised.append((lower, upper))

        if not normalised:
            raise ValueError("A box must have at least one dimension.")
        object.__setattr__(self, "bounds", tuple(normalised))

    @property
    def dimension(self):
        return len(self.bounds)

    @property
    def lower(self):
        return np.asarray([bound[0] for bound in self.bounds], dtype=float)

    @property
    def upper(self):
        return np.asarray([bound[1] for bound in self.bounds], dtype=float)

    @property
    def volume(self):
        return math.prod(upper - lower for lower, upper in self.bounds)

    def contains(self, points):
        values = _finite_points(points, self.dimension)
        return np.all((values >= self.lower) & (values <= self.upper), axis=-1)


@dataclass(frozen=True)
class EuclideanBall:
    """A closed Euclidean ball with a finite center and positive radius."""

    center: tuple[float, ...]
    radius: float

    def __init__(self, center, radius):
        try:
            raw_center = tuple(center)
        except TypeError as error:
            raise ValueError(
                "A Euclidean-ball center must contain finite real numbers."
            ) from error
        if not raw_center:
            raise ValueError(
                "A Euclidean-ball center must contain finite real numbers."
            )

        values = []
        for value in raw_center:
            if isinstance(value, (bool, str, bytes, complex)):
                raise ValueError(
                    "A Euclidean-ball center must contain finite real numbers."
                )
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "A Euclidean-ball center must contain finite real numbers."
                ) from error
            if not np.isfinite(number):
                raise ValueError(
                    "A Euclidean-ball center must contain finite real numbers."
                )
            values.append(number)

        if isinstance(radius, (bool, str, bytes, complex)):
            raise ValueError(
                "A Euclidean-ball radius must be a positive finite number."
            )
        try:
            radius = float(radius)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "A Euclidean-ball radius must be a positive finite number."
            ) from error
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(
                "A Euclidean-ball radius must be a positive finite number."
            )

        object.__setattr__(self, "center", tuple(values))
        object.__setattr__(self, "radius", radius)

    @property
    def dimension(self):
        return len(self.center)

    def contains(self, points):
        values = _finite_points(points, self.dimension)
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.hypot.reduce(
                values - np.asarray(self.center), axis=-1
            )
        return distances <= self.radius


@dataclass(frozen=True)
class WholeSpace:
    """The complete real state space ``R^dimension``."""

    dimension: int

    def __post_init__(self):
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, Integral)
            or self.dimension < 1
        ):
            raise ValueError("dimension must be a positive integer.")

    def contains(self, points):
        values = _finite_points(points, self.dimension)
        return np.ones(values.shape[:-1], dtype=bool)


@dataclass(frozen=True)
class SublevelSet:
    """The closed set ``{x : certificate.value(x) <= level}``."""

    certificate: object
    level: float

    def __post_init__(self):
        if self.certificate is None:
            raise ValueError("A sublevel set requires a certificate.")
        if isinstance(self.level, (bool, str, bytes, complex)):
            raise ValueError("A sublevel level must be positive and finite.")
        try:
            level = float(self.level)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "A sublevel level must be positive and finite."
            ) from error
        if not np.isfinite(level) or level <= 0:
            raise ValueError("A sublevel level must be positive and finite.")
        object.__setattr__(self, "level", level)

    def contains(self, points):
        return self.certificate.value(points) <= self.level


__all__ = ["Box", "EuclideanBall", "SublevelSet", "WholeSpace"]
