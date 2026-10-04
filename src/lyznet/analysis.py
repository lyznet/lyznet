"""Linearization and quadratic Lyapunov candidate construction."""

from dataclasses import dataclass

import numpy as np
from scipy import linalg
import sympy as sp

from .systems import ControlAffineSystem


def _real_finite_array(value, label):
    """Return a copied real floating-point array without lossy conversion."""

    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "{} must contain real numeric values.".format(label)
        ) from error
    if np.iscomplexobj(source):
        raise ValueError("{} must contain real values.".format(label))
    try:
        array = np.array(source, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "{} must contain real numeric values.".format(label)
        ) from error
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values.".format(label))
    return array


def _immutable_array(value, label):
    array = np.ascontiguousarray(_real_finite_array(value, label))
    immutable = np.frombuffer(array.tobytes(), dtype=array.dtype)
    return immutable.reshape(array.shape)


def _point(point, dimension, label):
    values = (point,) if np.isscalar(point) and dimension == 1 else tuple(point)
    if len(values) != dimension:
        raise ValueError(
            "{} must contain {} values, received {}.".format(
                label, dimension, len(values)
            )
        )
    return values


def linearize(system, at=None, input_at=None):
    """Return the state Jacobian at a declared operating point."""

    point = _point(
        system.equilibrium if at is None else at,
        system.state_dimension,
        "Linearization point",
    )
    _real_finite_array(point, "Linearization point")
    substitutions = dict(zip(system.states, point))

    if isinstance(system, ControlAffineSystem):
        control = _point(
            system.equilibrium_input if input_at is None else input_at,
            system.input_dimension,
            "Linearization input",
        )
        _real_finite_array(control, "Linearization input")
        dynamics = (
            system.symbolic.drift
            + system.symbolic.input_matrix * sp.ImmutableMatrix(control)
        )
        jacobian = dynamics.jacobian(system.states).subs(substitutions)
    else:
        if input_at is not None:
            raise ValueError("input_at applies only to a control-affine system.")
        jacobian = system.symbolic.jacobian.subs(substitutions)

    if jacobian.free_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in jacobian.free_symbols))
        raise ValueError(
            "Linearization still contains free symbols: {}.".format(names)
        )
    return _real_finite_array(jacobian.tolist(), "Linearization")


def _positive_definite_matrix(value, dimension, label):
    matrix = _real_finite_array(value, label)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            "{} must have shape ({}, {}), received {}.".format(
                label, dimension, dimension, matrix.shape
            )
        )
    if not np.array_equal(matrix, matrix.T):
        raise ValueError(
            "{} must be exactly symmetric after conversion to float.".format(label)
        )
    if np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise ValueError("{} must be positive definite.".format(label))
    return matrix


def _require_small_residual(residual, terms):
    """Check the floating-point Lyapunov equation solution explicitly."""

    scale = max(
        1.0,
        sum(float(np.linalg.norm(term, ord=np.inf)) for term in terms),
    )
    tolerance = 1e-12 + 1e-9 * scale
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    if residual_norm > tolerance:
        raise ValueError(
            "Lyapunov equation residual {:.3g} exceeds the numerical "
            "tolerance {:.3g}.".format(residual_norm, tolerance)
        )


@dataclass(frozen=True)
class QuadraticLyapunov:
    """Numerical data for ``V(x)=(x-centre)^T P (x-centre)``."""

    matrix: np.ndarray
    q_matrix: np.ndarray
    linearization: np.ndarray
    centre: np.ndarray

    def __post_init__(self):
        for name in ("matrix", "q_matrix", "linearization", "centre"):
            object.__setattr__(
                self, name, _immutable_array(getattr(self, name), name)
            )
        if (
            self.matrix.ndim != 2
            or self.matrix.shape[0] < 1
            or self.matrix.shape[0] != self.matrix.shape[1]
        ):
            raise ValueError("matrix must be a nonempty square matrix.")
        dimension = self.matrix.shape[0]
        for name, shape in (
            ("q_matrix", (dimension, dimension)),
            ("linearization", (dimension, dimension)),
            ("centre", (dimension,)),
        ):
            if getattr(self, name).shape != shape:
                raise ValueError("{} must have shape {}.".format(name, shape))

    @property
    def state_dimension(self):
        return self.matrix.shape[0]

    def value(self, states):
        values = np.asarray(states)
        if values.ndim == 0 or values.shape[-1] != self.state_dimension:
            raise ValueError(
                "States must have shape (..., {}).".format(self.state_dimension)
            )
        shifted = values - self.centre
        return np.einsum(
            "...i,ij,...j->...", shifted, self.matrix, shifted
        )

    @property
    def residual(self):
        return (
            self.linearization.T @ self.matrix
            + self.matrix @ self.linearization
            + self.q_matrix
        )


def quadratic_lyapunov(system, q_matrix=None, at=None):
    """Construct a quadratic candidate from a stable linearization.

    This function performs numerical checks only. Formal verification of the
    nonlinear system is a separate operation.
    """

    if isinstance(system, ControlAffineSystem):
        raise TypeError(
            "quadratic_lyapunov requires an autonomous system; close the "
            "loop first for a controlled plant."
        )
    centre = _real_finite_array(
        _point(
            system.equilibrium if at is None else at,
            system.state_dimension,
            "Lyapunov centre",
        ),
        "Lyapunov centre",
    )
    linearization = linearize(system, at=centre)
    dimension = system.state_dimension
    q_matrix = _positive_definite_matrix(
        np.eye(dimension) if q_matrix is None else q_matrix,
        dimension,
        "q_matrix",
    )

    # Candidate construction:
    #
    #   A^T P + P A = -Q,  Q > 0.
    #
    # A Hurwitz matrix gives the unique solution P > 0.
    if np.max(np.real(np.linalg.eigvals(linearization))) >= 0:
        raise ValueError("The continuous-time linearization is not Hurwitz.")
    try:
        matrix = linalg.solve_continuous_lyapunov(
            linearization.T, -q_matrix
        )
    except linalg.LinAlgError as error:
        raise ValueError(
            "The continuous Lyapunov equation could not be solved numerically."
        ) from error

    matrix = _real_finite_array(matrix, "Lyapunov solution")
    matrix = 0.5 * (matrix + matrix.T)
    if np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise ValueError("The Lyapunov solution is not positive definite.")
    terms = (
        linearization.T @ matrix,
        matrix @ linearization,
        q_matrix,
    )
    _require_small_residual(sum(terms), terms)
    return QuadraticLyapunov(
        matrix=matrix,
        q_matrix=q_matrix,
        linearization=linearization,
        centre=centre,
    )


__all__ = ["QuadraticLyapunov", "linearize", "quadratic_lyapunov"]
