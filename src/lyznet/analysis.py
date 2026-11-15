"""Linearization and quadratic Lyapunov candidate construction."""

from dataclasses import dataclass

import numpy as np
from scipy import linalg
import sympy as sp

from .systems import ControlAffineSystem


def homogeneous_degree(system):
    """Return the exact ordinary degree of a homogeneous vector field.

    Zero components impose no degree. The positive scale symbol lets SymPy
    simplify radicals such as ``sqrt(scale**6 * p(x))`` without introducing
    an absolute value.
    """

    if isinstance(system, ControlAffineSystem):
        raise TypeError("homogeneous_degree requires an autonomous system.")
    scale = sp.Symbol("_homogeneity_scale", positive=True)
    substitutions = {
        state: scale * state for state in system.states
    }
    common_degree = None
    for expression in system.symbolic.dynamics:
        expression = sp.sympify(expression)
        if expression == 0:
            continue
        scaled = expression.subs(substitutions, simultaneous=True)
        ratio = sp.powsimp(
            sp.powdenest(sp.simplify(scaled / expression), force=True),
            force=True,
        )
        powers = ratio.as_powers_dict()
        degree = sp.simplify(powers.get(scale, 0))
        if degree.free_symbols or sp.simplify(ratio - scale**degree) != 0:
            return None
        if common_degree is None:
            common_degree = degree
        elif sp.simplify(degree - common_degree) != 0:
            return None
    return common_degree


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


@dataclass(frozen=True)
class QuadraticControlLyapunov:
    """Numerical data for an LQR quadratic control-Lyapunov function."""

    matrix: np.ndarray
    q_matrix: np.ndarray
    r_matrix: np.ndarray
    linearization: np.ndarray
    input_linearization: np.ndarray
    centre: np.ndarray

    def __post_init__(self):
        for name in (
            "matrix",
            "q_matrix",
            "r_matrix",
            "linearization",
            "input_linearization",
            "centre",
        ):
            object.__setattr__(
                self, name, _immutable_array(getattr(self, name), name)
            )
        if (
            self.matrix.ndim != 2
            or self.matrix.shape[0] < 1
            or self.matrix.shape[0] != self.matrix.shape[1]
        ):
            raise ValueError("matrix must be a nonempty square matrix.")
        if (
            self.r_matrix.ndim != 2
            or self.r_matrix.shape[0] < 1
            or self.r_matrix.shape[0] != self.r_matrix.shape[1]
        ):
            raise ValueError("r_matrix must be a nonempty square matrix.")
        state_dimension = self.matrix.shape[0]
        input_dimension = self.r_matrix.shape[0]
        for name, shape in (
            ("matrix", (state_dimension, state_dimension)),
            ("q_matrix", (state_dimension, state_dimension)),
            ("r_matrix", (input_dimension, input_dimension)),
            ("linearization", (state_dimension, state_dimension)),
            (
                "input_linearization",
                (state_dimension, input_dimension),
            ),
            ("centre", (state_dimension,)),
        ):
            if getattr(self, name).shape != shape:
                raise ValueError("{} must have shape {}.".format(name, shape))

    @property
    def state_dimension(self):
        return self.matrix.shape[0]

    @property
    def input_dimension(self):
        return self.r_matrix.shape[0]

    @property
    def gain(self):
        """Return the LQR feedback under ``u-u*=K(x-x*)``."""

        return -np.linalg.solve(
            self.r_matrix,
            self.input_linearization.T @ self.matrix,
        )

    @property
    def residual(self):
        feedback = (
            self.matrix
            @ self.input_linearization
            @ np.linalg.solve(
                self.r_matrix,
                self.input_linearization.T @ self.matrix,
            )
        )
        return (
            self.linearization.T @ self.matrix
            + self.matrix @ self.linearization
            - feedback
            + self.q_matrix
        )

    def value(self, states):
        values = np.asarray(states)
        if values.ndim == 0 or values.shape[-1] != self.state_dimension:
            raise ValueError(
                "States must have shape (..., {}).".format(
                    self.state_dimension
                )
            )
        shifted = values - self.centre
        return np.einsum(
            "...i,ij,...j->...", shifted, self.matrix, shifted
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


def quadratic_control_lyapunov(system, q_matrix=None, r_matrix=None):
    """Construct a continuous-time LQR quadratic CLF at equilibrium."""

    if not isinstance(system, ControlAffineSystem):
        raise TypeError(
            "quadratic_control_lyapunov requires a control-affine system."
        )
    q_matrix = _positive_definite_matrix(
        np.eye(system.state_dimension) if q_matrix is None else q_matrix,
        system.state_dimension,
        "q_matrix",
    )
    r_matrix = _positive_definite_matrix(
        np.eye(system.input_dimension) if r_matrix is None else r_matrix,
        system.input_dimension,
        "r_matrix",
    )
    state_matrix = linearize(system)
    substitutions = dict(zip(system.states, system.equilibrium))
    input_matrix = _real_finite_array(
        system.symbolic.input_matrix.subs(
            substitutions, simultaneous=True
        ).tolist(),
        "Input linearization",
    )

    # Continuous-time algebraic Riccati equation:
    # A.T P + P A - P B R^-1 B.T P + Q = 0.
    try:
        matrix = linalg.solve_continuous_are(
            state_matrix, input_matrix, q_matrix, r_matrix
        )
    except linalg.LinAlgError as error:
        raise ValueError(
            "The linearization is not stabilizable for these LQR weights."
        ) from error
    matrix = _real_finite_array(
        0.5 * (matrix + matrix.T), "Riccati solution"
    )
    if np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise ValueError("The Riccati solution must be positive definite.")
    feedback_term = (
        matrix
        @ input_matrix
        @ np.linalg.solve(r_matrix, input_matrix.T @ matrix)
    )
    residual_terms = (
        state_matrix.T @ matrix,
        matrix @ state_matrix,
        -feedback_term,
        q_matrix,
    )
    _require_small_residual(sum(residual_terms), residual_terms)
    gain = -np.linalg.solve(r_matrix, input_matrix.T @ matrix)
    closed_loop = state_matrix + input_matrix @ gain
    if np.max(np.real(np.linalg.eigvals(closed_loop))) >= 0:
        raise ValueError("The LQR closed-loop linearization is not Hurwitz.")
    return QuadraticControlLyapunov(
        matrix=matrix,
        q_matrix=q_matrix,
        r_matrix=r_matrix,
        linearization=state_matrix,
        input_linearization=input_matrix,
        centre=_real_finite_array(system.equilibrium, "LQR centre"),
    )


__all__ = [
    "homogeneous_degree",
    "QuadraticControlLyapunov",
    "QuadraticLyapunov",
    "linearize",
    "quadratic_control_lyapunov",
    "quadratic_lyapunov",
]
