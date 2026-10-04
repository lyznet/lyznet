"""Numerical equilibrium discovery with retained residual evidence."""

from dataclasses import dataclass

import numpy as np
from scipy import optimize
import sympy as sp


def _normalise_problem(states, dynamics):
    states = tuple(states)
    if not states or not all(isinstance(state, sp.Symbol) for state in states):
        raise TypeError("states must contain one or more SymPy symbols.")
    if len(set(states)) != len(states):
        raise ValueError("states must not contain duplicate symbols.")
    expressions = tuple(sp.sympify(value) for value in dynamics)
    if len(expressions) != len(states):
        raise ValueError(
            "dynamics must contain {} expressions, received {}.".format(
                len(states), len(expressions)
            )
        )
    free_symbols = set().union(*(
        expression.free_symbols for expression in expressions
    ))
    undeclared = free_symbols.difference(states)
    if undeclared:
        names = ", ".join(sorted(str(symbol) for symbol in undeclared))
        raise ValueError("dynamics contains undeclared symbols: {}.".format(names))
    return states, expressions


def _finite_vector(values, dimension, label):
    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "{} must contain {} finite real values.".format(label, dimension)
        ) from error
    if np.iscomplexobj(source):
        raise ValueError(
            "{} must contain {} finite real values.".format(label, dimension)
        )
    try:
        array = np.asarray(source, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "{} must contain {} finite real values.".format(label, dimension)
        ) from error
    if array.shape != (dimension,) or not np.all(np.isfinite(array)):
        raise ValueError(
            "{} must contain {} finite real values.".format(label, dimension)
        )
    return array


def _positive_finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be positive and finite.".format(label)) from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError("{} must be positive and finite.".format(label))
    return number


def _compile_dynamics(states, expressions):
    function = sp.lambdify(states, expressions, modules="numpy")

    def evaluate(values):
        with np.errstate(all="ignore"):
            result = function(*values)
        result = np.asarray(result)
        if np.iscomplexobj(result):
            raise ValueError("Dynamics evaluated to complex values.")
        return np.asarray(result, dtype=float).reshape(-1)

    return evaluate


@dataclass(frozen=True)
class NumericalEquilibrium:
    """An approximate equilibrium and its checked numerical residual."""

    states: tuple
    dynamics: tuple
    point: tuple
    residual: tuple
    residual_tolerance: float
    method: str

    def __post_init__(self):
        states, dynamics = _normalise_problem(self.states, self.dynamics)
        dimension = len(states)
        point = _finite_vector(self.point, dimension, "point")
        residual = _finite_vector(self.residual, dimension, "residual")
        tolerance = _positive_finite(
            self.residual_tolerance, "residual_tolerance"
        )
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a nonempty string.")

        try:
            evaluated = _compile_dynamics(states, dynamics)(point)
        except (TypeError, ValueError, ArithmeticError) as error:
            raise ValueError(
                "dynamics must evaluate to a finite real residual at point."
            ) from error
        if (
            evaluated.shape != (dimension,)
            or not np.all(np.isfinite(evaluated))
        ):
            raise ValueError(
                "dynamics must evaluate to a finite real residual at point."
            )
        scale = max(
            1.0,
            float(np.max(np.abs(evaluated))),
            float(np.max(np.abs(residual))),
        )
        roundoff = 32 * np.finfo(float).eps * scale
        if not np.allclose(
            residual,
            evaluated,
            rtol=32 * np.finfo(float).eps,
            atol=roundoff,
        ):
            raise ValueError(
                "Stored residual does not match dynamics evaluated at point."
            )
        if np.linalg.norm(evaluated, ord=np.inf) > tolerance:
            raise ValueError(
                "point does not meet the declared residual_tolerance."
            )

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "dynamics", dynamics)
        object.__setattr__(self, "point", tuple(float(value) for value in point))
        object.__setattr__(
            self, "residual", tuple(float(value) for value in residual)
        )
        object.__setattr__(self, "residual_tolerance", tolerance)

    @property
    def dimension(self):
        return len(self.point)

    @property
    def residual_norm(self):
        return float(np.linalg.norm(self.residual, ord=np.inf))


def solve_equilibrium(
    *,
    states,
    dynamics,
    initial_guess,
    method="hybr",
    residual_tolerance=1e-10,
    solver_options=None,
):
    """Solve ``dynamics = 0`` numerically from ``initial_guess``."""

    states, dynamics = _normalise_problem(states, dynamics)
    guess = _finite_vector(initial_guess, len(states), "initial_guess")
    tolerance = _positive_finite(residual_tolerance, "residual_tolerance")
    evaluate = _compile_dynamics(states, dynamics)
    solution = optimize.root(
        evaluate,
        guess,
        method=method,
        options={} if solver_options is None else dict(solver_options),
    )
    point = np.asarray(solution.x, dtype=float)
    residual = evaluate(point)
    if (
        not solution.success
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(residual))
        or np.linalg.norm(residual, ord=np.inf) > tolerance
    ):
        raise RuntimeError(
            "Equilibrium solve failed to meet residual tolerance {}: {}"
            .format(tolerance, solution.message)
        )
    return NumericalEquilibrium(
        states=states,
        dynamics=dynamics,
        point=tuple(point),
        residual=tuple(residual),
        residual_tolerance=tolerance,
        method="scipy.optimize.root:{}".format(method),
    )


__all__ = ["NumericalEquilibrium", "solve_equilibrium"]
