"""Immutable representations of nonquadratic certificates."""

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class PolynomialLyapunov:
    """A fixed polynomial Lyapunov candidate in declared state variables."""

    states: tuple
    expression: sp.Expr

    def __post_init__(self):
        states = tuple(self.states)
        if not states or not all(
            isinstance(state, sp.Symbol) for state in states
        ):
            raise TypeError("states must contain one or more SymPy symbols.")
        expression = sp.expand(sp.sympify(self.expression))
        undeclared = expression.free_symbols.difference(states)
        if undeclared:
            raise ValueError("expression contains undeclared symbols.")
        if not expression.is_polynomial(*states):
            raise ValueError("expression must be a polynomial.")
        polynomial = sp.Poly(expression, *states)
        if any(
            coefficient.is_real is not True
            or coefficient.is_finite is not True
            for coefficient in polynomial.coeffs()
        ):
            raise ValueError("polynomial coefficients must be finite and real.")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "expression", expression)

    @property
    def state_dimension(self):
        return len(self.states)

    def value(self, points):
        values = np.asarray(points, dtype=float)
        if values.ndim == 0 or values.shape[-1] != self.state_dimension:
            raise ValueError(
                "points must have final dimension {}.".format(
                    self.state_dimension
                )
            )
        function = sp.lambdify(self.states, self.expression, modules="numpy")
        flat = values.reshape(-1, self.state_dimension)
        result = np.asarray(function(*flat.T), dtype=float)
        if result.ndim == 0:
            result = np.full(len(flat), result.item())
        return result.reshape(values.shape[:-1])


__all__ = ["PolynomialLyapunov"]
