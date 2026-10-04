"""Validation for the state-set convention ``h_i(x) <= 1``."""

import numpy as np
import sympy as sp


def normalise_state_constraints(system, state_constraints):
    constraints = tuple(sp.sympify(value) for value in state_constraints)
    undeclared = set().union(
        *(value.free_symbols for value in constraints)
    ).difference(system.states)
    if undeclared:
        raise ValueError("state constraints contain undeclared symbols.")
    substitutions = dict(zip(system.states, system.equilibrium))
    for expression in constraints:
        at_equilibrium = float(sp.N(expression.subs(substitutions), 17))
        if not np.isfinite(at_equilibrium) or at_equilibrium >= 1:
            raise ValueError(
                "The equilibrium must lie strictly inside every h_i<=1 "
                "constraint."
            )
    return constraints


def normalise_viable_faces(system, viable_faces):
    result = []
    for face in viable_faces:
        if len(face) != 2:
            raise ValueError("A viable face must be (coordinate, side).")
        coordinate, side = face
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, int)
            or not 0 <= coordinate < system.state_dimension
            or side not in ("lower", "upper")
        ):
            raise ValueError(
                "A viable face requires a valid coordinate and "
                "'lower' or 'upper'."
            )
        item = (coordinate, side)
        if item in result:
            raise ValueError("viable_faces must not contain duplicates.")
        result.append(item)
    return tuple(result)


__all__ = ["normalise_state_constraints", "normalise_viable_faces"]
