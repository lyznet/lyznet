"""Fixed symbolic expressions shared by quadratic CLF backends."""

from dataclasses import dataclass

import sympy as sp

from .quadratic import _freeze_matrix


@dataclass(frozen=True)
class QuadraticCLFProblem:
    states: tuple
    centre: tuple
    candidate: sp.Expr
    drift_derivative: sp.Expr
    input_derivatives: tuple
    closed_loop_dynamics: tuple
    closed_loop_derivative: sp.Expr
    contraction_principal_minors: tuple
    candidate_principal_minors: tuple


def build_quadratic_clf_problem(system, certificate):
    """Write the Artstein and local LQR expressions in relative states."""

    states = sp.ImmutableMatrix(system.states)
    matrix = _freeze_matrix(certificate.matrix)
    centre = _freeze_matrix(certificate.centre)
    equilibrium_input = sp.ImmutableMatrix(system.equilibrium_input)
    physical_states = states + centre
    substitutions = dict(zip(system.states, physical_states))
    operating_drift = (
        system.symbolic.drift
        + system.symbolic.input_matrix * equilibrium_input
    )
    drift = operating_drift.subs(substitutions, simultaneous=True)
    input_matrix = system.symbolic.input_matrix.subs(
        substitutions, simultaneous=True
    )
    origin = {state: 0 for state in states}
    if any(sp.simplify(value.subs(origin)) != 0 for value in drift):
        raise ValueError(
            "The frozen centre and equilibrium input must define an exact "
            "controlled equilibrium."
        )

    # Relative coordinates y=x-x*:
    # V(y)=y.T P y,
    # L_f V=grad(V) [f(x*+y)+g(x*+y)u*],
    # L_g V=grad(V) g(x*+y).
    candidate = sp.expand((states.T * matrix * states)[0])
    gradient = sp.ImmutableMatrix([candidate]).jacobian(states)
    drift_derivative = sp.expand((gradient * drift)[0])
    input_derivatives = tuple(
        sp.expand(value) for value in gradient * input_matrix
    )

    # Local LQR controller v=Ky for the input deviation v=u-u*.
    feedback = _freeze_matrix(certificate.gain) * states
    closed_loop = (drift + input_matrix * feedback).applyfunc(sp.expand)
    closed_loop_derivative = sp.expand((gradient * closed_loop)[0])
    jacobian = closed_loop.jacobian(states)
    negative_contraction = -(jacobian.T * matrix + matrix * jacobian)
    contraction_minors = tuple(
        sp.expand(sp.det(negative_contraction[:size, :size]))
        for size in range(1, system.state_dimension + 1)
    )
    candidate_minors = tuple(
        sp.det(matrix[:size, :size])
        for size in range(1, system.state_dimension + 1)
    )
    return QuadraticCLFProblem(
        states=tuple(states),
        centre=tuple(centre),
        candidate=candidate,
        drift_derivative=drift_derivative,
        input_derivatives=input_derivatives,
        closed_loop_dynamics=tuple(closed_loop),
        closed_loop_derivative=closed_loop_derivative,
        contraction_principal_minors=contraction_minors,
        candidate_principal_minors=candidate_minors,
    )


__all__ = ["QuadraticCLFProblem", "build_quadratic_clf_problem"]
