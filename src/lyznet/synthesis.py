"""Exact Z3 synthesis for explicit polynomial Lyapunov templates."""

from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import sympy as sp

from ._symbolic_z3 import sympy_to_z3
from .certificates import PolynomialLyapunov
from .systems import ControlAffineSystem
from .verification.quadratic import _load_z3


class SynthesisStatus(Enum):
    FOUND = "found"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PolynomialLyapunovSynthesisResult:
    status: SynthesisStatus
    candidate: PolynomialLyapunov | None
    template: sp.Expr
    coefficients: tuple
    coefficient_values: tuple
    reduction_constraints: tuple
    elapsed_time: float
    backend_status: str
    message: str = ""

    @property
    def found(self):
        return self.status is SynthesisStatus.FOUND


def _terms(system, terms):
    expressions = tuple(sp.expand(sp.sympify(term)) for term in terms)
    if not expressions:
        raise ValueError("terms must contain at least one polynomial term.")
    seen = set()
    for expression in expressions:
        if expression == 0 or not expression.is_polynomial(*system.states):
            raise ValueError("Every synthesis term must be a nonzero polynomial.")
        if expression.free_symbols.difference(system.states):
            raise ValueError("Synthesis terms contain undeclared symbols.")
        identity = sp.srepr(expression)
        if identity in seen:
            raise ValueError("Synthesis terms must not contain duplicates.")
        seen.add(identity)
    return expressions


def _dominant_odd_degree_reduction(derivative, states, coefficients):
    """Derive necessary coefficient equalities from odd extreme degrees."""

    equations = []
    substitutions = {}
    for _ in range(len(coefficients) + 1):
        polynomial = sp.Poly(
            sp.expand(derivative.subs(substitutions)), *states
        )
        terms = tuple(
            (powers, sp.expand(coefficient))
            for powers, coefficient in polynomial.terms()
            if sp.expand(coefficient) != 0
        )
        if not terms:
            break
        degrees = tuple(sum(powers) for powers, _ in terms)
        odd_extremes = {
            degree
            for degree in (min(degrees), max(degrees))
            if degree % 2
        }
        if not odd_extremes:
            break
        existing = {sp.srepr(value) for value in equations}
        equations.extend(
            coefficient
            for (_, coefficient), degree in zip(terms, degrees)
            if degree in odd_extremes and sp.srepr(coefficient) not in existing
        )
        solution = next(iter(sp.linsolve(equations, coefficients)))
        updated = {
            coefficient: sp.expand(value)
            for coefficient, value in zip(coefficients, solution)
            if value != coefficient
        }
        if updated == substitutions:
            break
        substitutions = updated
    return tuple(
        sp.Eq(coefficient, substitutions[coefficient])
        for coefficient in coefficients
        if coefficient in substitutions
    )


def _z3_rational(value, z3):
    if z3.is_int_value(value):
        return sp.Integer(value.as_long())
    if z3.is_rational_value(value):
        return sp.Rational(
            value.numerator_as_long(), value.denominator_as_long()
        )
    raise TypeError("Z3 returned a non-rational coefficient.")


def synthesize_global_polynomial_lyapunov(
    system, terms, *, symbolic_reduction=True, timeout=60
):
    """Search ``V=sum c_i t_i`` under global positivity and decrease."""

    if isinstance(system, ControlAffineSystem):
        raise TypeError("Polynomial synthesis requires an autonomous system.")
    if system.numerical_equilibrium is not None or any(
        sp.sympify(value).is_zero is not True for value in system.equilibrium
    ):
        raise ValueError("Global synthesis requires the exact origin.")
    if not all(
        expression.is_polynomial(*system.states)
        for expression in system.symbolic.dynamics
    ):
        raise ValueError("Global synthesis requires polynomial dynamics.")
    terms = _terms(system, terms)
    if not isinstance(symbolic_reduction, bool):
        raise TypeError("symbolic_reduction must be a boolean.")
    if timeout is not None and float(timeout) <= 0:
        raise ValueError("timeout must be positive.")
    z3 = _load_z3()
    coefficients = tuple(
        sp.Dummy("lyznet_c{}".format(index))
        for index in range(len(terms))
    )
    template = sp.expand(sum(
        coefficient * term
        for coefficient, term in zip(coefficients, terms)
    ))
    derivative = sp.expand(sum(
        sp.diff(template, state) * component
        for state, component in zip(system.states, system.symbolic.dynamics)
    ))
    reduction = (
        _dominant_odd_degree_reduction(
            derivative, system.states, coefficients
        )
        if symbolic_reduction
        else ()
    )
    z3_states = tuple(
        z3.Real("synthesis_x{}".format(index + 1))
        for index in range(system.state_dimension)
    )
    z3_coefficients = tuple(
        z3.Real("synthesis_c{}".format(index))
        for index in range(len(coefficients))
    )
    substitutions = dict(zip(
        tuple(system.states) + coefficients,
        z3_states + z3_coefficients,
    ))
    candidate_z3 = sympy_to_z3(template, substitutions, z3)
    derivative_z3 = sympy_to_z3(derivative, substitutions, z3)
    nonzero = z3.Or(*(state != 0 for state in z3_states))
    basis_values = tuple(
        template.subs({
            state: int(index == active)
            for index, state in enumerate(system.states)
        })
        for active in range(system.state_dimension)
    )

    # Search condition:
    # exists c, forall x!=0: V_c(x)>0 and L_f V_c(x)<0.
    solver = z3.Solver()
    if timeout is not None:
        solver.set("timeout", max(1, int(float(timeout) * 1000)))
    solver.add(z3.ForAll(
        z3_states,
        z3.Implies(
            nonzero,
            z3.And(candidate_z3 > 0, derivative_z3 < 0),
        ),
    ))
    solver.add(*(
        sympy_to_z3(value, substitutions, z3) > 0
        for value in basis_values
    ))
    solver.add(
        sympy_to_z3(sum(basis_values), substitutions, z3) == 1
    )
    solver.add(*(
        sympy_to_z3(
            constraint.lhs - constraint.rhs, substitutions, z3
        ) == 0
        for constraint in reduction
    ))
    started = perf_counter()
    answer = solver.check()
    elapsed = perf_counter() - started
    if answer != z3.sat:
        status = (
            SynthesisStatus.INFEASIBLE
            if answer == z3.unsat
            else SynthesisStatus.UNKNOWN
        )
        return PolynomialLyapunovSynthesisResult(
            status,
            None,
            template,
            coefficients,
            (),
            reduction,
            elapsed,
            str(answer),
            solver.reason_unknown() if answer == z3.unknown else "",
        )
    model = solver.model()
    try:
        values = tuple(
            _z3_rational(model.eval(value, model_completion=True), z3)
            for value in z3_coefficients
        )
    except TypeError as error:
        return PolynomialLyapunovSynthesisResult(
            SynthesisStatus.UNKNOWN,
            None,
            template,
            coefficients,
            (),
            reduction,
            elapsed,
            str(answer),
            str(error),
        )
    expression = sp.expand(template.subs(dict(zip(coefficients, values))))
    return PolynomialLyapunovSynthesisResult(
        SynthesisStatus.FOUND,
        PolynomialLyapunov(tuple(system.states), expression),
        template,
        coefficients,
        values,
        reduction,
        elapsed,
        str(answer),
    )


__all__ = [
    "PolynomialLyapunovSynthesisResult",
    "SynthesisStatus",
    "synthesize_global_polynomial_lyapunov",
]
