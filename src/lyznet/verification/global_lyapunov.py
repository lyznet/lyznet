"""Whole-space symbolic Lyapunov verification with Z3."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import sympy as sp

from lyznet._symbolic_z3 import sympy_to_z3
from lyznet.certificates import PolynomialLyapunov
from lyznet.sets import WholeSpace
from lyznet.systems import ControlAffineSystem

from .core.global_lyapunov_z3 import (
    candidate_positivity_violation,
    leading_form_positivity_violation,
    lyapunov_lie_derivative,
    polynomial_at_infinity,
    radial_unboundedness_violation,
    strict_decrease_violation,
)
from .quadratic import (
    OptionalVerificationDependencyError,
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
    _load_z3,
)


@dataclass(frozen=True)
class GlobalLyapunovVerificationResult:
    system: object
    certificate: object
    claim: VerificationClaim
    status: VerificationStatus
    verified_set: WholeSpace | None
    obligations: tuple[VerificationObligation, ...]
    backend: str
    elapsed_time: float
    properness_verified: bool
    uses_auxiliary_bounds: bool = False

    @property
    def is_verified(self):
        return self.status is VerificationStatus.VERIFIED


def _expression(candidate):
    return (
        candidate.expression
        if isinstance(candidate, PolynomialLyapunov)
        else sp.sympify(candidate)
    )


def verify_global_lyapunov(
    system,
    candidate,
    *,
    timeout=None,
    allow_trigonometric_bounds=False,
):
    """Prove positivity, strict decrease, and properness over ``R^n``."""

    if isinstance(system, ControlAffineSystem):
        raise TypeError("Global Lyapunov verification requires autonomy.")
    if system.numerical_equilibrium is not None or any(
        sp.sympify(value).is_zero is not True for value in system.equilibrium
    ):
        raise ValueError("Global verification requires the exact origin.")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not np.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ValueError("timeout must be positive and finite.")
    if not isinstance(allow_trigonometric_bounds, bool):
        raise TypeError("allow_trigonometric_bounds must be a boolean.")

    try:
        z3 = _load_z3()
    except OptionalVerificationDependencyError:
        raise
    started = perf_counter()
    states = tuple(system.states)
    dynamics = tuple(system.symbolic.dynamics)
    if (
        isinstance(candidate, PolynomialLyapunov)
        and candidate.states != states
    ):
        raise ValueError(
            "Polynomial candidate states must match the system states."
        )
    candidate_expression = sp.expand(_expression(candidate))
    if candidate_expression.free_symbols.difference(states):
        raise ValueError("candidate contains undeclared symbols.")
    origin = {state: 0 for state in states}
    dynamics_at_origin = tuple(
        sp.simplify(value.subs(origin)) for value in dynamics
    )
    candidate_at_origin = sp.simplify(candidate_expression.subs(origin))
    lie_derivative = lyapunov_lie_derivative(
        states, dynamics, candidate_expression
    )
    z3_states = tuple(
        z3.Real("global_x{}".format(index + 1))
        for index in range(len(states))
    )
    substitutions = dict(zip(states, z3_states))

    def encode(expression):
        functions, constraints = {}, []
        encoded = sympy_to_z3(
            expression,
            substitutions,
            z3,
            allow_trigonometric_bounds=allow_trigonometric_bounds,
            function_substitutions=functions,
            extra_constraints=constraints,
        )
        return encoded, tuple(constraints), bool(functions)

    encoded_candidate, positivity_constraints, positivity_conservative = (
        encode(candidate_expression)
    )
    encoded_derivative, decrease_constraints, decrease_conservative = (
        encode(lie_derivative)
    )
    nonzero = z3.Or(*(state != 0 for state in z3_states))

    def check(
        name,
        violation,
        constraints=(),
        conservative=False,
        quantified=False,
    ):
        query_started = perf_counter()
        solver = z3.SolverFor("NRA") if quantified else z3.Solver()
        if timeout is not None:
            solver.set("timeout", max(1, int(float(timeout) * 1000)))
        solver.add(*constraints, violation)
        answer = solver.check()
        if answer == z3.unsat:
            status = VerificationStatus.VERIFIED
        elif answer == z3.sat and not conservative:
            status = VerificationStatus.COUNTEREXAMPLE
        else:
            status = VerificationStatus.UNKNOWN
        return VerificationObligation(
            name=name,
            status=status,
            elapsed_time=perf_counter() - query_started,
            backend_status=str(answer),
            message=(
                "The trigonometric enclosure admits this model."
                if answer == z3.sat and conservative
                else (solver.reason_unknown() if answer == z3.unknown else "")
            ),
        )

    obligations = [
        VerificationObligation(
            "global_origin_is_equilibrium",
            VerificationStatus.VERIFIED
            if all(value == 0 for value in dynamics_at_origin)
            else VerificationStatus.COUNTEREXAMPLE,
            0.0,
            "symbolic",
        ),
        VerificationObligation(
            "global_candidate_zero_at_equilibrium",
            VerificationStatus.VERIFIED
            if candidate_at_origin == 0
            else VerificationStatus.COUNTEREXAMPLE,
            0.0,
            "symbolic",
        ),
    ]
    if all(item.status is VerificationStatus.VERIFIED for item in obligations):
        obligations.append(
            check(
                "global_candidate_positivity",
                candidate_positivity_violation(
                    z3, nonzero=nonzero, candidate=encoded_candidate
                ),
                positivity_constraints,
                positivity_conservative,
            )
        )
    if all(item.status is VerificationStatus.VERIFIED for item in obligations):
        obligations.append(
            check(
                "global_strict_decrease",
                strict_decrease_violation(
                    z3,
                    nonzero=nonzero,
                    lie_derivative=encoded_derivative,
                ),
                decrease_constraints,
                decrease_conservative,
            )
        )

    if all(item.status is VerificationStatus.VERIFIED for item in obligations):
        scale_symbol = sp.Dummy("radial_scale")
        try:
            degree, leading, homogenized = polynomial_at_infinity(
                states, candidate_expression, scale_symbol
            )
        except (sp.PolynomialError, ValueError):
            radial = VerificationObligation(
                "global_radial_unboundedness",
                VerificationStatus.UNSUPPORTED,
                0.0,
                "unsupported:nonpolynomial",
            )
        else:
            if sp.expand(candidate_expression - leading) == 0:
                radial = VerificationObligation(
                    "global_radial_unboundedness",
                    VerificationStatus.VERIFIED,
                    0.0,
                    "implied-by-homogeneous-positivity",
                )
            else:
                norm_squared = z3.Sum([state**2 for state in z3_states])
                leading_query = leading_form_positivity_violation(
                    z3,
                    direction_norm_squared=norm_squared,
                    leading_form=sympy_to_z3(leading, substitutions, z3),
                )
                radial = check("global_radial_unboundedness", leading_query)
                if radial.status is not VerificationStatus.VERIFIED:
                    directions = tuple(
                        z3.Real("radial_u{}".format(index + 1))
                        for index in range(len(states))
                    )
                    scale = z3.Real("radial_t")
                    level = z3.Real("radial_level")
                    cutoff = z3.Real("radial_cutoff")
                    radial_substitutions = dict(zip(states, directions))
                    radial_substitutions[scale_symbol] = scale
                    exact_query = radial_unboundedness_violation(
                        z3,
                        level=level,
                        cutoff=cutoff,
                        scale=scale,
                        directions=directions,
                        direction_norm_squared=z3.Sum(
                            [direction**2 for direction in directions]
                        ),
                        homogenized_candidate=sympy_to_z3(
                            homogenized, radial_substitutions, z3
                        ),
                        degree=degree,
                    )
                    radial = check(
                        "global_radial_unboundedness",
                        exact_query,
                        quantified=True,
                    )
        obligations.append(radial)

    status = next(
        (
            item.status
            for item in obligations
            if item.status is not VerificationStatus.VERIFIED
        ),
        VerificationStatus.VERIFIED,
    )
    properness = any(
        item.name == "global_radial_unboundedness"
        and item.status is VerificationStatus.VERIFIED
        for item in obligations
    )
    return GlobalLyapunovVerificationResult(
        system=system,
        certificate=candidate,
        claim=VerificationClaim.LYAPUNOV,
        status=status,
        verified_set=(
            WholeSpace(system.state_dimension)
            if status is VerificationStatus.VERIFIED
            else None
        ),
        obligations=tuple(obligations),
        backend="z3",
        elapsed_time=perf_counter() - started,
        properness_verified=properness,
        uses_auxiliary_bounds=positivity_conservative or decrease_conservative,
    )


__all__ = ["GlobalLyapunovVerificationResult", "verify_global_lyapunov"]
