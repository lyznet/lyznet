"""dReal adapter for two-stage quadratic CLF verification."""

from numbers import Integral
from time import perf_counter

import numpy as np

from lyznet.sets import SublevelSet

from ._level_search import search_verified_level
from ._quadratic_clf import build_quadratic_clf_problem
from .core.quadratic_clf_dreal import (
    domain_boundary_violation,
    local_contraction_violation,
    outer_artstein_violation,
)
from .core.quadratic_dreal import positive_definite_by_sylvester
from .quadratic import (
    QuadraticVerificationResult,
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
)
from .quadratic_dreal import (
    _dreal_number,
    _load_dreal,
    _sympy_to_dreal,
)


def verify_quadratic_clf_dreal(
    system,
    certificate,
    *,
    delta,
    local_search_precision,
    outer_search_precision,
    local_max_level,
    maximum_level,
    jobs,
):
    """Verify an LQR-contracting core and an outer Artstein annulus."""

    for value, label in (
        (delta, "delta"),
        (local_search_precision, "local_search_precision"),
        (outer_search_precision, "outer_search_precision"),
        (local_max_level, "local_max_level"),
        (maximum_level, "maximum_level"),
    ):
        if (
            isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("{} must be positive and finite.".format(label))
    if local_max_level > maximum_level:
        raise ValueError("local_max_level must not exceed maximum_level.")
    if isinstance(jobs, bool) or not isinstance(jobs, Integral) or jobs < 1:
        raise ValueError("jobs must be a positive integer.")

    dreal = _load_dreal()
    started = perf_counter()
    problem = build_quadratic_clf_problem(system, certificate)
    config = dreal.Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = float(delta)
    config.number_of_jobs = int(jobs)
    states = tuple(
        dreal.Variable("clf_y{}".format(index + 1))
        for index in range(system.state_dimension)
    )
    substitutions = dict(zip(problem.states, states))
    candidate = _sympy_to_dreal(
        problem.candidate, substitutions, dreal
    )
    drift_derivative = _sympy_to_dreal(
        problem.drift_derivative, substitutions, dreal
    )
    input_derivatives = tuple(
        _sympy_to_dreal(value, substitutions, dreal)
        for value in problem.input_derivatives
    )
    contraction_minors = tuple(
        _sympy_to_dreal(value, substitutions, dreal)
        for value in problem.contraction_principal_minors
    )
    centre = tuple(
        _sympy_to_dreal(value, {}, dreal) for value in problem.centre
    )
    physical_states = tuple(
        state + value for state, value in zip(states, centre)
    )
    in_domain = dreal.And(*(
        dreal.And(
            state >= _dreal_number(lower, dreal),
            state <= _dreal_number(upper, dreal),
        )
        for state, (lower, upper) in zip(
            physical_states, system.domain.bounds
        )
    ))
    on_boundary = dreal.Or(*(
        dreal.Or(
            state == _dreal_number(lower, dreal),
            state == _dreal_number(upper, dreal),
        )
        for state, (lower, upper) in zip(
            physical_states, system.domain.bounds
        )
    ))
    no_control_effect = dreal.And(*(
        derivative == 0 for derivative in input_derivatives
    ))

    def check(name, violation):
        query_started = perf_counter()
        answer = dreal.CheckSatisfiability(violation, config)
        verified = answer is None
        return VerificationObligation(
            name,
            (
                VerificationStatus.VERIFIED
                if verified
                else VerificationStatus.UNKNOWN
            ),
            perf_counter() - query_started,
            "unsat" if verified else "delta-sat",
        )

    positive = positive_definite_by_sylvester(
        problem.candidate_principal_minors
    )
    positivity = VerificationObligation(
        "candidate_positive_definite",
        (
            VerificationStatus.VERIFIED
            if positive
            else VerificationStatus.COUNTEREXAMPLE
        ),
        0.0,
        "exact-rational",
    )
    if not positive:
        return QuadraticVerificationResult(
            system,
            certificate,
            VerificationClaim.CONTROL_LYAPUNOV,
            VerificationStatus.COUNTEREXAMPLE,
            None,
            None,
            (positivity,),
            "dreal",
            perf_counter() - started,
        )

    def boundary(level, name):
        return check(
            name,
            domain_boundary_violation(
                dreal,
                in_domain=in_domain,
                on_domain_boundary=on_boundary,
                candidate=candidate,
                level=_dreal_number(level, dreal),
            ),
        )

    def check_local(level):
        contraction = check(
            "local_lqr_contraction",
            local_contraction_violation(
                dreal,
                in_domain=in_domain,
                candidate=candidate,
                level=_dreal_number(level, dreal),
                contraction_principal_minors=contraction_minors,
            ),
        )
        if contraction.status is not VerificationStatus.VERIFIED:
            return (contraction,)
        invariance = VerificationObligation(
            "local_lqr_positive_invariance",
            VerificationStatus.VERIFIED,
            0.0,
            "implied-by-contraction",
        )
        return (
            contraction,
            invariance,
            boundary(level, "local_boundary_exclusion"),
        )

    local_level, local_proof = search_verified_level(
        local_max_level,
        float(local_search_precision),
        check_local,
    )
    if local_level is None:
        return QuadraticVerificationResult(
            system,
            certificate,
            VerificationClaim.CONTROL_LYAPUNOV,
            VerificationStatus.UNKNOWN,
            None,
            None,
            (positivity,) + tuple(local_proof),
            "dreal",
            perf_counter() - started,
        )

    def check_outer(level):
        decrease = check(
            "annular_clf_decrease",
            outer_artstein_violation(
                dreal,
                in_domain=in_domain,
                candidate=candidate,
                local_level=_dreal_number(local_level, dreal),
                outer_level=_dreal_number(level, dreal),
                no_control_effect=no_control_effect,
                drift_derivative=drift_derivative,
            ),
        )
        if decrease.status is not VerificationStatus.VERIFIED:
            return (decrease,)
        return decrease, boundary(level, "outer_boundary_exclusion")

    outer_level, outer_proof = search_verified_level(
        maximum_level,
        float(outer_search_precision),
        check_outer,
        lower=local_level,
    )
    obligations = (
        (positivity,) + tuple(local_proof) + tuple(outer_proof)
    )
    if outer_level is None:
        return QuadraticVerificationResult(
            system,
            certificate,
            VerificationClaim.CONTROL_LYAPUNOV,
            VerificationStatus.UNKNOWN,
            SublevelSet(certificate, local_level),
            None,
            obligations,
            "dreal",
            perf_counter() - started,
        )
    return QuadraticVerificationResult(
        system,
        certificate,
        VerificationClaim.CONTROL_LYAPUNOV,
        VerificationStatus.VERIFIED,
        SublevelSet(certificate, local_level),
        SublevelSet(certificate, outer_level),
        obligations,
        "dreal",
        perf_counter() - started,
    )


__all__ = ["verify_quadratic_clf_dreal"]
