"""Whole-space verification of a fixed quadratic Lyapunov candidate."""

from dataclasses import replace
from time import perf_counter

import numpy as np
import sympy as sp

from lyznet.analysis import (
    QuadraticLyapunov,
    linearize,
    quadratic_lyapunov,
)
from lyznet.sets import WholeSpace
from lyznet.systems import ControlAffineSystem

from .core.global_quadratic_dreal import (
    certified_decrease_principal_minors,
    global_remainder_violation,
)
from .core.quadratic_crown import positive_definite_by_sylvester
from .global_lyapunov import (
    GlobalLyapunovVerificationResult,
    verify_global_lyapunov,
)
from .quadratic import (
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
    _build_problem,
    _freeze_decimal,
)
from .quadratic_dreal import _load_dreal, _sympy_to_dreal


def verify_global_quadratic(
    system,
    *,
    certificate=None,
    backend="z3",
    timeout=None,
    dreal_delta=1e-4,
    decrease_margin=1e-4,
    jobs=1,
    allow_trigonometric_bounds=True,
):
    """Verify a quadratic Lyapunov function over all of ``R^n``."""

    if isinstance(system, ControlAffineSystem):
        raise TypeError(
            "Global quadratic Lyapunov verification requires autonomy."
        )
    certificate = (
        quadratic_lyapunov(system)
        if certificate is None
        else certificate
    )
    if not isinstance(certificate, QuadraticLyapunov):
        raise TypeError("certificate must be a QuadraticLyapunov candidate.")
    if certificate.state_dimension != system.state_dimension:
        raise ValueError("Certificate dimension does not match the system.")
    if system.numerical_equilibrium is not None or any(
        sp.sympify(value).is_zero is not True for value in system.equilibrium
    ):
        raise ValueError("Global quadratic verification requires the exact origin.")
    if not np.array_equal(
        certificate.centre, np.asarray(system.equilibrium, dtype=float)
    ):
        raise ValueError("Certificate centre must match the equilibrium exactly.")
    if not np.allclose(certificate.linearization, linearize(system)):
        raise ValueError("Certificate linearization does not match the system.")
    if backend == "z3":
        problem = _build_problem(system, certificate)
        result = verify_global_lyapunov(
            system,
            problem.candidate,
            timeout=timeout,
            allow_trigonometric_bounds=allow_trigonometric_bounds,
        )
        return replace(result, certificate=certificate)
    if backend != "dreal":
        raise ValueError("Global quadratic verification requires Z3 or dReal.")
    if timeout is not None:
        raise ValueError("A hard dReal timeout is not implemented yet.")
    if (
        isinstance(dreal_delta, bool)
        or not np.isfinite(float(dreal_delta))
        or float(dreal_delta) <= 0
    ):
        raise ValueError("dreal_delta must be positive and finite.")
    if (
        isinstance(decrease_margin, bool)
        or not np.isfinite(float(decrease_margin))
        or float(decrease_margin) <= 0
    ):
        raise ValueError("decrease_margin must be positive and finite.")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer.")
    problem = _build_problem(system, certificate, include_crown=True)
    started = perf_counter()
    positivity = positive_definite_by_sylvester(problem.principal_minors)
    obligations = [
        VerificationObligation(
            "candidate_positive_definite",
            (
                VerificationStatus.VERIFIED
                if positivity
                else VerificationStatus.COUNTEREXAMPLE
            ),
            0.0,
            "exact-rational",
        )
    ]
    decay = float(
        np.min(
            np.linalg.eigvalsh(
                np.asarray(problem.linear_decrease_matrix, dtype=float)
            )
        )
    )
    if not np.isfinite(decay) or decay <= 0:
        raise ValueError(
            "The frozen linear decrease matrix must be positive definite."
        )
    q = _freeze_decimal(0.99 * decay)
    minors = certified_decrease_principal_minors(
        problem.linear_decrease_matrix, lower_bound=q
    )
    decrease_verified = positive_definite_by_sylvester(minors)
    obligations.append(
        VerificationObligation(
            "linear_decrease_lower_bound",
            (
                VerificationStatus.VERIFIED
                if decrease_verified
                else VerificationStatus.COUNTEREXAMPLE
            ),
            0.0,
            "exact-rational",
            "Certified q = {}.".format(q) if decrease_verified else "",
        )
    )
    if not positivity or not decrease_verified:
        status = VerificationStatus.COUNTEREXAMPLE
    else:
        mu = _freeze_decimal(decrease_margin)
        if mu >= q:
            raise ValueError("decrease_margin must be smaller than q.")
        dreal = _load_dreal()
        config = dreal.Config()
        config.precision = float(dreal_delta)
        config.number_of_jobs = int(jobs)
        states = tuple(
            dreal.Variable("global_x{}".format(index + 1))
            for index in range(system.state_dimension)
        )
        substitutions = dict(zip(problem.states, states))
        violation = global_remainder_violation(
            remainder_bound=_sympy_to_dreal(
                problem.remainder_bound, substitutions, dreal
            ),
            limit=_sympy_to_dreal((q - mu) ** 2, {}, dreal),
        )
        query_started = perf_counter()
        answer = dreal.CheckSatisfiability(violation, config)
        verified = answer is None
        obligations.append(
            VerificationObligation(
                "global_remainder_bound",
                (
                    VerificationStatus.VERIFIED
                    if verified
                    else VerificationStatus.UNKNOWN
                ),
                perf_counter() - query_started,
                "unsat" if verified else "delta-sat",
            )
        )
        status = (
            VerificationStatus.VERIFIED
            if verified
            else VerificationStatus.UNKNOWN
        )
    return GlobalLyapunovVerificationResult(
        system=system,
        certificate=certificate,
        claim=VerificationClaim.LYAPUNOV,
        status=status,
        verified_set=(
            WholeSpace(system.state_dimension)
            if status is VerificationStatus.VERIFIED
            else None
        ),
        obligations=tuple(obligations),
        backend="dreal",
        elapsed_time=perf_counter() - started,
        properness_verified=positivity,
    )


__all__ = ["verify_global_quadratic"]
