"""CROWN adapter for one proposed two-stage quadratic CLF funnel."""

from numbers import Integral
from time import perf_counter

import numpy as np
import sympy as sp

from lyznet.sets import SublevelSet

from ._crown import (
    CrownRunner,
    expression_graph,
    load_crown,
    normalise_device,
    require_supported_expressions,
)
from ._quadratic_clf import build_quadratic_clf_problem
from .core.quadratic_clf_crown import (
    annular_artstein_condition,
    local_contraction_condition,
)
from .core.quadratic_crown import (
    exact_quadratic_face_separation,
    positive_definite_by_sylvester,
)
from .quadratic import (
    QuadraticVerificationResult,
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
    _freeze_decimal,
)
from .quadratic_crown import _status_of, _sublevel_enclosure


def verify_quadratic_clf_crown(
    system,
    certificate,
    *,
    local_level,
    outer_level,
    timeout,
    device,
    batch_size,
    proof_margin,
    verbose,
):
    """Verify one LQR core and one outer Artstein annulus."""

    for value, label in (
        (local_level, "local_level"),
        (outer_level, "outer_level"),
        (proof_margin, "proof_margin"),
    ):
        if (
            isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("{} must be positive and finite.".format(label))
    if local_level > outer_level:
        raise ValueError("local_level must not exceed outer_level.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean.")

    started = perf_counter()
    problem = build_quadratic_clf_problem(system, certificate)
    positivity_verified = positive_definite_by_sylvester(
        problem.candidate_principal_minors
    )
    positivity = VerificationObligation(
        "candidate_positive_definite",
        (
            VerificationStatus.VERIFIED
            if positivity_verified
            else VerificationStatus.COUNTEREXAMPLE
        ),
        0.0,
        "exact-rational",
    )
    if not positivity_verified:
        return QuadraticVerificationResult(
            system,
            certificate,
            VerificationClaim.CONTROL_LYAPUNOV,
            VerificationStatus.COUNTEREXAMPLE,
            None,
            None,
            (positivity,),
            "crown",
            perf_counter() - started,
        )

    centre = np.asarray(certificate.centre, dtype=float)
    relative_domain = tuple(
        (lower - value, upper - value)
        for value, (lower, upper) in zip(
            centre, system.domain.bounds
        )
    )
    matrix = sp.hessian(problem.candidate, problem.states) / 2
    inverse = matrix.inv()
    exact_outer = _freeze_decimal(outer_level)
    exact_margin = _freeze_decimal(proof_margin)
    boundary_verified = all(
        exact_quadratic_face_separation(
            _freeze_decimal(face) ** 2 / inverse[index, index],
            level=exact_outer,
            proof_margin=exact_margin,
        )
        for index, bounds in enumerate(relative_domain)
        for face in bounds
    )
    boundary = VerificationObligation(
        "domain_boundary_exclusion",
        (
            VerificationStatus.VERIFIED
            if boundary_verified
            else VerificationStatus.COUNTEREXAMPLE
        ),
        0.0,
        "exact-rational",
    )
    if not boundary_verified:
        return QuadraticVerificationResult(
            system,
            certificate,
            VerificationClaim.CONTROL_LYAPUNOV,
            VerificationStatus.COUNTEREXAMPLE,
            None,
            None,
            (positivity, boundary),
            "crown",
            perf_counter() - started,
        )

    require_supported_expressions(
        problem.states,
        (
            problem.candidate,
            problem.drift_derivative,
            *problem.input_derivatives,
            *problem.contraction_principal_minors,
        ),
    )
    torch, api = load_crown()
    runner = CrownRunner(
        torch=torch,
        api=api,
        device=normalise_device(torch, device),
        timeout=timeout,
        batch_size=batch_size,
        verbose=verbose,
    )
    local_graph = expression_graph(
        torch,
        problem.states,
        (problem.candidate,) + problem.contraction_principal_minors,
    )

    def local_condition(output):
        return local_contraction_condition(
            output,
            level=local_level,
            proof_margin=proof_margin,
            state_dimension=system.state_dimension,
        )

    local = runner.solve(
        "local_lqr_contraction",
        local_graph,
        1 + system.state_dimension,
        local_condition,
        _sublevel_enclosure(problem, local_level, relative_domain),
    )
    obligations = [positivity, local, boundary]
    if local.status is not VerificationStatus.VERIFIED:
        return QuadraticVerificationResult(
            system,
            certificate,
            VerificationClaim.CONTROL_LYAPUNOV,
            _status_of(obligations),
            None,
            None,
            tuple(obligations),
            "crown",
            perf_counter() - started,
        )
    invariance = VerificationObligation(
        "local_lqr_positive_invariance",
        VerificationStatus.VERIFIED,
        0.0,
        "implied-by-contraction",
    )
    obligations.insert(2, invariance)
    outer_graph = expression_graph(
        torch,
        problem.states,
        (
            problem.candidate,
            problem.drift_derivative,
            *problem.input_derivatives,
        ),
    )

    def outer_condition(output):
        return annular_artstein_condition(
            output,
            local_level=local_level,
            outer_level=outer_level,
            proof_margin=proof_margin,
            input_dimension=system.input_dimension,
        )

    outer = runner.solve(
        "annular_clf_decrease",
        outer_graph,
        2 + system.input_dimension,
        outer_condition,
        relative_domain,
    )
    obligations.append(outer)
    status = _status_of(obligations)
    return QuadraticVerificationResult(
        system,
        certificate,
        VerificationClaim.CONTROL_LYAPUNOV,
        status,
        SublevelSet(certificate, local_level),
        (
            SublevelSet(certificate, outer_level)
            if status is VerificationStatus.VERIFIED
            else None
        ),
        tuple(obligations),
        "crown",
        perf_counter() - started,
    )


__all__ = ["verify_quadratic_clf_crown"]
