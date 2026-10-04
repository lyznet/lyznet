"""CROWN adapter for one proposed two-stage quadratic funnel."""

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
from .core.quadratic_crown import (
    annular_decrease_condition,
    exact_quadratic_face_separation,
    local_remainder_condition,
    positive_definite_by_sylvester,
)
from .quadratic import (
    QuadraticVerificationResult,
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
    _build_problem,
    _freeze_decimal,
    _validate_inputs,
)


_LINEAR_DECAY_FACTOR = 0.99


def _status_of(obligations):
    statuses = {item.status for item in obligations}
    if VerificationStatus.UNKNOWN in statuses:
        return VerificationStatus.UNKNOWN
    if VerificationStatus.COUNTEREXAMPLE in statuses:
        return VerificationStatus.COUNTEREXAMPLE
    return VerificationStatus.VERIFIED


def _exact_obligation(name, verified):
    return VerificationObligation(
        name=name,
        status=(
            VerificationStatus.VERIFIED
            if verified
            else VerificationStatus.COUNTEREXAMPLE
        ),
        elapsed_time=0.0,
        backend_status="exact-rational",
    )


def _sublevel_enclosure(problem, level, domain):
    """Return an outward box enclosing ``y.T P y <= level``."""

    matrix = sp.hessian(problem.candidate, problem.states) / 2
    inverse = matrix.inv()
    exact_level = _freeze_decimal(level)
    bounds = []
    for index, (lower, upper) in enumerate(domain):
        radius_squared = exact_level * inverse[index, index]
        radius = float(sp.N(sp.sqrt(radius_squared), 80))
        radius = float(np.nextafter(radius, np.inf))
        while _freeze_decimal(radius) ** 2 < radius_squared:
            radius = float(np.nextafter(radius, np.inf))
        bounds.append((max(lower, -radius), min(upper, radius)))
    return tuple(bounds)


def verify_quadratic_roa_crown(
    system,
    certificate,
    *,
    local_level,
    outer_level,
    timeout=None,
    device="auto",
    batch_size=64,
    proof_margin=1e-6,
    local_decrease_margin=1e-4,
    verbose=False,
):
    """Verify one local level and one outer level with CROWN.

    This backend deliberately does not bisect formal queries. Empirical level
    proposal belongs to the later screening stage; every level reported here
    is exactly the level supplied by the caller and directly reverified.
    """

    _validate_inputs(system, certificate, None, outer_level)
    for value, label in (
        (local_level, "local_level"),
        (outer_level, "outer_level"),
        (timeout, "timeout"),
        (proof_margin, "proof_margin"),
        (local_decrease_margin, "local_decrease_margin"),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("{} must be positive and finite.".format(label))
    local_level = float(local_level)
    outer_level = float(outer_level)
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
    problem = _build_problem(system, certificate, include_crown=True)
    if any(value != 0 for value in problem.dynamics_at_centre):
        raise ValueError(
            "CROWN quadratic verification requires the stored centre to be "
            "an exact equilibrium; use dReal for an approximate centre."
        )

    positivity = _exact_obligation(
        "candidate_positive_definite",
        positive_definite_by_sylvester(problem.principal_minors),
    )
    estimated_decay = float(np.min(np.linalg.eigvalsh(
        np.asarray(problem.linear_decrease_matrix, dtype=float)
    )))
    decay_lower_bound = _freeze_decimal(
        _LINEAR_DECAY_FACTOR * estimated_decay
    )
    certified_decrease = problem.linear_decrease_matrix - (
        decay_lower_bound * sp.eye(system.state_dimension)
    )
    decrease_minors = tuple(
        sp.det(certified_decrease[:size, :size])
        for size in range(1, system.state_dimension + 1)
    )
    linear_decrease = _exact_obligation(
        "linear_decrease_lower_bound",
        positive_definite_by_sylvester(decrease_minors),
    )
    exact_obligations = [positivity, linear_decrease]
    if _status_of(exact_obligations) is not VerificationStatus.VERIFIED:
        return QuadraticVerificationResult(
            system=system,
            certificate=certificate,
            claim=VerificationClaim.LYAPUNOV,
            status=VerificationStatus.COUNTEREXAMPLE,
            local_set=None,
            verified_set=None,
            obligations=tuple(exact_obligations),
            backend="crown",
            elapsed_time=perf_counter() - started,
        )

    if float(decay_lower_bound) <= local_decrease_margin:
        raise ValueError(
            "local_decrease_margin must be smaller than the certified "
            "linear decay bound."
        )
    remainder_limit = (
        float(decay_lower_bound) - float(local_decrease_margin)
    ) ** 2 - float(proof_margin)
    if remainder_limit <= 0:
        raise ValueError("The selected margins leave no remainder allowance.")

    centre = tuple(float(value) for value in problem.centre)
    relative_domain = tuple(
        (lower - value, upper - value)
        for value, (lower, upper) in zip(centre, system.domain.bounds)
    )
    matrix = sp.hessian(problem.candidate, problem.states) / 2
    inverse = matrix.inv()
    boundary_verified = True
    exact_outer = _freeze_decimal(outer_level)
    exact_margin = _freeze_decimal(proof_margin)
    for index, bounds in enumerate(relative_domain):
        for face in bounds:
            minimum = _freeze_decimal(face) ** 2 / inverse[index, index]
            boundary_verified &= exact_quadratic_face_separation(
                minimum,
                level=exact_outer,
                proof_margin=exact_margin,
            )
    boundary = _exact_obligation(
        "domain_boundary_exclusion", boundary_verified
    )
    exact_obligations.append(boundary)
    if not boundary_verified:
        return QuadraticVerificationResult(
            system=system,
            certificate=certificate,
            claim=VerificationClaim.LYAPUNOV,
            status=VerificationStatus.COUNTEREXAMPLE,
            local_set=None,
            verified_set=None,
            obligations=tuple(exact_obligations),
            backend="crown",
            elapsed_time=perf_counter() - started,
        )

    require_supported_expressions(
        problem.states,
        (
            problem.candidate,
            problem.remainder_bound,
            problem.lie_derivative,
        ),
    )
    torch, api = load_crown()
    crown_device = normalise_device(torch, device)
    runner = CrownRunner(
        torch=torch,
        api=api,
        device=crown_device,
        timeout=timeout,
        batch_size=batch_size,
        verbose=verbose,
    )
    local_graph = expression_graph(
        torch,
        problem.states,
        (problem.candidate, problem.remainder_bound),
    )

    def local_condition(output):
        return local_remainder_condition(
            output,
            level=local_level,
            remainder_limit=remainder_limit,
            proof_margin=proof_margin,
        )

    local = runner.solve(
        "local_remainder_bound",
        local_graph,
        2,
        local_condition,
        _sublevel_enclosure(problem, local_level, relative_domain),
    )
    obligations = exact_obligations[:2] + [local, boundary]
    if local.status is not VerificationStatus.VERIFIED:
        return QuadraticVerificationResult(
            system=system,
            certificate=certificate,
            claim=VerificationClaim.LYAPUNOV,
            status=_status_of(obligations),
            local_set=None,
            verified_set=None,
            obligations=tuple(obligations),
            backend="crown",
            elapsed_time=perf_counter() - started,
        )

    outer_graph = expression_graph(
        torch,
        problem.states,
        (problem.candidate, problem.lie_derivative),
    )

    def outer_condition(output):
        return annular_decrease_condition(
            output,
            local_level=local_level,
            outer_level=outer_level,
            proof_margin=proof_margin,
        )

    outer = runner.solve(
        "annular_decrease",
        outer_graph,
        2,
        outer_condition,
        relative_domain,
    )
    obligations.append(outer)
    status = _status_of(obligations)
    local_set = SublevelSet(certificate, local_level)
    return QuadraticVerificationResult(
        system=system,
        certificate=certificate,
        claim=VerificationClaim.LYAPUNOV,
        status=status,
        local_set=local_set,
        verified_set=(
            SublevelSet(certificate, outer_level)
            if status is VerificationStatus.VERIFIED
            else None
        ),
        obligations=tuple(obligations),
        backend="crown",
        elapsed_time=perf_counter() - started,
    )


__all__ = ["verify_quadratic_roa_crown"]
