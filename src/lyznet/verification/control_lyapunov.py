"""Quadratic control-Lyapunov verification and orchestration."""

from time import perf_counter

import numpy as np

from lyznet._symbolic_z3 import sympy_to_z3
from lyznet.analysis import (
    QuadraticControlLyapunov,
    linearize,
    quadratic_control_lyapunov,
)
from lyznet.sets import SublevelSet, WholeSpace
from lyznet.systems import ControlAffineSystem

from ._level_search import all_verified, search_verified_level
from ._quadratic_clf import build_quadratic_clf_problem
from .core.quadratic_clf_z3 import (
    candidate_positivity_violation,
    domain_boundary_violation,
    global_artstein_violation,
    sublevel_artstein_violation,
)
from .global_lyapunov import GlobalLyapunovVerificationResult
from .quadratic import (
    QuadraticVerificationResult,
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
    _automatic_level_cap,
    _load_z3,
)


def _validate_candidate(system, certificate, *, require_domain):
    if not isinstance(system, ControlAffineSystem):
        raise TypeError("Quadratic CLF verification requires a control-affine system.")
    if require_domain and system.domain is None:
        raise ValueError("Bounded quadratic CLF verification requires a domain.")
    if system.input_set is not None:
        raise ValueError(
            "Input-constrained CLF verification is added with the "
            "constraint stage."
        )
    if system.numerical_equilibrium is not None:
        raise ValueError("Quadratic CLF verification requires an exact equilibrium.")
    candidate = (
        quadratic_control_lyapunov(system)
        if certificate is None
        else certificate
    )
    if not isinstance(candidate, QuadraticControlLyapunov):
        raise TypeError("certificate must be a QuadraticControlLyapunov.")
    if candidate.state_dimension != system.state_dimension:
        raise ValueError("Certificate dimension does not match the system.")
    if not np.array_equal(
        candidate.centre, np.asarray(system.equilibrium, dtype=float)
    ):
        raise ValueError("Certificate centre must match the equilibrium exactly.")
    if not np.allclose(candidate.linearization, linearize(system)):
        raise ValueError("Certificate linearization does not match the system.")
    return candidate


def _timeout(value):
    if value is None:
        return None
    if isinstance(value, bool) or not np.isfinite(float(value)) or value <= 0:
        raise ValueError("timeout must be positive and finite.")
    return float(value)


def _positive_finite(value, label):
    if (
        isinstance(value, bool)
        or not np.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("{} must be positive and finite.".format(label))
    return float(value)


def _encode_z3(problem, system, z3, allow_trigonometric_bounds):
    states = tuple(
        z3.Real("clf_y{}".format(index + 1))
        for index in range(system.state_dimension)
    )
    substitutions = dict(zip(problem.states, states))

    def encode(expressions):
        functions = {}
        constraints = []
        values = tuple(
            sympy_to_z3(
                expression,
                substitutions,
                z3,
                allow_trigonometric_bounds=allow_trigonometric_bounds,
                function_substitutions=functions,
                extra_constraints=constraints,
            )
            for expression in expressions
        )
        return values, tuple(constraints), bool(functions)

    (candidate,), candidate_constraints, candidate_conservative = encode(
        (problem.candidate,)
    )
    decrease, decrease_constraints, decrease_conservative = encode(
        (problem.drift_derivative,) + problem.input_derivatives
    )
    return {
        "states": states,
        "candidate": candidate,
        "drift_derivative": decrease[0],
        "input_derivatives": decrease[1:],
        "candidate_constraints": candidate_constraints,
        "decrease_constraints": decrease_constraints,
        "candidate_conservative": candidate_conservative,
        "decrease_conservative": decrease_conservative,
    }


def _z3_obligation(
    z3, name, violation, constraints, timeout, *, conservative=False
):
    started = perf_counter()
    solver = z3.Solver()
    if timeout is not None:
        solver.set("timeout", max(1, int(1000 * timeout)))
    solver.add(*constraints, violation)
    answer = solver.check()
    if answer == z3.unsat:
        status = VerificationStatus.VERIFIED
    elif answer == z3.sat and not conservative:
        status = VerificationStatus.COUNTEREXAMPLE
    else:
        status = VerificationStatus.UNKNOWN
    return VerificationObligation(
        name,
        status,
        perf_counter() - started,
        str(answer),
        (
            "The trigonometric enclosure admits this model."
            if answer == z3.sat and conservative
            else (solver.reason_unknown() if answer == z3.unknown else "")
        ),
    )


def verify_global_quadratic_clf(
    system,
    certificate=None,
    *,
    backend="z3",
    timeout=None,
    allow_trigonometric_bounds=True,
):
    """Verify Artstein's condition over the complete real state space."""

    if backend != "z3":
        raise ValueError("Global quadratic CLF verification requires Z3.")
    timeout = _timeout(timeout)
    if not isinstance(allow_trigonometric_bounds, bool):
        raise TypeError("allow_trigonometric_bounds must be a boolean.")
    candidate = _validate_candidate(
        system, certificate, require_domain=False
    )
    problem = build_quadratic_clf_problem(system, candidate)
    z3 = _load_z3()
    started = perf_counter()
    encoded = _encode_z3(
        problem, system, z3, allow_trigonometric_bounds
    )
    nonzero = z3.Or(*(state != 0 for state in encoded["states"]))
    no_control_effect = z3.And(*(
        derivative == 0 for derivative in encoded["input_derivatives"]
    ))
    positivity = _z3_obligation(
        z3,
        "global_candidate_positivity",
        candidate_positivity_violation(
            z3,
            state_is_nonzero=nonzero,
            candidate=encoded["candidate"],
        ),
        encoded["candidate_constraints"],
        timeout,
        conservative=encoded["candidate_conservative"],
    )
    obligations = [positivity]
    if positivity.status is VerificationStatus.VERIFIED:
        obligations.append(_z3_obligation(
            z3,
            "global_clf_decrease",
            global_artstein_violation(
                z3,
                state_is_nonzero=nonzero,
                no_control_effect=no_control_effect,
                drift_derivative=encoded["drift_derivative"],
            ),
            encoded["decrease_constraints"],
            timeout,
            conservative=encoded["decrease_conservative"],
        ))
    if all_verified(obligations):
        obligations.append(VerificationObligation(
            "global_radial_unboundedness",
            VerificationStatus.VERIFIED,
            0.0,
            "implied-by-positive-quadratic",
        ))
    status = next(
        (
            item.status
            for item in obligations
            if item.status is not VerificationStatus.VERIFIED
        ),
        VerificationStatus.VERIFIED,
    )
    return GlobalLyapunovVerificationResult(
        system=system,
        certificate=candidate,
        claim=VerificationClaim.CONTROL_LYAPUNOV,
        status=status,
        verified_set=(
            WholeSpace(system.state_dimension)
            if status is VerificationStatus.VERIFIED
            else None
        ),
        obligations=tuple(obligations),
        backend="z3",
        elapsed_time=perf_counter() - started,
        properness_verified=(status is VerificationStatus.VERIFIED),
        uses_auxiliary_bounds=(
            encoded["candidate_conservative"]
            or encoded["decrease_conservative"]
        ),
    )


def _verify_quadratic_clf_z3(
    system,
    candidate,
    *,
    maximum_level,
    search_precision,
    timeout,
    allow_trigonometric_bounds,
):
    problem = build_quadratic_clf_problem(system, candidate)
    z3 = _load_z3()
    started = perf_counter()
    encoded = _encode_z3(
        problem, system, z3, allow_trigonometric_bounds
    )
    states = encoded["states"]
    nonzero = z3.Or(*(state != 0 for state in states))
    no_control_effect = z3.And(*(
        derivative == 0 for derivative in encoded["input_derivatives"]
    ))
    physical_states = tuple(
        state + sympy_to_z3(centre, {}, z3)
        for state, centre in zip(states, problem.centre)
    )
    in_domain = z3.And(*(
        z3.And(
            state >= z3.RealVal(repr(lower)),
            state <= z3.RealVal(repr(upper)),
        )
        for state, (lower, upper) in zip(
            physical_states, system.domain.bounds
        )
    ))
    on_boundary = z3.Or(*(
        z3.Or(
            state == z3.RealVal(repr(lower)),
            state == z3.RealVal(repr(upper)),
        )
        for state, (lower, upper) in zip(
            physical_states, system.domain.bounds
        )
    ))
    positivity = _z3_obligation(
        z3,
        "candidate_positive_definite",
        candidate_positivity_violation(
            z3,
            state_is_nonzero=nonzero,
            candidate=encoded["candidate"],
        ),
        encoded["candidate_constraints"],
        timeout,
        conservative=encoded["candidate_conservative"],
    )
    if positivity.status is not VerificationStatus.VERIFIED:
        return QuadraticVerificationResult(
            system,
            candidate,
            VerificationClaim.CONTROL_LYAPUNOV,
            positivity.status,
            None,
            None,
            (positivity,),
            "z3",
            perf_counter() - started,
        )

    def check_level(level):
        exact_level = z3.RealVal(repr(float(level)))
        decrease = _z3_obligation(
            z3,
            "sublevel_clf_decrease",
            sublevel_artstein_violation(
                z3,
                in_domain=in_domain,
                state_is_nonzero=nonzero,
                candidate=encoded["candidate"],
                level=exact_level,
                no_control_effect=no_control_effect,
                drift_derivative=encoded["drift_derivative"],
            ),
            encoded["decrease_constraints"],
            timeout,
            conservative=encoded["decrease_conservative"],
        )
        if decrease.status is not VerificationStatus.VERIFIED:
            return (decrease,)
        boundary = _z3_obligation(
            z3,
            "domain_boundary_exclusion",
            domain_boundary_violation(
                z3,
                in_domain=in_domain,
                on_domain_boundary=on_boundary,
                candidate=encoded["candidate"],
                level=exact_level,
            ),
            encoded["candidate_constraints"],
            timeout,
            conservative=encoded["candidate_conservative"],
        )
        return decrease, boundary

    level, proof = search_verified_level(
        maximum_level, search_precision, check_level
    )
    status = (
        VerificationStatus.VERIFIED
        if level is not None
        else next(
            (
                item.status
                for item in proof
                if item.status is not VerificationStatus.VERIFIED
            ),
            VerificationStatus.UNKNOWN,
        )
    )
    verified_set = None if level is None else SublevelSet(candidate, level)
    return QuadraticVerificationResult(
        system,
        candidate,
        VerificationClaim.CONTROL_LYAPUNOV,
        status,
        verified_set,
        verified_set,
        (positivity,) + tuple(proof),
        "z3",
        perf_counter() - started,
    )


def verify_quadratic_clf(
    system,
    certificate=None,
    *,
    backend="z3",
    maximum_level=None,
    search_precision=1e-4,
    timeout=None,
    dreal_delta=1e-4,
    c1_search_precision=1e-4,
    c2_search_precision=1e-4,
    local_max_level=None,
    jobs=1,
    allow_trigonometric_bounds=False,
    crown_device="auto",
    crown_batch_size=64,
    crown_proof_margin=1e-6,
    crown_verbose=False,
):
    """Find or verify an unrestricted-input quadratic CLF sublevel."""

    if backend not in ("z3", "dreal", "crown"):
        raise ValueError("backend must be 'z3', 'dreal', or 'crown'.")
    candidate = _validate_candidate(system, certificate, require_domain=True)
    cap = (
        _automatic_level_cap(system, candidate)
        if maximum_level is None
        else _positive_finite(maximum_level, "maximum_level")
    )
    if backend == "z3":
        if not isinstance(allow_trigonometric_bounds, bool):
            raise TypeError("allow_trigonometric_bounds must be a boolean.")
        return _verify_quadratic_clf_z3(
            system,
            candidate,
            maximum_level=cap,
            search_precision=_positive_finite(
                search_precision, "search_precision"
            ),
            timeout=_timeout(timeout),
            allow_trigonometric_bounds=allow_trigonometric_bounds,
        )
    if backend == "dreal":
        if timeout is not None:
            raise ValueError("A hard dReal timeout is not implemented yet.")
        from .quadratic_clf_dreal import verify_quadratic_clf_dreal
        return verify_quadratic_clf_dreal(
            system,
            candidate,
            delta=dreal_delta,
            local_search_precision=c1_search_precision,
            outer_search_precision=c2_search_precision,
            local_max_level=(cap if local_max_level is None else local_max_level),
            maximum_level=cap,
            jobs=jobs,
        )
    if backend == "crown":
        if local_max_level is None or maximum_level is None:
            raise ValueError(
                "CROWN requires proposed local_max_level and maximum_level."
            )
        from .quadratic_clf_crown import verify_quadratic_clf_crown
        return verify_quadratic_clf_crown(
            system,
            candidate,
            local_level=local_max_level,
            outer_level=maximum_level,
            timeout=_timeout(timeout),
            device=crown_device,
            batch_size=crown_batch_size,
            proof_margin=crown_proof_margin,
            verbose=crown_verbose,
        )


__all__ = ["verify_global_quadratic_clf", "verify_quadratic_clf"]
