"""dReal adapter for two-stage quadratic Lyapunov verification."""

from importlib import import_module
from numbers import Integral
from time import perf_counter

import numpy as np
import sympy as sp

from lyznet.sets import SublevelSet

from .core.quadratic_dreal import (
    domain_boundary_violation,
    local_contraction_violation,
    local_invariance_violation,
    outer_annulus_violation,
    positive_definite_by_sylvester,
)
from .quadratic import (
    OptionalVerificationDependencyError,
    QuadraticVerificationResult,
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
    _automatic_level_cap,
    _build_problem,
    _floor_to_precision,
    _freeze_decimal,
    _validate_inputs,
)


def _load_dreal():
    try:
        return import_module("dreal")
    except ImportError as error:
        raise OptionalVerificationDependencyError(
            "dReal verification requires the optional dependency dreal; "
            "install LyZNet with `pip install -e '.[dreal]'`."
        ) from error


def _sympy_to_dreal(expression, substitutions, dreal):
    """Translate supported SymPy expressions to dReal expressions."""

    value = sp.sympify(expression)
    if value in substitutions:
        return dreal.Expression(substitutions[value])
    if value.is_Integer:
        return dreal.Expression(int(value))
    if value.is_Rational:
        return (
            dreal.Expression(int(value.p))
            / dreal.Expression(int(value.q))
        )
    if value.is_Float:
        if value._prec > 64:
            raise ValueError(
                "dReal conversion cannot preserve a high-precision SymPy "
                "Float; use an exact Rational or a binary64 coefficient."
            )
        return _sympy_to_dreal(_freeze_decimal(value), substitutions, dreal)
    if value.is_NumberSymbol:
        raise NotImplementedError(
            "Replace symbolic constants such as pi by an explicit fixed "
            "coefficient or a sound enclosure."
        )
    if value.is_Add:
        return sum(
            (
                _sympy_to_dreal(term, substitutions, dreal)
                for term in value.args
            ),
            dreal.Expression(0),
        )
    if value.is_Mul:
        result = dreal.Expression(1)
        for factor in value.args:
            result *= _sympy_to_dreal(factor, substitutions, dreal)
        return result
    if value.is_Pow:
        base, exponent = value.args
        return _sympy_to_dreal(
            base, substitutions, dreal
        ) ** _sympy_to_dreal(exponent, substitutions, dreal)
    if value.func is sp.Abs:
        return abs(_sympy_to_dreal(value.args[0], substitutions, dreal))
    if value.is_Function:
        function = getattr(dreal, value.func.__name__, None)
        if function is None:
            raise NotImplementedError(
                "dReal conversion does not support function {!s}.".format(
                    value.func
                )
            )
        return function(*(
            _sympy_to_dreal(argument, substitutions, dreal)
            for argument in value.args
        ))
    raise NotImplementedError(
        "dReal conversion does not support expression {!s}.".format(value)
    )


def _dreal_number(value, dreal):
    return _sympy_to_dreal(_freeze_decimal(value), {}, dreal)


def _all_verified(obligations):
    return all(
        item.status is VerificationStatus.VERIFIED for item in obligations
    )


def _search_verified_level(upper, precision, check, *, lower=0.0):
    """Bisect a monotone formal condition and recheck the reported level."""

    low = float(lower)
    high = float(upper)
    if low > 0 and not _all_verified(check(low)):
        return None, check(low)
    while high - low > precision:
        level = (low + high) / 2
        obligations = check(level)
        if _all_verified(obligations):
            low = level
        else:
            high = level
    level = max(float(lower), _floor_to_precision(low, precision))
    if level <= 0:
        return None, check(high)
    obligations = check(level)
    if not _all_verified(obligations):
        return None, obligations
    return level, obligations


def verify_quadratic_roa_dreal(
    system,
    certificate,
    *,
    delta=1e-4,
    c1_search_precision=1e-4,
    c2_search_precision=1e-4,
    local_max_level=None,
    maximum_level=None,
    jobs=1,
):
    """Verify a contracting local set and an outer decreasing annulus."""

    _validate_inputs(
        system, certificate, c2_search_precision, maximum_level
    )
    for value, label in (
        (delta, "delta"),
        (c1_search_precision, "c1_search_precision"),
        (c2_search_precision, "c2_search_precision"),
        (local_max_level, "local_max_level"),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("{} must be positive and finite.".format(label))
    if isinstance(jobs, bool) or not isinstance(jobs, Integral) or jobs < 1:
        raise ValueError("jobs must be a positive integer.")

    dreal = _load_dreal()
    started = perf_counter()
    problem = _build_problem(system, certificate)
    centre_is_exact = all(
        value == 0 for value in problem.dynamics_at_centre
    )
    claim = (
        VerificationClaim.LYAPUNOV
        if centre_is_exact
        else VerificationClaim.STABLE_EQUILIBRIUM_ROA
    )
    cap = (
        _automatic_level_cap(system, certificate)
        if maximum_level is None
        else float(maximum_level)
    )
    local_cap = cap if local_max_level is None else min(
        float(local_max_level), cap
    )

    config = dreal.Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = float(delta)
    config.number_of_jobs = int(jobs)

    states = tuple(
        dreal.Variable("y{}".format(index + 1))
        for index in range(system.state_dimension)
    )
    substitutions = dict(zip(problem.states, states))
    candidate = _sympy_to_dreal(
        problem.candidate, substitutions, dreal
    )
    lie_derivative = _sympy_to_dreal(
        problem.lie_derivative, substitutions, dreal
    )
    contraction_minors = tuple(
        _sympy_to_dreal(value, substitutions, dreal)
        for value in problem.contraction_principal_minors
    )
    centre = tuple(
        _sympy_to_dreal(value, {}, dreal) for value in problem.centre
    )
    physical_states = tuple(
        state + center for state, center in zip(states, centre)
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

    def check(name, violation):
        check_started = perf_counter()
        answer = dreal.CheckSatisfiability(violation, config)
        verified = answer is None
        return VerificationObligation(
            name=name,
            status=(
                VerificationStatus.VERIFIED
                if verified
                else VerificationStatus.UNKNOWN
            ),
            elapsed_time=perf_counter() - check_started,
            backend_status="unsat" if verified else "delta-sat",
        )

    positivity_verified = positive_definite_by_sylvester(
        problem.principal_minors
    )
    positivity = VerificationObligation(
        name="candidate_positive_definite",
        status=(
            VerificationStatus.VERIFIED
            if positivity_verified
            else VerificationStatus.COUNTEREXAMPLE
        ),
        elapsed_time=0.0,
        backend_status="exact-rational",
    )
    if not positivity_verified:
        return QuadraticVerificationResult(
            system=system,
            certificate=certificate,
            claim=claim,
            status=VerificationStatus.COUNTEREXAMPLE,
            local_set=None,
            verified_set=None,
            obligations=(positivity,),
            backend="dreal",
            elapsed_time=perf_counter() - started,
        )

    def boundary_check(level, name):
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

    def check_local_shape(level):
        contraction = check(
            "local_contraction",
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
        boundary = boundary_check(level, "local_boundary_exclusion")
        return (contraction, boundary)

    def check_local(level):
        shape_obligations = check_local_shape(level)
        if not _all_verified(shape_obligations):
            return shape_obligations
        if centre_is_exact:
            invariance = VerificationObligation(
                name="local_positive_invariance",
                status=VerificationStatus.VERIFIED,
                elapsed_time=0.0,
                backend_status="implied-by-contraction",
            )
        else:
            invariance = check(
                "local_positive_invariance",
                local_invariance_violation(
                    dreal,
                    in_domain=in_domain,
                    candidate=candidate,
                    level=_dreal_number(level, dreal),
                    lie_derivative=lie_derivative,
                ),
            )
        return (
            shape_obligations[0],
            invariance,
            shape_obligations[1],
        )

    local_level, local_obligations = _search_verified_level(
        local_cap,
        float(c1_search_precision),
        check_local_shape if not centre_is_exact else check_local,
    )
    if local_level is not None and not centre_is_exact:
        while local_level >= float(c1_search_precision):
            local_obligations = check_local(local_level)
            if _all_verified(local_obligations):
                break
            local_level = _floor_to_precision(
                local_level / 2, float(c1_search_precision)
            )
        if local_level < float(c1_search_precision):
            local_level = None

    if local_level is None:
        return QuadraticVerificationResult(
            system=system,
            certificate=certificate,
            claim=claim,
            status=VerificationStatus.UNKNOWN,
            local_set=None,
            verified_set=None,
            obligations=(positivity,) + tuple(local_obligations),
            backend="dreal",
            elapsed_time=perf_counter() - started,
        )

    local_set = SublevelSet(certificate, local_level)

    def check_outer(level):
        decrease = check(
            "annular_decrease",
            outer_annulus_violation(
                dreal,
                in_domain=in_domain,
                candidate=candidate,
                local_level=_dreal_number(local_level, dreal),
                outer_level=_dreal_number(level, dreal),
                lie_derivative=lie_derivative,
            ),
        )
        if decrease.status is not VerificationStatus.VERIFIED:
            return (decrease,)
        boundary = boundary_check(level, "outer_boundary_exclusion")
        return (decrease, boundary)

    outer_level, outer_obligations = _search_verified_level(
        cap,
        float(c2_search_precision),
        check_outer,
        lower=local_level,
    )
    obligations = (
        (positivity,) + tuple(local_obligations) + tuple(outer_obligations)
    )
    if outer_level is None:
        return QuadraticVerificationResult(
            system=system,
            certificate=certificate,
            claim=claim,
            status=VerificationStatus.UNKNOWN,
            local_set=local_set,
            verified_set=None,
            obligations=obligations,
            backend="dreal",
            elapsed_time=perf_counter() - started,
        )

    return QuadraticVerificationResult(
        system=system,
        certificate=certificate,
        claim=claim,
        status=VerificationStatus.VERIFIED,
        local_set=local_set,
        verified_set=SublevelSet(certificate, outer_level),
        obligations=obligations,
        backend="dreal",
        elapsed_time=perf_counter() - started,
    )


__all__ = ["verify_quadratic_roa_dreal"]
