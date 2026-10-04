"""Z3 adapter for direct verification of quadratic Lyapunov sublevels."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
from importlib import import_module
from time import perf_counter

import numpy as np
import sympy as sp

from lyznet.analysis import QuadraticLyapunov, linearize, quadratic_lyapunov
from lyznet.sets import SublevelSet
from lyznet.systems import ControlAffineSystem

from .core.quadratic_z3 import (
    domain_boundary_violation,
    positive_definite_by_sylvester,
    sublevel_decrease_violation,
)


class OptionalVerificationDependencyError(ImportError):
    """A requested verification backend is not installed."""


class VerificationStatus(Enum):
    VERIFIED = "verified"
    COUNTEREXAMPLE = "counterexample"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VerificationObligation:
    """The outcome of one explicit mathematical proof obligation."""

    name: str
    status: VerificationStatus
    elapsed_time: float
    backend_status: str


@dataclass(frozen=True)
class QuadraticVerificationResult:
    """Formal evidence for one quadratic Lyapunov sublevel."""

    system: object
    certificate: QuadraticLyapunov
    status: VerificationStatus
    verified_set: SublevelSet | None
    obligations: tuple[VerificationObligation, ...]
    backend: str
    elapsed_time: float

    @property
    def verified_level(self):
        return None if self.verified_set is None else self.verified_set.level

    @property
    def is_verified(self):
        return self.status is VerificationStatus.VERIFIED


def _load_z3():
    try:
        return import_module("z3")
    except ImportError as error:
        raise OptionalVerificationDependencyError(
            "Z3 verification requires the optional dependency z3-solver; "
            "install LyZNet with `pip install -e '.[z3]'`."
        ) from error


def _freeze_decimal(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Frozen coefficients must be finite.")
    return sp.Rational(repr(value))


def _freeze_matrix(values):
    array = np.asarray(values)
    return sp.ImmutableMatrix(
        array.shape[0],
        array.size // array.shape[0],
        [_freeze_decimal(value) for value in array.flat],
    )


def _sympy_to_z3(expression, substitutions, z3):
    """Translate polynomial SymPy arithmetic to exact Z3 real arithmetic."""

    value = sp.sympify(expression)
    if value in substitutions:
        return substitutions[value]
    if value.is_Integer:
        return z3.RealVal(int(value))
    if value.is_Rational:
        return z3.RealVal("{}/{}".format(value.p, value.q))
    if value.is_Float:
        return z3.RealVal(str(value))
    if value.is_Add:
        return z3.Sum(
            [_sympy_to_z3(term, substitutions, z3) for term in value.args]
        )
    if value.is_Mul:
        result = z3.RealVal(1)
        for factor in value.args:
            result *= _sympy_to_z3(factor, substitutions, z3)
        return result
    if value.is_Pow:
        base, exponent = value.args
        if not exponent.is_Integer or exponent < 0:
            raise NotImplementedError(
                "Z3 quadratic verification supports polynomial dynamics."
            )
        return _sympy_to_z3(base, substitutions, z3) ** int(exponent)
    raise NotImplementedError(
        "Z3 quadratic verification does not support expression {!s}.".format(
            value
        )
    )


@dataclass(frozen=True)
class _QuadraticProblem:
    states: tuple
    centre: tuple
    candidate: sp.Expr
    lie_derivative: sp.Expr
    principal_minors: tuple


def _build_problem(system, certificate):
    relative_states = sp.ImmutableMatrix(system.states)
    matrix = _freeze_matrix(certificate.matrix)
    centre = _freeze_matrix(certificate.centre)
    if matrix != matrix.T:
        raise ValueError("The frozen quadratic matrix must be exactly symmetric.")

    # Express the original dynamics in relative coordinates y=x-x*.
    physical_states = relative_states + centre
    dynamics = system.symbolic.dynamics.subs(
        dict(zip(system.states, physical_states)), simultaneous=True
    )
    candidate = (relative_states.T * matrix * relative_states)[0]
    lie_derivative = (
        sp.ImmutableMatrix([candidate]).jacobian(relative_states) * dynamics
    )[0]
    principal_minors = tuple(
        sp.det(matrix[:size, :size])
        for size in range(1, system.state_dimension + 1)
    )
    return _QuadraticProblem(
        states=tuple(relative_states),
        centre=tuple(centre),
        candidate=sp.expand(candidate),
        lie_derivative=sp.expand(lie_derivative),
        principal_minors=principal_minors,
    )


def _validate_inputs(system, certificate, search_precision, maximum_level):
    if isinstance(system, ControlAffineSystem):
        raise TypeError(
            "Quadratic Lyapunov verification requires an autonomous system."
        )
    if system.domain is None:
        raise ValueError("Quadratic Lyapunov verification requires a box domain.")
    if not isinstance(certificate, QuadraticLyapunov):
        raise TypeError("certificate must be a QuadraticLyapunov candidate.")
    if certificate.state_dimension != system.state_dimension:
        raise ValueError("Certificate dimension does not match the system.")
    if not np.array_equal(
        certificate.centre, np.asarray(system.equilibrium, dtype=float)
    ):
        raise ValueError("Certificate centre must match the equilibrium exactly.")
    if not np.allclose(certificate.linearization, linearize(system)):
        raise ValueError("Certificate linearization does not match the system.")
    if any(
        not lower < value < upper
        for value, (lower, upper) in zip(
            certificate.centre, system.domain.bounds
        )
    ):
        raise ValueError("The equilibrium must lie in the domain interior.")
    for value, label in (
        (search_precision, "search_precision"),
        (maximum_level, "maximum_level"),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("{} must be positive and finite.".format(label))


def _automatic_level_cap(system, certificate):
    """Return a numerical upper bound used only to bracket formal search."""

    radii = np.asarray([
        max(abs(lower - centre), abs(upper - centre))
        for centre, (lower, upper) in zip(
            certificate.centre, system.domain.bounds
        )
    ])
    return float(np.linalg.eigvalsh(certificate.matrix)[-1] * radii @ radii)


def _floor_to_precision(level, precision):
    value = Decimal(str(level))
    quantum = Decimal(str(precision))
    return float(
        (value / quantum).to_integral_value(rounding=ROUND_FLOOR) * quantum
    )


def verify_quadratic_roa(
    system,
    *,
    certificate=None,
    backend="z3",
    search_precision=1e-4,
    maximum_level=None,
    timeout=None,
):
    """Find and formally verify a quadratic Lyapunov sublevel.

    Z3 reasons exactly about the rational numbers obtained from the stored
    floating-point candidate. ``timeout`` is an optional per-query limit in
    seconds. Every returned positive level has been checked directly after
    rounding to ``search_precision``.
    """

    if backend != "z3":
        raise ValueError("This stage supports only backend='z3'.")
    certificate = (
        quadratic_lyapunov(system) if certificate is None else certificate
    )
    _validate_inputs(system, certificate, search_precision, maximum_level)
    search_precision = float(search_precision)
    cap = (
        _automatic_level_cap(system, certificate)
        if maximum_level is None
        else float(maximum_level)
    )
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not np.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError("timeout must be positive and finite.")
        timeout = float(timeout)

    z3 = _load_z3()
    started = perf_counter()
    problem = _build_problem(system, certificate)
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
            system,
            certificate,
            VerificationStatus.COUNTEREXAMPLE,
            None,
            (positivity,),
            "z3",
            perf_counter() - started,
        )

    z3_states = tuple(
        z3.Real("y{}".format(index + 1))
        for index in range(system.state_dimension)
    )
    substitutions = dict(zip(problem.states, z3_states))
    candidate = _sympy_to_z3(problem.candidate, substitutions, z3)
    lie_derivative = _sympy_to_z3(
        problem.lie_derivative, substitutions, z3
    )
    centre = tuple(
        _sympy_to_z3(value, substitutions, z3) for value in problem.centre
    )
    physical_states = tuple(
        state + center for state, center in zip(z3_states, centre)
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
    nonzero = z3.Or(*(state != 0 for state in z3_states))

    def check(name, violation):
        check_started = perf_counter()
        solver = z3.Solver()
        if timeout is not None:
            solver.set("timeout", max(1, int(1000 * timeout)))
        solver.add(violation)
        answer = solver.check()
        if answer == z3.unsat:
            status = VerificationStatus.VERIFIED
        elif answer == z3.sat:
            status = VerificationStatus.COUNTEREXAMPLE
        else:
            status = VerificationStatus.UNKNOWN
        return VerificationObligation(
            name=name,
            status=status,
            elapsed_time=perf_counter() - check_started,
            backend_status=str(answer),
        )

    def check_level(level):
        exact_level = z3.RealVal(repr(float(level)))
        decrease = check(
            "sublevel_decrease",
            sublevel_decrease_violation(
                z3,
                in_domain=in_domain,
                state_is_nonzero=nonzero,
                candidate=candidate,
                lie_derivative=lie_derivative,
                level=exact_level,
            ),
        )
        if decrease.status is not VerificationStatus.VERIFIED:
            return (decrease,)
        boundary = check(
            "domain_boundary_exclusion",
            domain_boundary_violation(
                z3,
                in_domain=in_domain,
                on_domain_boundary=on_boundary,
                candidate=candidate,
                level=exact_level,
            ),
        )
        return (decrease, boundary)

    low, high = 0.0, cap
    while high - low > search_precision:
        level = (low + high) / 2
        obligations = check_level(level)
        if all(
            item.status is VerificationStatus.VERIFIED
            for item in obligations
        ):
            low = level
        else:
            high = level

    verified_level = _floor_to_precision(low, search_precision)
    if verified_level <= 0:
        failed = check_level(high)
        statuses = {item.status for item in failed}
        status = (
            VerificationStatus.UNKNOWN
            if VerificationStatus.UNKNOWN in statuses
            else VerificationStatus.COUNTEREXAMPLE
        )
        return QuadraticVerificationResult(
            system,
            certificate,
            status,
            None,
            (positivity,) + failed,
            "z3",
            perf_counter() - started,
        )

    best_obligations = check_level(verified_level)
    if not all(
        item.status is VerificationStatus.VERIFIED
        for item in best_obligations
    ):
        raise RuntimeError("The rounded reporting level failed re-verification.")
    return QuadraticVerificationResult(
        system,
        certificate,
        VerificationStatus.VERIFIED,
        SublevelSet(certificate, verified_level),
        (positivity,) + best_obligations,
        "z3",
        perf_counter() - started,
    )


__all__ = [
    "OptionalVerificationDependencyError",
    "QuadraticVerificationResult",
    "VerificationObligation",
    "VerificationStatus",
    "verify_quadratic_roa",
]
