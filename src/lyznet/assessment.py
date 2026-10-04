"""Empirical proposals for formal quadratic verification levels."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

import numpy as np
from scipy.stats import qmc
import sympy as sp

from .analysis import quadratic_control_lyapunov, quadratic_lyapunov
from .systems import ControlAffineSystem
from .verification._quadratic_clf import build_quadratic_clf_problem
from .verification.quadratic import (
    _automatic_level_cap,
    _build_problem,
)
from .verification.quadratic_crown import _LINEAR_DECAY_FACTOR


@dataclass(frozen=True)
class QuadraticLevelProposal:
    """A sampled level proposal; this record is not formal evidence."""

    stage: str
    suggested_level: float
    observed_failure_level: float | None
    containment_level: float
    sample_count: int
    seed: int

    @property
    def is_formal(self):
        return False


@dataclass(frozen=True)
class QuadraticScreeningResult:
    """Empirical proposals for the inner and outer proof stages."""

    local: QuadraticLevelProposal
    outer: QuadraticLevelProposal

    @property
    def is_formal(self):
        return False


def _positive_power_of_two(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 16
        or value & (value - 1)
    ):
        raise ValueError("sample_count must be a power of two at least 16.")
    return value


def _samples(bounds, count, seed):
    engine = qmc.Sobol(len(bounds), scramble=True, seed=seed)
    unit = engine.random_base2(int(np.log2(count)))
    bounds = np.asarray(bounds, dtype=float)
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])


def _values(states, expressions, points):
    functions = tuple(
        sp.lambdify(states, expression, modules="numpy")
        for expression in expressions
    )
    columns = []
    for function in functions:
        values = np.asarray(function(*points.T), dtype=float)
        if values.ndim == 0:
            values = np.full(len(points), values.item())
        columns.append(values)
    return tuple(columns)


def _floor(value, precision):
    value = Decimal(str(value))
    quantum = Decimal(str(precision))
    return float(
        (value / quantum).to_integral_value(rounding=ROUND_FLOOR)
        * quantum
    )


def _containment_level(system, candidate):
    inverse = np.linalg.inv(np.asarray(candidate.matrix, dtype=float))
    centre = np.asarray(candidate.centre, dtype=float)
    levels = []
    for index, (lower, upper) in enumerate(system.domain.bounds):
        levels.extend((
            (centre[index] - lower) ** 2 / inverse[index, index],
            (upper - centre[index]) ** 2 / inverse[index, index],
        ))
    return float(min(levels))


def _proposal(
    stage,
    candidate_values,
    failures,
    containment,
    cap,
    precision,
    sample_count,
    seed,
):
    failure_level = (
        float(np.min(candidate_values[failures]))
        if np.any(failures)
        else None
    )
    observed = min(
        containment,
        cap,
        np.inf if failure_level is None else failure_level,
    )
    # CROWN receives one proposal rather than a formal bisection. Retain a
    # ten-percent gap from the first sampled failure or containment boundary.
    safety = max(8 * precision, 0.1 * observed)
    suggested = _floor(max(0.0, observed - safety), precision)
    return QuadraticLevelProposal(
        stage=stage,
        suggested_level=suggested,
        observed_failure_level=failure_level,
        containment_level=containment,
        sample_count=sample_count,
        seed=seed,
    )


def screen_quadratic_roa(
    system,
    certificate=None,
    *,
    sample_count=2**16,
    seed=20260823,
    maximum_level=None,
    precision=1e-4,
    proof_margin=1e-6,
    local_decrease_margin=1e-4,
):
    """Sample the same remainder and decrease predicates used by CROWN."""

    sample_count = _positive_power_of_two(sample_count)
    certificate = (
        quadratic_lyapunov(system)
        if certificate is None
        else certificate
    )
    problem = _build_problem(system, certificate, include_crown=True)
    centre = np.asarray(certificate.centre, dtype=float)
    relative_bounds = tuple(
        (lower - value, upper - value)
        for value, (lower, upper) in zip(centre, system.domain.bounds)
    )
    points = _samples(relative_bounds, sample_count, seed)
    values, derivatives, remainder = _values(
        problem.states,
        (problem.candidate, problem.lie_derivative, problem.remainder_bound),
        points,
    )
    decay = float(np.min(np.linalg.eigvalsh(
        np.asarray(problem.linear_decrease_matrix, dtype=float)
    )))
    remainder_limit = (
        _LINEAR_DECAY_FACTOR * decay - local_decrease_margin
    ) ** 2 - proof_margin
    local_failures = remainder >= remainder_limit
    outer_failures = (
        (values > np.finfo(float).eps)
        & (derivatives >= -proof_margin)
    )
    containment = _containment_level(system, certificate)
    cap = (
        _automatic_level_cap(system, certificate)
        if maximum_level is None
        else float(maximum_level)
    )
    return QuadraticScreeningResult(
        local=_proposal(
            "local_remainder_bound",
            values,
            local_failures,
            containment,
            cap,
            precision,
            sample_count,
            seed,
        ),
        outer=_proposal(
            "annular_decrease",
            values,
            outer_failures,
            containment,
            cap,
            precision,
            sample_count,
            seed,
        ),
    )


def screen_quadratic_clf(
    system,
    certificate=None,
    *,
    sample_count=2**16,
    seed=20260823,
    maximum_level=None,
    precision=1e-4,
    proof_margin=1e-6,
):
    """Sample the CROWN contraction and Artstein proof predicates."""

    if not isinstance(system, ControlAffineSystem):
        raise TypeError("CLF screening requires a control-affine system.")
    sample_count = _positive_power_of_two(sample_count)
    certificate = (
        quadratic_control_lyapunov(system)
        if certificate is None
        else certificate
    )
    problem = build_quadratic_clf_problem(system, certificate)
    centre = np.asarray(certificate.centre, dtype=float)
    relative_bounds = tuple(
        (lower - value, upper - value)
        for value, (lower, upper) in zip(centre, system.domain.bounds)
    )
    points = _samples(relative_bounds, sample_count, seed)
    columns = _values(
        problem.states,
        (
            problem.candidate,
            problem.drift_derivative,
            *problem.input_derivatives,
            *problem.contraction_principal_minors,
        ),
        points,
    )
    values = columns[0]
    derivatives = columns[1]
    effects = np.max(
        np.abs(np.column_stack(columns[2:2 + system.input_dimension])),
        axis=1,
    )
    minors = np.min(
        np.column_stack(columns[2 + system.input_dimension:]),
        axis=1,
    )
    local_failures = minors <= proof_margin
    outer_failures = (
        (values > np.finfo(float).eps)
        & (effects <= proof_margin)
        & (derivatives >= -proof_margin)
    )
    containment = _containment_level(system, certificate)
    cap = (
        _automatic_level_cap(system, certificate)
        if maximum_level is None
        else float(maximum_level)
    )
    return QuadraticScreeningResult(
        local=_proposal(
            "local_lqr_contraction",
            values,
            local_failures,
            containment,
            cap,
            precision,
            sample_count,
            seed,
        ),
        outer=_proposal(
            "annular_clf_decrease",
            values,
            outer_failures,
            containment,
            cap,
            precision,
            sample_count,
            seed,
        ),
    )


__all__ = [
    "QuadraticLevelProposal",
    "QuadraticScreeningResult",
    "screen_quadratic_clf",
    "screen_quadratic_roa",
]
