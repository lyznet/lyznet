"""Global dReal verification for positive homogeneous neural candidates."""

from time import perf_counter

import numpy as np
import sympy as sp

from lyznet.analysis import homogeneous_degree
from lyznet.neural import HomogeneousNeuralLyapunov
from lyznet.sets import WholeSpace
from lyznet.systems import ControlAffineSystem

from .core.homogeneous_dreal import (
    candidate_positivity_violation,
    dynamics_well_defined_violation,
    strict_decrease_violation,
)
from .global_lyapunov import GlobalLyapunovVerificationResult
from .quadratic import (
    VerificationClaim,
    VerificationObligation,
    VerificationStatus,
)
from .quadratic_dreal import _dreal_number, _load_dreal, _sympy_to_dreal


def _origin_extension_branch(expression, states):
    """Return the punctured branch of the canonical zero extension.

    The accepted representation is

    ``Piecewise((0, Eq(sum(x_i**2), 0)), (f(x), True))``.

    On the unit sphere only ``f`` applies. Requiring this exact form keeps the
    extension visible in the system definition and avoids interpreting a
    general symbolic piecewise expression inside the proof adapter.
    """

    expression = sp.sympify(expression)
    if not isinstance(expression, sp.Piecewise):
        return expression
    if len(expression.args) != 2 or expression.args[1].cond != sp.true:
        raise ValueError(
            "Homogeneous verification supports only the canonical zero "
            "extension at the origin."
        )
    origin_value, condition = expression.args[0]
    if sp.simplify(origin_value) != 0 or not isinstance(condition, sp.Equality):
        raise ValueError(
            "The first homogeneous Piecewise branch must set the origin to zero."
        )
    condition_expression = sp.expand(condition.lhs - condition.rhs)
    norm_squared = sum(state**2 for state in states)
    if (
        sp.simplify(condition_expression - norm_squared) != 0
        and sp.simplify(condition_expression + norm_squared) != 0
    ):
        raise ValueError(
            "The homogeneous Piecewise condition must be "
            "Eq(sum(x_i**2), 0)."
        )
    return sp.sympify(expression.args[1].expr)


def _domain_requirements(expressions):
    denominators = set()
    radicands = set()
    for expression in expressions:
        denominator = sp.factor(sp.denom(sp.together(expression)))
        if denominator != 1:
            denominators.add(denominator)
        for subexpression in sp.preorder_traversal(expression):
            if not subexpression.is_Pow:
                continue
            base, exponent = subexpression.args
            if exponent.is_Rational and int(exponent.q) % 2 == 0:
                radicands.add(sp.factor(base))
    key = sp.default_sort_key
    return tuple(sorted(denominators, key=key)), tuple(
        sorted(radicands, key=key)
    )


def _candidate_expression(candidate, variables, dreal):
    model = candidate.model
    weights = model.layers[0].weight.detach().cpu().numpy()
    biases = model.layers[0].bias.detach().cpu().numpy()
    coefficients = model.output_layer.weight.detach().cpu().numpy()[0]
    offset = float(model.output_layer.bias.detach().cpu().numpy()[0])
    norm = dreal.sqrt(sum(variable**2 for variable in variables))
    direction = [variable / norm for variable in variables]
    hidden = []
    for row, bias in zip(weights, biases):
        preactivation = _dreal_number(bias, dreal)
        preactivation += sum(
            _dreal_number(weight, dreal) * value
            for weight, value in zip(row, direction)
        )
        hidden.append(dreal.tanh(preactivation))
    log_shape = _dreal_number(offset, dreal) + sum(
        _dreal_number(coefficient, dreal) * value
        for coefficient, value in zip(coefficients, hidden)
        if coefficient != 0
    )
    return norm**candidate.degree * dreal.exp(log_shape)


def verify_homogeneous_neural_lyapunov_dreal(
    system,
    candidate,
    *,
    dreal_delta=1e-4,
    positivity_margin=1e-4,
    decrease_margin=1e-4,
    well_defined_margin=1e-8,
    jobs=1,
):
    r"""Prove global stability using homogeneity and the unit sphere.

    For a positive degree-``p`` homogeneous candidate and degree-``d``
    homogeneous dynamics, proving

    ``V(u)>0`` and ``grad(V)(u)^T f(u)<0`` for every ``||u||=1``

    proves the corresponding inequalities for every nonzero state by radial
    scaling. Positive homogeneity also makes ``V`` radially unbounded.
    """

    if isinstance(system, ControlAffineSystem):
        raise TypeError("Homogeneous Lyapunov verification requires autonomy.")
    if type(candidate) is not HomogeneousNeuralLyapunov:
        raise TypeError(
            "candidate must be exactly a HomogeneousNeuralLyapunov."
        )
    if candidate.state_dimension != system.state_dimension:
        raise ValueError("Candidate dimension does not match the system.")
    if system.numerical_equilibrium is not None or any(
        sp.sympify(value).is_zero is not True
        for value in system.equilibrium
    ):
        raise ValueError("Global homogeneous verification requires the origin.")
    vector_field_degree = homogeneous_degree(system)
    if vector_field_degree is None or vector_field_degree <= 0:
        raise ValueError(
            "The autonomous dynamics must be homogeneous of positive degree."
        )
    for value, label in (
        (dreal_delta, "dreal_delta"),
        (positivity_margin, "positivity_margin"),
        (decrease_margin, "decrease_margin"),
        (well_defined_margin, "well_defined_margin"),
    ):
        if (
            isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("{} must be positive and finite.".format(label))
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer.")

    dreal = _load_dreal()
    config = dreal.Config()
    config.precision = float(dreal_delta)
    config.number_of_jobs = int(jobs)
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    variables = tuple(
        dreal.Variable("homogeneous_x{}".format(index + 1))
        for index in range(system.state_dimension)
    )
    substitutions = dict(zip(system.states, variables))
    punctured_dynamics = tuple(
        _origin_extension_branch(expression, system.states)
        for expression in system.symbolic.dynamics
    )
    dynamics = tuple(
        _sympy_to_dreal(expression, substitutions, dreal)
        for expression in punctured_dynamics
    )
    candidate_expression = _candidate_expression(candidate, variables, dreal)
    lie_derivative = sum(
        candidate_expression.Differentiate(variable) * dynamics[index]
        for index, variable in enumerate(variables)
    )
    on_unit_sphere = sum(variable**2 for variable in variables) == 1
    started = perf_counter()

    def check(name, violation):
        query_started = perf_counter()
        answer = dreal.CheckSatisfiability(violation, config)
        verified = answer is None
        return VerificationObligation(
            name=name,
            status=(
                VerificationStatus.VERIFIED
                if verified
                else VerificationStatus.UNKNOWN
            ),
            elapsed_time=perf_counter() - query_started,
            backend_status="unsat" if verified else "delta-sat",
            message="" if verified else str(answer),
        )

    obligations = [
        VerificationObligation(
            "homogeneous_structure",
            VerificationStatus.VERIFIED,
            0.0,
            "symbolic",
            "Dynamics degree {}; candidate degree {}."
            .format(vector_field_degree, candidate.degree),
        )
    ]
    denominators, radicands = _domain_requirements(punctured_dynamics)
    if denominators or radicands:
        encoded_denominators = tuple(
            _sympy_to_dreal(value, substitutions, dreal)
            for value in denominators
        )
        encoded_radicands = tuple(
            _sympy_to_dreal(value, substitutions, dreal)
            for value in radicands
        )
        obligations.append(check(
            "homogeneous_dynamics_well_defined",
            dynamics_well_defined_violation(
                dreal,
                on_unit_sphere=on_unit_sphere,
                denominators=encoded_denominators,
                radicands=encoded_radicands,
                margin=_dreal_number(well_defined_margin, dreal),
            ),
        ))
    else:
        obligations.append(VerificationObligation(
            "homogeneous_dynamics_well_defined",
            VerificationStatus.VERIFIED,
            0.0,
            "symbolic:no-denominators-or-even-roots",
        ))
    if all(
        item.status is VerificationStatus.VERIFIED for item in obligations
    ):
        obligations.append(check(
            "homogeneous_candidate_positivity",
            candidate_positivity_violation(
                dreal,
                on_unit_sphere=on_unit_sphere,
                candidate=candidate_expression,
                margin=_dreal_number(positivity_margin, dreal),
            ),
        ))
    if all(
        item.status is VerificationStatus.VERIFIED for item in obligations
    ):
        obligations.append(check(
            "homogeneous_strict_decrease",
            strict_decrease_violation(
                dreal,
                on_unit_sphere=on_unit_sphere,
                lie_derivative=lie_derivative,
                margin=_dreal_number(decrease_margin, dreal),
            ),
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
        properness_verified=(status is VerificationStatus.VERIFIED),
    )


__all__ = ["verify_homogeneous_neural_lyapunov_dreal"]
