"""Deterministic convex synthesis of homogeneous neural Lyapunov functions."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtri
from scipy.stats import qmc
import sympy as sp
import torch

from .analysis import homogeneous_degree
from .neural import HomogeneousNeuralLyapunov, TanhNetwork


@dataclass(frozen=True)
class HomogeneousSynthesisDiagnostics:
    """Numerical evidence produced before formal verification."""

    optimal_margin: float
    fitted_margin: float
    validation_maximum_ratio: float
    validation_minimum_ratio: float
    validation_violations: int
    validation_directions: int
    active_features: int
    log_shape_range: float


@dataclass(frozen=True)
class HomogeneousSynthesisResult:
    """A fixed candidate and the empirical record of its construction."""

    candidate: HomogeneousNeuralLyapunov
    model_path: Path
    fingerprint: str
    diagnostics: HomogeneousSynthesisDiagnostics
    elapsed_time: float
    loaded: bool


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be an integer.".format(label))
    if value < 1:
        raise ValueError("{} must be positive.".format(label))
    return int(value)


def _finite_positive(value, label):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("{} must be positive and finite.".format(label))
    return value


def _unit_directions(dimension, count):
    """Return deterministic directions; this is screening, not proof."""

    count = _positive_integer(count, "direction count")
    if count < 2:
        raise ValueError("direction count must be at least two.")
    if dimension == 2:
        angles = 2 * np.pi * np.arange(count, dtype=np.float64) / count
        return np.column_stack((np.cos(angles), np.sin(angles)))

    exponent = int(math.ceil(math.log2(count + 1)))
    unit = qmc.Sobol(dimension, scramble=False).random_base2(exponent)
    normal = ndtri(np.clip(unit[1:count + 1], 1e-12, 1 - 1e-12))
    norms = np.linalg.norm(normal, axis=1, keepdims=True)
    zero = norms[:, 0] == 0
    normal[zero, 0] = 1
    norms[zero] = 1
    return normal / norms


def _feature_parameters(dimension, width, feature_scale):
    exponent = int(math.ceil(math.log2(width + 1)))
    samples = qmc.Sobol(
        dimension + 2, scramble=False
    ).random_base2(exponent)[1:width + 1]
    normal = ndtri(np.clip(samples[:, :dimension], 1e-9, 1 - 1e-9))
    norms = np.linalg.norm(normal, axis=1, keepdims=True)
    zero = norms[:, 0] == 0
    normal[zero, 0] = 1
    norms[zero] = 1
    normal /= norms
    row_scales = np.exp(
        np.log(0.25 * feature_scale)
        + samples[:, dimension] * np.log(16)
    )
    weights = normal * row_scales[:, None]
    biases = -2 + 4 * samples[:, dimension + 1]
    return weights, biases


def _log_derivative_terms(system, directions, weights, biases, degree):
    """Return affine terms in ``(grad V f)/V`` on the unit sphere."""

    dynamics = np.asarray(system.numpy.dynamics(directions), dtype=np.float64)
    if dynamics.shape != directions.shape:
        raise ValueError(
            "System dynamics returned shape {}, expected {}."
            .format(dynamics.shape, directions.shape)
        )
    radial_rate = np.sum(directions * dynamics, axis=1)
    tangent_dynamics = dynamics - radial_rate[:, None] * directions
    activations = np.tanh(directions @ weights.T + biases)
    feature_rates = (
        (1 - activations**2) * (tangent_dynamics @ weights.T)
    )
    return degree * radial_rate, feature_rates, activations


def _maximum_margin(base, feature_rates, coefficient_bound):
    width = feature_rates.shape[1]
    objective = np.zeros(width + 1)
    objective[-1] = -1
    solution = linprog(
        objective,
        A_ub=np.column_stack((feature_rates, np.ones(len(base)))),
        b_ub=-base,
        bounds=[(-coefficient_bound, coefficient_bound)] * width
        + [(None, None)],
        method="highs",
        options={"presolve": True},
    )
    if not solution.success:
        raise RuntimeError(
            "Homogeneous max-margin synthesis failed: {}."
            .format(solution.message)
        )
    return float(solution.x[-1])


def _minimum_l1_coefficients(
    base,
    feature_rates,
    *,
    target_margin,
    coefficient_bound,
):
    width = feature_rates.shape[1]
    identity = np.eye(width)
    solution = linprog(
        np.concatenate((np.zeros(width), np.ones(width))),
        A_ub=np.block([
            [feature_rates, np.zeros_like(feature_rates)],
            [identity, -identity],
            [-identity, -identity],
        ]),
        b_ub=np.concatenate((
            -target_margin - base,
            np.zeros(width),
            np.zeros(width),
        )),
        bounds=[(-coefficient_bound, coefficient_bound)] * width
        + [(0, coefficient_bound)] * width,
        method="highs",
        options={"presolve": True},
    )
    if not solution.success:
        raise RuntimeError(
            "Homogeneous coefficient fitting failed: {}."
            .format(solution.message)
        )
    return solution.x[:width]


def _validate_candidate(
    system,
    weights,
    biases,
    coefficients,
    *,
    degree,
    count,
    batch_size=8192,
):
    directions = _unit_directions(weights.shape[1], count)
    maximum = -np.inf
    minimum = np.inf
    violations = 0
    log_minimum = np.inf
    log_maximum = -np.inf
    for begin in range(0, count, batch_size):
        batch = directions[begin:begin + batch_size]
        base, rates, activations = _log_derivative_terms(
            system, batch, weights, biases, degree
        )
        ratios = base + rates @ coefficients
        log_shapes = activations @ coefficients
        maximum = max(maximum, float(np.max(ratios)))
        minimum = min(minimum, float(np.min(ratios)))
        violations += int(np.count_nonzero(ratios >= 0))
        log_minimum = min(log_minimum, float(np.min(log_shapes)))
        log_maximum = max(log_maximum, float(np.max(log_shapes)))
    return maximum, minimum, violations, log_minimum, log_maximum


def _safe_name(system):
    name = system.name or "system"
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name
    ).strip("_")
    return safe or "system"


def _configuration_fingerprint(configuration):
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]


def _candidate(weights, biases, coefficients, offset, degree):
    model = TanhNetwork(
        weights.shape[1],
        hidden_layers=1,
        width=weights.shape[0],
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.from_numpy(weights))
        model.layers[0].bias.copy_(torch.from_numpy(biases))
        model.output_layer.weight.copy_(
            torch.from_numpy(coefficients.reshape(1, -1))
        )
        model.output_layer.bias.fill_(offset)
    return HomogeneousNeuralLyapunov(model, degree=degree)


def synthesize_homogeneous_neural_lyapunov(
    system,
    *,
    results_dir="results",
    width=50,
    direction_count=8192,
    validation_directions=262144,
    feature_scale=1.0,
    coefficient_bound=10.0,
    margin_fraction=0.1,
    maximum_fitted_margin=0.1,
    candidate_degree=1,
    overwrite=False,
    verbose=False,
):
    r"""Fit ``V=||x||^p exp(a^T tanh(W x/||x||+b))`` by two LPs.

    The first linear program maximizes the sampled decay margin. The second
    retains part of that margin while minimizing the coefficient one-norm.
    Dense validation is empirical only; dReal verification is separate.
    """

    if system.state_dimension < 2:
        raise ValueError(
            "Homogeneous neural synthesis requires at least two states."
        )
    if system.numerical_equilibrium is not None or any(
        sp.sympify(value).is_zero is not True
        for value in system.equilibrium
    ):
        raise ValueError("Homogeneous synthesis requires the exact origin.")
    vector_field_degree = homogeneous_degree(system)
    if vector_field_degree is None or vector_field_degree <= 0:
        raise ValueError(
            "The autonomous dynamics must be homogeneous of positive degree."
        )
    width = _positive_integer(width, "width")
    direction_count = _positive_integer(direction_count, "direction_count")
    validation_directions = _positive_integer(
        validation_directions, "validation_directions"
    )
    candidate_degree = _positive_integer(candidate_degree, "candidate_degree")
    feature_scale = _finite_positive(feature_scale, "feature_scale")
    coefficient_bound = _finite_positive(
        coefficient_bound, "coefficient_bound"
    )
    maximum_fitted_margin = _finite_positive(
        maximum_fitted_margin, "maximum_fitted_margin"
    )
    margin_fraction = float(margin_fraction)
    if not np.isfinite(margin_fraction) or not 0 < margin_fraction < 1:
        raise ValueError("margin_fraction must lie strictly between zero and one.")

    configuration = {
        "kind": "homogeneous-neural-convex-v2",
        "states": [str(state) for state in system.states],
        "dynamics": [
            sp.srepr(expression) for expression in system.symbolic.dynamics
        ],
        "vector_field_degree": str(vector_field_degree),
        "candidate_degree": candidate_degree,
        "width": width,
        "direction_count": direction_count,
        "validation_directions": validation_directions,
        "feature_scale": feature_scale,
        "coefficient_bound": coefficient_bound,
        "margin_fraction": margin_fraction,
        "maximum_fitted_margin": maximum_fitted_margin,
    }
    fingerprint = _configuration_fingerprint(configuration)
    directory = Path(results_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "{}-homogeneous-{}.pt".format(
        _safe_name(system), fingerprint
    )
    started = perf_counter()
    loaded = path.exists() and not overwrite
    if loaded:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        weights = checkpoint["weights"].numpy()
        biases = checkpoint["biases"].numpy()
        coefficients = checkpoint["coefficients"].numpy()
        offset = float(checkpoint["offset"])
        diagnostics = HomogeneousSynthesisDiagnostics(
            **checkpoint["diagnostics"]
        )
    else:
        directions = _unit_directions(system.state_dimension, direction_count)
        weights, biases = _feature_parameters(
            system.state_dimension, width, feature_scale
        )
        base, feature_rates, _ = _log_derivative_terms(
            system, directions, weights, biases, candidate_degree
        )
        optimal_margin = _maximum_margin(
            base, feature_rates, coefficient_bound
        )
        if optimal_margin <= 0:
            raise RuntimeError(
                "The fixed angular features admit no sampled Lyapunov "
                "function: optimal margin={:.6g}.".format(optimal_margin)
            )
        fitted_margin = min(
            margin_fraction * optimal_margin, maximum_fitted_margin
        )
        coefficients = _minimum_l1_coefficients(
            base,
            feature_rates,
            target_margin=fitted_margin,
            coefficient_bound=coefficient_bound,
        )
        (
            validation_maximum,
            validation_minimum,
            violations,
            log_minimum,
            log_maximum,
        ) = _validate_candidate(
            system,
            weights,
            biases,
            coefficients,
            degree=candidate_degree,
            count=validation_directions,
        )
        if violations:
            raise RuntimeError(
                "The homogeneous candidate failed empirical validation: "
                "{} of {} directions had a nonnegative derivative ratio."
                .format(violations, validation_directions)
            )
        offset = -log_minimum
        active = np.abs(coefficients) > 1e-12
        active_count = int(np.count_nonzero(active))
        if active_count == 0:
            active[0] = True
        weights = weights[active]
        biases = biases[active]
        coefficients = coefficients[active]
        diagnostics = HomogeneousSynthesisDiagnostics(
            optimal_margin=optimal_margin,
            fitted_margin=fitted_margin,
            validation_maximum_ratio=validation_maximum,
            validation_minimum_ratio=validation_minimum,
            validation_violations=violations,
            validation_directions=validation_directions,
            active_features=active_count,
            log_shape_range=log_maximum - log_minimum,
        )
        torch.save(
            {
                "configuration": configuration,
                "weights": torch.from_numpy(weights),
                "biases": torch.from_numpy(biases),
                "coefficients": torch.from_numpy(coefficients),
                "offset": offset,
                "diagnostics": asdict(diagnostics),
            },
            path,
        )
    candidate = _candidate(
        weights, biases, coefficients, offset, candidate_degree
    )
    if verbose:
        action = "loaded" if loaded else "synthesized"
        print("Homogeneous candidate {}: {}".format(action, path))
        print(
            "  sampled fitted margin: {:.6g}; validation maximum: {:.6g}"
            .format(
                diagnostics.fitted_margin,
                diagnostics.validation_maximum_ratio,
            )
        )
    return HomogeneousSynthesisResult(
        candidate=candidate,
        model_path=path,
        fingerprint=fingerprint,
        diagnostics=diagnostics,
        elapsed_time=perf_counter() - started,
        loaded=loaded,
    )


__all__ = [
    "HomogeneousSynthesisDiagnostics",
    "HomogeneousSynthesisResult",
    "synthesize_homogeneous_neural_lyapunov",
]
