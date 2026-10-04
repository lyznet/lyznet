"""Empirical level proposals for a neural Lyapunov funnel."""

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

import numpy as np
import sympy as sp
import torch

from .verification._crown import expression_graph, normalise_device


@dataclass(frozen=True)
class NeuralScreeningResult:
    """Sampled neural levels; this record is not formal evidence."""

    local_level: float
    outer_level: float
    containment_limit: float
    decrease_limit: float | None
    boundary_limit: float
    constraint_limit: float | None
    estimated_volume: float
    sample_count: int
    seed: int
    device: str

    @property
    def is_formal(self):
        return False


def _floor(value, precision):
    value = Decimal(str(value))
    quantum = Decimal(str(precision))
    return float(
        (value / quantum).to_integral_value(rounding=ROUND_FLOOR) * quantum
    )


def _candidate_values(candidate, points, batch_size):
    values = []
    gradients = []
    with torch.no_grad():
        for begin in range(0, len(points), batch_size):
            value, gradient = candidate.value_and_gradient(
                points[begin:begin + batch_size]
            )
            values.append(value.reshape(-1))
            gradients.append(gradient)
    return torch.cat(values), torch.cat(gradients)


def _box_samples(torch_device, bounds, count, seed):
    engine = torch.quasirandom.SobolEngine(
        len(bounds), scramble=True, seed=int(seed)
    )
    points = engine.draw(count, dtype=torch.float32).to(torch_device)
    lower = torch.tensor(
        [item[0] for item in bounds],
        dtype=torch.float32,
        device=torch_device,
    )
    upper = torch.tensor(
        [item[1] for item in bounds],
        dtype=torch.float32,
        device=torch_device,
    )
    return lower + points * (upper - lower)


def screen_neural_lyapunov(
    system,
    candidate,
    quadratic_set,
    *,
    sample_count=2**14,
    seed=20260823,
    device="auto",
    precision=1e-4,
    proof_margin=1e-6,
    batch_size=4096,
    state_constraints=(),
    viable_faces=(),
    state_constraint_band=1e-3,
    state_constraint_viability_margin=1e-5,
    face_viability_margin=1e-5,
):
    """Sample the same containment, decrease, and boundary predicates.

    The returned levels are conservative proposals for one subsequent formal
    CROWN call. No conclusion from this function is a proof.
    """

    if sample_count < 256:
        raise ValueError("sample_count must be at least 256.")
    torch_device = normalise_device(torch, device)
    state_constraints = tuple(
        sp.sympify(value) for value in state_constraints
    )
    viable_faces = set(viable_faces)
    model = deepcopy(candidate).to(torch_device).eval()
    bounds = system.domain.bounds
    points = _box_samples(torch_device, bounds, sample_count, seed)
    values, gradients = _candidate_values(model, points, batch_size)
    dynamics = expression_graph(
        torch, system.states, tuple(system.symbolic.dynamics)
    ).to(torch_device)
    with torch.no_grad():
        derivative = torch.sum(
            gradients * dynamics(points), dim=1
        )
        centre = torch.tensor(
            tuple(map(float, quadratic_set.certificate.centre)),
            dtype=torch.float32,
            device=torch_device,
        )
        matrix = torch.tensor(
            np.asarray(quadratic_set.certificate.matrix, dtype=float),
            dtype=torch.float32,
            device=torch_device,
        )
        displacement = points - centre
        quadratic_values = torch.sum(
            (displacement @ matrix) * displacement, dim=1
        )
        if state_constraints:
            constraint_graph = expression_graph(
                torch, system.states, state_constraints
            ).to(torch_device)
            constraint_values = constraint_graph(points)
            admissible = torch.all(constraint_values <= 1, dim=1)
        else:
            constraint_graph = None
            constraint_values = None
            admissible = torch.ones(
                len(points), dtype=torch.bool, device=torch_device
            )

    outside_bootstrap = quadratic_values >= (
        quadratic_set.level - proof_margin
    )
    if not bool(outside_bootstrap.any()):
        raise RuntimeError("Screening did not sample outside the quadratic core.")
    containment_limit = float(values[outside_bootstrap].min().cpu())
    local_level = _floor(
        max(precision, 0.9 * containment_limit), precision
    )
    local_level = max(precision, local_level)

    bad_decrease = (
        admissible
        & (values >= local_level)
        & (derivative >= -proof_margin)
    )
    decrease_limit = (
        float(values[bad_decrease].min().cpu())
        if bool(bad_decrease.any())
        else None
    )
    face_limits = []
    per_face = max(64, sample_count // (2 * system.state_dimension))
    for coordinate, bound in enumerate(bounds):
        for side_name, side in (("lower", bound[0]), ("upper", bound[1])):
            face = _box_samples(
                torch_device,
                bounds,
                per_face,
                seed + 100 + len(face_limits),
            )
            face[:, coordinate] = float(side)
            face_values, _ = _candidate_values(model, face, batch_size)
            with torch.no_grad():
                face_admissible = (
                    torch.all(constraint_graph(face) <= 1, dim=1)
                    if constraint_graph is not None
                    else torch.ones(
                        len(face), dtype=torch.bool, device=torch_device
                    )
                )
                if (coordinate, side_name) in viable_faces:
                    outward_sign = -1.0 if side_name == "lower" else 1.0
                    outward = outward_sign * dynamics(face)[:, coordinate]
                    face_failure = face_admissible & (
                        outward >= -face_viability_margin
                    )
                else:
                    face_failure = face_admissible
            face_limits.append(
                float(face_values[face_failure].min().cpu())
                if bool(face_failure.any())
                else float("inf")
            )
    boundary_limit = min(face_limits)

    constraint_limits = []
    for index, constraint in enumerate(state_constraints):
        lie_derivative = sp.expand(sum(
            sp.diff(constraint, state) * dynamics_expression
            for state, dynamics_expression in zip(
                system.states, system.symbolic.dynamics
            )
        ))
        lie_graph = expression_graph(
            torch, system.states, (lie_derivative,)
        ).to(torch_device)
        with torch.no_grad():
            lie_values = lie_graph(points).reshape(-1)
            boundary_failure = (
                admissible
                & (constraint_values[:, index] >= 1 - state_constraint_band)
                & (lie_values >= -state_constraint_viability_margin)
            )
        constraint_limits.append(
            float(values[boundary_failure].min().cpu())
            if bool(boundary_failure.any())
            else float("inf")
        )
    constraint_limit = (
        min(constraint_limits) if constraint_limits else None
    )
    observed_outer = min(
        boundary_limit,
        float("inf") if decrease_limit is None else decrease_limit,
        float("inf") if constraint_limit is None else constraint_limit,
        1.0 if candidate.architecture == "anchored" else float("inf"),
    )
    outer_level = _floor(max(local_level, 0.9 * observed_outer), precision)
    outer_level = max(local_level, outer_level)
    if candidate.architecture == "anchored":
        outer_level = min(outer_level, _floor(1 - 10 * precision, precision))
    fraction = float(
        (admissible & (values <= outer_level)).float().mean().cpu()
    )
    return NeuralScreeningResult(
        local_level=local_level,
        outer_level=outer_level,
        containment_limit=containment_limit,
        decrease_limit=decrease_limit,
        boundary_limit=boundary_limit,
        constraint_limit=constraint_limit,
        estimated_volume=system.domain.volume * fraction,
        sample_count=sample_count,
        seed=seed,
        device=torch_device,
    )


__all__ = ["NeuralScreeningResult", "screen_neural_lyapunov"]
