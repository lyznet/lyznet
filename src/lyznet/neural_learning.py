"""Deterministic trajectory data and PINN training for neural Lyapunov functions."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import sympy as sp
import torch

from .neural import (
    CenteredNeuralLyapunov,
    QuadraticAnchoredNeuralLyapunov,
    TanhNetwork,
)
from .neural_options import PINNCorrectionStage
from .verification._crown import expression_graph, normalise_device


_LOG_LATENT_SATURATION_LEVEL = float(
    np.log(np.arctanh(1.0 - 1e-5))
)


@dataclass(frozen=True)
class TrajectoryData:
    states: np.ndarray
    targets: np.ndarray
    path: Path
    fingerprint: str
    converged: int
    saturated: int


@dataclass(frozen=True)
class NeuralTrainingResult:
    candidate: object
    model_path: Path
    fingerprint: str
    elapsed_time: float
    loaded: bool
    final_loss: float


_PLAIN_EFFORTS = {
    "low": {
        "samples": 2048,
        "width": 30,
        "warmup_steps": 1000,
        "correction_steps": 1000,
        "collocation_points": 8192,
    },
    "mid": {
        "samples": 4096,
        "width": 30,
        "warmup_steps": 2000,
        "correction_steps": 2000,
        "collocation_points": 16384,
    },
    "high": {
        "samples": 8192,
        "width": 30,
        "warmup_steps": 3000,
        "correction_steps": 3000,
        "collocation_points": 32768,
    },
}


def _anchored_stages(multiplier):
    """Return the published anchored data/PDE/boundary schedule."""

    return (
        PINNCorrectionStage(
            steps=int(4_000 * multiplier),
            learning_rate=1e-3,
            pde_weight=0.0,
            decrease_weight=0.0,
            condition_source="uniform",
            condition_seed=987,
            data_batch_size=8_192,
        ),
        PINNCorrectionStage(
            steps=int(6_000 * multiplier),
            learning_rate=2e-5,
            data_weight=0.1,
            pde_weight=1.0,
            decrease_weight=80.0,
            condition_source="target_data",
            condition_target_bounds=(0.5, 0.99),
            data_batch_size=8_192,
        ),
        PINNCorrectionStage(
            steps=int(3_000 * multiplier),
            learning_rate=1e-5,
            data_weight=0.1,
            pde_weight=1.0,
            decrease_weight=80.0,
            boundary_weight=10.0,
            boundary_lower_bound=0.9,
            boundary_seed=654,
            condition_source="target_data",
            condition_target_bounds=(0.5, 0.99),
            data_batch_size=8_192,
        ),
    )


_ANCHORED_EFFORTS = {
    "low": {
        "samples": 3_000,
        "correction_samples": 100_000,
        "width": 30,
        "warmup_steps": 0,
        "correction_steps": 0,
        "correction_stages": _anchored_stages(1.0),
        "collocation_points": 300_000,
        "collocation_batch_size": 256,
        "correction_points": 2**19,
        "correction_batch_size": 4_096,
        "boundary_points": 2**14,
        "max_time": 100.0,
    },
    "mid": {
        "samples": 4_096,
        "correction_samples": 100_000,
        "width": 30,
        "warmup_steps": 0,
        "correction_steps": 0,
        "correction_stages": _anchored_stages(0.5),
        "collocation_points": 300_000,
        "collocation_batch_size": 256,
        "correction_points": 2**19,
        "correction_batch_size": 4_096,
        "boundary_points": 2**14,
        "max_time": 100.0,
    },
    "high": {
        "samples": 8_192,
        "correction_samples": 100_000,
        "width": 30,
        "warmup_steps": 0,
        "correction_steps": 0,
        "correction_stages": _anchored_stages(0.5),
        "collocation_points": 300_000,
        "collocation_batch_size": 256,
        "correction_points": 2**19,
        "correction_batch_size": 4_096,
        "boundary_points": 2**14,
        "max_time": 100.0,
    },
}


def effort_configuration(effort, architecture):
    configurations = (
        _ANCHORED_EFFORTS if architecture == "anchored" else _PLAIN_EFFORTS
    )
    try:
        return dict(configurations[effort])
    except KeyError as error:
        raise ValueError("effort must be 'low', 'mid', or 'high'.") from error


def _configuration_fingerprint(configuration):
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]


def _system_configuration(system):
    return {
        "states": [str(state) for state in system.states],
        "dynamics": [
            sp.srepr(expression) for expression in system.symbolic.dynamics
        ],
        "domain": list(system.domain.bounds),
        "equilibrium": [str(value) for value in system.equilibrium],
    }


def _constraint_configuration(state_constraints):
    return [sp.srepr(sp.sympify(value)) for value in state_constraints]


def _anchored_log_latent_data_loss(
    model, states, targets, centre, matrix, scale
):
    r"""Fit the unsaturated latent of an anchored candidate.

    For ``q=scale*(x-x*)^T P (x-x*)`` and
    ``H=q*exp(N(x)-N(x*))``, an interior label ``0<w<1`` specifies

        log(H) = log(atanh(w)).

    A saturated label ``w=1`` supplies the one-sided lower bound
    ``log(H)>=log(atanh(1-1e-5))``. The equilibrium label is imposed by the
    quadratic anchor and contributes no data error.
    """

    displacement = states - centre
    quadratic = scale * torch.sum(
        (displacement @ matrix) * displacement,
        dim=1,
        keepdim=True,
    )
    correction = model(states) - model(centre.reshape(1, -1))
    log_latent = torch.log(quadratic.clamp_min(1e-30)) + correction
    interior = (targets > 0) & (targets < 1)
    saturated = targets >= 1
    bounded_targets = targets.clamp(1e-7, 1 - 1e-7)
    target_log_latent = torch.log(torch.atanh(bounded_targets))
    interior_error = torch.where(
        interior,
        (log_latent - target_log_latent).square(),
        torch.zeros_like(log_latent),
    )
    saturation_error = torch.where(
        saturated,
        torch.relu(
            _LOG_LATENT_SATURATION_LEVEL - log_latent
        ).square(),
        torch.zeros_like(log_latent),
    )
    return (interior_error + saturation_error).mean()


def _admissible_sobol(
    system, state_constraints, count, seed, torch_device
):
    """Draw a deterministic Sobol prefix inside every ``h_i<=1``."""

    engine = torch.quasirandom.SobolEngine(
        system.state_dimension, scramble=True, seed=int(seed)
    )
    lower = torch.tensor(
        system.domain.lower, dtype=torch.float32, device=torch_device
    )
    upper = torch.tensor(
        system.domain.upper, dtype=torch.float32, device=torch_device
    )
    constraint_graph = (
        expression_graph(torch, system.states, state_constraints).to(
            torch_device
        )
        if state_constraints
        else None
    )
    accepted = []
    accepted_count = 0
    for _ in range(100):
        unit = engine.draw(max(256, count), dtype=torch.float32).to(
            torch_device
        )
        points = lower + unit * (upper - lower)
        if constraint_graph is not None:
            with torch.no_grad():
                admissible = torch.all(
                    constraint_graph(points) <= 1, dim=1
                )
            points = points[admissible]
        accepted.append(points)
        accepted_count += len(points)
        if accepted_count >= count:
            return torch.cat(accepted)[:count]
    raise RuntimeError(
        "Could not draw enough states satisfying all h_i<=1 constraints."
    )


def _result_directory(path):
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _box_boundary_points(bounds, count, seed, torch_device):
    """Sample all faces of a box with deterministic Sobol sequences."""

    dimension = len(bounds)
    per_face = max(2, int(np.ceil(count / (2 * dimension))))
    lows = torch.tensor(
        [pair[0] for pair in bounds],
        dtype=torch.float32,
        device=torch_device,
    )
    highs = torch.tensor(
        [pair[1] for pair in bounds],
        dtype=torch.float32,
        device=torch_device,
    )
    faces = []
    for coordinate, (lower, upper) in enumerate(bounds):
        for side in (lower, upper):
            engine = torch.quasirandom.SobolEngine(
                dimension, scramble=True, seed=int(seed) + len(faces)
            )
            unit = engine.draw(per_face, dtype=torch.float32).to(
                torch_device
            )
            points = lows + unit * (highs - lows)
            points[:, coordinate] = float(side)
            faces.append(points)
    return torch.cat(faces)[:count]


def _safe_name(system):
    name = system.name or "system"
    result = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name
    ).strip("_")
    return result or "system"


def generate_trajectory_data(
    system,
    *,
    sample_count,
    results_dir,
    device="auto",
    seed=20260823,
    step_size=0.02,
    max_time=30.0,
    transform_mu=0.1,
    convergence_radius=1e-3,
    cost_limit=200.0,
    sampling="sobol",
    include_equilibrium=True,
    terminate_on_domain_exit=True,
    require_termination=False,
    state_constraints=(),
    overwrite=False,
    verbose=False,
):
    r"""Generate ``(x,W(x))`` data by forward simulation.

    The target is

    ``W(x)=tanh(mu * integral_0^infinity ||phi(t,x)-x*||^2 dt)``.

    By default, trajectories that leave the computational box or accumulate
    ``cost_limit`` units of running cost receive the saturated target one.
    Domain-exit termination can be disabled when the box is only a sampling
    domain. This is empirical training data, not verification evidence.
    """

    if system.domain is None:
        raise ValueError("Neural trajectory data require a bounded domain.")
    if sample_count < 16:
        raise ValueError("sample_count must be at least 16.")
    if step_size <= 0 or max_time <= 0 or transform_mu <= 0:
        raise ValueError("step_size, max_time, and transform_mu must be positive.")
    if cost_limit <= 0:
        raise ValueError("cost_limit must be positive.")
    if sampling not in ("sobol", "random"):
        raise ValueError("sampling must be 'sobol' or 'random'.")
    torch_device = normalise_device(torch, device)
    state_constraints = tuple(
        sp.sympify(value) for value in state_constraints
    )
    configuration = {
        "kind": "trajectory-data-v2",
        "system": _system_configuration(system),
        "sample_count": int(sample_count),
        "seed": int(seed),
        "step_size": float(step_size),
        "max_time": float(max_time),
        "transform_mu": float(transform_mu),
        "convergence_radius": float(convergence_radius),
        "cost_limit": float(cost_limit),
        "sampling": sampling,
        "include_equilibrium": bool(include_equilibrium),
        "terminate_on_domain_exit": bool(terminate_on_domain_exit),
        "require_termination": bool(require_termination),
        "state_constraints": _constraint_configuration(state_constraints),
    }
    fingerprint = _configuration_fingerprint(configuration)
    path = _result_directory(results_dir) / (
        "{}-trajectory-{}.npz".format(_safe_name(system), fingerprint)
    )
    if path.exists() and not overwrite:
        stored = np.load(path)
        result = TrajectoryData(
            states=stored["states"],
            targets=stored["targets"],
            path=path,
            fingerprint=fingerprint,
            converged=int(stored["converged"]),
            saturated=int(stored["saturated"]),
        )
        if verbose:
            print(
                "Trajectory data: loaded {} samples from {} (device: {})."
                .format(len(result.states), path, torch_device)
            )
        return result

    dtype = torch.float32
    lower = torch.tensor(
        system.domain.lower, dtype=dtype, device=torch_device
    )
    upper = torch.tensor(
        system.domain.upper, dtype=dtype, device=torch_device
    )
    if sampling == "sobol":
        initial = _admissible_sobol(
            system,
            state_constraints,
            int(sample_count),
            seed,
            torch_device,
        )
    else:
        generator = np.random.default_rng(int(seed))
        constraint_graph = (
            expression_graph(torch, system.states, state_constraints).to(
                torch_device
            )
            if state_constraints
            else None
        )
        accepted = []
        accepted_count = 0
        while accepted_count < sample_count:
            values = generator.uniform(
                np.asarray(system.domain.lower, dtype=float),
                np.asarray(system.domain.upper, dtype=float),
                size=(max(sample_count, 256), system.state_dimension),
            )
            points = torch.as_tensor(
                values, dtype=dtype, device=torch_device
            )
            if constraint_graph is not None:
                with torch.no_grad():
                    points = points[
                        torch.all(constraint_graph(points) <= 1, dim=1)
                    ]
            accepted.append(points)
            accepted_count += len(points)
        initial = torch.cat(accepted)[:sample_count]
    centre = torch.tensor(
        tuple(map(float, system.equilibrium)),
        dtype=dtype,
        device=torch_device,
    )
    if include_equilibrium:
        initial[0] = centre
    states = initial.clone()
    cost = torch.zeros(sample_count, dtype=dtype, device=torch_device)
    active = torch.ones(sample_count, dtype=torch.bool, device=torch_device)
    converged = torch.zeros_like(active)
    saturated = torch.zeros_like(active)
    dynamics = expression_graph(
        torch, system.states, tuple(system.symbolic.dynamics)
    ).to(torch_device)
    constraint_graph = (
        expression_graph(torch, system.states, state_constraints).to(
            torch_device
        )
        if state_constraints
        else None
    )
    steps = int(np.ceil(max_time / step_size))
    if verbose:
        print(
            "Trajectory data: generating {} samples on {}."
            .format(sample_count, torch_device)
        )
    with torch.no_grad():
        for _ in range(steps):
            if not bool(active.any()):
                break
            displacement = states - centre
            k1 = dynamics(states)
            q1 = torch.sum(displacement.square(), dim=1)
            stage2 = states + 0.5 * step_size * k1
            k2 = dynamics(stage2)
            q2 = torch.sum((stage2 - centre).square(), dim=1)
            stage3 = states + 0.5 * step_size * k2
            k3 = dynamics(stage3)
            q3 = torch.sum((stage3 - centre).square(), dim=1)
            stage4 = states + step_size * k3
            k4 = dynamics(stage4)
            q4 = torch.sum((stage4 - centre).square(), dim=1)
            next_states = states + step_size * (k1 + 2*k2 + 2*k3 + k4) / 6
            next_cost = cost + step_size * (q1 + 2*q2 + 2*q3 + q4) / 6
            states = torch.where(active[:, None], next_states, states)
            cost = torch.where(active, next_cost, cost)
            inside = torch.all(
                (states >= lower) & (states <= upper), dim=1
            )
            if constraint_graph is not None:
                inside &= torch.all(
                    constraint_graph(states) <= 1, dim=1
                )
            near = torch.linalg.vector_norm(states - centre, dim=1) <= (
                convergence_radius
            )
            newly_converged = active & near
            outside = (~inside) if terminate_on_domain_exit else False
            newly_saturated = active & (outside | (cost >= cost_limit))
            converged |= newly_converged
            saturated |= newly_saturated
            active &= ~(newly_converged | newly_saturated)
            states = torch.where(active[:, None], states, centre)

        unresolved = int(active.sum().item())
        if require_termination and unresolved:
            raise RuntimeError(
                "{} trajectories were unresolved at max_time={}."
                .format(unresolved, max_time)
            )
        # A finite horizon cannot classify every point near a basin boundary.
        # Conservatively use the plateau label for unresolved trajectories.
        saturated |= active
        targets = torch.tanh(transform_mu * cost)
        targets[saturated] = 1.0
        if include_equilibrium:
            targets[0] = 0.0

    state_array = initial.cpu().numpy()
    target_array = targets.cpu().numpy()
    np.savez_compressed(
        path,
        states=state_array,
        targets=target_array,
        converged=int(converged.sum().item()),
        saturated=int(saturated.sum().item()),
    )
    result = TrajectoryData(
        states=state_array,
        targets=target_array,
        path=path,
        fingerprint=fingerprint,
        converged=int(converged.sum().item()),
        saturated=int(saturated.sum().item()),
    )
    if verbose:
        print(
            "Trajectory data: {} converged, {} saturated; saved to {}."
            .format(result.converged, result.saturated, path)
        )
    return result


def _live_value_and_gradient(model, states, architecture, centre, matrix, scale):
    values, gradients = model.value_and_gradient(states)
    centre_values, centre_gradients = model.value_and_gradient(
        centre.reshape(1, -1)
    )
    if architecture == "plain":
        displacement = states - centre
        affine = torch.sum(
            displacement * centre_gradients, dim=1, keepdim=True
        )
        return values - centre_values - affine, gradients - centre_gradients

    displacement = states - centre
    quadratic = torch.sum(
        (displacement @ matrix) * displacement, dim=1, keepdim=True
    )
    quadratic_gradient = 2 * displacement @ matrix
    exponential = torch.exp(values - centre_values)
    latent = scale * quadratic * exponential
    latent_gradient = scale * exponential * (
        quadratic_gradient + quadratic * gradients
    )
    transformed = torch.tanh(latent)
    return transformed, (1 - transformed.square()) * latent_gradient


def train_neural_lyapunov(
    system,
    quadratic,
    data,
    *,
    architecture,
    results_dir,
    device="auto",
    seed=20260823,
    width=20,
    hidden_layers=2,
    warmup_steps=1000,
    correction_steps=1000,
    collocation_points=8192,
    batch_size=512,
    data_batch_size=512,
    collocation_batch_size=None,
    correction_points=None,
    correction_batch_size=None,
    correction_data=None,
    correction_stages=(),
    warmup_learning_rate=1e-3,
    correction_learning_rate=1e-3,
    warmup_data_weight=1.0,
    data_weight=1.0,
    pde_weight=1.0,
    positivity_weight=1.0,
    decrease_weight=0.0,
    decrease_margin=1e-5,
    positivity_margin=1e-5,
    correction_level_bounds=None,
    range_weight=0.0,
    plateau_gradient_weight=0.0,
    plateau_gradient_level=None,
    warmup_range_weight=0.0,
    warmup_plateau_gradient_weight=0.0,
    boundary_points=1024,
    boundary_weight=0.0,
    warmup_boundary_weight=0.0,
    compile_training=False,
    sampling_seed=None,
    report_every=1000,
    transform_mu=0.1,
    anchor_scale=0.25,
    quadratic_anchored_data_loss="value",
    state_constraints=(),
    initial_model_path=None,
    overwrite=False,
    verbose=False,
):
    """Fit trajectory values, then correct the PDE and LF inequalities.

    Correction directly penalizes sampled violations of

    ``W(x) >= positivity_margin*||x-x*||^2`` and
    ``grad(W)(x) f(x) <= -decrease_margin``.

    These losses shape a candidate only. Formal verification remains an
    independent subsequent stage.
    """

    if architecture not in ("plain", "anchored"):
        raise ValueError("architecture must be 'plain' or 'anchored'.")
    if quadratic_anchored_data_loss not in ("value", "log_latent"):
        raise ValueError(
            "quadratic_anchored_data_loss must be 'value' or 'log_latent'."
        )
    if architecture != "anchored" and (
        quadratic_anchored_data_loss != "value"
    ):
        raise ValueError(
            "quadratic_anchored_data_loss='log_latent' requires "
            "architecture='anchored'."
        )
    torch_device = normalise_device(torch, device)
    state_constraints = tuple(
        sp.sympify(value) for value in state_constraints
    )
    initial_checkpoint = None
    initial_model_fingerprint = None
    if initial_model_path is not None:
        initial_model_path = Path(initial_model_path).expanduser().resolve()
        if not initial_model_path.is_file():
            raise FileNotFoundError(
                "Initial neural checkpoint not found: {}"
                .format(initial_model_path)
            )
        initial_checkpoint = torch.load(
            initial_model_path, map_location="cpu", weights_only=True
        )
        if "state_dict" not in initial_checkpoint or (
            "configuration" not in initial_checkpoint
        ):
            raise ValueError(
                "Initial neural checkpoint lacks training metadata: {}"
                .format(initial_model_path)
            )
        initial_model_fingerprint = _configuration_fingerprint(
            initial_checkpoint["configuration"]
        )
    configuration = {
        "kind": "neural-lyapunov-v4",
        "system": _system_configuration(system),
        "data": data.fingerprint,
        "architecture": architecture,
        "seed": int(seed),
        "width": int(width),
        "hidden_layers": int(hidden_layers),
        "warmup_steps": int(warmup_steps),
        "correction_steps": int(correction_steps),
        "collocation_points": int(collocation_points),
        "batch_size": int(batch_size),
        "data_batch_size": data_batch_size,
        "collocation_batch_size": collocation_batch_size,
        "correction_points": correction_points,
        "correction_batch_size": correction_batch_size,
        "correction_data": (
            None if correction_data is None else correction_data.fingerprint
        ),
        "correction_stages": [stage.__dict__ for stage in correction_stages],
        "warmup_learning_rate": float(warmup_learning_rate),
        "correction_learning_rate": float(correction_learning_rate),
        "warmup_data_weight": float(warmup_data_weight),
        "data_weight": float(data_weight),
        "pde_weight": float(pde_weight),
        "positivity_weight": float(positivity_weight),
        "decrease_weight": float(decrease_weight),
        "decrease_margin": float(decrease_margin),
        "positivity_margin": float(positivity_margin),
        "correction_level_bounds": correction_level_bounds,
        "range_weight": float(range_weight),
        "plateau_gradient_weight": float(plateau_gradient_weight),
        "plateau_gradient_level": plateau_gradient_level,
        "warmup_range_weight": float(warmup_range_weight),
        "warmup_plateau_gradient_weight": float(
            warmup_plateau_gradient_weight
        ),
        "boundary_points": int(boundary_points),
        "boundary_weight": float(boundary_weight),
        "warmup_boundary_weight": float(warmup_boundary_weight),
        "compile_training": bool(compile_training),
        "sampling_seed": sampling_seed,
        "transform_mu": float(transform_mu),
        "anchor_scale": float(anchor_scale),
        "quadratic_anchored_data_loss": quadratic_anchored_data_loss,
        "state_constraints": _constraint_configuration(state_constraints),
        "matrix": np.asarray(quadratic.matrix, dtype=float).tolist(),
    }
    if initial_model_fingerprint is not None:
        configuration["initial_model"] = initial_model_fingerprint
    fingerprint = _configuration_fingerprint(configuration)
    path = _result_directory(results_dir) / (
        "{}-pinn-{}-{}.pt".format(
            _safe_name(system), architecture, fingerprint
        )
    )
    torch.manual_seed(int(seed))
    model = TanhNetwork(
        system.state_dimension,
        hidden_layers=hidden_layers,
        width=width,
        device=torch_device,
        dtype=torch.float32,
    )
    loaded = path.exists() and not overwrite
    started = perf_counter()
    final_loss = float("nan")
    if loaded:
        checkpoint = torch.load(
            path, map_location=torch_device, weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        final_loss = float(checkpoint["final_loss"])
        if verbose:
            print("PINN training: loaded {} (device: {}).".format(path, torch_device))
    else:
        if initial_checkpoint is not None:
            try:
                model.load_state_dict(initial_checkpoint["state_dict"])
            except RuntimeError as error:
                raise ValueError(
                    "The initial checkpoint architecture does not match "
                    "the requested neural model."
                ) from error
            if verbose:
                print(
                    "PINN training: continuing from {}."
                    .format(initial_model_path)
                )
        states = torch.tensor(
            data.states, dtype=torch.float32, device=torch_device
        )
        targets = torch.tensor(
            data.targets, dtype=torch.float32, device=torch_device
        ).reshape(-1, 1)
        centre = torch.tensor(
            tuple(map(float, system.equilibrium)),
            dtype=torch.float32,
            device=torch_device,
        )
        matrix = torch.tensor(
            np.asarray(quadratic.matrix, dtype=float),
            dtype=torch.float32,
            device=torch_device,
        )
        sampling_seed = int(seed) if sampling_seed is None else int(sampling_seed)
        collocation = _admissible_sobol(
            system,
            state_constraints,
            int(collocation_points),
            sampling_seed + 1009,
            torch_device,
        )
        correction_source = data if correction_data is None else correction_data
        correction_states = torch.tensor(
            correction_source.states,
            dtype=torch.float32,
            device=torch_device,
        )
        correction_targets = torch.tensor(
            correction_source.targets,
            dtype=torch.float32,
            device=torch_device,
        ).reshape(-1, 1)
        correction_points = (
            int(collocation_points)
            if correction_points is None
            else int(correction_points)
        )
        correction_bank = _admissible_sobol(
            system,
            state_constraints,
            correction_points,
            sampling_seed + 2017,
            torch_device,
        )
        boundary = _box_boundary_points(
            system.domain.bounds,
            int(boundary_points),
            sampling_seed + 3011,
            torch_device,
        )
        primary_plateau_states = states[targets.reshape(-1) == 1]
        correction_plateau_states = correction_states[
            correction_targets.reshape(-1) == 1
        ]
        plateau_states = (
            correction_plateau_states
            if correction_data is None
            else torch.cat(
                (primary_plateau_states, correction_plateau_states), dim=0
            )
        )
        dynamics = expression_graph(
            torch, system.states, tuple(system.symbolic.dynamics)
        ).to(torch_device)
        stages = tuple(correction_stages)
        if not stages and correction_steps:
            stages = (
                PINNCorrectionStage(
                    steps=int(correction_steps),
                    learning_rate=float(correction_learning_rate),
                    data_weight=float(data_weight),
                    pde_weight=float(pde_weight),
                    decrease_weight=float(decrease_weight),
                    boundary_weight=float(boundary_weight),
                ),
            )
        optimizer_arguments = {"lr": float(warmup_learning_rate)}
        if torch_device == "cuda":
            optimizer_arguments["fused"] = True
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_arguments)
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 2)
        if verbose:
            print(
                "PINN training: {} architecture, {}+{} updates on {}."
                .format(
                    architecture,
                    warmup_steps,
                    sum(stage.steps for stage in stages),
                    torch_device,
                )
            )

        collocation_batch_size = (
            int(batch_size)
            if collocation_batch_size is None
            else int(collocation_batch_size)
        )
        correction_batch_size = (
            collocation_batch_size
            if correction_batch_size is None
            else int(correction_batch_size)
        )
        compiled = bool(compile_training and torch_device == "cuda")
        if compiled:
            torch.compiler.reset()

        def cyclic(bank, size, step):
            start = step * int(size)
            indices = (
                torch.arange(int(size), device=torch_device) + start
            ) % len(bank)
            return bank[indices]

        def fit_batch(data_states, data_targets, size, step, *, random):
            if size is None:
                return data_states, data_targets
            if random:
                indices = torch.randint(
                    len(data_states),
                    (min(int(size), len(data_states)),),
                    generator=generator,
                ).to(torch_device)
                return data_states[indices], data_targets[indices]
            return (
                cyclic(data_states, int(size), step),
                cyclic(data_targets, int(size), step),
            )

        zero = torch.zeros((), dtype=torch.float32, device=torch_device)

        def objective_terms(
            fit_states,
            fit_targets,
            pde_states,
            condition_states,
            plateau_batch,
            boundary_batch,
            *,
            include_conditions,
            selected_pde_weight,
            selected_data_weight,
            selected_decrease_weight,
            selected_boundary_weight,
            selected_range_weight,
            selected_plateau_weight,
            boundary_lower_bound,
        ):
            if quadratic_anchored_data_loss == "log_latent":
                data_loss = _anchored_log_latent_data_loss(
                    model,
                    fit_states,
                    fit_targets,
                    centre,
                    matrix,
                    anchor_scale,
                )
            else:
                values, _ = _live_value_and_gradient(
                    model,
                    fit_states,
                    architecture,
                    centre,
                    matrix,
                    anchor_scale,
                )
                data_loss = (values - fit_targets).square().mean()
            pde_loss = zero
            range_loss = zero
            if selected_pde_weight != 0 or selected_range_weight != 0:
                pde_values, pde_gradients = _live_value_and_gradient(
                    model,
                    pde_states,
                    architecture,
                    centre,
                    matrix,
                    anchor_scale,
                )
                if selected_pde_weight != 0:
                    derivative = torch.sum(
                        pde_gradients * dynamics(pde_states),
                        dim=1,
                        keepdim=True,
                    )
                    omega = torch.sum(
                        (pde_states - centre).square(),
                        dim=1,
                        keepdim=True,
                    )
                    residual = derivative + transform_mu * (
                        1 - pde_values.square()
                    ) * omega
                    pde_loss = residual.square().mean()
                if selected_range_weight != 0:
                    range_loss = (
                        torch.relu(-pde_values).square()
                        + torch.relu(pde_values - 1).square()
                    ).mean()

            positivity_loss = zero
            decrease_loss = zero
            if include_conditions:
                condition_values, condition_gradients = (
                    _live_value_and_gradient(
                        model,
                        condition_states,
                        architecture,
                        centre,
                        matrix,
                        anchor_scale,
                    )
                )
                radius_squared = torch.sum(
                    (condition_states - centre).square(), dim=1
                )
                positivity_loss = torch.relu(
                    positivity_margin * radius_squared
                    - condition_values.reshape(-1)
                ).mean()
                if selected_decrease_weight != 0:
                    condition_derivative = torch.sum(
                        condition_gradients * dynamics(condition_states),
                        dim=1,
                    )
                    active = torch.ones_like(
                        condition_derivative, dtype=torch.bool
                    )
                    if correction_level_bounds is not None:
                        lower_level, upper_level = correction_level_bounds
                        detached = condition_values.detach().reshape(-1)
                        active &= detached >= lower_level
                        active &= detached <= upper_level
                    violations = torch.relu(
                        condition_derivative + decrease_margin
                    )
                    weights = active.to(violations.dtype)
                    decrease_loss = (
                        (violations * weights).sum()
                        / weights.sum().clamp_min(1)
                    )

            plateau_loss = zero
            if selected_plateau_weight != 0:
                plateau_values, plateau_gradients = _live_value_and_gradient(
                    model,
                    plateau_batch,
                    architecture,
                    centre,
                    matrix,
                    anchor_scale,
                )
                gradient_error = plateau_gradients.square().sum(dim=1)
                if plateau_gradient_level is None:
                    plateau_loss = gradient_error.mean()
                else:
                    active = (
                        plateau_values.detach().reshape(-1)
                        >= plateau_gradient_level
                    )
                    weights = active.to(gradient_error.dtype)
                    plateau_loss = (
                        (gradient_error * weights).sum()
                        / weights.sum().clamp_min(1)
                    )

            boundary_loss = zero
            if selected_boundary_weight != 0:
                boundary_values, boundary_gradients = (
                    _live_value_and_gradient(
                        model,
                        boundary_batch,
                        architecture,
                        centre,
                        matrix,
                        anchor_scale,
                    )
                )
                if boundary_lower_bound is None:
                    boundary_loss = (boundary_values - 1).square().mean()
                    boundary_loss = boundary_loss + (
                        plateau_gradient_weight
                        * boundary_gradients.square().sum(dim=1).mean()
                    )
                else:
                    boundary_loss = torch.relu(
                        boundary_lower_bound - boundary_values
                    ).square().mean()
            return (
                selected_data_weight * data_loss
                + selected_pde_weight * pde_loss
                + positivity_weight * positivity_loss
                + selected_decrease_weight * decrease_loss
                + selected_boundary_weight * boundary_loss
                + selected_range_weight * range_loss
                + selected_plateau_weight * plateau_loss
            )

        def warmup_objective(
            fit_states, fit_targets, pde_states, plateau_batch
        ):
            return objective_terms(
                fit_states,
                fit_targets,
                pde_states,
                pde_states,
                plateau_batch,
                boundary,
                include_conditions=False,
                selected_pde_weight=float(pde_weight),
                selected_data_weight=float(warmup_data_weight),
                selected_decrease_weight=0.0,
                selected_boundary_weight=float(warmup_boundary_weight),
                selected_range_weight=float(warmup_range_weight),
                selected_plateau_weight=float(
                    warmup_plateau_gradient_weight
                ),
                boundary_lower_bound=None,
            )

        warmup_loss = (
            torch.compile(
                warmup_objective, fullgraph=True, mode="reduce-overhead"
            )
            if compiled
            else warmup_objective
        )
        dummy_plateau = centre.reshape(1, -1)
        for step in range(int(warmup_steps)):
            optimizer.zero_grad(set_to_none=True)
            fit_states, fit_targets = fit_batch(
                states, targets, data_batch_size, step, random=False
            )
            points = cyclic(collocation, collocation_batch_size, step)
            plateau_batch = (
                plateau_states
                if data_batch_size is None and len(plateau_states)
                else (
                    cyclic(
                        plateau_states,
                        min(int(data_batch_size), len(plateau_states)),
                        step,
                    )
                    if len(plateau_states)
                    else dummy_plateau
                )
            )
            loss = warmup_loss(
                fit_states, fit_targets, points, plateau_batch
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if verbose and (step + 1) % max(1, int(report_every)) == 0:
                print(
                    "PINN warmup: {}/{}; loss={:.6g}."
                    .format(step + 1, warmup_steps, final_loss)
                )

        for stage_index, stage in enumerate(stages, start=1):
            optimizer_arguments = {"lr": float(stage.learning_rate)}
            if torch_device == "cuda":
                optimizer_arguments["fused"] = True
            optimizer = torch.optim.Adam(
                model.parameters(), **optimizer_arguments
            )
            if stage.condition_source == "target_data":
                lower_target, upper_target = stage.condition_target_bounds
                selected = (
                    (correction_targets.reshape(-1) >= lower_target)
                    & (correction_targets.reshape(-1) <= upper_target)
                )
                condition_bank = correction_states[selected]
                if not len(condition_bank):
                    raise ValueError(
                        "Correction stage selected no target-data states."
                    )
            else:
                condition_seed = (
                    sampling_seed + 2017
                    if stage.condition_seed is None
                    else int(stage.condition_seed)
                )
                condition_bank = (
                    correction_bank
                    if condition_seed == sampling_seed + 2017
                    else _admissible_sobol(
                        system,
                        state_constraints,
                        correction_points,
                        condition_seed,
                        torch_device,
                    )
                )
            stage_boundary = (
                boundary
                if stage.boundary_seed is None
                else _box_boundary_points(
                    system.domain.bounds,
                    int(boundary_points),
                    int(stage.boundary_seed),
                    torch_device,
                )
            )

            def correction_objective(
                fit_states,
                fit_targets,
                pde_states,
                condition_states,
                plateau_batch,
            ):
                return objective_terms(
                    fit_states,
                    fit_targets,
                    pde_states,
                    condition_states,
                    plateau_batch,
                    stage_boundary,
                    include_conditions=True,
                    selected_pde_weight=float(stage.pde_weight),
                    selected_data_weight=float(stage.data_weight),
                    selected_decrease_weight=float(stage.decrease_weight),
                    selected_boundary_weight=float(stage.boundary_weight),
                    selected_range_weight=float(range_weight),
                    selected_plateau_weight=float(
                        plateau_gradient_weight
                    ),
                    boundary_lower_bound=stage.boundary_lower_bound,
                )

            if compiled:
                torch.cuda.synchronize(torch_device)
                torch.compiler.reset()
                correction_loss = torch.compile(
                    correction_objective,
                    fullgraph=True,
                    mode="reduce-overhead",
                )
            else:
                correction_loss = correction_objective
            if verbose:
                print(
                    "PINN correction stage {}/{}: {} updates."
                    .format(stage_index, len(stages), stage.steps)
                )
            for step in range(stage.steps):
                optimizer.zero_grad(set_to_none=True)
                fit_states, fit_targets = fit_batch(
                    correction_states,
                    correction_targets,
                    stage.data_batch_size,
                    step,
                    random=True,
                )
                condition_batch = cyclic(
                    condition_bank,
                    min(correction_batch_size, len(condition_bank)),
                    step,
                )
                pde_batch = (
                    condition_batch
                    if stage.condition_source == "target_data"
                    else cyclic(
                        collocation, collocation_batch_size, step
                    )
                )
                plateau_batch = (
                    plateau_states
                    if stage.data_batch_size is None and len(plateau_states)
                    else (
                        cyclic(
                            plateau_states,
                            min(
                                int(stage.data_batch_size),
                                len(plateau_states),
                            ),
                            step,
                        )
                        if len(plateau_states)
                        else dummy_plateau
                    )
                )
                loss = correction_loss(
                    fit_states,
                    fit_targets,
                    pde_batch,
                    condition_batch,
                    plateau_batch,
                )
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())
                if verbose and (step + 1) % max(1, int(report_every)) == 0:
                    print(
                        "PINN correction stage {}/{}: {}/{}; loss={:.6g}."
                        .format(
                            stage_index,
                            len(stages),
                            step + 1,
                            stage.steps,
                            final_loss,
                        )
                    )
        if torch_device == "cuda":
            torch.cuda.synchronize(torch_device)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "configuration": configuration,
                "final_loss": final_loss,
            },
            path,
        )
        if verbose:
            print("PINN training: saved {}.".format(path))

    model.eval()
    if architecture == "plain":
        candidate = CenteredNeuralLyapunov(
            model, tuple(map(float, system.equilibrium))
        )
    else:
        candidate = QuadraticAnchoredNeuralLyapunov(
            model,
            tuple(map(float, system.equilibrium)),
            quadratic.matrix,
            scale=anchor_scale,
        )
    return NeuralTrainingResult(
        candidate=candidate,
        model_path=path,
        fingerprint=fingerprint,
        elapsed_time=perf_counter() - started,
        loaded=loaded,
        final_loss=final_loss,
    )


__all__ = [
    "NeuralTrainingResult",
    "PINNCorrectionStage",
    "TrajectoryData",
    "effort_configuration",
    "generate_trajectory_data",
    "train_neural_lyapunov",
]
