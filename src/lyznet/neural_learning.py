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
from .verification._crown import expression_graph, normalise_device


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


_EFFORTS = {
    "low": {
        "samples": 2048,
        "width": 20,
        "warmup_steps": 1000,
        "correction_steps": 1000,
        "collocation_points": 8192,
    },
    "mid": {
        "samples": 4096,
        "width": 30,
        "warmup_steps": 3000,
        "correction_steps": 3000,
        "collocation_points": 16384,
    },
    "high": {
        "samples": 8192,
        "width": 30,
        "warmup_steps": 6000,
        "correction_steps": 6000,
        "collocation_points": 32768,
    },
}


def effort_configuration(effort):
    try:
        return dict(_EFFORTS[effort])
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
    state_constraints=(),
    overwrite=False,
    verbose=False,
):
    r"""Generate ``(x,W(x))`` data by forward simulation.

    The target is

    ``W(x)=tanh(mu * integral_0^infinity ||phi(t,x)-x*||^2 dt)``.

    Trajectories that leave the computational box or reach ``W>=0.999`` are
    assigned the saturated target one. This is empirical training data, not
    verification evidence.
    """

    if system.domain is None:
        raise ValueError("Neural trajectory data require a bounded domain.")
    if sample_count < 16:
        raise ValueError("sample_count must be at least 16.")
    if step_size <= 0 or max_time <= 0 or transform_mu <= 0:
        raise ValueError("step_size, max_time, and transform_mu must be positive.")
    torch_device = normalise_device(torch, device)
    state_constraints = tuple(
        sp.sympify(value) for value in state_constraints
    )
    configuration = {
        "kind": "trajectory-data-v1",
        "system": _system_configuration(system),
        "sample_count": int(sample_count),
        "seed": int(seed),
        "step_size": float(step_size),
        "max_time": float(max_time),
        "transform_mu": float(transform_mu),
        "convergence_radius": float(convergence_radius),
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
    initial = _admissible_sobol(
        system,
        state_constraints,
        int(sample_count),
        seed,
        torch_device,
    )
    centre = torch.tensor(
        tuple(map(float, system.equilibrium)),
        dtype=dtype,
        device=torch_device,
    )
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
            running_cost = torch.sum(displacement.square(), dim=1)
            cost = cost + step_size * running_cost * active
            k1 = dynamics(states)
            k2 = dynamics(states + 0.5 * step_size * k1)
            k3 = dynamics(states + 0.5 * step_size * k2)
            k4 = dynamics(states + step_size * k3)
            next_states = states + step_size * (k1 + 2*k2 + 2*k3 + k4) / 6
            states = torch.where(active[:, None], next_states, states)
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
            transformed = torch.tanh(transform_mu * cost)
            newly_converged = active & near
            newly_saturated = active & ((~inside) | (transformed >= 0.999))
            converged |= newly_converged
            saturated |= newly_saturated
            active &= ~(newly_converged | newly_saturated)

        # A finite horizon cannot classify every point near a basin boundary.
        # Conservatively use the plateau label for unresolved trajectories.
        saturated |= active
        targets = torch.tanh(transform_mu * cost)
        targets[saturated] = 1.0
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
    transform_mu=0.1,
    anchor_scale=0.25,
    state_constraints=(),
    overwrite=False,
    verbose=False,
):
    """Fit trajectory values, then correct the Zubov PDE residual."""

    if architecture not in ("plain", "anchored"):
        raise ValueError("architecture must be 'plain' or 'anchored'.")
    torch_device = normalise_device(torch, device)
    state_constraints = tuple(
        sp.sympify(value) for value in state_constraints
    )
    configuration = {
        "kind": "neural-lyapunov-v1",
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
        "transform_mu": float(transform_mu),
        "anchor_scale": float(anchor_scale),
        "state_constraints": _constraint_configuration(state_constraints),
        "matrix": np.asarray(quadratic.matrix, dtype=float).tolist(),
    }
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
    ).to(torch_device)
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
        collocation = _admissible_sobol(
            system,
            state_constraints,
            int(collocation_points),
            int(seed) + 1,
            torch_device,
        )
        dynamics = expression_graph(
            torch, system.states, tuple(system.symbolic.dynamics)
        ).to(torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 2)
        if verbose:
            print(
                "PINN training: {} architecture, {}+{} updates on {}."
                .format(
                    architecture, warmup_steps, correction_steps, torch_device
                )
            )

        total_steps = int(warmup_steps) + int(correction_steps)
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            indices = torch.randint(
                len(states),
                (min(batch_size, len(states)),),
                generator=generator,
            ).to(torch_device)
            values, _ = _live_value_and_gradient(
                model,
                states[indices],
                architecture,
                centre,
                matrix,
                anchor_scale,
            )
            data_loss = (values - targets[indices]).square().mean()
            loss = data_loss
            if step >= warmup_steps:
                begin = (
                    (step - warmup_steps) * batch_size
                ) % len(collocation)
                pde_indices = (
                    torch.arange(batch_size, device=torch_device) + begin
                ) % len(collocation)
                points = collocation[pde_indices]
                pde_values, gradients = _live_value_and_gradient(
                    model,
                    points,
                    architecture,
                    centre,
                    matrix,
                    anchor_scale,
                )
                vector_field = dynamics(points)
                derivative = torch.sum(
                    gradients * vector_field, dim=1, keepdim=True
                )
                omega = torch.sum((points - centre).square(), dim=1, keepdim=True)
                residual = derivative + transform_mu * (
                    1 - pde_values.square()
                ) * omega
                pde_loss = residual.square().mean()
                positivity = torch.relu(-pde_values).square().mean()
                loss = data_loss + pde_loss + positivity
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            if verbose and (step + 1) % max(1, total_steps // 5) == 0:
                print(
                    "PINN training: {}/{}; loss={:.6g}."
                    .format(step + 1, total_steps, final_loss)
                )
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
    "TrajectoryData",
    "effort_configuration",
    "generate_trajectory_data",
    "train_neural_lyapunov",
]
