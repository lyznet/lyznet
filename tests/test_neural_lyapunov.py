"""Focused tests for neural candidates, artifacts, and proof orchestration."""

from pathlib import Path

import numpy as np
import pytest
import sympy as sp


torch = pytest.importorskip("torch")

import lyznet as lz
from lyznet.neural import (
    CenteredNeuralLyapunov,
    QuadraticAnchoredNeuralLyapunov,
    TanhNetwork,
)
from lyznet.neural_assessment import screen_neural_lyapunov
from lyznet.neural_learning import (
    TrajectoryData,
    generate_trajectory_data,
    train_neural_lyapunov,
)
from lyznet.verification import neural_crown as adapter


def stable_linear_system():
    state = sp.symbols("x")
    return lz.System.continuous(
        states=(state,),
        dynamics=(-state,),
        domain=((-1.0, 1.0),),
        name="stable_linear",
    )


@pytest.mark.parametrize("architecture", ("plain", "anchored"))
def test_neural_candidate_gradient_matches_autograd(architecture):
    torch.manual_seed(1)
    model = TanhNetwork(2, hidden_layers=2, width=5)
    candidate = (
        CenteredNeuralLyapunov(model, (0.0, 0.0))
        if architecture == "plain"
        else QuadraticAnchoredNeuralLyapunov(
            model, (0.0, 0.0), np.eye(2)
        )
    )
    points = torch.randn(8, 2, requires_grad=True)
    values, analytic = candidate.value_and_gradient(points)
    automatic = torch.autograd.grad(candidate(points).sum(), points)[0]

    assert torch.allclose(analytic, automatic, atol=1e-6)
    assert candidate.value(np.zeros((1, 2)))[0] == pytest.approx(0.0)
    if architecture == "anchored":
        assert bool(torch.all(values >= 0))


def test_trajectory_data_uses_a_deterministic_fingerprint(tmp_path):
    options = dict(
        sample_count=16,
        results_dir=tmp_path,
        device="cpu",
        max_time=0.02,
        step_size=0.02,
    )
    first = generate_trajectory_data(stable_linear_system(), **options)
    second = generate_trajectory_data(stable_linear_system(), **options)

    assert first.path == second.path
    assert first.fingerprint == second.fingerprint
    assert np.array_equal(first.states, second.states)
    assert first.path.exists()


def test_training_reuses_only_the_matching_checkpoint(tmp_path):
    system = stable_linear_system()
    quadratic = lz.quadratic_lyapunov(system)
    states = np.linspace(-1, 1, 32, dtype=np.float32).reshape(-1, 1)
    data = TrajectoryData(
        states=states,
        targets=np.tanh(0.05 * states[:, 0] ** 2),
        path=Path(tmp_path) / "data.npz",
        fingerprint="fixed-data",
        converged=32,
        saturated=0,
    )
    options = dict(
        architecture="anchored",
        results_dir=tmp_path,
        device="cpu",
        width=4,
        warmup_steps=1,
        correction_steps=1,
        collocation_points=32,
        batch_size=16,
    )
    first = train_neural_lyapunov(system, quadratic, data, **options)
    second = train_neural_lyapunov(system, quadratic, data, **options)

    assert not first.loaded
    assert second.loaded
    assert first.model_path == second.model_path
    assert (
        first.candidate.proof_fingerprint()
        == second.candidate.proof_fingerprint()
    )


def test_neural_screening_is_explicitly_nonformal():
    system = stable_linear_system()
    quadratic = lz.quadratic_lyapunov(system)
    quadratic_set = lz.SublevelSet(quadratic, 0.1)
    candidate = QuadraticAnchoredNeuralLyapunov(
        TanhNetwork(1, hidden_layers=1, width=3),
        (0.0,),
        quadratic.matrix,
    )

    result = screen_neural_lyapunov(
        system,
        candidate,
        quadratic_set,
        sample_count=256,
        device="cpu",
    )

    assert not result.is_formal
    assert 0 < result.local_level <= result.outer_level < 1


def test_neural_crown_reports_the_two_stage_claim(monkeypatch):
    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def solve(self, name, model, output_dimension, condition, bounds):
            return lz.VerificationObligation(
                name=name,
                status=lz.VerificationStatus.VERIFIED,
                elapsed_time=0.0,
                backend_status="verified",
            )

    system = stable_linear_system()
    quadratic = lz.quadratic_lyapunov(system)
    quadratic_set = lz.SublevelSet(quadratic, 0.1)
    bootstrap = lz.QuadraticVerificationResult(
        system=system,
        certificate=quadratic,
        claim=lz.VerificationClaim.LYAPUNOV,
        status=lz.VerificationStatus.VERIFIED,
        local_set=quadratic_set,
        verified_set=quadratic_set,
        obligations=(),
        backend="test",
        elapsed_time=0.0,
    )
    candidate = QuadraticAnchoredNeuralLyapunov(
        TanhNetwork(1, hidden_layers=1, width=3),
        (0.0,),
        quadratic.matrix,
    )
    monkeypatch.setattr(adapter, "load_crown", lambda: (torch, object()))
    monkeypatch.setattr(adapter, "normalise_device", lambda *args: "cpu")
    monkeypatch.setattr(adapter, "CrownRunner", FakeRunner)

    result = adapter.verify_neural_lyapunov_crown(
        system,
        candidate,
        bootstrap,
        local_level=0.01,
        outer_level=0.02,
    )

    assert result.is_verified
    assert result.claim is lz.VerificationClaim.STABLE_EQUILIBRIUM_ROA
    assert result.verified_level == pytest.approx(0.02)

