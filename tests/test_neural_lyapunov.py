"""Focused tests for neural candidates, artifacts, and proof orchestration."""

from inspect import signature
from pathlib import Path

import numpy as np
import pytest
import sympy as sp


torch = pytest.importorskip("torch")

import lyznet as lz
from lyznet.constraints import normalise_viable_faces
from lyznet.neural import (
    CenteredNeuralLyapunov,
    QuadraticAnchoredNeuralLyapunov,
    TanhNetwork,
)
from lyznet.neural_assessment import screen_neural_lyapunov
from lyznet.neural_learning import (
    TrajectoryData,
    effort_configuration,
    generate_trajectory_data,
    train_neural_lyapunov,
)
from lyznet.neural_options import PINNCorrectionStage
import lyznet.neural_workflow as neural_workflow
from lyznet.verification import neural_crown as adapter


def test_public_neural_architecture_default_is_anchored():
    parameters = signature(lz.find_lyapunov).parameters

    assert parameters["architecture"].default == "anchored"
    assert parameters["effort"].default is None


@pytest.mark.parametrize(
    ("learner_options", "expected_effort"),
    ((None, "low"), ({"width": 7}, "fixed")),
)
def test_omitted_effort_selects_low_or_one_fixed_recipe(
    monkeypatch, tmp_path, learner_options, expected_effort
):
    calls = []

    def candidate(*args, effort, learner_options, **kwargs):
        calls.append((effort, learner_options))
        return effort

    monkeypatch.setattr(
        neural_workflow, "_find_single_neural_lyapunov", candidate
    )
    result = neural_workflow.find_neural_lyapunov(
        stable_linear_system(),
        architecture="anchored",
        effort=None,
        verify="crown",
        device="cpu",
        results_dir=tmp_path,
        seed=1,
        data_overwrite=False,
        model_overwrite=False,
        data_options=None,
        learner_options=learner_options,
        verification_options=None,
        learner_verbose=False,
        verifier_verbose=False,
        plot=False,
        plot_options=None,
        state_constraints=(),
        viable_faces=(),
    )

    assert result == expected_effort
    assert calls == [(expected_effort, learner_options)]


def test_neural_effort_recipes_are_architecture_specific():
    plain_low = effort_configuration("low", "plain")
    plain_mid = effort_configuration("mid", "plain")
    plain_high = effort_configuration("high", "plain")
    anchored_low = effort_configuration("low", "anchored")
    anchored_mid = effort_configuration("mid", "anchored")
    anchored_high = effort_configuration("high", "anchored")

    assert "correction_stages" not in plain_low
    assert {
        plain_low["width"], plain_mid["width"], plain_high["width"]
    } == {30}
    assert sum(
        stage.steps for stage in anchored_low["correction_stages"]
    ) == 13_000
    assert sum(
        stage.steps for stage in anchored_mid["correction_stages"]
    ) == 6_500
    assert sum(
        stage.steps for stage in anchored_high["correction_stages"]
    ) == 6_500
    assert anchored_low["correction_samples"] == 100_000


def test_mid_effort_continues_low_and_verifies_best_screen(
    monkeypatch, tmp_path
):
    calls = []
    volumes = {"low": 2.0, "mid": 3.0}

    def candidate(*args, effort, verify, learner_options, **kwargs):
        initial = learner_options.get("initial_model_path")
        calls.append((effort, verify, initial))
        verification = None
        if verify == "crown":
            verification = type(
                "Verification",
                (),
                {"status": lz.VerificationStatus.VERIFIED},
            )()
        return lz.LyapunovSearchResult(
            system=args[0],
            family="neural",
            candidate=effort,
            verification=verification,
            timings=lz.LyapunovSearchTimings(
                1.0 if verify is False else 0.0,
                2.0 if verify == "crown" else 0.0,
                3.0,
            ),
            assessment=type(
                "Assessment",
                (),
                {"estimated_volume": volumes[effort]},
            )(),
            model_path=tmp_path / "{}.pt".format(effort),
        )

    monkeypatch.setattr(
        neural_workflow, "_find_single_neural_lyapunov", candidate
    )
    result = neural_workflow.find_neural_lyapunov(
        stable_linear_system(),
        architecture="anchored",
        effort="mid",
        verify="crown",
        device="cpu",
        results_dir=tmp_path,
        seed=1,
        data_overwrite=False,
        model_overwrite=False,
        data_options=None,
        learner_options=None,
        verification_options=None,
        learner_verbose=False,
        verifier_verbose=False,
        plot=False,
        plot_options=None,
        state_constraints=(),
        viable_faces=(),
    )

    low_path = tmp_path / "low.pt"
    assert calls == [
        ("low", False, None),
        ("mid", False, low_path),
        ("mid", "crown", low_path),
    ]
    assert result.candidate == "mid"
    assert result.timings.candidate == pytest.approx(2.0)
    assert result.timings.verification == pytest.approx(2.0)


def test_high_effort_falls_back_after_best_screen_fails_proof(
    monkeypatch, tmp_path
):
    calls = []
    volumes = {"low": 2.0, "mid": 3.0, "high": 4.0}

    def candidate(*args, effort, verify, learner_options, **kwargs):
        initial = learner_options.get("initial_model_path")
        calls.append((effort, verify, initial))
        verification = None
        if verify == "crown":
            status = (
                lz.VerificationStatus.UNKNOWN
                if effort == "high"
                else lz.VerificationStatus.VERIFIED
            )
            verification = type("Verification", (), {"status": status})()
        return lz.LyapunovSearchResult(
            system=args[0],
            family="neural",
            candidate=effort,
            verification=verification,
            timings=lz.LyapunovSearchTimings(
                1.0 if verify is False else 0.0,
                2.0 if verify == "crown" else 0.0,
                3.0,
            ),
            assessment=type(
                "Assessment", (), {"estimated_volume": volumes[effort]}
            )(),
            model_path=tmp_path / "{}.pt".format(effort),
        )

    monkeypatch.setattr(
        neural_workflow, "_find_single_neural_lyapunov", candidate
    )
    result = neural_workflow.find_neural_lyapunov(
        stable_linear_system(),
        architecture="anchored",
        effort="high",
        verify="crown",
        device="cpu",
        results_dir=tmp_path,
        seed=1,
        data_overwrite=False,
        model_overwrite=False,
        data_options=None,
        learner_options=None,
        verification_options=None,
        learner_verbose=False,
        verifier_verbose=False,
        plot=False,
        plot_options=None,
        state_constraints=(),
        viable_faces=(),
    )

    assert calls == [
        ("low", False, None),
        ("mid", False, tmp_path / "low.pt"),
        ("high", False, tmp_path / "mid.pt"),
        ("high", "crown", tmp_path / "mid.pt"),
        ("mid", "crown", tmp_path / "low.pt"),
    ]
    assert result.candidate == "mid"
    assert result.timings.candidate == pytest.approx(3.0)
    assert result.timings.verification == pytest.approx(4.0)


def stable_linear_system():
    state = sp.symbols("x")
    return lz.System.continuous(
        states=(state,),
        dynamics=(-state,),
        domain=((-1.0, 1.0),),
        name="stable_linear",
    )


def test_all_viable_faces_expands_from_the_system_dimension():
    assert normalise_viable_faces(
        stable_linear_system(), "all"
    ) == ((0, "lower"), (0, "upper"))

    with pytest.raises(ValueError, match="'all' or a sequence"):
        normalise_viable_faces(stable_linear_system(), "automatic")


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


def test_trajectory_data_samples_only_the_admissible_set(tmp_path):
    state = stable_linear_system().states[0]
    data = generate_trajectory_data(
        stable_linear_system(),
        sample_count=32,
        results_dir=tmp_path,
        device="cpu",
        max_time=0.02,
        step_size=0.02,
        state_constraints=(4 * state**2,),
    )

    assert np.all(4 * data.states[:, 0] ** 2 <= 1 + 1e-6)


def test_random_trajectory_bank_matches_the_published_sampling_rule(tmp_path):
    data = generate_trajectory_data(
        stable_linear_system(),
        sample_count=16,
        results_dir=tmp_path,
        device="cpu",
        seed=123,
        sampling="random",
        include_equilibrium=False,
        terminate_on_domain_exit=False,
        max_time=0.02,
        step_size=0.02,
    )
    expected = np.random.default_rng(123).uniform(-1, 1, size=(16, 1))

    assert np.array_equal(data.states, expected.astype(np.float32))


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

    staged = train_neural_lyapunov(
        system,
        quadratic,
        data,
        **{
            **options,
            "correction_steps": 0,
            "correction_stages": (
                PINNCorrectionStage(steps=1, learning_rate=1e-3),
            ),
        },
    )
    assert staged.model_path != first.model_path

    continued = train_neural_lyapunov(
        system,
        quadratic,
        data,
        **{
            **options,
            "warmup_steps": 0,
            "correction_steps": 0,
            "initial_model_path": first.model_path,
        },
    )
    assert continued.model_path != first.model_path
    assert (
        continued.candidate.proof_fingerprint()
        == first.candidate.proof_fingerprint()
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


def test_neural_crown_proves_constraints_and_selected_viable_faces(
    monkeypatch,
):
    names = []

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def solve(self, name, model, output_dimension, condition, bounds):
            names.append(name)
            return lz.VerificationObligation(
                name=name,
                status=lz.VerificationStatus.VERIFIED,
                elapsed_time=0.0,
                backend_status="verified",
            )

    system = stable_linear_system()
    state = system.states[0]
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
        state_constraints=(4 * state**2,),
        viable_faces="all",
    )

    assert result.is_verified
    assert isinstance(result.verified_set, lz.RestrictedSublevelSet)
    assert "neural_state_constraint_1_viability" in names
    assert "neural_boundary_x1_lower" in names
    assert "neural_boundary_x1_upper" in names
