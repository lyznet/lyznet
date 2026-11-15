"""Tests for homogeneous neural synthesis and global verification."""

import numpy as np
import pytest
import sympy as sp

import lyznet as lz


torch = pytest.importorskip("torch")

from lyznet.homogeneous_learning import (  # noqa: E402
    synthesize_homogeneous_neural_lyapunov,
)
from lyznet.neural import (  # noqa: E402
    HomogeneousNeuralLyapunov,
    TanhNetwork,
)


def radial_system():
    x1, x2 = sp.symbols("x1 x2")
    radius = sp.sqrt(x1**2 + x2**2)
    return lz.System.continuous(
        states=(x1, x2),
        dynamics=(-x1 * radius, -x2 * radius),
        domain=((-1.0, 1.0),) * 2,
        name="homogeneous_radial_test",
    )


def test_homogeneous_candidate_is_positive_and_scales_exactly():
    torch.manual_seed(7)
    model = TanhNetwork(
        2, hidden_layers=1, width=4, dtype=torch.float64
    )
    candidate = HomogeneousNeuralLyapunov(model, degree=2)
    points = np.asarray(((1.0, 2.0), (-2.0, 1.0)))

    assert candidate.value((0.0, 0.0)) == 0
    assert np.all(candidate.value(points) > 0)
    np.testing.assert_allclose(
        candidate.value(3 * points),
        9 * candidate.value(points),
        rtol=1e-12,
        atol=1e-12,
    )


def test_convex_synthesis_is_deterministic_and_reuses_its_checkpoint(tmp_path):
    options = {
        "results_dir": tmp_path,
        "width": 5,
        "direction_count": 32,
        "validation_directions": 128,
    }
    first = synthesize_homogeneous_neural_lyapunov(
        radial_system(), **options
    )
    second = synthesize_homogeneous_neural_lyapunov(
        radial_system(), **options
    )

    assert not first.loaded
    assert second.loaded
    assert first.model_path == second.model_path
    assert first.diagnostics.validation_violations == 0
    assert first.diagnostics.validation_maximum_ratio < 0
    np.testing.assert_allclose(
        first.candidate.value(((1.0, 0.0), (0.0, 1.0))),
        second.candidate.value(((1.0, 0.0), (0.0, 1.0))),
    )


def test_global_homogeneous_workflow_constructs_a_candidate(tmp_path):
    result = lz.find_lyapunov(
        radial_system(),
        family="neural",
        architecture="homogeneous",
        scope="global",
        verify=False,
        results_dir=tmp_path,
        learner_options={
            "width": 5,
            "direction_count": 32,
            "validation_directions": 128,
        },
    )

    assert result.is_global
    assert result.verification is None
    assert result.candidate.architecture == "homogeneous"


def test_dreal_proves_the_visible_unit_sphere_conditions():
    pytest.importorskip("dreal")
    from lyznet.verification.homogeneous_dreal import (
        verify_homogeneous_neural_lyapunov_dreal,
    )

    model = TanhNetwork(
        2, hidden_layers=1, width=1, dtype=torch.float64
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    candidate = HomogeneousNeuralLyapunov(model, degree=1)
    proof = verify_homogeneous_neural_lyapunov_dreal(
        radial_system(),
        candidate,
        dreal_delta=1e-5,
        positivity_margin=0.5,
        decrease_margin=0.5,
        well_defined_margin=0.5,
    )

    assert proof.status is lz.VerificationStatus.VERIFIED
    assert proof.properness_verified
    assert [item.name for item in proof.obligations] == [
        "homogeneous_structure",
        "homogeneous_dynamics_well_defined",
        "homogeneous_candidate_positivity",
        "homogeneous_strict_decrease",
    ]
