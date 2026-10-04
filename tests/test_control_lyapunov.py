"""Tests for quadratic control-Lyapunov construction and verification."""

from dataclasses import replace

import numpy as np
import pytest
import sympy as sp

import lyznet as lz
from lyznet.verification._quadratic_clf import (
    build_quadratic_clf_problem,
)


def controlled_vdp():
    x1, x2, u = sp.symbols("x1 x2 u")
    return lz.System.control_affine(
        states=(x1, x2),
        inputs=(u,),
        drift=(x2, -x1 + (1 - x1**2) * x2),
        input_matrix=((0,), (1,)),
        domain=((-4.0, 4.0),) * 2,
    )


def test_lqr_candidate_matches_the_migrated_construction():
    candidate = lz.quadratic_control_lyapunov(controlled_vdp())

    np.testing.assert_allclose(
        candidate.matrix,
        [[3.37841423, 0.41421356], [0.41421356, 2.68179283]],
        rtol=2e-8,
        atol=2e-8,
    )
    assert np.linalg.norm(candidate.residual, ord=np.inf) < 1e-12
    assert np.max(np.real(np.linalg.eigvals(
        candidate.linearization
        + candidate.input_linearization @ candidate.gain
    ))) < 0


def test_problem_displays_the_artstein_derivatives_directly():
    system = controlled_vdp()
    certificate = lz.quadratic_control_lyapunov(system)
    problem = build_quadratic_clf_problem(system, certificate)
    states = sp.ImmutableMatrix(problem.states)
    gradient = sp.ImmutableMatrix([problem.candidate]).jacobian(states)

    expected_drift = (gradient * system.symbolic.drift)[0]
    expected_input = tuple(gradient * system.symbolic.input_matrix)
    assert sp.simplify(problem.drift_derivative - expected_drift) == 0
    assert all(
        sp.simplify(actual - expected) == 0
        for actual, expected in zip(
            problem.input_derivatives, expected_input
        )
    )


def test_verification_rejects_a_mismatched_centre():
    system = controlled_vdp()
    candidate = lz.quadratic_control_lyapunov(system)
    corrupted = replace(
        candidate,
        centre=candidate.centre + np.asarray((1e-9, 0.0)),
    )

    with pytest.raises(ValueError, match="match the equilibrium"):
        lz.verify_quadratic_clf(system, corrupted)


def test_z3_verifies_the_global_quadratic_clf():
    result = lz.find_control_lyapunov(
        controlled_vdp(), scope="global", verify="z3"
    )

    assert result.is_verified
    assert result.verification.claim is lz.VerificationClaim.CONTROL_LYAPUNOV
    assert isinstance(result.verified_set, lz.WholeSpace)
    assert tuple(item.name for item in result.verification.obligations) == (
        "global_candidate_positivity",
        "global_clf_decrease",
        "global_radial_unboundedness",
    )


def test_dreal_verifies_the_two_stage_quadratic_clf():
    pytest.importorskip("dreal")
    result = lz.verify_quadratic_clf(
        controlled_vdp(),
        backend="dreal",
        maximum_level=45.0,
        local_max_level=4.0,
        c1_search_precision=1e-2,
        c2_search_precision=1e-2,
        jobs=1,
    )

    assert result.is_verified
    assert result.local_set.level == pytest.approx(2.86)
    assert result.verified_set.level == pytest.approx(42.09)
    assert tuple(item.name for item in result.obligations) == (
        "candidate_positive_definite",
        "local_lqr_contraction",
        "local_lqr_positive_invariance",
        "local_boundary_exclusion",
        "annular_clf_decrease",
        "outer_boundary_exclusion",
    )


def test_crown_verifies_a_linear_quadratic_clf():
    pytest.importorskip("torch")
    pytest.importorskip("abcrown")
    x, u = sp.symbols("x u")
    system = lz.System.control_affine(
        states=(x,),
        inputs=(u,),
        drift=(x,),
        input_matrix=((1,),),
        domain=((-1.0, 1.0),),
    )

    result = lz.verify_quadratic_clf(
        system,
        backend="crown",
        local_max_level=0.1,
        maximum_level=0.2,
        crown_device="cpu",
        timeout=15,
    )

    assert result.is_verified
    assert result.local_set.level == pytest.approx(0.1)
    assert result.verified_set.level == pytest.approx(0.2)
