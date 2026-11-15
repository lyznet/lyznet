"""Tests for linearization and quadratic Lyapunov construction."""

from dataclasses import replace
import importlib

import numpy as np
import pytest
import sympy as sp

import lyznet as lz


def stable_van_der_pol():
    x1, x2 = sp.symbols("x1 x2")
    return lz.System.continuous(
        states=(x1, x2),
        dynamics=(-x2, x1 - (1 - x1**2) * x2),
        domain=((-2.5, 2.5), (-3.5, 3.5)),
    )


def test_quadratic_candidate_matches_the_reference():
    candidate = lz.quadratic_lyapunov(stable_van_der_pol())

    np.testing.assert_allclose(
        candidate.linearization, ((0.0, -1.0), (1.0, -1.0))
    )
    np.testing.assert_allclose(
        candidate.matrix, ((1.5, -0.5), (-0.5, 1.0)), atol=1e-12
    )
    np.testing.assert_allclose(candidate.residual, 0.0, atol=1e-12)
    assert candidate.value((1.0, 0.0)) == pytest.approx(1.5)


def test_candidate_arrays_are_immutable():
    candidate = lz.quadratic_lyapunov(stable_van_der_pol())

    for name in ("matrix", "q_matrix", "linearization", "centre"):
        values = getattr(candidate, name)
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)


def test_linearize_control_affine_system_at_an_input():
    state, control = sp.symbols("x u")
    system = lz.System.control_affine(
        states=(state,),
        inputs=(control,),
        drift=(-state,),
        input_matrix=((state,),),
    )

    np.testing.assert_allclose(lz.linearize(system), ((-1.0,),))
    np.testing.assert_allclose(
        lz.linearize(system, input_at=(2.0,)), ((1.0,),)
    )


def test_quadratic_constructor_rejects_an_unstable_linearization():
    state = sp.symbols("x")
    system = lz.System.continuous(states=(state,), dynamics=(state,))

    with pytest.raises(ValueError, match="not Hurwitz"):
        lz.quadratic_lyapunov(system)


def test_quadratic_constructor_requires_an_autonomous_system():
    state, control = sp.symbols("x u")
    system = lz.System.control_affine(
        states=(state,),
        inputs=(control,),
        drift=(-state,),
        input_matrix=((1,),),
    )

    with pytest.raises(TypeError, match="autonomous"):
        lz.quadratic_lyapunov(system)


def test_quadratic_weights_must_be_exactly_symmetric_and_positive():
    system = stable_van_der_pol()

    with pytest.raises(ValueError, match="exactly symmetric"):
        lz.quadratic_lyapunov(
            system, q_matrix=((2.0, 1.0), (1.0 + 1e-12, 2.0))
        )
    with pytest.raises(ValueError, match="positive definite"):
        lz.quadratic_lyapunov(system, q_matrix=((1.0, 0.0), (0.0, 0.0)))


def test_candidate_record_checks_shapes_without_claiming_a_proof():
    candidate = lz.QuadraticLyapunov(
        matrix=np.eye(1),
        q_matrix=np.eye(1),
        linearization=np.eye(1),
        centre=np.zeros(1),
    )

    assert candidate.residual.item() == 3.0
    with pytest.raises(ValueError, match="q_matrix"):
        replace(candidate, q_matrix=np.eye(2))


def test_constructor_rejects_a_bad_solver_residual(monkeypatch):
    module = importlib.import_module("lyznet.analysis")
    monkeypatch.setattr(
        module.linalg,
        "solve_continuous_lyapunov",
        lambda matrix, q_matrix: np.eye(2),
    )

    with pytest.raises(ValueError, match="Lyapunov equation residual"):
        lz.quadratic_lyapunov(stable_van_der_pol())


def test_homogeneous_degree_is_exact_for_polynomial_and_radical_fields():
    x1, x2 = sp.symbols("x1 x2")
    polynomial = lz.System.continuous(
        states=(x1, x2),
        dynamics=(x1**3 + x1*x2**2, -x2**3),
    )
    radical = lz.System.continuous(
        states=(x1, x2),
        dynamics=(
            -x1 * sp.sqrt(x1**2 + x2**2),
            -x2 * sp.sqrt(x1**2 + x2**2),
        ),
    )
    nonhomogeneous = lz.System.continuous(
        states=(x1, x2),
        dynamics=(-x1 - x1**3, -x2),
    )

    assert lz.homogeneous_degree(polynomial) == 3
    assert lz.homogeneous_degree(radical) == 2
    assert lz.homogeneous_degree(nonhomogeneous) is None
