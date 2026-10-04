"""Tests for two-stage quadratic verification with dReal."""

from dataclasses import replace

import numpy as np
import pytest
import sympy as sp

import lyznet as lz


pytest.importorskip("dreal")


def test_dreal_verifies_a_nonpolynomial_pendulum_funnel():
    x1, x2 = sp.symbols("x1 x2")
    system = lz.System.continuous(
        states=(x1, x2),
        dynamics=(x2, -sp.sin(x1) - x2),
        domain=((-2.0, 2.0),) * 2,
    )

    result = lz.verify_quadratic_roa(system, backend="dreal")

    assert result.status is lz.VerificationStatus.VERIFIED
    assert result.claim is lz.VerificationClaim.LYAPUNOV
    assert 1.7 < result.local_set.level < 1.8
    assert 3.3 < result.verified_set.level < 3.4
    assert tuple(item.name for item in result.obligations) == (
        "candidate_positive_definite",
        "local_contraction",
        "local_positive_invariance",
        "local_boundary_exclusion",
        "annular_decrease",
        "outer_boundary_exclusion",
    )
    assert result.obligations[2].backend_status == "implied-by-contraction"


def test_dreal_handles_an_approximately_known_equilibrium():
    state = sp.symbols("x")
    dynamics = (2 - state**2,)
    equilibrium = lz.solve_equilibrium(
        states=(state,),
        dynamics=dynamics,
        initial_guess=(1.4,),
        residual_tolerance=1e-12,
    )
    system = lz.System.continuous(
        states=(state,),
        dynamics=dynamics,
        equilibrium=equilibrium,
        domain=((1.0, 2.0),),
    )

    result = lz.verify_quadratic_roa(
        system,
        backend="dreal",
        local_max_level=0.01,
        maximum_level=0.05,
    )

    assert result.status is lz.VerificationStatus.VERIFIED
    assert result.claim is lz.VerificationClaim.STABLE_EQUILIBRIUM_ROA
    assert result.obligations[2].name == "local_positive_invariance"
    assert result.obligations[2].backend_status == "unsat"


def test_z3_rejects_a_frozen_center_that_is_not_an_exact_equilibrium():
    state = sp.symbols("x")
    system = lz.System.continuous(
        states=(state,),
        dynamics=(sp.Rational(1, 3) - state,),
        equilibrium=(sp.Rational(1, 3),),
        domain=((0.0, 1.0),),
    )

    with pytest.raises(ValueError, match="exact equilibrium.*dReal"):
        lz.verify_quadratic_roa(system, backend="z3")


def test_dreal_delta_sat_is_unknown_not_a_counterexample():
    x1, x2 = sp.symbols("x1 x2")
    system = lz.System.continuous(
        states=(x1, x2),
        dynamics=(-x2, x1 - (1 - x1**2) * x2),
        domain=((-2.5, 2.5), (-3.5, 3.5)),
    )
    reference = lz.quadratic_lyapunov(system)
    candidate = replace(
        reference, matrix=np.eye(2), q_matrix=np.eye(2)
    )

    result = lz.verify_quadratic_roa(
        system,
        certificate=candidate,
        backend="dreal",
        local_max_level=0.1,
        maximum_level=0.1,
        c1_search_precision=1e-3,
        c2_search_precision=1e-3,
    )

    assert result.status is lz.VerificationStatus.UNKNOWN
    assert result.verified_set is None
    assert result.obligations[-1].backend_status == "delta-sat"
