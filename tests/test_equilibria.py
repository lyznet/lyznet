"""Tests for numerical equilibrium evidence retained by systems."""

import numpy as np
import pytest
import sympy as sp

import lyznet as lz


def test_solve_equilibrium_retains_checked_residual_evidence():
    state = sp.symbols("x")
    equilibrium = lz.solve_equilibrium(
        states=(state,),
        dynamics=(2 - state**2,),
        initial_guess=(1.4,),
        residual_tolerance=1e-12,
    )
    system = lz.System.continuous(
        states=(state,),
        dynamics=(2 - state**2,),
        equilibrium=equilibrium,
        domain=((1, 2),),
    )

    assert system.numerical_equilibrium is equilibrium
    assert equilibrium.point == pytest.approx((np.sqrt(2),))
    assert equilibrium.residual_norm < 1e-12


def test_numerical_equilibrium_cannot_be_reused_for_other_dynamics():
    state = sp.symbols("x")
    equilibrium = lz.solve_equilibrium(
        states=(state,), dynamics=(2 - state**2,), initial_guess=(1.4,)
    )

    with pytest.raises(ValueError, match="dynamics do not match"):
        lz.System.continuous(
            states=(state,),
            dynamics=(3 - state**2,),
            equilibrium=equilibrium,
        )


def test_numerical_equilibrium_rejects_false_residual_evidence():
    state = sp.symbols("x")

    with pytest.raises(ValueError, match="Stored residual does not match"):
        lz.NumericalEquilibrium(
            states=(state,),
            dynamics=(state,),
            point=(0.0,),
            residual=(1.0,),
            residual_tolerance=2.0,
            method="test",
        )


@pytest.mark.parametrize("initial_guess", ((1j,), (np.nan,), (np.inf,)))
def test_equilibrium_solver_requires_a_finite_real_initial_guess(initial_guess):
    state = sp.symbols("x")

    with pytest.raises(ValueError, match="finite real"):
        lz.solve_equilibrium(
            states=(state,), dynamics=(-state,), initial_guess=initial_guess
        )


def test_closed_loop_must_preserve_a_numerical_equilibrium():
    state, control = sp.symbols("x u")
    equilibrium = lz.solve_equilibrium(
        states=(state,), dynamics=(2 - state**2,), initial_guess=(1.4,)
    )
    system = lz.System.control_affine(
        states=(state,),
        inputs=(control,),
        drift=(2 - state**2,),
        input_matrix=((1,),),
        equilibrium=equilibrium,
    )

    assert system.closed_loop((0,)).numerical_equilibrium is not None
    with pytest.raises(ValueError, match="does not preserve"):
        system.closed_loop((1,))
