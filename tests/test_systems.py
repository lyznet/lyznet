"""Tests for continuous-time autonomous and control-affine systems."""

import numpy as np
import pytest
import sympy as sp

import lyznet as lz


def test_continuous_system_preserves_symbolic_and_numpy_views():
    x1, x2 = sp.symbols("x1 x2")
    system = lz.System.continuous(
        states=(x1, x2),
        dynamics=(1 - x1, -x2),
        equilibrium=(1, 0),
        domain=((-1, 2), (-2, 2)),
        name="example",
    )

    assert isinstance(system, lz.ContinuousTimeSystem)
    assert system.symbolic.dynamics == sp.Matrix((1 - x1, -x2))
    assert system.symbolic.jacobian == sp.Matrix(((-1, 0), (0, -1)))
    assert system.domain.bounds == ((-1.0, 2.0), (-2.0, 2.0))
    assert repr(system) == (
        "ContinuousTimeSystem(name='example', state_dimension=2)"
    )

    single = system.numpy.dynamics(np.zeros(2, dtype=np.float32))
    batch = system.numpy.dynamics(np.zeros((3, 2), dtype=np.float64))
    assert single.shape == (2,)
    assert batch.shape == (3, 2)
    assert single.dtype == np.float32


def test_system_rejects_undeclared_symbols_and_false_equilibria():
    x, y = sp.symbols("x y")

    with pytest.raises(ValueError, match="undeclared symbols: y"):
        lz.System.continuous(states=(x,), dynamics=(-x + y,))
    with pytest.raises(ValueError, match="not an equilibrium"):
        lz.System.continuous(
            states=(x,), dynamics=(-x + 1,), equilibrium=(0,)
        )


@pytest.mark.parametrize(
    "equilibrium", ((sp.oo,), (sp.I,), (sp.symbols("undeclared"),))
)
def test_equilibrium_must_be_a_closed_finite_real_point(equilibrium):
    state = sp.symbols("x")

    with pytest.raises(ValueError, match="closed finite real scalars"):
        lz.System.continuous(
            states=(state,), dynamics=(0,), equilibrium=equilibrium
        )


def test_system_rejects_dynamics_undefined_at_the_equilibrium():
    state = sp.symbols("x")

    with pytest.raises(ValueError, match="not an equilibrium"):
        lz.System.continuous(
            states=(state,),
            dynamics=(sp.sin(state) / state - 1,),
            equilibrium=(0,),
        )


def test_control_system_preserves_symbolic_and_numpy_views():
    state, control = sp.symbols("x u")
    system = lz.System.control_affine(
        states=(state,),
        inputs=(control,),
        drift=(-state,),
        input_matrix=((1 + state**2,),),
        input_set=lz.Box(((-2, 2),)),
        name="controlled",
    )
    states = np.asarray([[0.0], [2.0]], dtype=np.float32)
    inputs = np.asarray([[2.0], [3.0]], dtype=np.float32)

    assert system.state_dimension == 1
    assert system.input_dimension == 1
    assert system.symbolic.drift == sp.Matrix((-state,))
    assert system.symbolic.input_matrix == sp.Matrix(((1 + state**2,),))
    np.testing.assert_array_equal(
        system.numpy.input_matrix(states),
        np.asarray([[[1.0]], [[5.0]]]),
    )
    np.testing.assert_array_equal(
        system.numpy.dynamics(states, inputs), np.asarray([[2.0], [13.0]])
    )


@pytest.mark.parametrize(
    "input_set",
    (lz.Box(((1.0, 3.0),)), lz.EuclideanBall(center=(2.5,), radius=0.5)),
)
def test_control_system_accepts_a_nonzero_equilibrium_input(input_set):
    state, control = sp.symbols("x u")
    system = lz.System.control_affine(
        states=(state,),
        inputs=(control,),
        drift=(-state - 2,),
        input_matrix=((1,),),
        equilibrium_input=(2,),
        input_set=input_set,
    )

    assert system.equilibrium == (0,)
    assert system.equilibrium_input == (2,)


def test_control_input_set_must_match_and_contain_equilibrium_input():
    state, control = sp.symbols("x u")

    with pytest.raises(TypeError, match="Box, EuclideanBall, or None"):
        lz.System.control_affine(
            states=(state,),
            inputs=(control,),
            drift=(-state,),
            input_matrix=((1,),),
            input_set=((-1, 1),),
        )
    with pytest.raises(ValueError, match="Input-set dimension 2"):
        lz.System.control_affine(
            states=(state,),
            inputs=(control,),
            drift=(-state,),
            input_matrix=((1,),),
            input_set=lz.Box(((-1, 1),) * 2),
        )
    with pytest.raises(ValueError, match="physical input coordinates"):
        lz.System.control_affine(
            states=(state,),
            inputs=(control,),
            drift=(-state - 2,),
            input_matrix=((1,),),
            equilibrium_input=(2,),
            input_set=lz.EuclideanBall((0,), 1),
        )


def test_control_input_membership_uses_exact_stored_bounds():
    state, control = sp.symbols("x u")
    tiny = sp.Rational(1, 10**400)

    with pytest.raises(ValueError, match="must belong to input_set"):
        lz.System.control_affine(
            states=(state,),
            inputs=(control,),
            drift=(tiny,),
            input_matrix=((1,),),
            equilibrium_input=(-tiny,),
            input_set=lz.Box(((0, 1),)),
        )


def test_closed_loop_returns_an_autonomous_system():
    state, control = sp.symbols("x u")
    system = lz.System.control_affine(
        states=(state,),
        inputs=(control,),
        drift=(state,),
        input_matrix=((1,),),
        domain=((-2, 2),),
        name="plant",
    )

    closed_loop = system.closed_loop((-2 * state,))

    assert isinstance(closed_loop, lz.ContinuousTimeSystem)
    assert closed_loop.name == "plant-closed-loop"
    assert closed_loop.domain is system.domain
    assert closed_loop.symbolic.dynamics == sp.Matrix((-state,))


def test_systems_and_generated_views_are_read_only():
    state = sp.symbols("x")
    system = lz.System.continuous(states=(state,), dynamics=(-state,))
    numpy_view = system.numpy

    with pytest.raises(AttributeError, match="read-only"):
        system.equilibrium = (1,)
    with pytest.raises(AttributeError, match="read-only"):
        system.symbolic.dynamics = sp.Matrix((state,))
    with pytest.raises(AttributeError, match="read-only"):
        numpy_view.dynamics = None


@pytest.mark.parametrize(
    "values, message",
    (
        (np.asarray([1.0 + 1.0j]), "real values"),
        (np.asarray([np.nan]), "finite values"),
        (np.asarray([np.inf]), "finite values"),
    ),
)
def test_numpy_dynamics_requires_finite_real_states(values, message):
    state = sp.symbols("x")
    system = lz.System.continuous(states=(state,), dynamics=(-state,))

    with pytest.raises(ValueError, match=message):
        system.numpy.dynamics(values)


def test_numpy_view_rejects_invalid_expression_values():
    state = sp.symbols("x")
    complex_system = lz.System.continuous(
        states=(state,), dynamics=(sp.I * state,)
    )
    singular_system = lz.System.continuous(
        states=(state,),
        dynamics=(state / (state - 1),),
        equilibrium=(0,),
    )

    with pytest.raises(ValueError, match="Dynamics.*complex values"):
        complex_system.numpy.dynamics(np.asarray([1.0]))
    with pytest.raises(ValueError, match="Dynamics.*non-finite values"):
        singular_system.numpy.dynamics(np.asarray([1.0]))


def test_system_base_is_not_directly_constructible():
    with pytest.raises(TypeError, match="base class"):
        lz.System()
