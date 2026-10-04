"""Tests for empirical quadratic proposals and compact plotting."""

import pytest
import sympy as sp

import lyznet as lz


def reversed_vdp():
    x1, x2 = sp.symbols("x1 x2")
    return lz.System.continuous(
        states=(x1, x2),
        dynamics=(-x2, x1 - (1 - x1**2) * x2),
        domain=((-2.5, 2.5), (-3.5, 3.5)),
        name="reversed_vdp",
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


def test_lf_screen_is_deterministic_and_explicitly_empirical():
    first = lz.screen_quadratic_roa(
        reversed_vdp(), sample_count=2**10, seed=7
    )
    second = lz.screen_quadratic_roa(
        reversed_vdp(), sample_count=2**10, seed=7
    )

    assert first == second
    assert not first.is_formal
    assert not first.local.is_formal
    assert 0 < first.local.suggested_level
    assert first.local.suggested_level <= first.outer.suggested_level


def test_clf_screen_samples_the_artstein_surface_condition():
    screen = lz.screen_quadratic_clf(
        controlled_vdp(), sample_count=2**10
    )

    assert not screen.is_formal
    assert screen.local.stage == "local_lqr_contraction"
    assert screen.outer.stage == "annular_clf_decrease"
    assert screen.outer.suggested_level > screen.local.suggested_level


def test_verified_result_plot_is_saved_without_showing(tmp_path):
    pytest.importorskip("matplotlib")
    result = lz.find_lyapunov(reversed_vdp(), verify="z3")
    destination = tmp_path / "quadratic.pdf"

    figure = result.plot(
        surface=True,
        phase_portrait=False,
        show=False,
        save=destination,
        resolution=60,
    )

    assert destination.is_file()
    assert len(figure.axes) >= 2
