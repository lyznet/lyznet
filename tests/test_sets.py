"""Tests for the basic mathematical set value objects."""

from decimal import Decimal

import numpy as np
import pytest
import sympy as sp

import lyznet as lz


def test_box_preserves_definition_and_contains_points():
    box = lz.Box(((-1, 1), (-2, 2)))

    assert box.bounds == ((-1.0, 1.0), (-2.0, 2.0))
    assert box.dimension == 2
    assert box.volume == 8.0
    assert box.contains(np.asarray([0.0, 2.0]))
    np.testing.assert_array_equal(
        box.contains(np.asarray([[0.0, 0.0], [2.0, 0.0]])),
        np.asarray([True, False]),
    )


@pytest.mark.parametrize(
    "bounds",
    ((), ((0, 0),), ((1, -1),), ((0, np.inf),), ((-1e308, 1e308),)),
)
def test_box_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        lz.Box(bounds)


def test_euclidean_ball_preserves_definition_and_contains_points():
    ball = lz.EuclideanBall(
        center=(sp.Rational(1, 2), -1), radius=Decimal("2.5")
    )

    assert ball.center == (0.5, -1.0)
    assert ball.radius == 2.5
    assert ball.dimension == 2
    assert ball.contains(np.asarray([3.0, -1.0]))
    np.testing.assert_array_equal(
        ball.contains(np.asarray([[0.5, -1.0], [3.1, -1.0]])),
        np.asarray([True, False]),
    )


@pytest.mark.parametrize(
    "center", ((), (0, np.nan), (0, np.inf), (0, 1j), ("0",))
)
def test_euclidean_ball_requires_a_finite_real_center(center):
    with pytest.raises(ValueError, match="center.*finite real"):
        lz.EuclideanBall(center=center, radius=1)


@pytest.mark.parametrize("radius", (0, -1, np.nan, np.inf, True, "1"))
def test_euclidean_ball_requires_a_positive_finite_radius(radius):
    with pytest.raises(ValueError, match="radius.*positive finite"):
        lz.EuclideanBall(center=(0,), radius=radius)


@pytest.mark.parametrize(
    "points",
    (
        np.asarray([0.0, 1.0j]),
        np.asarray(["0", "1"]),
        np.asarray([object(), object()]),
        np.asarray([np.nan, 0.0]),
        np.asarray([np.inf, 0.0]),
    ),
)
def test_numerical_sets_reject_nonfinite_or_nonreal_points(points):
    with pytest.raises(ValueError, match="finite real numbers"):
        lz.Box(((-1, 1), (-2, 2))).contains(points)
    with pytest.raises(ValueError, match="finite real numbers"):
        lz.EuclideanBall((0, 0), 1).contains(points)


def test_whole_space_validates_dimension_and_point_shape():
    space = lz.WholeSpace(2)

    np.testing.assert_array_equal(
        space.contains(np.asarray([[0.0, 0.0], [1e300, -1e300]])),
        np.asarray([True, True]),
    )
    with pytest.raises(ValueError, match="positive integer"):
        lz.WholeSpace(True)
    with pytest.raises(ValueError, match="final dimension 2"):
        space.contains(np.asarray([0.0]))


def test_sets_are_immutable():
    box = lz.Box(((0, 1),))

    with pytest.raises(AttributeError):
        box.bounds = ((0, 2),)


def test_sublevel_set_uses_the_candidate_value_function():
    class Candidate:
        @staticmethod
        def value(points):
            return np.asarray(points)[..., 0] ** 2

    sublevel = lz.SublevelSet(Candidate(), 1.0)

    np.testing.assert_array_equal(
        sublevel.contains(np.asarray([[-1.0], [0.0], [2.0]])),
        np.asarray([True, True, False]),
    )
    with pytest.raises(ValueError, match="positive and finite"):
        lz.SublevelSet(Candidate(), 0.0)


def test_restricted_sublevel_intersects_all_defining_sets():
    x1, x2 = sp.symbols("x1 x2")
    candidate = lz.PolynomialLyapunov((x1, x2), x1**2 + x2**2)
    region = lz.RestrictedSublevelSet(
        candidate,
        2.0,
        lz.Box(((-1.0, 1.0), (-1.0, 1.0))),
        (x1, x2),
        (4 * x1**2 + x2**2,),
    )

    assert region.contains(np.asarray([0.25, 0.25]))
    assert not region.contains(np.asarray([0.75, 0.0]))
    assert not region.contains(np.asarray([0.0, 1.1]))
