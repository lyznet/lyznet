"""Focused tests for two-stage quadratic CROWN verification."""

from importlib.util import find_spec

import pytest
import sympy as sp

import lyznet as lz
from lyznet.verification import quadratic_crown as adapter


def stable_van_der_pol():
    x1, x2 = sp.symbols("x1 x2")
    return lz.System.continuous(
        states=(x1, x2),
        dynamics=(-x2, x1 - (1 - x1**2) * x2),
        domain=((-2.5, 2.5), (-3.5, 3.5)),
    )


def test_crown_requires_proposed_levels():
    with pytest.raises(ValueError, match="requires proposed"):
        lz.verify_quadratic_roa(stable_van_der_pol(), backend="crown")


def test_crown_rejects_an_approximate_equilibrium_before_loading_backend(
    monkeypatch,
):
    state = sp.symbols("x")
    dynamics = (2 - state**2,)
    equilibrium = lz.solve_equilibrium(
        states=(state,), dynamics=dynamics, initial_guess=(1.4,)
    )
    system = lz.System.continuous(
        states=(state,),
        dynamics=dynamics,
        equilibrium=equilibrium,
        domain=((1.0, 2.0),),
    )
    monkeypatch.setattr(
        adapter,
        "load_crown",
        lambda: pytest.fail("CROWN must not load for an approximate centre"),
    )

    with pytest.raises(ValueError, match="exact equilibrium.*dReal"):
        lz.verify_quadratic_roa(
            system,
            backend="crown",
            local_max_level=0.01,
            maximum_level=0.05,
        )


def test_crown_reports_an_unsupported_symbolic_operation_before_loading(
    monkeypatch,
):
    state = sp.symbols("x")
    system = lz.System.continuous(
        states=(state,),
        dynamics=(-sp.sinh(state),),
        domain=((-1.0, 1.0),),
    )
    monkeypatch.setattr(
        adapter,
        "load_crown",
        lambda: pytest.fail("unsupported expressions must fail before import"),
    )

    with pytest.raises(NotImplementedError, match="does not support"):
        lz.verify_quadratic_roa(
            system,
            backend="crown",
            local_max_level=0.1,
            maximum_level=0.2,
        )


def test_crown_adapter_reports_a_verified_two_stage_proof(monkeypatch):
    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def solve(self, name, model, output_dimension, condition, bounds):
            return lz.VerificationObligation(
                name=name,
                status=lz.VerificationStatus.VERIFIED,
                elapsed_time=0.01,
                backend_status="verified",
            )

    monkeypatch.setattr(adapter, "load_crown", lambda: (object(), object()))
    monkeypatch.setattr(adapter, "normalise_device", lambda torch, value: "cpu")
    monkeypatch.setattr(adapter, "expression_graph", lambda *args: object())
    monkeypatch.setattr(adapter, "CrownRunner", FakeRunner)

    result = lz.verify_quadratic_roa(
        stable_van_der_pol(),
        backend="crown",
        local_max_level=0.4,
        maximum_level=2.3,
    )

    assert result.status is lz.VerificationStatus.VERIFIED
    assert result.local_set.level == pytest.approx(0.4)
    assert result.verified_set.level == pytest.approx(2.3)
    assert tuple(item.name for item in result.obligations) == (
        "candidate_positive_definite",
        "linear_decrease_lower_bound",
        "local_remainder_bound",
        "domain_boundary_exclusion",
        "annular_decrease",
    )


def test_crown_verifies_van_der_pol_with_external_backend():
    if find_spec("torch") is None or find_spec("abcrown") is None:
        pytest.skip("The optional CROWN stack is not installed.")

    result = lz.verify_quadratic_roa(
        stable_van_der_pol(),
        backend="crown",
        local_max_level=0.4,
        maximum_level=2.3,
        timeout=30,
        crown_device="cpu",
    )

    assert result.status is lz.VerificationStatus.VERIFIED
    assert result.local_set.level == pytest.approx(0.4)
    assert result.verified_set.level == pytest.approx(2.3)
