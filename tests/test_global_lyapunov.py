"""Tests for global quadratic and polynomial Lyapunov workflows."""

from dataclasses import replace

import pytest
import sympy as sp

import lyznet as lz
import lyznet.workflows as workflows


def inverted_pendulum():
    x1, x2 = sp.symbols("x1 x2")
    return lz.System.continuous(
        states=(x1, x2),
        dynamics=(x2, sp.sin(x1) - 4.4142 * x1 - 3.3163 * x2),
        domain=((-16.0, 16.0),) * 2,
    )


def polynomial_system():
    x1, x2 = sp.symbols("x1 x2")
    return lz.System.continuous(
        states=(x1, x2),
        dynamics=(-x1**3 + x1**5 * x2, -x2**3 - x1**6),
        domain=((-10.0, 10.0),) * 2,
    )


def test_z3_verifies_global_quadratic_with_trigonometric_bounds():
    result = lz.find_lyapunov(
        inverted_pendulum(), scope="global", verify="z3"
    )

    certificate = result.require_verified()
    assert isinstance(certificate.verified_set, lz.WholeSpace)
    assert result.verification.uses_auxiliary_bounds
    assert result.verification.properness_verified


def test_dreal_verifies_global_quadratic_remainder_bound():
    pytest.importorskip("dreal")
    result = lz.find_lyapunov(
        inverted_pendulum(), scope="global", verify="dreal"
    )

    assert result.is_verified
    assert result.verification.obligations[-1].name == (
        "global_remainder_bound"
    )


def test_polynomial_synthesis_is_followed_by_independent_verification():
    system = polynomial_system()
    x1, x2 = system.states
    result = lz.find_lyapunov(
        system,
        family="polynomial",
        search="z3",
        scope="global",
        terms=(x1**2, x2**2, x1 * x2),
        verify="z3",
    )

    assert result.synthesis.found
    assert result.is_verified
    assert isinstance(result.candidate, lz.PolynomialLyapunov)


def test_fixed_verifier_rejects_a_corrupted_synthesis_result(monkeypatch):
    system = polynomial_system()
    x1, x2 = system.states
    synthesis = lz.synthesize_global_polynomial_lyapunov(
        system, (x1**2, x2**2, x1 * x2)
    )
    corrupted = replace(
        synthesis,
        candidate=lz.PolynomialLyapunov(
            system.states, -synthesis.candidate.expression
        ),
    )
    monkeypatch.setattr(
        workflows,
        "synthesize_global_polynomial_lyapunov",
        lambda *args, **kwargs: corrupted,
    )

    result = lz.find_lyapunov(
        system,
        family="polynomial",
        search="z3",
        scope="global",
        terms=(x1**2, x2**2, x1 * x2),
        verify="z3",
    )

    assert result.synthesis.found
    assert result.verification.status is lz.VerificationStatus.COUNTEREXAMPLE
    assert not result.is_verified
