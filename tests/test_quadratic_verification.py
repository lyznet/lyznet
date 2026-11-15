"""Tests for direct Z3 verification of quadratic Lyapunov sublevels."""

from dataclasses import replace

import numpy as np
import pytest
import sympy as sp

import lyznet as lz


pytest.importorskip("z3")


def stable_van_der_pol():
    x1, x2 = sp.symbols("x1 x2")
    return lz.System.continuous(
        states=(x1, x2),
        dynamics=(-x2, x1 - (1 - x1**2) * x2),
        domain=((-2.5, 2.5), (-3.5, 3.5)),
    )


def test_z3_verifies_a_van_der_pol_sublevel():
    system = stable_van_der_pol()
    result = lz.verify_quadratic_roa(system, search_precision=1e-4)

    assert result.status is lz.VerificationStatus.VERIFIED
    assert result.verified_level > 0
    assert tuple(item.name for item in result.obligations) == (
        "candidate_positive_definite",
        "sublevel_decrease",
        "domain_boundary_exclusion",
    )
    assert all(
        item.status is lz.VerificationStatus.VERIFIED
        for item in result.obligations
    )


def test_z3_rejects_a_candidate_without_strict_decrease():
    system = stable_van_der_pol()
    reference = lz.quadratic_lyapunov(system)
    candidate = replace(
        reference,
        matrix=np.eye(2),
        q_matrix=np.eye(2),
    )

    result = lz.verify_quadratic_roa(
        system,
        certificate=candidate,
        maximum_level=0.1,
        search_precision=1e-3,
    )

    assert result.status is lz.VerificationStatus.COUNTEREXAMPLE
    assert result.verified_set is None


def test_workflow_returns_only_formally_verified_certificate():
    result = lz.find_lyap(stable_van_der_pol(), verify="z3")
    certificate = result.require_verified()

    assert result.is_verified
    assert certificate.verification is result.verification
    assert certificate.verified_set is result.verification.verified_set
    assert certificate.candidate is result.candidate


def test_workflow_without_backend_does_not_claim_verification():
    result = lz.find_lyapunov(stable_van_der_pol())

    assert result.verification is None
    assert not result.is_verified
    with pytest.raises(RuntimeError, match="status: not run"):
        result.require_verified()


def test_verification_requires_a_domain_and_matching_centre():
    state = sp.symbols("x")
    unbounded = lz.System.continuous(states=(state,), dynamics=(-state,))

    with pytest.raises(ValueError, match="box domain"):
        lz.verify_quadratic_roa(unbounded)

    system = lz.System.continuous(
        states=(state,), dynamics=(-state,), domain=((-1, 1),)
    )
    candidate = replace(
        lz.quadratic_lyapunov(system), centre=np.asarray((0.1,))
    )
    with pytest.raises(ValueError, match="centre"):
        lz.verify_quadratic_roa(system, certificate=candidate)
