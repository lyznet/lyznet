"""Verify a quadratic Lyapunov funnel for a nonlinear pendulum with dReal."""

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(x2, -sp.sin(x1) - x2),
    domain=((-2.0, 2.0),) * 2,
    name="damped_pendulum",
)

if __name__ == "__main__":
    result = lz.find_lyapunov(system, verify="dreal")
    certificate = result.require_verified()
    print("Quadratic Lyapunov matrix:\n", certificate.candidate.matrix)
    print("Verified local level:", certificate.local_set.level)
    print("Verified outer level:", certificate.verified_set.level)
    print("Formal claim:", result.verification.claim.value)
    print("Timings:", result.timings)
