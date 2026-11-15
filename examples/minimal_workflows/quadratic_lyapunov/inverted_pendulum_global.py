"""Verify a global quadratic Lyapunov function with Z3."""

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(x2, sp.sin(x1) - 4.4142 * x1 - 3.3163 * x2),
    domain=((-16.0, 16.0),) * 2,
    name="inverted_pendulum_linear_control",
)

if __name__ == "__main__":
    result = lz.find_lyapunov(
        system, scope="global", verify="z3"
    )
    certificate = result.require_verified()
    print("Global Lyapunov matrix:\n", certificate.candidate.matrix)
    print("Verification status:", result.verification.status.value)
