"""Synthesize and independently verify a global polynomial Lyapunov function."""

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(-x1**3 + x1**5 * x2, -x2**3 - x1**6),
    domain=((-10.0, 10.0),) * 2,
    name="global_polynomial_2d",
)

if __name__ == "__main__":
    result = lz.find_lyapunov(
        system,
        family="polynomial",
        search="z3",
        scope="global",
        terms=(x1**2, x2**2, x1 * x2),
        verify="z3",
    )
    certificate = result.require_verified()
    print("Global Lyapunov function:", certificate.candidate.expression)
    print("Verification status:", result.verification.status.value)
