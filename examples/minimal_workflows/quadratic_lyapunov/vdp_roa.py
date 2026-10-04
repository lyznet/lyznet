"""Find and formally verify a quadratic Lyapunov function with Z3."""

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(-x2, x1 - (1 - x1**2) * x2),
    domain=((-2.5, 2.5), (-3.5, 3.5)),
    name="van_der_pol_mu_1",
)

if __name__ == "__main__":
    result = lz.find_lyapunov(system, verify="z3")
    certificate = result.require_verified()
    print("Quadratic Lyapunov matrix:\n", certificate.candidate.matrix)
    print("Verified level:", certificate.verified_set.level)
    print("Verification status:", result.verification.status.value)
    print("Timings:", result.timings)
