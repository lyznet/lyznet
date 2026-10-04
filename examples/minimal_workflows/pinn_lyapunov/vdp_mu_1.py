"""Learn and verify a small anchored neural Lyapunov function."""

from pathlib import Path

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
    result = lz.find_lyapunov(
        system,
        family="neural",
        effort="low",
        verify="crown",
        plot=True,
        results_dir=Path(__file__).parent / "results",
        learner_verbose=True,
        verifier_verbose=True,
    )
    result.require_verified()
