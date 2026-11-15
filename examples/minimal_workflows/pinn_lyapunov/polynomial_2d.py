"""Learn and verify a PINN Lyapunov function for the CDC 2023 example."""

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    dynamics=(x2, -2 * x1 + sp.Rational(1, 3) * x1**3 - x2),
    domain=((-6.0, 6.0),) * 2,
    name="polynomial_2d",
)


if __name__ == "__main__":
    lz.find_lyap(
        system,
        family="neural",
        effort="high",
        verify="crown",
        plot=True,
        learner_verbose=True,
        verifier_verbose=True,
    ).require_verified()
