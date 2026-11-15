"""Learn and verify a PINN Lyapunov function for Van der Pol, mu=1."""

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    dynamics=(-x2, x1 - (1 - x1**2) * x2),
    domain=((-2.5, 2.5), (-3.5, 3.5)),
    name="minimal_vdp_mu_1",
)


if __name__ == "__main__":
    lz.find_lyap(
        system,
        family="neural",
        effort="low",
        verify="crown",
        plot=True,
        learner_verbose=True,
        verifier_verbose=True,
    ).require_verified()
