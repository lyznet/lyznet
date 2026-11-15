"""Learn and verify an anchored PINN Lyapunov function for two machines."""

from pathlib import Path

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
delta = sp.Float("3.141592653589793") / 3
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(
        x2,
        -sp.Rational(1, 2) * x2
        - (sp.sin(x1 + delta) - sp.sin(delta)),
    ),
    domain=((-2.0, 3.0), (-3.0, 1.5)),
    name="minimal_neural_two_machine",
)


if __name__ == "__main__":
    result = lz.find_lyap(
        system,
        family="neural",
        verify="crown",
        effort="low",
        seed=123,
        verification_options={"bootstrap_backend": "dreal"},
        plot=True,
        results_dir=Path(__file__).parent / "results",
        learner_verbose=True,
        verifier_verbose=True,
    )
    result.require_verified()
