"""Learn and verify an anchored PINN Lyapunov function for a known basin."""

from pathlib import Path

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(-x1 + 2 * x1**2 * x2, -x2),
    domain=((-3.0, 3.0),) * 2,
    name="minimal_neural_exact_basin",
)


if __name__ == "__main__":
    result = lz.find_lyap(
        system,
        family="neural",
        verify="crown",
        effort="mid",
        seed=123,
        data_options={"transform_mu": 0.33},
        plot=True,
        results_dir=Path(__file__).parent / "results",
        learner_verbose=True,
        verifier_verbose=True,
    )
    result.require_verified()
