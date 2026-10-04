"""Verify a neural Lyapunov set whose boundary includes a state constraint."""

from pathlib import Path

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(-x1, -x2),
    domain=((-1.0, 1.0),) * 2,
    name="stable_linear_disk_constraint",
)

if __name__ == "__main__":
    result = lz.find_lyapunov(
        system,
        family="neural",
        verify="crown",
        state_constraints=(sp.Rational(25, 36) * (x1**2 + x2**2),),
        viable_faces=((0, "lower"), (0, "upper"),
                      (1, "lower"), (1, "upper")),
        plot=True,
        results_dir=Path(__file__).parent / "results",
        learner_verbose=True,
        verifier_verbose=True,
    )
    result.require_verified()
