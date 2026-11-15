"""Find a global homogeneous neural LF for the degree-five benchmark."""

import math
from pathlib import Path

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
theta = 1e-2
first = x1**4 + 2*x1**2*x2**2 - x2**4
second = -x1**4 + 2*x1**2*x2**2 + x2**4
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(
        -2*math.sin(theta)*x1*first - 2*math.cos(theta)*x2*second,
        2*math.cos(theta)*x1*first - 2*math.sin(theta)*x2*second,
    ),
    domain=((-100.0, 100.0),) * 2,
    name="degree_five_homogeneous",
)


if __name__ == "__main__":
    result = lz.find_lyapunov(
        system,
        family="neural",
        architecture="homogeneous",
        scope="global",
        verify="dreal",
        results_dir=Path(__file__).parent / "results",
        learner_verbose=True,
        verifier_verbose=True,
        verification_options={"jobs": 4},
        plot=True,
        plot_options={"surface": True},
    )
    result.require_verified()
