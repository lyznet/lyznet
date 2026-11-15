"""Find a global homogeneous neural LF for the irrational benchmark."""

from pathlib import Path

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
numerator = (
    2*x1**6 + 8*x1**5*x2 + 22*x1**4*x2**2 + 18*x1**3*x2**3
    + 26*x1**2*x2**4 - 16*x1*x2**5 + 13*x2**6
)
denominator = (
    2*x1**4 - 4*x1**3*x2 + 5*x1**2*x2**2
    + 22*x1*x2**3 + 14*x2**4
)
scale = sp.sqrt(numerator) / denominator
first = -sp.Rational(1, 3) * scale * (
    8*x1**11 + 22*x1**10*x2 + 93*x1**9*x2**2 + 126*x1**8*x2**3
    + 483*x1**7*x2**4 + 186*x1**6*x2**5 + 445*x1**5*x2**6
    - 1189*x1**4*x2**7 + 234*x1**3*x2**8 - 713*x1**2*x2**9
    + 1772*x1*x2**10 - 414*x2**11
)
second = sp.Rational(1, 3) * scale * (
    6*x1**11 + 16*x1**10*x2 + 74*x1**9*x2**2 + 123*x1**8*x2**3
    + 180*x1**7*x2**4 - 267*x1**6*x2**5 - 1083*x1**5*x2**6
    - 1729*x1**4*x2**7 - 1580*x1**3*x2**8 + 687*x1**2*x2**9
    + 50*x1*x2**10 - 194*x2**11
)
# The displayed formula is defined away from zero and has a continuous zero
# extension because it is homogeneous of positive degree. State that
# extension explicitly so the system definition remains total.
at_origin = sp.Eq(x1**2 + x2**2, 0)
system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(
        sp.Piecewise((0, at_origin), (first, True)),
        sp.Piecewise((0, at_origin), (second, True)),
    ),
    domain=((-1.0, 1.0),) * 2,
    name="irrational_homogeneous",
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
