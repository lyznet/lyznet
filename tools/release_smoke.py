"""Smoke-test the installed wheel outside the source tree."""

import sympy as sp

import lyznet as lz


x = sp.symbols("x")
system = lz.System.continuous(
    states=(x,),
    dynamics=(-x,),
    domain=((-1.0, 1.0),),
    name="release_smoke_test",
)
result = lz.find_lyapunov(system, verify="z3")
result.require_verified()
print("LyZNet {} wheel smoke test: VERIFIED".format(lz.__version__))
