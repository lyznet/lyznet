"""Verify a global quadratic CLF for controlled Van der Pol."""

import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
u = sp.symbols("u")
system = lz.System.control_affine(
    states=(x1, x2),
    inputs=(u,),
    drift=(x2, -x1 + (1 - x1**2) * x2),
    input_matrix=((0,), (1,)),
    domain=((-4.0, 4.0),) * 2,
    name="controlled_van_der_pol",
)


if __name__ == "__main__":
    result = lz.find_control_lyapunov(
        system,
        scope="global",
        verify="z3",
    )
    certificate = result.require_verified()
    print("Global quadratic CLF verified.")
    print("Matrix:\n", certificate.candidate.matrix)
    print("LQR gain:\n", certificate.candidate.gain)
