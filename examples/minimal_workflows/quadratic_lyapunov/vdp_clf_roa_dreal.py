"""Verify a two-stage quadratic CLF funnel with dReal."""

from pathlib import Path

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
        verify="dreal",
        verification_options={"jobs": 1},
    )
    certificate = result.require_verified()
    print("Local LQR sublevel:", certificate.local_set.level)
    print("Verified CLF sublevel:", certificate.verified_set.level)
    result.plot(
        surface=True,
        save=Path(__file__).parent / "results" / "vdp_clf_roa_dreal.pdf",
    )
