"""Verify a proposed quadratic CLF funnel with optional CROWN."""

import sympy as sp

import lyznet as lz


x, u = sp.symbols("x u")
system = lz.System.control_affine(
    states=(x,),
    inputs=(u,),
    drift=(x,),
    input_matrix=((1,),),
    domain=((-1.0, 1.0),),
    name="linear_controlled_system",
)


if __name__ == "__main__":
    result = lz.find_control_lyapunov(
        system,
        verify="crown",
        verification_options={
            "local_max_level": 0.1,
            "maximum_level": 0.2,
        },
    )
    certificate = result.require_verified()
    print("Local LQR sublevel:", certificate.local_set.level)
    print("Verified CLF sublevel:", certificate.verified_set.level)
