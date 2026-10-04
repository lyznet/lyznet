"""Verify a two-stage quadratic CLF funnel with dReal."""

from vdp_global import system

import lyznet as lz


if __name__ == "__main__":
    result = lz.find_control_lyapunov(
        system,
        verify="dreal",
        verification_options={"jobs": 1},
    )
    certificate = result.require_verified()
    print("Local LQR sublevel:", certificate.local_set.level)
    print("Verified CLF sublevel:", certificate.verified_set.level)
