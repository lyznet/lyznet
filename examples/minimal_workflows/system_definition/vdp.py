"""Define and evaluate autonomous and controlled Van der Pol systems."""

import numpy as np
import sympy as sp

import lyznet as lz


x1, x2 = sp.symbols("x1 x2")
u = sp.symbols("u")
domain = ((-2.0, 2.0),) * 2

system = lz.System.continuous(
    states=(x1, x2),
    dynamics=(x2, -x1 + (1 - x1**2) * x2),
    domain=domain,
    name="van_der_pol",
)

controlled_system = lz.System.control_affine(
    states=(x1, x2),
    inputs=(u,),
    drift=tuple(system.symbolic.dynamics),
    input_matrix=((0,), (1,)),
    domain=domain,
    input_set=lz.Box(((-2.0, 2.0),)),
    name="controlled_van_der_pol",
)

if __name__ == "__main__":
    state = np.asarray([1.0, 0.0])
    print(system)
    print("f([1, 0]) =", system.numpy.dynamics(state))
    print(controlled_system)
    print("f([1, 0]) + g([1, 0])(-1) =", controlled_system.numpy.dynamics(
        state, np.asarray([-1.0])
    ))
