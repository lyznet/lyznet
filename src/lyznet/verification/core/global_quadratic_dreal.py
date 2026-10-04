"""Mathematical conditions for a global quadratic dReal proof."""

import sympy as sp


def certified_decrease_principal_minors(matrix, *, lower_bound):
    # Claim: Q-qI is positive definite for Q=-(A.T P+PA).
    shifted = matrix - lower_bound * sp.eye(matrix.rows)
    return tuple(
        sp.det(shifted[:size, :size])
        for size in range(1, matrix.rows + 1)
    )


def global_remainder_violation(*, remainder_bound, limit):
    # Claim for every x:
    # ||P Dr(x)+Dr(x).T P||_F^2 <= (q-mu)^2.
    # Query: the squared bound exceeds its limit.
    return remainder_bound > limit


__all__ = ["certified_decrease_principal_minors", "global_remainder_violation"]
