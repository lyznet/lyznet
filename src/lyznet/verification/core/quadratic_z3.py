"""Mathematical conditions for direct quadratic verification with Z3.

Let ``y = x - x*``, ``V(y) = y.T P y``, and let ``D`` be the declared box.
Each function below returns an existential violation. Therefore, an ``unsat``
answer proves the universal mathematical statement written above the query.
"""


def positive_definite_by_sylvester(principal_minors):
    # Claim: P is positive definite.
    # Sylvester: det(P[1:k, 1:k]) > 0 for every k=1,...,n.
    return all(minor.is_positive is True for minor in principal_minors)


def sublevel_decrease_violation(
    z3, *, in_domain, state_is_nonzero, candidate, lie_derivative, level
):
    # Claim:
    #   y+x* in D, y != 0, V(y) <= c  ==>  L_f V(y) < 0.
    # Violation:
    #   y+x* in D, y != 0, V(y) <= c, L_f V(y) >= 0.
    return z3.And(
        in_domain,
        state_is_nonzero,
        candidate <= level,
        lie_derivative >= 0,
    )


def domain_boundary_violation(
    z3, *, in_domain, on_domain_boundary, candidate, level
):
    # Claim:
    #   y+x* in boundary(D)  ==>  V(y) > c.
    # Violation:
    #   y+x* in boundary(D), V(y) <= c.
    return z3.And(in_domain, on_domain_boundary, candidate <= level)


__all__ = [
    "domain_boundary_violation",
    "positive_definite_by_sylvester",
    "sublevel_decrease_violation",
]
