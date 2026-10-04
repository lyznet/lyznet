"""Mathematical conditions for a two-stage dReal quadratic proof.

Let ``y=x-x_c``, ``V(y)=y.T P y``, and let ``D`` be the declared box. Each
function returns an existential violation. An ``unsat`` result proves the
universal claim written above it; ``delta-sat`` is reported as unknown.
"""


def positive_definite_by_sylvester(principal_minors):
    # Claim: P is positive definite.
    # Sylvester: det(P[1:k, 1:k]) > 0 for every k=1,...,n.
    return all(minor.is_positive is True for minor in principal_minors)


def local_contraction_violation(
    dreal, *, in_domain, candidate, level, contraction_principal_minors
):
    # Claim on V(y)<=c1:
    #
    #   Df(x_c+y)^T P + P Df(x_c+y) < 0.
    #
    # Sylvester's criterion fails exactly when a leading principal minor of
    # -(Df^T P + P Df) is nonpositive.
    return dreal.And(
        in_domain,
        candidate <= level,
        dreal.Or(*(
            minor <= 0 for minor in contraction_principal_minors
        )),
    )


def local_invariance_violation(
    dreal, *, in_domain, candidate, level, lie_derivative
):
    # Claim on the local boundary:
    #
    #   V(y)=c1  ==>  L_f V(y)<0.
    #
    # This extra condition is needed when x_c is only a numerical estimate.
    return dreal.And(
        in_domain,
        candidate == level,
        lie_derivative >= 0,
    )


def outer_annulus_violation(
    dreal,
    *,
    in_domain,
    candidate,
    local_level,
    outer_level,
    lie_derivative,
):
    # Claim on the annulus:
    #
    #   c1<=V(y)<=c2  ==>  L_f V(y)<0.
    return dreal.And(
        in_domain,
        candidate >= local_level,
        candidate <= outer_level,
        lie_derivative >= 0,
    )


def domain_boundary_violation(
    dreal, *, in_domain, on_domain_boundary, candidate, level
):
    # Claim:
    #
    #   x_c+y in boundary(D)  ==>  V(y)>c.
    return dreal.And(in_domain, on_domain_boundary, candidate <= level)


__all__ = [
    "domain_boundary_violation",
    "local_contraction_violation",
    "local_invariance_violation",
    "outer_annulus_violation",
    "positive_definite_by_sylvester",
]
