"""Mathematical conditions for two-stage dReal quadratic-CLF proofs."""


def local_contraction_violation(
    dreal,
    *,
    in_domain,
    candidate,
    level,
    contraction_principal_minors,
):
    # For f_K=f+gKy, claim on V<=c1:
    # Df_K(y).T P + P Df_K(y) < 0.
    # Query (Sylvester): a leading principal minor of its negative is <=0.
    return dreal.And(
        in_domain,
        candidate <= level,
        dreal.Or(*(minor <= 0 for minor in contraction_principal_minors)),
    )


def outer_artstein_violation(
    dreal,
    *,
    in_domain,
    candidate,
    local_level,
    outer_level,
    no_control_effect,
    drift_derivative,
):
    # Claim on c1<=V<=c2: L_g V=0 implies L_f V<0.
    # Query: L_g V=0 and L_f V>=0 at a point in the annulus.
    return dreal.And(
        in_domain,
        candidate >= local_level,
        candidate <= outer_level,
        no_control_effect,
        drift_derivative >= 0,
    )


def domain_boundary_violation(
    dreal, *, in_domain, on_domain_boundary, candidate, level
):
    # Claim: V(y)>c on the boundary of the computational box.
    return dreal.And(in_domain, on_domain_boundary, candidate <= level)


__all__ = [
    "domain_boundary_violation",
    "local_contraction_violation",
    "outer_artstein_violation",
]
