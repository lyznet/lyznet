"""Mathematical conditions for Z3 quadratic-CLF proofs."""


def candidate_positivity_violation(
    z3, *, state_is_nonzero, candidate
):
    # Claim: y != 0 implies V(y)>0.
    # Query: y != 0 and V(y)<=0.
    return z3.And(state_is_nonzero, candidate <= 0)


def global_artstein_violation(
    z3, *, state_is_nonzero, no_control_effect, drift_derivative
):
    # Artstein claim for unrestricted input deviations:
    # y != 0 and L_g V(y)=0 imply L_f V(y)<0.
    # Query: y != 0, L_g V(y)=0, and L_f V(y)>=0.
    return z3.And(
        state_is_nonzero,
        no_control_effect,
        drift_derivative >= 0,
    )


def sublevel_artstein_violation(
    z3,
    *,
    in_domain,
    state_is_nonzero,
    candidate,
    level,
    no_control_effect,
    drift_derivative,
):
    # Claim on the box: y != 0 and V(y)<=c and L_g V(y)=0 imply
    # L_f V(y)<0. Query: all hypotheses hold and L_f V(y)>=0.
    return z3.And(
        in_domain,
        state_is_nonzero,
        candidate <= level,
        no_control_effect,
        drift_derivative >= 0,
    )


def domain_boundary_violation(
    z3, *, in_domain, on_domain_boundary, candidate, level
):
    # Claim: V(y)>c on the boundary of the computational box.
    # Query: x*+y is on that boundary and V(y)<=c.
    return z3.And(in_domain, on_domain_boundary, candidate <= level)


__all__ = [
    "candidate_positivity_violation",
    "domain_boundary_violation",
    "global_artstein_violation",
    "sublevel_artstein_violation",
]
