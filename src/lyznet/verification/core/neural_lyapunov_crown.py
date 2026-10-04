"""CROWN conditions for a neural Lyapunov reach-and-stay funnel.

The local quadratic sublevel ``{V_q<=c_q}`` has already been verified.  For a
plain neural candidate ``W``, CROWN proves

    {W<=c_1} subset {V_q<c_q},
    L_f W<0 on {c_1<=W<=c_2},
    {W<=c_2} intersect boundary(D) is empty.

For an anchored candidate,

    H=s*q*exp(G),  W=tanh(H),  q=(x-x*)^T P (x-x*).

The verifier uses ``H`` because tanh is strictly increasing.  It uses
``R=(grad(q)+q*grad(G))^T f`` for decrease because

    L_f W = s*exp(G)*(1-W^2)*R,

and the omitted factor is strictly positive.

With state constraints, the certified set is

    S = D intersect (intersection_i {h_i<=1}) intersect {W<=c_2}.

Decrease is required only in the admissible annulus.  On every active
``h_i=1`` boundary the proof requires ``L_f h_i<0``.  A box face either stays
separated from ``S`` or, when explicitly selected as viable, satisfies the
same strict inward-pointing condition ``n^T f<0``.  These conditions make
``S`` positively invariant by Nagumo's tangent-cone criterion.
"""


def equilibrium_membership(output, *, local_level, proof_margin):
    # Claim: W(x*) < c_1-eps.
    return output[0] < local_level - proof_margin


def quadratic_bootstrap_containment(
    output, *, local_level, quadratic_level, proof_margin
):
    # Outputs: (W,V_q). Claim: W>c_1+eps or V_q<c_q-eps.
    return (
        (output[0] > local_level + proof_margin)
        | (output[1] < quadratic_level - proof_margin)
    )


def annular_decrease(
    output, *, local_level, outer_level, proof_margin
):
    # Outputs: (W,L_f W). Claim on D:
    # W<c_1-eps or W>c_2+eps or L_f W<-eps.
    return (
        (output[0] < local_level - proof_margin)
        | (output[0] > outer_level + proof_margin)
        | (output[1] < -proof_margin)
    )


def boundary_separation(output, *, outer_level, proof_margin):
    # Claim on each face of D: W>c_2+eps.
    return output[0] > outer_level + proof_margin


def constrained_annular_decrease(
    output, *, local_level, outer_level, proof_margin, constraint_count
):
    # Outputs: (W,L_f W,h_1,...,h_m). On D prove
    # W<c_1-eps or W>c_2+eps or some h_i>1 or L_f W<-eps.
    outside = output[2] > 1
    for index in range(3, 2 + constraint_count):
        outside = outside | (output[index] > 1)
    return (
        (output[0] < local_level - proof_margin)
        | (output[0] > outer_level + proof_margin)
        | outside
        | (output[1] < -proof_margin)
    )


def constrained_boundary_separation(
    output, *, outer_level, proof_margin, constraint_count
):
    # Outputs: (W,h_1,...,h_m) on a box face. Prove that every admissible
    # point is outside the neural sublevel.
    separated = output[0] > outer_level + proof_margin
    for index in range(1, 1 + constraint_count):
        separated = separated | (output[index] > 1)
    return separated


def boundary_viability(
    output, *, outer_level, proof_margin, viability_margin
):
    # Outputs: (W,n^T f) on a box face with outward normal n. Prove
    # W>c_2+eps or n^T f<-eps_D. This is a strict, robust form of Nagumo's
    # tangent-cone condition wherever the sublevel touches that face.
    return (
        (output[0] > outer_level + proof_margin)
        | (output[1] < -viability_margin)
    )


def constrained_boundary_viability(
    output,
    *,
    outer_level,
    proof_margin,
    viability_margin,
    constraint_count,
):
    # Outputs: (W,n^T f,h_1,...,h_m). Outside K is irrelevant; every
    # admissible sublevel point on this face must have strict inward flow.
    viable = (
        (output[0] > outer_level + proof_margin)
        | (output[1] < -viability_margin)
    )
    for index in range(2, 2 + constraint_count):
        viable = viable | (output[index] > 1)
    return viable


def state_constraint_viability(
    output,
    *,
    outer_level,
    proof_margin,
    constraint_index,
    constraint_count,
    boundary_band,
    viability_margin,
):
    # Outputs: (W,h_1,...,h_m,L_f h_i). On D prove
    # W>c_2+eps or some h_j>1 or h_i<1-delta or L_f h_i<-eps_h.
    # Hence strict Nagumo viability holds on every active h_i=1 boundary,
    # including intersections of several active constraints.
    viable = (
        (output[0] > outer_level + proof_margin)
        | (output[1 + constraint_index] < 1 - boundary_band)
        | (output[1 + constraint_count] < -viability_margin)
    )
    for index in range(1, 1 + constraint_count):
        viable = viable | (output[index] > 1)
    return viable


def anchored_equilibrium_membership(output, *, inner_lower_latent):
    # Claim: H(x*)<atanh(c_1-eps), hence W(x*)<c_1-eps.
    return output[0] < inner_lower_latent


def anchored_bootstrap_containment(output, *, minimum_correction):
    # Output: G=N-N(x*).  Here
    # G_min=log(atanh(c_1+eps)/(s*(c_q-eps))).
    # G>G_min implies W<=c_1+eps only if q<c_q-eps.
    return output[0] > minimum_correction


def anchored_annular_decrease(
    output, *, inner_lower_latent, outer_upper_latent, proof_margin
):
    # Outputs: (H,R). Claim:
    # H<atanh(c_1-eps) or H>atanh(c_2+eps) or R<-eps.
    return (
        (output[0] < inner_lower_latent)
        | (output[0] > outer_upper_latent)
        | (output[1] < -proof_margin)
    )


def anchored_boundary_separation(output, *, outer_upper_latent):
    # Claim on each face of D: H>atanh(c_2+eps).
    return output[0] > outer_upper_latent


def anchored_constrained_annular_decrease(
    output,
    *,
    inner_lower_latent,
    outer_upper_latent,
    proof_margin,
    constraint_count,
):
    # Outputs: (H,R,h_1,...,h_m). The same constrained annular claim holds
    # because L_f W and R have the same strict sign.
    outside = output[2] > 1
    for index in range(3, 2 + constraint_count):
        outside = outside | (output[index] > 1)
    return (
        (output[0] < inner_lower_latent)
        | (output[0] > outer_upper_latent)
        | outside
        | (output[1] < -proof_margin)
    )


def anchored_constrained_boundary_separation(
    output, *, outer_upper_latent, constraint_count
):
    # Outputs: (H,h_1,...,h_m) on a box face.
    separated = output[0] > outer_upper_latent
    for index in range(1, 1 + constraint_count):
        separated = separated | (output[index] > 1)
    return separated


def anchored_boundary_viability(
    output, *, outer_upper_latent, viability_margin
):
    # Outputs: (H,n^T f) on a box face.
    return (
        (output[0] > outer_upper_latent)
        | (output[1] < -viability_margin)
    )


def anchored_constrained_boundary_viability(
    output, *, outer_upper_latent, viability_margin, constraint_count
):
    # Outputs: (H,n^T f,h_1,...,h_m) on a box face.
    viable = (
        (output[0] > outer_upper_latent)
        | (output[1] < -viability_margin)
    )
    for index in range(2, 2 + constraint_count):
        viable = viable | (output[index] > 1)
    return viable


def anchored_state_constraint_viability(
    output,
    *,
    outer_upper_latent,
    constraint_index,
    constraint_count,
    boundary_band,
    viability_margin,
):
    # Outputs: (H,h_1,...,h_m,L_f h_i). Strict monotonicity of tanh turns
    # H>atanh(c_2+eps) into W>c_2+eps; the remaining disjuncts are the exact
    # Nagumo condition used for a plain candidate.
    viable = (
        (output[0] > outer_upper_latent)
        | (output[1 + constraint_index] < 1 - boundary_band)
        | (output[1 + constraint_count] < -viability_margin)
    )
    for index in range(1, 1 + constraint_count):
        viable = viable | (output[index] > 1)
    return viable


__all__ = [
    "anchored_annular_decrease",
    "anchored_boundary_viability",
    "anchored_bootstrap_containment",
    "anchored_boundary_separation",
    "anchored_constrained_annular_decrease",
    "anchored_constrained_boundary_separation",
    "anchored_constrained_boundary_viability",
    "anchored_equilibrium_membership",
    "anchored_state_constraint_viability",
    "annular_decrease",
    "boundary_viability",
    "boundary_separation",
    "constrained_annular_decrease",
    "constrained_boundary_separation",
    "constrained_boundary_viability",
    "equilibrium_membership",
    "quadratic_bootstrap_containment",
    "state_constraint_viability",
]
