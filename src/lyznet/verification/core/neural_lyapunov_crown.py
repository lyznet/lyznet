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


__all__ = [
    "anchored_annular_decrease",
    "anchored_bootstrap_containment",
    "anchored_boundary_separation",
    "anchored_equilibrium_membership",
    "annular_decrease",
    "boundary_separation",
    "equilibrium_membership",
    "quadratic_bootstrap_containment",
]
