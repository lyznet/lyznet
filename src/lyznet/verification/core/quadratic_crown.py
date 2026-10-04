"""Mathematical conditions for a two-stage CROWN quadratic proof.

Let ``y=x-x*``, ``V(y)=y.T P y``, and write

    f(x*+y) = A y + r(y),      Q = -(A.T P + P A).

CROWN proves each predicate below for every ``y`` in its declared input box.
The adapter supplies graphs whose outputs are exactly the expressions named
beside each predicate. Exact-rational checks establish ``P>0``, ``Q>qI``,
and separation from the box boundary before CROWN is called.
"""


def positive_definite_by_sylvester(principal_minors):
    # Claim: M is positive definite.
    # Sylvester: det(M[1:k, 1:k]) > 0 for every k=1,...,n.
    return all(minor.is_positive is True for minor in principal_minors)


def local_remainder_condition(
    output, *, level, remainder_limit, proof_margin
):
    # Outputs: (V(y), R(y)), where
    #
    #   R(y) = ||P Dr(y) + Dr(y).T P||_F^2.
    #
    # Claim on the supplied box:
    #
    #   V(y) > c1 + eps  or  R(y) < (q-mu)^2 - eps.
    #
    # Since ||M||_2 <= ||M||_F, this proves
    #
    #   Df(x*+y).T P + P Df(x*+y) < -mu I
    #
    # throughout V(y)<=c1.
    return (
        (output[0] > level + proof_margin)
        | (output[1] < remainder_limit)
    )


def annular_decrease_condition(
    output, *, local_level, outer_level, proof_margin
):
    # Outputs: (V(y), L_f V(y)).
    #
    # Claim on the declared domain D:
    #
    #   V(y) < c1-eps  or  V(y) > c2+eps
    #   or L_f V(y) < -eps.
    #
    # Thus L_f V<0 throughout c1<=V<=c2 in D.
    return (
        (output[0] < local_level - proof_margin)
        | (output[0] > outer_level + proof_margin)
        | (output[1] < -proof_margin)
    )


def exact_quadratic_face_separation(
    minimum_face_value, *, level, proof_margin
):
    # For V(y)=y.T P y on the complete hyperplane y_i=b,
    #
    #   min V(y) = b^2 / (P^-1)_ii.
    #
    # Claim for every face of D: min V > c2+eps. Since the bounded face is a
    # subset of its hyperplane, this proves {V<=c2} lies in interior(D).
    return minimum_face_value > level + proof_margin


__all__ = [
    "annular_decrease_condition",
    "exact_quadratic_face_separation",
    "local_remainder_condition",
    "positive_definite_by_sylvester",
]
