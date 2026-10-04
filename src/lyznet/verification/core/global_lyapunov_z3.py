"""Mathematical conditions for a whole-space Z3 Lyapunov proof."""

import sympy as sp


def lyapunov_lie_derivative(states, dynamics, candidate):
    # Definition: L_f V(x) = grad(V(x)) f(x).
    return sp.expand(sum(
        sp.diff(candidate, state) * component
        for state, component in zip(states, dynamics)
    ))


def candidate_positivity_violation(z3, *, nonzero, candidate):
    # Claim: x != 0 ==> V(x)>0.
    # Query: x != 0 and V(x)<=0.
    return z3.And(nonzero, candidate <= 0)


def strict_decrease_violation(z3, *, nonzero, lie_derivative):
    # Claim: x != 0 ==> L_f V(x)<0.
    # Query: x != 0 and L_f V(x)>=0.
    return z3.And(nonzero, lie_derivative >= 0)


def polynomial_at_infinity(states, candidate, scale):
    """Return degree, leading homogeneous form, and homogenization."""

    polynomial = sp.Poly(candidate, *states)
    if polynomial.is_zero:
        raise ValueError("The zero polynomial has no leading form.")
    degree = polynomial.total_degree()
    leading = sp.Integer(0)
    homogenized = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        term = coefficient
        for state, power in zip(states, powers):
            term *= state**power
        term_degree = sum(powers)
        homogenized += scale ** (degree - term_degree) * term
        if term_degree == degree:
            leading += term
    return int(degree), sp.expand(leading), sp.expand(homogenized)


def leading_form_positivity_violation(
    z3, *, direction_norm_squared, leading_form
):
    # Sufficient claim: ||u||^2=1 ==> V_d(u)>0.
    # Query: ||u||^2=1 and V_d(u)<=0.
    return z3.And(direction_norm_squared == 1, leading_form <= 0)


def radial_unboundedness_violation(
    z3,
    *,
    level,
    cutoff,
    scale,
    directions,
    direction_norm_squared,
    homogenized_candidate,
    degree,
):
    # With V_hat(t,u)=t^d V(u/t), radial unboundedness is
    #
    # forall M exists delta in (0,1] forall 0<t<=delta, ||u||=1:
    #     V_hat(t,u) >= M t^d.
    #
    # Query: the exact quantified negation.
    claim = z3.ForAll(
        [level],
        z3.Exists(
            [cutoff],
            z3.And(
                cutoff > 0,
                cutoff <= 1,
                z3.ForAll(
                    [scale, *directions],
                    z3.Implies(
                        z3.And(
                            scale > 0,
                            scale <= cutoff,
                            direction_norm_squared == 1,
                        ),
                        homogenized_candidate >= level * scale**degree,
                    ),
                ),
            ),
        ),
    )
    return z3.Not(claim)


__all__ = [
    "candidate_positivity_violation",
    "leading_form_positivity_violation",
    "lyapunov_lie_derivative",
    "polynomial_at_infinity",
    "radial_unboundedness_violation",
    "strict_decrease_violation",
]
