"""Mathematical conditions for global homogeneous dReal verification."""


def dynamics_well_defined_violation(
    dreal,
    *,
    on_unit_sphere,
    denominators,
    radicands,
    margin,
):
    # Claim on ||x||=1:
    #   denominator(x)^2 >= margin for every denominator, and
    #   radicand(x) >= margin for every even-root radicand.
    # These strict bounds imply that the punctured dynamics are continuous
    # and well defined on the complete unit sphere.
    failures = [value**2 < margin for value in denominators]
    failures.extend(value < margin for value in radicands)
    return dreal.And(on_unit_sphere, dreal.Or(*failures))


def candidate_positivity_violation(
    dreal, *, on_unit_sphere, candidate, margin
):
    # Claim: ||x||=1 implies V(x) >= margin.
    return dreal.And(on_unit_sphere, candidate < margin)


def strict_decrease_violation(
    dreal, *, on_unit_sphere, lie_derivative, margin
):
    # Claim: ||x||=1 implies grad(V)(x)^T f(x) <= -margin.
    return dreal.And(on_unit_sphere, lie_derivative > -margin)


__all__ = [
    "candidate_positivity_violation",
    "dynamics_well_defined_violation",
    "strict_decrease_violation",
]
