"""Sound rational enclosures for sine and cosine auxiliaries."""


def sine_auxiliary_constraints(z3, *, argument, auxiliary):
    # Claim: -1 <= sin(t) <= 1 for every t.
    constraints = (auxiliary >= -1, auxiliary <= 1)
    positive = z3.And(argument >= 0, argument <= z3.RealVal("333/212"))
    negative = z3.And(
        argument >= z3.RealVal("-333/212"), argument < 0
    )
    return constraints + (
        z3.Implies(
            positive,
            z3.And(
                auxiliary >= argument - argument**3 / 6,
                auxiliary <= 6 * argument / (6 + argument**2),
            ),
        ),
        z3.Implies(
            negative,
            z3.And(
                auxiliary >= 6 * argument / (6 + argument**2),
                auxiliary <= argument - argument**3 / 6,
            ),
        ),
    )


def cosine_auxiliary_constraints(z3, *, argument, auxiliary):
    # Claim: -1 <= cos(t) <= 1 for every t. On |t|<=333/212,
    # 1-t^2/2 <= cos(t) <= 1-4t^2/(355/113)^2.
    central = z3.And(
        argument >= z3.RealVal("-333/212"),
        argument <= z3.RealVal("333/212"),
    )
    return (
        auxiliary >= -1,
        auxiliary <= 1,
        z3.Implies(
            central,
            z3.And(
                auxiliary >= 1 - argument**2 / 2,
                auxiliary
                <= 1 - 4 * argument**2 / z3.RealVal("355/113")**2,
            ),
        ),
    )


__all__ = ["cosine_auxiliary_constraints", "sine_auxiliary_constraints"]
