"""Mathematical predicates for CROWN quadratic-CLF proofs."""


def local_contraction_condition(
    output, *, level, proof_margin, state_dimension
):
    # Outputs: V, m_1,...,m_n, where m_k is a leading principal minor of
    # -(Df_K.T P+P Df_K). Claim: V>c1+eps or every m_k>eps.
    every_minor_positive = output[1] > proof_margin
    for index in range(2, 1 + state_dimension):
        every_minor_positive = (
            every_minor_positive & (output[index] > proof_margin)
        )
    return (output[0] > level + proof_margin) | every_minor_positive


def annular_artstein_condition(
    output,
    *,
    local_level,
    outer_level,
    proof_margin,
    input_dimension,
):
    # Outputs: V, L_f V, L_g1 V,...,L_gm V. Claim:
    # V<c1-eps or V>c2+eps or some |L_gj V|>eps or L_f V<-eps.
    some_control_effect = (
        (output[2] > proof_margin) | (output[2] < -proof_margin)
    )
    for index in range(3, 2 + input_dimension):
        some_control_effect = (
            some_control_effect
            | (output[index] > proof_margin)
            | (output[index] < -proof_margin)
        )
    return (
        (output[0] < local_level - proof_margin)
        | (output[0] > outer_level + proof_margin)
        | some_control_effect
        | (output[1] < -proof_margin)
    )


__all__ = ["annular_artstein_condition", "local_contraction_condition"]
