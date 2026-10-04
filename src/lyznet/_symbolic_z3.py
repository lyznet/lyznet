"""Translate supported SymPy arithmetic into exact Z3 arithmetic."""

import sympy as sp


def sympy_to_z3(
    expression,
    substitutions,
    z3,
    *,
    allow_trigonometric_bounds=False,
    function_substitutions=None,
    extra_constraints=None,
):
    """Encode arithmetic, optionally enclosing sine and cosine auxiliaries."""

    functions = {} if function_substitutions is None else function_substitutions
    constraints = [] if extra_constraints is None else extra_constraints

    def convert(value):
        value = sp.sympify(value)
        if value in substitutions:
            return substitutions[value]
        if value.is_Integer:
            return z3.RealVal(int(value))
        if value.is_Rational:
            return z3.RealVal("{}/{}".format(value.p, value.q))
        if value.is_Float:
            return z3.RealVal(str(value))
        if value.is_Add:
            return z3.Sum([convert(term) for term in value.args])
        if value.is_Mul:
            result = z3.RealVal(1)
            for factor in value.args:
                result *= convert(factor)
            return result
        if value.is_Pow:
            base, exponent = value.args
            if not exponent.is_Integer or exponent < 0:
                raise NotImplementedError(
                    "Z3 conversion supports nonnegative integer powers."
                )
            return convert(base) ** int(exponent)
        if value.func is sp.Abs and allow_trigonometric_bounds:
            argument = convert(value.args[0])
            return z3.If(argument >= 0, argument, -argument)
        if value.func in (sp.sin, sp.cos) and allow_trigonometric_bounds:
            if value in functions:
                return functions[value]
            auxiliary = z3.Real(
                "{}_{}".format(value.func.__name__, len(functions))
            )
            functions[value] = auxiliary
            from .verification.core.trigonometric_z3 import (
                cosine_auxiliary_constraints,
                sine_auxiliary_constraints,
            )
            builder = (
                sine_auxiliary_constraints
                if value.func is sp.sin
                else cosine_auxiliary_constraints
            )
            constraints.extend(builder(
                z3, argument=convert(value.args[0]), auxiliary=auxiliary
            ))
            return auxiliary
        if value.is_Function:
            raise NotImplementedError(
                "Z3 supports polynomial arithmetic unless trigonometric "
                "bounds are enabled."
            )
        raise NotImplementedError(
            "Z3 does not support expression {!s}.".format(value)
        )

    return convert(expression)


__all__ = ["sympy_to_z3"]
