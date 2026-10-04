"""Optional alpha-beta-CROWN integration.

This module contains dependency loading and solver plumbing only. The proof
conditions live in ``verification/core/quadratic_crown.py``.
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from importlib import import_module
from io import StringIO
import logging
import math
import sys
from time import perf_counter
import warnings

import sympy as sp

from .quadratic import (
    OptionalVerificationDependencyError,
    VerificationObligation,
    VerificationStatus,
)


_PATH_NOTICE = "Adding complete_verifier to sys.path"
_HUGEPAGE_NOTICE = "Hugepage enabled for branch-and-bound domains."


def require_supported_expressions(states, expressions):
    """Reject symbolic operations outside the explicit CROWN compiler."""

    states = set(states)
    functions = {
        sp.Abs,
        sp.cos,
        sp.exp,
        sp.log,
        sp.sin,
        sp.tan,
        sp.tanh,
    }

    def visit(expression):
        expression = sp.sympify(expression)
        if isinstance(expression, sp.Symbol):
            if expression not in states:
                raise NotImplementedError(
                    "CROWN expressions may use only declared states."
                )
            return
        if not expression.free_symbols and expression.is_number:
            return
        if expression.is_Add or expression.is_Mul:
            for argument in expression.args:
                visit(argument)
            return
        if expression.is_Pow:
            base, exponent = expression.args
            if exponent.free_symbols:
                raise NotImplementedError(
                    "CROWN does not support symbolic exponents."
                )
            visit(base)
            return
        if expression.func in functions and len(expression.args) == 1:
            visit(expression.args[0])
            return
        raise NotImplementedError(
            "CROWN does not support symbolic expression {!s}.".format(
                expression
            )
        )

    for expression in expressions:
        visit(expression)


def _compile_expression(expression, symbol_indices, torch):
    """Compile supported SymPy arithmetic to a CROWN-traceable graph."""

    expression = sp.sympify(expression)
    if isinstance(expression, sp.Symbol):
        return lambda values: values[..., symbol_indices[expression]]
    if not expression.free_symbols and expression.is_number:
        constant = float(expression)
        return lambda values: values[..., 0] * 0 + constant
    if expression.is_Add:
        terms = tuple(
            _compile_expression(term, symbol_indices, torch)
            for term in expression.args
        )

        def add(values):
            result = terms[0](values)
            for term in terms[1:]:
                result = result + term(values)
            return result

        return add
    if expression.is_Mul:
        factors = tuple(
            _compile_expression(factor, symbol_indices, torch)
            for factor in expression.args
        )

        def multiply(values):
            result = factors[0](values)
            for factor in factors[1:]:
                result = result * factor(values)
            return result

        return multiply
    if expression.is_Pow:
        base, exponent = expression.args
        if exponent.free_symbols:
            raise NotImplementedError(
                "CROWN does not support symbolic exponents."
            )
        compiled = _compile_expression(base, symbol_indices, torch)
        power = int(exponent) if exponent.is_Integer else float(exponent)
        return lambda values: compiled(values) ** power

    functions = {
        sp.Abs: torch.abs,
        sp.cos: torch.cos,
        sp.exp: torch.exp,
        sp.log: torch.log,
        sp.sin: torch.sin,
        sp.tan: torch.tan,
        sp.tanh: torch.tanh,
    }
    function = functions.get(expression.func)
    if function is not None and len(expression.args) == 1:
        argument = _compile_expression(
            expression.args[0], symbol_indices, torch
        )
        return lambda values: function(argument(values))
    raise NotImplementedError(
        "CROWN does not support symbolic expression {!s}.".format(expression)
    )


def expression_graph(torch, states, expressions):
    """Return one Torch module with the requested symbolic outputs."""

    indices = {state: index for index, state in enumerate(states)}
    compiled = tuple(
        _compile_expression(expression, indices, torch)
        for expression in expressions
    )

    class ExpressionGraph(torch.nn.Module):
        def forward(self, values):
            return torch.stack(
                tuple(expression(values) for expression in compiled),
                dim=-1,
            )

    return ExpressionGraph()


def _keep_import_log(record):
    return record.getMessage() != _HUGEPAGE_NOTICE


def _replay_import_output(output):
    for line in output.splitlines(keepends=True):
        if line.rstrip("\r\n") != _PATH_NOTICE:
            sys.stdout.write(line)


def _show_import_warning(showwarning, message, category, *args, **kwargs):
    if (
        category is DeprecationWarning
        and str(message).startswith("`torch.jit.script` is deprecated.")
    ):
        return
    showwarning(message, category, *args, **kwargs)


def load_crown():
    """Load PyTorch and alpha-beta-CROWN only when requested."""

    output = StringIO()
    root_logger = logging.getLogger()
    try:
        torch = import_module("torch")
        root_logger.addFilter(_keep_import_log)
        try:
            with redirect_stdout(output), warnings.catch_warnings():
                showwarning = warnings.showwarning

                def show_import_warning(message, category, *args, **kwargs):
                    _show_import_warning(
                        showwarning, message, category, *args, **kwargs
                    )

                warnings.showwarning = show_import_warning
                api = import_module("abcrown")
        finally:
            root_logger.removeFilter(_keep_import_log)
            _replay_import_output(output.getvalue())
    except ImportError as error:
        raise OptionalVerificationDependencyError(
            "CROWN verification requires optional external dependencies "
            "PyTorch and alpha-beta-CROWN."
        ) from error
    required = (
        "ABCrownSolver",
        "ConfigBuilder",
        "VerificationSpec",
        "output_vars",
    )
    if any(not hasattr(api, name) for name in required):
        raise OptionalVerificationDependencyError(
            "The installed alpha-beta-CROWN lacks its Python verification "
            "API."
        )
    return torch, api


def normalise_device(torch, device):
    if not isinstance(device, str):
        raise TypeError("crown_device must be 'auto', 'cpu', or 'cuda'.")
    result = device.strip().lower()
    if result == "auto":
        result = "cuda" if torch.cuda.is_available() else "cpu"
    if result not in ("cpu", "cuda"):
        raise ValueError("crown_device must be 'auto', 'cpu', or 'cuda'.")
    if result == "cuda" and not torch.cuda.is_available():
        raise ValueError("CROWN was asked to use unavailable CUDA hardware.")
    return result


def _negated_clauses(
    predicate, *, comparison_type, conjunction_type, disjunction_type
):
    """Return DNF clauses for the negation of a strict predicate."""

    if isinstance(predicate, comparison_type):
        if predicate.op == "<":
            return ((predicate.lhs >= predicate.rhs,),)
        if predicate.op == ">":
            return ((predicate.lhs <= predicate.rhs,),)
        raise ValueError("CROWN predicates require strict inequalities.")
    if isinstance(predicate, conjunction_type):
        return (
            _negated_clauses(
                predicate.left,
                comparison_type=comparison_type,
                conjunction_type=conjunction_type,
                disjunction_type=disjunction_type,
            )
            + _negated_clauses(
                predicate.right,
                comparison_type=comparison_type,
                conjunction_type=conjunction_type,
                disjunction_type=disjunction_type,
            )
        )
    if isinstance(predicate, disjunction_type):
        left = _negated_clauses(
            predicate.left,
            comparison_type=comparison_type,
            conjunction_type=conjunction_type,
            disjunction_type=disjunction_type,
        )
        right = _negated_clauses(
            predicate.right,
            comparison_type=comparison_type,
            conjunction_type=conjunction_type,
            disjunction_type=disjunction_type,
        )
        return tuple(a + b for a in left for b in right)
    raise TypeError("Unsupported CROWN predicate type.")


def _linear_clause(torch, clause, output_dimension, dtype):
    rows = []
    right_hand_sides = []
    for predicate in clause:
        expression = predicate.normalized_expr()
        row = [0.0] * output_dimension
        if not expression.coeffs:
            raise ValueError("A CROWN constraint must involve an output.")
        for (kind, index), coefficient in expression.coeffs.items():
            if kind != "output" or not 0 <= index < output_dimension:
                raise ValueError("Invalid CROWN output constraint.")
            row[index] += coefficient
        rows.append(row)
        right_hand_sides.append(-expression.constant)
    return (
        torch.tensor(rows, dtype=dtype),
        torch.tensor(right_hand_sides, dtype=dtype),
    )


def _specification(
    torch, api, bounds, output_dimension, output_constraint, dtype
):
    outputs = api.output_vars(output_dimension)
    sample = outputs[0] < 0
    predicate = output_constraint(outputs)
    clauses = _negated_clauses(
        predicate,
        comparison_type=type(sample),
        conjunction_type=type(sample & sample),
        disjunction_type=type(sample | sample),
    )
    unsafe_clauses = [[
        _linear_clause(torch, clause, output_dimension, dtype)
    ] for clause in clauses]
    lower = torch.tensor(
        [[float(bound[0]) for bound in bounds]], dtype=dtype
    )
    upper = torch.tensor(
        [[float(bound[1]) for bound in bounds]], dtype=dtype
    )
    count = max(1, len(unsafe_clauses))
    return api.VerificationSpec.build_spec(
        lower=lower.repeat(count, 1),
        upper=upper.repeat(count, 1),
        clauses=unsafe_clauses,
    )


def _status(result):
    backend_status = str(result.status)
    if bool(result.success) and (
        backend_status == "verified" or backend_status.startswith("safe")
    ):
        return VerificationStatus.VERIFIED, backend_status
    if bool(result.success) and (
        backend_status == "falsified" or backend_status.startswith("unsafe")
    ):
        return VerificationStatus.COUNTEREXAMPLE, backend_status
    return VerificationStatus.UNKNOWN, backend_status


class CrownRunner:
    """Prove output predicates on boxes with alpha-beta-CROWN."""

    def __init__(
        self, *, torch, api, device, timeout, batch_size, verbose
    ):
        self.torch = torch
        self.api = api
        self.device = device
        self.timeout = math.inf if timeout is None else float(timeout)
        self.batch_size = int(batch_size)
        self.verbose = bool(verbose)

    def solve(
        self, name, model, output_dimension, output_constraint, bounds
    ):
        stream = StringIO()
        started = perf_counter()
        original_dtype = self.torch.get_default_dtype()
        try:
            self.torch.set_default_dtype(self.torch.float64)
            output_context = (
                _NullContext() if self.verbose else redirect_stdout(stream)
            )
            error_context = (
                _NullContext() if self.verbose else redirect_stderr(stream)
            )
            with output_context, error_context:
                specification = _specification(
                    self.torch,
                    self.api,
                    bounds,
                    output_dimension,
                    output_constraint,
                    self.torch.float64,
                )
                graph = deepcopy(model).eval().to(
                    device=self.device, dtype=self.torch.float64
                )
                config = self.api.ConfigBuilder.from_defaults().set(
                    general__device=self.device,
                    general__double_fp=True,
                    attack__pgd_order="skip",
                    solver__batch_size=self.batch_size,
                    bab__timeout=self.timeout,
                )
                result = self.api.ABCrownSolver(
                    specification, graph, config=config
                ).solve()
        except (AttributeError, NotImplementedError, RuntimeError, ValueError):
            return VerificationObligation(
                name=name,
                status=VerificationStatus.UNKNOWN,
                elapsed_time=perf_counter() - started,
                backend_status="solver-error",
            )
        finally:
            self.torch.set_default_dtype(original_dtype)

        status, backend_status = _status(result)
        return VerificationObligation(
            name=name,
            status=status,
            elapsed_time=perf_counter() - started,
            backend_status=backend_status,
        )


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        return False


__all__ = [
    "CrownRunner",
    "expression_graph",
    "load_crown",
    "normalise_device",
    "require_supported_expressions",
]
