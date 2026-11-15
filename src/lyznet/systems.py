"""Canonical continuous-time system definitions and numerical views."""

import re

import numpy as np
import sympy as sp

from .equilibria import NumericalEquilibrium, _compile_dynamics
from .sets import Box, EuclideanBall


class _ReadOnly:
    """Permit construction and internal caching, but no later assignment."""

    _is_read_only = False

    def __setattr__(self, name, value):
        if self._is_read_only:
            raise AttributeError(
                "{} is read-only after construction.".format(type(self).__name__)
            )
        object.__setattr__(self, name, value)

    def _make_read_only(self):
        object.__setattr__(self, "_is_read_only", True)


class System(_ReadOnly):
    """Base class and namespace for concrete system constructors."""

    def __new__(cls, *args, **kwargs):
        if cls is System:
            raise TypeError(
                "System is a base class; construct a concrete system type."
            )
        return super().__new__(cls)

    @classmethod
    def continuous(cls, **kwargs):
        return ContinuousTimeSystem(**kwargs)

    @classmethod
    def control_affine(cls, **kwargs):
        return ControlAffineSystem(**kwargs)


def _normalise_symbols(symbols, label):
    values = tuple(symbols)
    if not values:
        raise ValueError("{} must contain at least one symbol.".format(label))
    if not all(isinstance(value, sp.Symbol) for value in values):
        raise TypeError("{} must contain only SymPy symbols.".format(label))
    if len(set(values)) != len(values):
        raise ValueError("{} must not contain duplicate symbols.".format(label))
    return values


def _infer_states(dynamics):
    """Infer an unambiguous conventional state sequence from dynamics."""

    expressions = tuple(sp.sympify(value) for value in list(sp.Matrix(dynamics)))
    symbols = set().union(*(expression.free_symbols for expression in expressions))
    if len(symbols) != len(expressions):
        raise ValueError(
            "Cannot infer states: dynamics must contain exactly one distinct "
            "state symbol per equation; pass states explicitly."
        )
    if len(symbols) == 1:
        return tuple(symbols)

    indexed = []
    for symbol in symbols:
        match = re.fullmatch(r"(.*?)(\d+)", symbol.name)
        if match is None:
            raise ValueError(
                "Cannot infer state order; pass states explicitly."
            )
        indexed.append((match.group(1), int(match.group(2)), symbol))
    prefixes = {prefix for prefix, _, _ in indexed}
    indices = sorted(index for _, index, _ in indexed)
    expected = list(range(indices[0], indices[0] + len(indices)))
    if len(prefixes) != 1 or indices != expected or indices[0] not in (0, 1):
        raise ValueError("Cannot infer state order; pass states explicitly.")
    return tuple(symbol for _, _, symbol in sorted(indexed, key=lambda item: item[1]))


def _normalise_vector(expressions, expected_size, label):
    try:
        matrix = sp.ImmutableMatrix(expressions)
    except TypeError:
        matrix = sp.ImmutableMatrix([expressions])
    values = tuple(sp.sympify(value) for value in list(matrix))
    if len(values) != expected_size:
        raise ValueError(
            "{} must contain {} expressions, received {}.".format(
                label, expected_size, len(values)
            )
        )
    return values


def _normalise_matrix(expressions, rows, columns, label):
    try:
        matrix = sp.ImmutableMatrix(expressions)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be a rectangular matrix.".format(label)) from error
    if matrix.shape != (rows, columns):
        raise ValueError(
            "{} must have shape ({}, {}), received {}.".format(
                label, rows, columns, matrix.shape
            )
        )
    return matrix


def _validate_free_symbols(expressions, allowed_symbols, label):
    free_symbols = set().union(
        *(expression.free_symbols for expression in expressions)
    )
    undeclared = free_symbols.difference(allowed_symbols)
    if undeclared:
        names = ", ".join(sorted(str(symbol) for symbol in undeclared))
        raise ValueError("{} contains undeclared symbols: {}.".format(label, names))


def _normalise_domain(domain, dimension):
    if domain is None:
        return None
    region = domain if isinstance(domain, Box) else Box(domain)
    if region.dimension != dimension:
        raise ValueError(
            "Domain dimension {} does not match state dimension {}.".format(
                region.dimension, dimension
            )
        )
    return region


def _normalise_input_set(input_set, dimension):
    if input_set is None:
        return None
    if not isinstance(input_set, (Box, EuclideanBall)):
        raise TypeError("input_set must be a Box, EuclideanBall, or None.")
    if input_set.dimension != dimension:
        raise ValueError(
            "Input-set dimension {} does not match input dimension {}.".format(
                input_set.dimension, dimension
            )
        )
    return input_set


def _freeze_float(value):
    """Interpret a stored float's round-trip decimal as an exact rational."""

    return sp.Rational(repr(float(value)))


def _input_set_contains_exact(input_set, point):
    if isinstance(input_set, Box):
        margins = []
        for value, (lower, upper) in zip(point, input_set.bounds):
            margins.extend(
                (sp.simplify(value - _freeze_float(lower)),
                 sp.simplify(_freeze_float(upper) - value))
            )
        return all(margin.is_nonnegative is True for margin in margins)

    distance_squared = sum(
        (value - _freeze_float(center)) ** 2
        for value, center in zip(point, input_set.center)
    )
    margin = sp.simplify(_freeze_float(input_set.radius) ** 2 - distance_squared)
    return margin.is_nonnegative is True


def _normalise_exact_point(values, dimension, label):
    try:
        values = tuple(sp.sympify(value) for value in values)
    except TypeError as error:
        raise TypeError(
            "{} must contain {} scalar values.".format(label, dimension)
        ) from error
    if len(values) != dimension:
        raise ValueError(
            "{} must contain {} values, received {}.".format(
                label, dimension, len(values)
            )
        )
    if any(
        not isinstance(value, sp.Expr)
        or value.free_symbols
        or value.is_number is not True
        or value.is_real is not True
        or value.is_finite is not True
        for value in values
    ):
        raise ValueError(
            "{} must contain only closed finite real scalars.".format(label)
        )
    return values


def _normalise_equilibrium(equilibrium, dimension):
    if equilibrium is None:
        return tuple(sp.Integer(0) for _ in range(dimension)), None
    if isinstance(equilibrium, NumericalEquilibrium):
        if equilibrium.dimension != dimension:
            raise ValueError(
                "Equilibrium must contain {} values, received {}.".format(
                    dimension, equilibrium.dimension
                )
            )
        return (
            tuple(sp.Float(repr(value)) for value in equilibrium.point),
            equilibrium,
        )
    return (
        _normalise_exact_point(equilibrium, dimension, "Equilibrium"),
        None,
    )


def _validate_numerical_equilibrium(symbolic, solution):
    if tuple(symbolic.states) != solution.states:
        raise ValueError(
            "Numerical equilibrium states do not match the system states."
        )
    if len(solution.dynamics) != len(symbolic.dynamics) or any(
        sp.simplify(left - right) != 0
        for left, right in zip(symbolic.dynamics, solution.dynamics)
    ):
        raise ValueError(
            "Numerical equilibrium dynamics do not match the system dynamics."
        )
    residual = _compile_dynamics(
        symbolic.states, tuple(symbolic.dynamics)
    )(solution.point)
    if (
        residual.shape != (len(symbolic.states),)
        or not np.all(np.isfinite(residual))
        or np.linalg.norm(residual, ord=np.inf)
        > solution.residual_tolerance
    ):
        raise ValueError(
            "Numerical equilibrium does not meet its residual tolerance."
        )


def _validate_equilibrium(states, dynamics, equilibrium, *, error_message):
    """Require the continuous-time equilibrium condition ``f(x*) = 0``."""

    substitutions = dict(zip(states, equilibrium))
    evaluated = sp.ImmutableMatrix(dynamics).subs(substitutions, simultaneous=True)
    residuals = tuple(sp.simplify(value) for value in evaluated)
    if any(residual != 0 for residual in residuals):
        raise ValueError(error_message)


def _normalise_numpy_input(values, width, label):
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != width:
        raise ValueError(
            "{} must have shape (..., {}), received {}.".format(
                label, width, array.shape
            )
        )
    if np.iscomplexobj(array):
        raise ValueError("{} must contain real values.".format(label))
    if not np.issubdtype(array.dtype, np.floating):
        try:
            array = array.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "{} must contain real numeric values.".format(label)
            ) from error
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain finite values.".format(label))
    return array


class _NumpyExpressions(_ReadOnly):
    def __init__(self, symbols, expressions, output_shape, label):
        self._symbols = tuple(symbols)
        self._functions = tuple(
            sp.lambdify(self._symbols, expression, modules="numpy")
            for expression in expressions
        )
        self._output_shape = tuple(output_shape)
        self._label = label
        self._make_read_only()

    def __call__(self, values):
        array = _normalise_numpy_input(
            values, len(self._symbols), "State values"
        )
        flat = array.reshape(-1, len(self._symbols))
        arguments = tuple(flat[:, index] for index in range(flat.shape[1]))
        columns = []
        for function in self._functions:
            with np.errstate(all="ignore"):
                result = np.asarray(function(*arguments))
            if np.iscomplexobj(result):
                raise ValueError("{} evaluated to complex values.".format(self._label))
            result = np.asarray(result, dtype=array.dtype)
            if result.ndim == 0:
                result = np.full(flat.shape[0], result.item(), dtype=array.dtype)
            else:
                result = np.broadcast_to(result, (flat.shape[0],)).astype(
                    array.dtype, copy=False
                )
            if not np.all(np.isfinite(result)):
                raise ValueError(
                    "{} evaluated to non-finite values.".format(self._label)
                )
            columns.append(result)
        stacked = np.stack(columns, axis=-1)
        return stacked.reshape(array.shape[:-1] + self._output_shape)


class _AutonomousSymbolicView(_ReadOnly):
    def __init__(self, states, dynamics):
        self.states = tuple(states)
        self.dynamics = sp.ImmutableMatrix(len(dynamics), 1, dynamics)
        self.jacobian = self.dynamics.jacobian(self.states)
        self._make_read_only()


class _AutonomousNumpyView(_ReadOnly):
    def __init__(self, states, dynamics):
        self._dynamics = _NumpyExpressions(
            states, dynamics, (len(dynamics),), "Dynamics"
        )
        self._make_read_only()

    def dynamics(self, states):
        """Evaluate ``f(x)`` at finite real states shaped ``(..., n)``."""

        return self._dynamics(states)


class _ControlSymbolicView(_ReadOnly):
    def __init__(self, states, inputs, drift, input_matrix):
        self.states = tuple(states)
        self.inputs = tuple(inputs)
        self.drift = sp.ImmutableMatrix(len(drift), 1, drift)
        self.input_matrix = sp.ImmutableMatrix(input_matrix)
        self.drift_jacobian = self.drift.jacobian(self.states)
        self._make_read_only()


class _ControlNumpyView(_ReadOnly):
    def __init__(self, states, inputs, drift, input_matrix):
        self._state_dimension = len(states)
        self._input_dimension = len(inputs)
        self._drift = _NumpyExpressions(
            states, drift, (self._state_dimension,), "Drift"
        )
        self._input_matrix = _NumpyExpressions(
            states,
            tuple(input_matrix),
            (self._state_dimension, self._input_dimension),
            "Input matrix",
        )
        self._make_read_only()

    def drift(self, states):
        return self._drift(states)

    def input_matrix(self, states):
        return self._input_matrix(states)

    def dynamics(self, states, inputs):
        state_values = _normalise_numpy_input(
            states, self._state_dimension, "State values"
        )
        input_values = _normalise_numpy_input(
            inputs, self._input_dimension, "Input values"
        )
        if state_values.shape[:-1] != input_values.shape[:-1]:
            raise ValueError(
                "State and input leading dimensions must match, received "
                "{} and {}.".format(state_values.shape, input_values.shape)
            )
        return self.drift(state_values) + np.matmul(
            self.input_matrix(state_values), input_values[..., None]
        )[..., 0]


class ContinuousTimeSystem(System):
    """An autonomous continuous-time system ``x_dot = f(x)``."""

    def __init__(
        self, *, dynamics, states=None, domain=None, equilibrium=None, name=""
    ):
        if states is None:
            states = _infer_states(dynamics)
        self.states = _normalise_symbols(states, "states")
        expressions = _normalise_vector(dynamics, len(self.states), "dynamics")
        _validate_free_symbols(expressions, set(self.states), "dynamics")
        self.name = str(name)
        self.domain = _normalise_domain(domain, len(self.states))
        self.equilibrium, self.numerical_equilibrium = _normalise_equilibrium(
            equilibrium, len(self.states)
        )
        self.symbolic = _AutonomousSymbolicView(self.states, expressions)
        if self.numerical_equilibrium is None:
            _validate_equilibrium(
                self.states,
                self.symbolic.dynamics,
                self.equilibrium,
                error_message=(
                    "Declared equilibrium is not an equilibrium of the "
                    "dynamics."
                ),
            )
        else:
            _validate_numerical_equilibrium(
                self.symbolic, self.numerical_equilibrium
            )
        self._numpy = None
        self._make_read_only()

    @property
    def state_dimension(self):
        return len(self.states)

    @property
    def numpy(self):
        if self._numpy is None:
            object.__setattr__(
                self,
                "_numpy",
                _AutonomousNumpyView(
                    self.states, tuple(self.symbolic.dynamics)
                ),
            )
        return self._numpy

    def __repr__(self):
        return "{}(name={!r}, state_dimension={})".format(
            type(self).__name__, self.name, self.state_dimension
        )


class ControlAffineSystem(System):
    """A continuous-time system ``x_dot = f(x) + g(x) u``."""

    def __init__(
        self,
        *,
        states,
        inputs,
        drift,
        input_matrix,
        domain=None,
        input_set=None,
        equilibrium=None,
        equilibrium_input=None,
        name="",
    ):
        self.states = _normalise_symbols(states, "states")
        self.inputs = _normalise_symbols(inputs, "inputs")
        overlap = set(self.states).intersection(self.inputs)
        if overlap:
            names = ", ".join(sorted(str(symbol) for symbol in overlap))
            raise ValueError(
                "State and input symbols must be distinct; overlap: {}.".format(names)
            )

        drift = _normalise_vector(drift, len(self.states), "drift")
        input_matrix = _normalise_matrix(
            input_matrix,
            len(self.states),
            len(self.inputs),
            "input_matrix",
        )
        _validate_free_symbols(drift, set(self.states), "drift")
        _validate_free_symbols(tuple(input_matrix), set(self.states), "input_matrix")

        self.name = str(name)
        self.domain = _normalise_domain(domain, len(self.states))
        self.input_set = _normalise_input_set(input_set, len(self.inputs))
        self.equilibrium, self.numerical_equilibrium = _normalise_equilibrium(
            equilibrium, len(self.states)
        )
        self.equilibrium_input = _normalise_exact_point(
            (
                (sp.Integer(0) for _ in self.inputs)
                if equilibrium_input is None
                else equilibrium_input
            ),
            len(self.inputs),
            "equilibrium_input",
        )
        if self.input_set is not None and not _input_set_contains_exact(
            self.input_set, self.equilibrium_input
        ):
            raise ValueError(
                "equilibrium_input must belong to input_set in physical "
                "input coordinates."
            )

        self.symbolic = _ControlSymbolicView(
            self.states, self.inputs, drift, input_matrix
        )
        controlled_dynamics = (
            self.symbolic.drift
            + self.symbolic.input_matrix * sp.ImmutableMatrix(self.equilibrium_input)
        )
        if self.numerical_equilibrium is None:
            _validate_equilibrium(
                self.states,
                controlled_dynamics,
                self.equilibrium,
                error_message=(
                    "Declared state and input are not an equilibrium of the "
                    "control-affine dynamics."
                ),
            )
        else:
            if any(value != 0 for value in self.equilibrium_input):
                raise ValueError(
                    "A numerical equilibrium currently requires zero "
                    "equilibrium input."
                )
            numerical_symbolic = _AutonomousSymbolicView(
                self.states, drift
            )
            _validate_numerical_equilibrium(
                numerical_symbolic, self.numerical_equilibrium
            )
        self._numpy = None
        self._make_read_only()

    @property
    def state_dimension(self):
        return len(self.states)

    @property
    def input_dimension(self):
        return len(self.inputs)

    @property
    def numpy(self):
        if self._numpy is None:
            object.__setattr__(
                self,
                "_numpy",
                _ControlNumpyView(
                    self.states,
                    self.inputs,
                    tuple(self.symbolic.drift),
                    self.symbolic.input_matrix,
                ),
            )
        return self._numpy

    def closed_loop(self, controller, name=None):
        controller = _normalise_vector(
            controller, self.input_dimension, "controller"
        )
        _validate_free_symbols(controller, set(self.states), "controller")
        dynamics = (
            self.symbolic.drift
            + self.symbolic.input_matrix * sp.ImmutableMatrix(controller)
        )
        equilibrium = self.equilibrium
        if self.numerical_equilibrium is not None:
            dynamics_tuple = tuple(dynamics)
            residual = _compile_dynamics(
                self.states, dynamics_tuple
            )(self.numerical_equilibrium.point)
            try:
                equilibrium = NumericalEquilibrium(
                    states=self.states,
                    dynamics=dynamics_tuple,
                    point=self.numerical_equilibrium.point,
                    residual=tuple(residual),
                    residual_tolerance=(
                        self.numerical_equilibrium.residual_tolerance
                    ),
                    method=(
                        self.numerical_equilibrium.method
                        + "; closed-loop residual evaluation"
                    ),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Controller does not preserve the numerical equilibrium "
                    "within its residual tolerance."
                ) from error
        return ContinuousTimeSystem(
            states=self.states,
            dynamics=tuple(dynamics),
            domain=self.domain,
            equilibrium=equilibrium,
            name=(
                name if name is not None else "{}-closed-loop".format(self.name)
            ),
        )

    def __repr__(self):
        return (
            "{}(name={!r}, state_dimension={}, input_dimension={})".format(
                type(self).__name__,
                self.name,
                self.state_dimension,
                self.input_dimension,
            )
        )


__all__ = ["ContinuousTimeSystem", "ControlAffineSystem", "System"]
