"""Restricted scalar f(x,y,z); deliberately distinct from a V1 vector curve."""
import numpy as np
from sympy import Symbol, diff, lambdify

from .._scalar_expression import parse_scalar_variables
from ..input.input_normalizer import InputNormalizer
from ..optimization._validation import finite_scalar, parameters
from ..optimization.objective import ObjectiveEvaluationError


class MultivariableFunction:
    dimension = 3
    parameter_names = ('x', 'y', 'z')
    coordinate_system = 'xyz'

    def __init__(self, expression: str):
        self.expression = InputNormalizer().normalize(expression)
        self.symbolic_form = parse_scalar_variables(self.expression, self.parameter_names)
        self.symbols = tuple(Symbol(name) for name in self.parameter_names)
        self.symbolic_gradient = tuple(diff(self.symbolic_form, symbol) for symbol in self.symbols)
        # Compilation only sees expressions produced by the validated grammar.
        try:
            self._value = lambdify(self.symbols, self.symbolic_form, modules='numpy', cse=True)
            self._value_grad = lambdify(
                self.symbols, (self.symbolic_form, *self.symbolic_gradient), modules='numpy', cse=True)
        except (KeyError, TypeError) as error:
            raise ValueError('Expression cannot be compiled as a real numerical function.') from error

    @classmethod
    def from_text(cls, expression: str):
        return cls(expression)

    def value(self, point):
        point = parameters(point, 3)
        try:
            with np.errstate(all='raise', under='ignore'):
                return finite_scalar(self._value(*point))
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            raise ObjectiveEvaluationError('Point is outside the real finite domain of f.') from error

    def value_and_grad(self, point):
        point = parameters(point, 3)
        try:
            with np.errstate(all='raise', under='ignore'):
                values = self._value_grad(*point)
                values = [finite_scalar(value) for value in values]
            return values[0], np.array(values[1:], dtype=np.float64)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            raise ObjectiveEvaluationError('Point is outside the real finite domain of f or its gradient.') from error

    def gradient(self, point):
        return self.value_and_grad(point)[1]
