from __future__ import annotations
from typing import Union
from sympy import Symbol, cos, diff, exp, expand, integrate, simplify, sin, sympify, tan


class ComponentFunction:
    def __init__(self, expression: str, variable: str = "t") -> None:
        self.expression = expression
        self.variable = variable
        self.symbol = Symbol(variable)
        self.symbolic_form = sympify(expression)

    def simplify(self) -> "ComponentFunction":
        simplified_expr = simplify(self.symbolic_form)
        return ComponentFunction(str(simplified_expr), self.variable)

    def expand(self) -> "ComponentFunction":
        expanded_expr = expand(self.symbolic_form)
        return ComponentFunction(str(expanded_expr), self.variable)

    def differentiate(self) -> "ComponentFunction":
        derivative_expr = diff(self.symbolic_form, self.symbol)
        return ComponentFunction(str(derivative_expr), self.variable)

    def integrate(self) -> "ComponentFunction":
        integrated_expr = integrate(self.symbolic_form, self.symbol)
        return ComponentFunction(str(integrated_expr), self.variable)

    def evaluate(self, value: float) -> float:
        result = self.symbolic_form.evalf(subs={self.symbol: value})
        return float(result)

    def is_equivalent_to(self, other: Union[str, "ComponentFunction"]) -> bool:
        other_expr = other.symbolic_form if isinstance(other, ComponentFunction) else sympify(other)
        equivalence = self.symbolic_form.equals(other_expr)
        return bool(equivalence)

    def get_type(self) -> str:
        expr = self.symbolic_form

        if expr.is_polynomial(self.symbol):
            degree = expr.as_poly(self.symbol).degree()
            if degree == 1:
                return "linear"
            return "polynomial"

        if expr.has(sin, cos, tan):
            return "trigonometric"

        if expr.has(exp):
            return "exponential"

        if expr.is_rational_function(self.symbol):
            return "rational"

        return "unknown"
