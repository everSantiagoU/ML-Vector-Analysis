from __future__ import annotations

import re

from sympy import Expr

from mlvectoranalysis.domain.component_function import ComponentFunction
from mlvectoranalysis.domain.vector_function import VectorFunction
from mlvectoranalysis.input.input_normalizer import InputNormalizer
from mlvectoranalysis.input.input_validator import InputValidator


class ExpressionParser:
    def __init__(
        self,
        variable: str = "t",
        normalizer: InputNormalizer | None = None,
        validator: InputValidator | None = None,
    ) -> None:
        self.variable = variable
        self.normalizer = normalizer if normalizer is not None else (
            validator.normalizer if validator is not None else InputNormalizer()
        )
        self.validator = validator if validator is not None else InputValidator(
            variable=variable,
            normalizer=self.normalizer,
        )
        if self.validator.variable != variable or self.validator.expected_components != 3:
            raise ValueError("Parser requires a validator with the same variable and three components.")

    def parse_component(self, expr: str) -> ComponentFunction:
        normalized_expr = self.normalizer.normalize(expr)
        if not normalized_expr:
            raise ValueError("Component expression cannot be empty.")

        symbolic_form = self.validator.parse_component_expression(normalized_expr)
        return self._create_component(normalized_expr, symbolic_form)

    def _create_component(self, normalized_expr: str, symbolic_form: Expr) -> ComponentFunction:
        component = ComponentFunction(symbolic_form, self.variable)
        component.expression = normalized_expr
        return component

    def parse_vector_expression(self, text: str) -> VectorFunction:
        normalized_text = self.normalizer.normalize(text)
        components = self.validator.extract_components(normalized_text)
        if len(components) != 3:
            raise ValueError("Vector expression must contain three valid components.")
        # Validate all three before constructing domain objects, without reinterpreting text.
        symbolic_forms = tuple(self.validator.parse_component_expression(expr) for expr in components)
        component_functions = tuple(
            self._create_component(expr, symbolic_form)
            for expr, symbolic_form in zip(components, symbolic_forms)
        )

        return VectorFunction(
            component_functions[0],
            component_functions[1],
            component_functions[2],
            name=self._extract_vector_name(normalized_text),
            variable=self.variable,
        )

    def _extract_vector_name(self, text: str) -> str:
        normalized_text = self.normalizer.normalize(text)
        match = re.match(r"^([A-Za-z_]\w*)\([^)]*\)=", normalized_text)
        if match:
            return match.group(1)

        return "r"
