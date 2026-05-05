from __future__ import annotations

import re

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
        self.normalizer = normalizer or InputNormalizer()
        self.validator = validator or InputValidator(
            variable=variable,
            normalizer=self.normalizer,
        )

    def parse_component(self, expr: str) -> ComponentFunction:
        normalized_expr = self.normalizer.normalize(expr)
        if not normalized_expr:
            raise ValueError("Component expression cannot be empty.")

        if not self.validator.validate_component_expression(normalized_expr):
            raise ValueError(f"Invalid component expression: {expr}")

        return ComponentFunction(normalized_expr, self.variable)

    def parse_vector_expression(self, text: str) -> VectorFunction:
        if not self.validator.validate_format(text):
            raise ValueError("Invalid vector expression format.")

        if not self.validator.validate_components(text):
            raise ValueError("Vector expression must contain three valid components.")

        if not self.validator.validate_allowed_symbols(text):
            raise ValueError("Vector expression contains unsupported symbols.")

        components = self.validator.extract_components(text)
        component_functions = tuple(
            self.parse_component(component) for component in components
        )

        return VectorFunction(
            component_functions[0],
            component_functions[1],
            component_functions[2],
            name=self._extract_vector_name(text),
            variable=self.variable,
        )

    def _extract_vector_name(self, text: str) -> str:
        normalized_text = self.normalizer.normalize(text)
        match = re.match(r"^([A-Za-z_]\w*)\([^)]*\)=", normalized_text)
        if match:
            return match.group(1)

        return "r"
