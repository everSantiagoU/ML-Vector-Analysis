from __future__ import annotations

import re

from sympy import Expr, Symbol

from mlvectoranalysis._scalar_expression import CONSTANTS, FUNCTIONS, parse_scalar, validate_variable

from mlvectoranalysis.input.input_normalizer import InputNormalizer


class InputValidator:
    def __init__(
        self,
        variable: str = "t",
        expected_components: int = 3,
        normalizer: InputNormalizer | None = None,
    ) -> None:
        validate_variable(variable)
        self.variable = variable
        self.expected_components = expected_components
        self.normalizer = normalizer if normalizer is not None else InputNormalizer()
        self.allowed_names = {variable, *FUNCTIONS, *CONSTANTS}
        self.sympy_locals = {variable: Symbol(variable), **FUNCTIONS, **CONSTANTS}

    def validate_format(self, text: str) -> bool:
        try:
            self.extract_components(text)
        except ValueError:
            return False
        return True

    def validate_components(self, text: str) -> bool:
        try:
            components = self.extract_components(text)
        except ValueError:
            return False

        if len(components) != self.expected_components:
            return False

        return all(self._is_valid_component(component) for component in components)

    def validate_allowed_symbols(self, text: str) -> bool:
        try:
            components = self.extract_components(text)
        except ValueError:
            return False

        return all(self._uses_allowed_symbols(component) for component in components)

    def validate_component_expression(self, expr: str) -> bool:
        try:
            self.parse_component_expression(expr)
        except ValueError:
            return False
        return True

    def parse_component_expression(self, expr: str) -> Expr:
        """Validate and construct once; shared by the parser and boolean checks."""
        return parse_scalar(self.normalizer.normalize(expr), self.variable)

    def extract_components(self, text: str) -> tuple[str, ...]:
        vector_expression = self._extract_vector_expression(text)
        if not self._has_wrapping_vector_delimiters(vector_expression):
            raise ValueError("Vector expression must be enclosed in angle brackets or parentheses.")

        inner_expression = vector_expression[1:-1]
        components = self._split_top_level_components(inner_expression)
        if any(component == "" for component in components):
            raise ValueError("Vector expression contains empty components.")

        return tuple(components)

    def _extract_vector_expression(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")

        normalized_text = self.normalizer.normalize(text)
        if not normalized_text:
            raise ValueError("Input text cannot be empty.")

        if "=" in normalized_text:
            if normalized_text.count("=") != 1:
                raise ValueError("Vector input must contain a single assignment.")
            header, normalized_text = normalized_text.split("=")
            match = re.fullmatch(r"([A-Za-z_]\w*)\(([^()]*)\)", header)
            if match is None or match.group(2) != self.variable:
                raise ValueError(f"Expected a vector header name({self.variable})=.")

        if normalized_text.startswith("<") or normalized_text.startswith("("):
            return normalized_text

        raise ValueError("Input text must contain a vector expression.")

    def _has_wrapping_vector_delimiters(self, text: str) -> bool:
        delimiter_pairs = {"<": ">", "(": ")"}
        if not text:
            return False
        opening_delimiter = text[0]
        closing_delimiter = delimiter_pairs.get(opening_delimiter)
        return closing_delimiter is not None and text.endswith(closing_delimiter)

    def _split_top_level_components(self, text: str) -> list[str]:
        components: list[str] = []
        current_component: list[str] = []
        depth = 0

        for character in text:
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("Vector expression has unbalanced parentheses.")

            if character == "," and depth == 0:
                components.append("".join(current_component))
                current_component = []
                continue

            current_component.append(character)

        if depth != 0:
            raise ValueError("Vector expression has unbalanced parentheses.")

        components.append("".join(current_component))
        return components

    def _is_valid_component(self, component: str) -> bool:
        return self.validate_component_expression(component)

    def _uses_allowed_symbols(self, component: str) -> bool:
        return self.validate_component_expression(component)
