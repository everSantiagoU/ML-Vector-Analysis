from __future__ import annotations

import re

from sympy import E, cos, exp, log, pi, sin, sqrt, sympify, tan
from sympy.core.sympify import SympifyError

from mlvectoranalysis.input.input_normalizer import InputNormalizer


class InputValidator:
    def __init__(
        self,
        variable: str = "t",
        expected_components: int = 3,
        normalizer: InputNormalizer | None = None,
    ) -> None:
        self.variable = variable
        self.expected_components = expected_components
        self.normalizer = normalizer or InputNormalizer()
        self.allowed_names = {
            variable,
            "sin",
            "cos",
            "tan",
            "sqrt",
            "exp",
            "log",
            "pi",
            "E",
        }
        self.sympy_locals = {
            variable: sympify(variable),
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "sqrt": sqrt,
            "exp": exp,
            "log": log,
            "pi": pi,
            "E": E,
        }

    def validate_format(self, text: str) -> bool:
        try:
            vector_expression = self._extract_vector_expression(text)
        except ValueError:
            return False

        if not vector_expression:
            return False

        return self._has_wrapping_vector_delimiters(vector_expression)

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
            _, normalized_text = normalized_text.rsplit("=", 1)

        if normalized_text.startswith("<") or normalized_text.startswith("("):
            return normalized_text

        raise ValueError("Input text must contain a vector expression.")

    def _has_wrapping_vector_delimiters(self, text: str) -> bool:
        delimiter_pairs = {"<": ">", "(": ")"}
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
        try:
            sympify(component, locals=self.sympy_locals)
        except (SympifyError, TypeError, SyntaxError):
            return False

        return True

    def _uses_allowed_symbols(self, component: str) -> bool:
        identifiers = self._extract_identifiers(component)
        if not identifiers.issubset(self.allowed_names):
            return False

        return self._is_valid_component(component)

    def _extract_identifiers(self, text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z_]\w*", text))
