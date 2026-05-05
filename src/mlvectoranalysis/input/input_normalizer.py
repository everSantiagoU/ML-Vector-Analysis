from __future__ import annotations

import re


class InputNormalizer:
    def normalize(self, text: str) -> str:
        text_without_extra_spaces = self.remove_extra_spaces(text)
        return self.standardize_syntax(text_without_extra_spaces)

    def remove_extra_spaces(self, text: str) -> str:
        compact_text = re.sub(r"\s+", " ", text.strip())
        return re.sub(r"\s*([,()<>+\-*/=])\s*", r"\1", compact_text)

    def standardize_syntax(self, text: str) -> str:
        standardized_text = text
        standardized_text = standardized_text.replace("^", "**")
        standardized_text = standardized_text.replace("π", "pi")
        standardized_text = self._standardize_square_roots(standardized_text)
        return self._standardize_function_names(standardized_text)

    def _standardize_square_roots(self, text: str) -> str:
        return re.sub(r"√\s*\(", "sqrt(", text)

    def _standardize_function_names(self, text: str) -> str:
        replacements = {
            "sen": "sin",
            "seno": "sin",
            "tg": "tan",
            "ln": "log",
        }

        standardized_text = text
        for original_name, sympy_name in replacements.items():
            standardized_text = re.sub(
                rf"\b{original_name}\s*\(",
                f"{sympy_name}(",
                standardized_text,
                flags=re.IGNORECASE,
            )
        return standardized_text
