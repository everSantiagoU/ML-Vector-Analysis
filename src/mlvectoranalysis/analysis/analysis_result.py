from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlvectoranalysis.domain.vector_function import VectorFunction


DEFAULT_OPERATIONS_AVAILABLE = [
    "simplify",
    "differentiate",
    "evaluate",
    "norm",
    "velocity",
    "acceleration",
    "speed",
]


@dataclass
class AnalysisResult:
    normalized_input: str
    component_types: dict[str, str]
    vector_type: str
    summary: str
    operations_available: list[str] = field(default_factory=lambda: DEFAULT_OPERATIONS_AVAILABLE.copy())

    @classmethod
    def from_vector(
        cls,
        vector: VectorFunction,
        normalized_input: str = "",
        vector_type: str = "unknown",
        summary: str | None = None,
        operations_available: list[str] | None = None,
    ) -> "AnalysisResult":
        component_types = cls._component_types_from_vector(vector)
        resolved_summary = summary or cls._build_default_summary(vector, vector_type)

        return cls(
            normalized_input=normalized_input,
            component_types=component_types,
            vector_type=vector_type,
            summary=resolved_summary,
            operations_available=operations_available or DEFAULT_OPERATIONS_AVAILABLE.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_input": self.normalized_input,
            "component_types": self.component_types,
            "vector_type": self.vector_type,
            "summary": self.summary,
            "operations_available": self.operations_available,
        }

    @staticmethod
    def _component_types_from_vector(vector: VectorFunction) -> dict[str, str]:
        component_x, component_y, component_z = vector.get_components()
        return {
            "x": component_x.get_type(),
            "y": component_y.get_type(),
            "z": component_z.get_type(),
        }

    @staticmethod
    def _build_default_summary(vector: VectorFunction, vector_type: str) -> str:
        return (
            f"Vector {vector.name}({vector.variable}) analyzed as {vector_type} "
            "with 3 components."
        )
