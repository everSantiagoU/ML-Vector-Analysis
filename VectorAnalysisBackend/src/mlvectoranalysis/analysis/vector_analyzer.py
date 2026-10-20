from mlvectoranalysis.analysis.analysis_result import AnalysisResult
from mlvectoranalysis.analysis.function_classifier import FunctionClassifier
from mlvectoranalysis.domain.vector_function import VectorFunction


class VectorAnalyzer:
    def __init__(self, classifier: FunctionClassifier | None = None) -> None:
        self.classifier = classifier if classifier is not None else FunctionClassifier()

    def analyze(self, vector: VectorFunction, normalized_input: str = "") -> AnalysisResult:
        component_types = self.identify_component_types(vector)
        vector_type = self.identify_vector_type(vector)
        return AnalysisResult(
            normalized_input=normalized_input,
            component_types=component_types,
            vector_type=vector_type,
            summary=self._summary(vector, vector_type),
        )

    def identify_component_types(self, vector: VectorFunction) -> dict[str, str]:
        return {
            axis: self.classifier.classify_component(component)
            for axis, component in zip(("x", "y", "z"), vector.get_components())
        }

    def identify_vector_type(self, vector: VectorFunction) -> str:
        return self.classifier.classify_vector(vector)

    def generate_summary(self, vector: VectorFunction) -> str:
        return self._summary(vector, self.identify_vector_type(vector))

    @staticmethod
    def _summary(vector: VectorFunction, vector_type: str) -> str:
        return f"Vector {vector.name}({vector.variable}) analyzed as {vector_type} with 3 components."
