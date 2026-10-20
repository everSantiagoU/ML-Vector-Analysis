from mlvectoranalysis.domain.component_function import ComponentFunction
from mlvectoranalysis.domain.vector_function import VectorFunction


class FunctionClassifier:
    """Symbolic categories, without geometric interpretation or ignoring constants."""

    def classify_component(self, component: ComponentFunction) -> str:
        return component.get_type()

    def classify_vector(self, vector: VectorFunction) -> str:
        labels = {self.classify_component(component) for component in vector.get_components()}
        if len(labels) == 1:
            return labels.pop()
        if labels <= {"linear", "polynomial"}:
            return "polynomial"
        return "mixed"
