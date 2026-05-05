from mlvectoranalysis.analysis.analysis_result import AnalysisResult
from mlvectoranalysis.domain.component_function import ComponentFunction
from mlvectoranalysis.domain.vector_function import VectorFunction


def build_sample_vector() -> VectorFunction:
    return VectorFunction(
        ComponentFunction("t"),
        ComponentFunction("t**2"),
        ComponentFunction("sin(t)"),
    )


def test_analysis_result_creation():
    result = AnalysisResult(
        normalized_input="<t,t**2,sin(t)>",
        component_types={"x": "linear", "y": "polynomial", "z": "trigonometric"},
        vector_type="mixed",
        summary="Sample vector analysis.",
        operations_available=["simplify", "differentiate"],
    )

    assert result.normalized_input == "<t,t**2,sin(t)>"
    assert result.component_types["x"] == "linear"
    assert result.vector_type == "mixed"
    assert result.summary == "Sample vector analysis."
    assert result.operations_available == ["simplify", "differentiate"]


def test_analysis_result_to_dict():
    result = AnalysisResult(
        normalized_input="<t,t**2,sin(t)>",
        component_types={"x": "linear", "y": "polynomial", "z": "trigonometric"},
        vector_type="mixed",
        summary="Sample vector analysis.",
        operations_available=["simplify", "differentiate"],
    )

    assert result.to_dict() == {
        "normalized_input": "<t,t**2,sin(t)>",
        "component_types": {"x": "linear", "y": "polynomial", "z": "trigonometric"},
        "vector_type": "mixed",
        "summary": "Sample vector analysis.",
        "operations_available": ["simplify", "differentiate"],
    }


def test_analysis_result_from_vector_identifies_component_types():
    vector = build_sample_vector()

    result = AnalysisResult.from_vector(
        vector,
        normalized_input="<t,t**2,sin(t)>",
        vector_type="mixed",
    )

    assert result.normalized_input == "<t,t**2,sin(t)>"
    assert result.component_types == {
        "x": "linear",
        "y": "polynomial",
        "z": "trigonometric",
    }
    assert result.vector_type == "mixed"


def test_analysis_result_from_vector_builds_default_summary():
    vector = build_sample_vector()

    result = AnalysisResult.from_vector(vector, vector_type="mixed")

    assert result.summary == "Vector r(t) analyzed as mixed with 3 components."


def test_analysis_result_from_vector_includes_current_vector_operations():
    vector = build_sample_vector()

    result = AnalysisResult.from_vector(vector)

    assert result.operations_available == [
        "simplify",
        "differentiate",
        "evaluate",
        "norm",
        "velocity",
        "acceleration",
        "speed",
    ]


def test_analysis_result_operations_are_not_shared_between_instances():
    first_result = AnalysisResult.from_vector(build_sample_vector())
    second_result = AnalysisResult.from_vector(build_sample_vector())

    first_result.operations_available.append("custom")

    assert "custom" not in second_result.operations_available


def test_analysis_result_can_be_imported_from_analysis_package():
    from mlvectoranalysis.analysis import AnalysisResult as ExportedAnalysisResult

    assert ExportedAnalysisResult is AnalysisResult
