import json

from mlvectoranalysis.analysis import AnalysisResult, VectorAnalyzer
from mlvectoranalysis.parsing import ExpressionParser


OPERATIONS = ['simplify', 'differentiate', 'evaluate', 'norm', 'velocity', 'acceleration', 'speed']


def test_analyzer_complete_result_and_methods():
    vector = ExpressionParser(variable='x').parse_vector_expression('path(x)=<x,x**2,sin(x)>')
    analyzer = VectorAnalyzer()
    source = '  exact supplied text  '
    result = analyzer.analyze(vector, source)
    expected_types = {'x': 'linear', 'y': 'polynomial', 'z': 'trigonometric'}
    summary = 'Vector path(x) analyzed as mixed with 3 components.'
    assert isinstance(result, AnalysisResult)
    assert result.to_dict() == {
        'normalized_input': source, 'component_types': expected_types,
        'vector_type': 'mixed', 'summary': summary, 'operations_available': OPERATIONS,
    }
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
    assert analyzer.identify_component_types(vector) == expected_types
    assert analyzer.identify_vector_type(vector) == 'mixed'
    assert analyzer.generate_summary(vector) == summary
    assert analyzer.analyze(vector).normalized_input == ''


def test_analyzer_respects_injected_classifier():
    class Classifier:
        def classify_component(self, component):
            return 'component:' + component.expression

        def classify_vector(self, vector):
            return 'custom'

    classifier = Classifier()
    analyzer = VectorAnalyzer(classifier)
    vector = ExpressionParser().parse_vector_expression('<t,0,1>')
    before = vector.to_matrix()
    result = analyzer.analyze(vector)
    assert analyzer.classifier is classifier
    assert result.component_types == {'x': 'component:t', 'y': 'component:0', 'z': 'component:1'}
    assert result.vector_type == 'custom'
    assert result.summary == analyzer.generate_summary(vector) == 'Vector r(t) analyzed as custom with 3 components.'
    assert vector.to_matrix() == before


def test_analyzers_have_independent_default_classifiers():
    assert VectorAnalyzer().classifier is not VectorAnalyzer().classifier
