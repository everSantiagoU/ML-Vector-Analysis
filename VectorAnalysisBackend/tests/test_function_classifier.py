import pytest

from mlvectoranalysis.analysis import FunctionClassifier
from mlvectoranalysis.domain.component_function import ComponentFunction
from mlvectoranalysis.parsing import ExpressionParser


@pytest.mark.parametrize(('expression', 'expected'), [
    ('2*t+1', 'linear'), ('t**2', 'polynomial'), ('3', 'polynomial'),
    ('0', 'polynomial'), ('sin(t)', 'trigonometric'), ('cos(t)', 'trigonometric'),
    ('tan(t)', 'trigonometric'), ('exp(t)', 'exponential'),
    ('1/(t+1)', 'rational'), ('sqrt(t)', 'unknown'), ('log(t)', 'unknown'),
    ('sin(t)+exp(t)', 'trigonometric'), ('exp(t)+1/t', 'exponential'),
    ('t**2+sin(t)', 'trigonometric'), ('sin(t)**2+cos(t)**2', 'trigonometric'),
])
def test_component_classification_policy(expression, expected):
    component = ComponentFunction(expression)
    assert component.get_type() == expected
    assert FunctionClassifier().classify_component(component) == expected


def test_component_classification_delegates(monkeypatch):
    component = ComponentFunction('t')
    monkeypatch.setattr(component, 'get_type', lambda: 'custom')
    assert FunctionClassifier().classify_component(component) == 'custom'


@pytest.mark.parametrize(('text', 'expected'), [
    ('<t,2*t,t+1>', 'linear'), ('<t**2,1,0>', 'polynomial'),
    ('<t,t**2,0>', 'polynomial'), ('<sin(t),cos(t),tan(t)>', 'trigonometric'),
    ('<exp(t),exp(2*t),exp(-t)>', 'exponential'),
    ('<1/t,1/(t+1),1/(t+2)>', 'rational'),
    ('<log(t),sqrt(t),log(t+1)>', 'unknown'), ('<log(t),t,0>', 'mixed'),
    ('<sin(t),cos(t),0>', 'mixed'), ('<0,0,0>', 'polynomial'),
    ('<1,2,3>', 'polynomial'), ('<t,t**2,sin(t)>', 'mixed'),
])
def test_vector_policy(text, expected):
    assert FunctionClassifier().classify_vector(ExpressionParser().parse_vector_expression(text)) == expected


def test_vector_classification_uses_component_method(monkeypatch):
    classifier = FunctionClassifier()
    monkeypatch.setattr(classifier, 'classify_component', lambda component: 'injected')
    vector = ExpressionParser().parse_vector_expression('<t,0,sin(t)>')
    assert classifier.classify_vector(vector) == 'injected'
