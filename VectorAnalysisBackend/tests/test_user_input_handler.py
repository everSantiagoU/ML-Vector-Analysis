import pytest

from mlvectoranalysis.input import InputNormalizer, InputValidator, UserInputHandler
from mlvectoranalysis.parsing import ExpressionParser
from mlvectoranalysis.analysis import VectorAnalyzer


def test_required_end_to_end_case():
    handler = UserInputHandler()
    raw = '  r(t) = < t , t^2 , sen(t) >  '
    assert handler.receive_input(raw) == raw
    result = handler.process_input()
    assert handler.raw_text == raw
    assert result.normalized_input == 'r(t)=<t,t**2,sin(t)>'
    assert result.component_types == {'x': 'linear', 'y': 'polynomial', 'z': 'trigonometric'}
    assert result.vector_type == 'mixed'
    assert result.summary == 'Vector r(t) analyzed as mixed with 3 components.'


def test_custom_variable_normalization_and_repeated_input():
    handler = UserInputHandler(variable='x')
    handler.receive_input(' v(x) = ( ln(x), √(x), π ) ')
    first = handler.process_input()
    assert first.normalized_input == 'v(x)=(log(x),sqrt(x),pi)'
    assert first.summary == 'Vector v(x) analyzed as mixed with 3 components.'
    handler.receive_input('<x,x,x>')
    second = handler.process_input()
    assert second.vector_type == 'linear'
    assert second.summary == 'Vector r(x) analyzed as linear with 3 components.'
    assert first.vector_type == 'mixed'
    handler.receive_input('<x,bad,x>')
    with pytest.raises(ValueError):
        handler.process_input()


@pytest.mark.parametrize('text', ['', ' ', '\n\t'])
def test_empty_input(text):
    handler = UserInputHandler()
    assert handler.raw_text == ''
    with pytest.raises(ValueError, match='nonempty'):
        handler.process_input()
    assert handler.receive_input(text) == text
    with pytest.raises(ValueError, match='nonempty'):
        handler.process_input()


@pytest.mark.parametrize('text', [None, 3, [], {}, b'<t,t,t>'])
def test_nontext_input(text):
    handler = UserInputHandler()
    with pytest.raises(ValueError, match='string'):
        handler.receive_input(text)
    assert handler.raw_text == ''


def test_injected_dependencies_run_real_flow():
    calls = []

    class Normalizer(InputNormalizer):
        def normalize(self, text):
            calls.append(('normalize', text))
            return super().normalize(text)

    class Validator(InputValidator):
        def parse_component_expression(self, expr):
            calls.append(('validate', expr))
            return super().parse_component_expression(expr)

    class Parser(ExpressionParser):
        def parse_vector_expression(self, text):
            calls.append(('parse', text))
            return super().parse_vector_expression(text)

    class Analyzer(VectorAnalyzer):
        def analyze(self, vector, normalized_input=''):
            calls.append(('analyze', vector.variable))
            return super().analyze(vector, normalized_input)

    normalizer = Normalizer()
    validator = Validator(variable='x', normalizer=normalizer)
    parser = Parser(variable='x', normalizer=normalizer, validator=validator)
    analyzer = Analyzer()
    handler = UserInputHandler('x', normalizer, validator, parser, analyzer)
    assert (handler.normalizer, handler.validator, handler.parser, handler.analyzer) == (normalizer, validator, parser, analyzer)
    handler.receive_input('v(x)=<x,x^2,sen(x)>')
    assert handler.process_input().vector_type == 'mixed'
    assert [value for kind, value in calls if kind == 'validate'] == ['x', 'x**2', 'sin(x)']
    assert ('parse', 'v(x)=<x,x**2,sin(x)>') in calls
    assert ('analyze', 'x') in calls


def test_defaults_reuse_injected_collaborators():
    normalizer = InputNormalizer()
    validator = InputValidator(variable='x', normalizer=normalizer)
    handler = UserInputHandler(variable='x', validator=validator)
    assert handler.normalizer is handler.parser.normalizer is normalizer
    assert handler.validator is handler.parser.validator is validator
    other = UserInputHandler(variable='x', parser=handler.parser)
    assert other.normalizer is normalizer
    assert other.validator is validator
    other.receive_input('<x,x,x>')
    assert other.process_input().vector_type == 'linear'


def test_inconsistent_dependencies_rejected():
    with pytest.raises(ValueError):
        UserInputHandler(validator=InputValidator(variable='x'))
    with pytest.raises(ValueError):
        UserInputHandler(parser=ExpressionParser(variable='x'))
    with pytest.raises(ValueError):
        UserInputHandler(parser=ExpressionParser(), validator=InputValidator())


def test_analyzer_errors_propagate():
    class Analyzer:
        def analyze(self, vector, normalized_input=''):
            raise ValueError('analysis failed')

    handler = UserInputHandler(analyzer=Analyzer())
    handler.receive_input('<t,t,t>')
    with pytest.raises(ValueError, match='analysis failed'):
        handler.process_input()


def test_public_imports():
    from mlvectoranalysis.input.user_input_handler import UserInputHandler as Handler
    from mlvectoranalysis.analysis.function_classifier import FunctionClassifier
    from mlvectoranalysis.analysis.vector_analyzer import VectorAnalyzer as Analyzer
    from mlvectoranalysis.analysis import FunctionClassifier as Classifier
    assert Handler is UserInputHandler
    assert Analyzer is VectorAnalyzer
    assert Classifier is FunctionClassifier
