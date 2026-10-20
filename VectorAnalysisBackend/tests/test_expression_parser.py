import pytest

from mlvectoranalysis.domain.component_function import ComponentFunction
from mlvectoranalysis.domain.vector_function import VectorFunction
from mlvectoranalysis.parsing.expression_parser import ExpressionParser


def test_expression_parser_parse_component_returns_component_function():
    parser = ExpressionParser()

    component = parser.parse_component("t^2 + 1")

    assert isinstance(component, ComponentFunction)
    assert component.variable == "t"
    assert component.is_equivalent_to("t**2+1")


def test_expression_parser_parse_component_rejects_unknown_symbol():
    parser = ExpressionParser()

    with pytest.raises(ValueError):
        parser.parse_component("x + 1")


def test_expression_parser_parse_vector_expression_returns_vector_function():
    parser = ExpressionParser()

    vector = parser.parse_vector_expression("<t,t^2,sen(t)>")

    assert isinstance(vector, VectorFunction)
    assert vector.name == "r"
    assert vector.variable == "t"
    assert vector.component_x.is_equivalent_to("t")
    assert vector.component_y.is_equivalent_to("t**2")
    assert vector.component_z.is_equivalent_to("sin(t)")


def test_expression_parser_parse_vector_expression_preserves_vector_name():
    parser = ExpressionParser()

    vector = parser.parse_vector_expression("v(t)=<t,t**2,cos(t)>")

    assert vector.name == "v"


def test_expression_parser_parse_vector_expression_preserves_custom_variable():
    parser = ExpressionParser(variable="x")

    vector = parser.parse_vector_expression("r(x)=<x,x^2,sin(x)>")

    assert vector.variable == "x"
    assert vector.component_x.variable == "x"
    assert vector.component_y.is_equivalent_to("x**2")


def test_expression_parser_parse_vector_expression_rejects_invalid_format():
    parser = ExpressionParser()

    with pytest.raises(ValueError):
        parser.parse_vector_expression("t,t**2,sin(t)")


def test_expression_parser_parse_vector_expression_rejects_wrong_component_count():
    parser = ExpressionParser()

    with pytest.raises(ValueError):
        parser.parse_vector_expression("<t,t**2>")


def test_expression_parser_parse_vector_expression_rejects_invalid_component():
    parser = ExpressionParser()

    with pytest.raises(ValueError):
        parser.parse_vector_expression("<t,t**,sin(t)>")


def test_expression_parser_can_be_imported_from_parsing_package():
    from mlvectoranalysis.parsing import ExpressionParser as ExportedExpressionParser

    assert ExportedExpressionParser is ExpressionParser


def test_parser_rejects_invalid_vectors(invalid_vector):
    with pytest.raises(ValueError):
        ExpressionParser().parse_vector_expression(invalid_vector)


def test_parser_rejects_invalid_scalar_grammar(invalid_component):
    with pytest.raises(ValueError):
        ExpressionParser().parse_component(invalid_component)


def test_parser_nested_comma_and_custom_symbol():
    parser = ExpressionParser(variable='u')
    vector = parser.parse_vector_expression('curve(u)=(log(u,2),u**2,E)')
    assert vector.name == 'curve'
    assert vector.variable == 'u'
    assert all(component.variable == 'u' for component in vector.get_components())
    assert vector.component_x.is_equivalent_to('log(u)/log(2)')
    assert vector.evaluate(2) == pytest.approx((1, 4, 2.718281828459045))


def test_external_parsing_never_sends_strings_to_sympify(monkeypatch):
    import mlvectoranalysis.domain.component_function as domain
    original = domain.sympify

    def guard(expression, *args, **kwargs):
        assert not isinstance(expression, str)
        return original(expression, *args, **kwargs)

    monkeypatch.setattr(domain, 'sympify', guard)
    vector = ExpressionParser().parse_vector_expression('<t,t**2,log(t,2)>')
    assert vector.evaluate(2) == (2.0, 4.0, 1.0)


@pytest.mark.parametrize('variable', ['x', 'u', 'beta', 'Q'])
def test_configured_symbols_are_consistent(variable):
    vector = ExpressionParser(variable=variable).parse_vector_expression(f'v({variable})=<{variable},0,1>')
    assert vector.component_x.symbolic_form == vector.component_x.symbol
    assert vector.component_x.differentiate().evaluate(1) == 1.0


def test_incompatible_validator_configuration():
    from mlvectoranalysis.input import InputValidator
    with pytest.raises(ValueError):
        ExpressionParser(validator=InputValidator(variable='x'))
    with pytest.raises(ValueError):
        ExpressionParser(validator=InputValidator(expected_components=2))


def test_invalid_vector_does_not_construct_partial_domain(monkeypatch):
    import mlvectoranalysis.parsing.expression_parser as parsing

    def forbidden(*args):
        pytest.fail('Domain object created for an invalid vector')

    monkeypatch.setattr(parsing, 'ComponentFunction', forbidden)
    with pytest.raises(ValueError):
        ExpressionParser().parse_vector_expression('<t,t,t.real>')


@pytest.mark.parametrize(('text', 'expected'), [
    ('-t+2', 0.0), ('+t/2', 1.0), ('2.5*t', 5.0), ('1e-3*t', 0.002),
    ('cos(t)', -0.4161468365471424), ('sqrt(t)', 1.4142135623730951),
])
def test_restricted_grammar_preserves_arithmetic(text, expected):
    assert ExpressionParser().parse_component(text).evaluate(2) == pytest.approx(expected)
