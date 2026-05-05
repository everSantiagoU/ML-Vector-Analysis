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
