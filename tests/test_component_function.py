from sympy import sympify

from mlvectoranalysis.domain.component_function import ComponentFunction


def test_component_function_creation():
    component = ComponentFunction("t**2 + 2*t + 1")
    assert component.expression == "t**2 + 2*t + 1"
    assert component.variable == "t"
    assert str(component.symbolic_form) == "t**2 + 2*t + 1"


def test_component_function_simplify():
    component = ComponentFunction("(t + 1)**2")
    simplified = component.simplify()
    assert simplified.is_equivalent_to("t**2 + 2*t + 1")


def test_component_function_expand():
    component = ComponentFunction("(t + 1)**2")
    expanded = component.expand()
    assert str(expanded.symbolic_form) == "t**2 + 2*t + 1"


def test_component_function_differentiate():
    component = ComponentFunction("t**2")
    derivative = component.differentiate()
    assert derivative == sympify("2*t")


def test_component_function_integrate():
    component = ComponentFunction("2*t")
    integrated = component.integrate()
    assert integrated.is_equivalent_to("t**2")


def test_component_function_evaluate():
    component = ComponentFunction("t**2 + 1")
    result = component.evaluate(2)
    assert result == 5.0


def test_component_function_is_equivalent():
    component = ComponentFunction("(t + 1)**2")
    assert component.is_equivalent_to("t**2 + 2*t + 1")


def test_component_function_type_linear():
    component = ComponentFunction("2*t + 3")
    assert component.get_type() == "linear"


def test_component_function_type_polynomial():
    component = ComponentFunction("t**3 + 2*t")
    assert component.get_type() == "polynomial"


def test_component_function_type_trigonometric():
    component = ComponentFunction("sin(t)")
    assert component.get_type() == "trigonometric"
