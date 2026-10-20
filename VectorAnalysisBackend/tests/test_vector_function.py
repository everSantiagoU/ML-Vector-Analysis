from sympy import sqrt, sympify

from mlvectoranalysis.domain.component_function import ComponentFunction
from mlvectoranalysis.domain.vector_function import VectorFunction


def build_sample_vector() -> VectorFunction:
    return VectorFunction(
        ComponentFunction("t"),
        ComponentFunction("t**2"),
        ComponentFunction("sin(t)")
    )


def test_vector_function_creation():
    vector = build_sample_vector()
    components = vector.get_components()

    assert vector.name == "r"
    assert vector.variable == "t"
    assert len(components) == 3


def test_vector_function_simplify_all():
    vector = VectorFunction(
        ComponentFunction("(t + 1)**2"),
        ComponentFunction("sin(t)**2 + cos(t)**2"),
        ComponentFunction("2*t")
    )

    simplified = vector.simplify_all()
    components = simplified.get_components()

    assert components[0].is_equivalent_to("t**2 + 2*t + 1")
    assert components[1].is_equivalent_to("1")
    assert components[2].is_equivalent_to("2*t")


def test_vector_function_differentiate():
    vector = build_sample_vector()
    derivative = vector.differentiate()
    components = derivative.get_components()

    assert components[0].is_equivalent_to("1")
    assert components[1].is_equivalent_to("2*t")
    assert components[2].is_equivalent_to("cos(t)")


def test_vector_function_velocity():
    vector = build_sample_vector()
    velocity = vector.velocity()
    components = velocity.get_components()

    assert components[0].is_equivalent_to("1")
    assert components[1].is_equivalent_to("2*t")
    assert components[2].is_equivalent_to("cos(t)")


def test_vector_function_acceleration():
    vector = build_sample_vector()
    acceleration = vector.acceleration()
    components = acceleration.get_components()

    assert components[0].is_equivalent_to("0")
    assert components[1].is_equivalent_to("2")
    assert components[2].is_equivalent_to("-sin(t)")


def test_vector_function_evaluate():
    vector = VectorFunction(
        ComponentFunction("t"),
        ComponentFunction("t**2"),
        ComponentFunction("t + 1")
    )

    result = vector.evaluate(2)
    assert result == (2.0, 4.0, 3.0)


def test_vector_function_norm():
    vector = VectorFunction(
        ComponentFunction("t"),
        ComponentFunction("0"),
        ComponentFunction("0")
    )

    assert vector.norm().equals(sympify("sqrt(t**2)"))


def test_vector_function_speed():
    vector = VectorFunction(
        ComponentFunction("t"),
        ComponentFunction("t**2"),
        ComponentFunction("0")
    )

    assert vector.speed().equals(sympify("sqrt(4*t**2 + 1)"))


def test_custom_vector_transformations_preserve_metadata_and_original():
    import pytest
    vector = VectorFunction(ComponentFunction('x**3', 'x'), ComponentFunction('sin(x)', 'x'), ComponentFunction('4', 'x'), name='path', variable='x')
    before = vector.to_matrix()
    assert before.shape == (3, 1)
    assert isinstance(vector.get_components(), tuple)
    for transformed in [vector.simplify_all(), vector.differentiate(), vector.velocity(), vector.acceleration()]:
        assert transformed is not vector
        assert transformed.name == 'path'
        assert transformed.variable == 'x'
        assert all(c.variable == 'x' for c in transformed.get_components())
    assert vector.to_matrix() == before
    assert vector.acceleration().evaluate(0) == pytest.approx((0, 0, 0))
    assert vector.velocity().evaluate(0) == pytest.approx((0, 1, 0))
    assert float(vector.speed().subs(vector.component_x.symbol, 0)) == pytest.approx(1)


def test_zero_and_constant_vectors_and_shared_components():
    for expression in ['0', '3']:
        component = ComponentFunction(expression)
        vector = VectorFunction(component, component, component)
        assert all(c is component for c in vector.get_components())
        assert vector.to_matrix().shape == (3, 1)
        assert vector.evaluate(4) == (float(expression),) * 3
        assert vector.velocity().evaluate(4) == (0.0, 0.0, 0.0)
        assert vector.acceleration().evaluate(4) == (0.0, 0.0, 0.0)
        assert vector.speed() == 0
        assert vector.norm().equals(sqrt(3 * int(expression)**2))
