import numpy as np
import pytest
from sympy import diff

from mlvectoranalysis._scalar_expression import parse_scalar, parse_scalar_variables
from mlvectoranalysis.domain.multivariable_function import MultivariableFunction
from mlvectoranalysis.optimization import DifferentiableObjective, GradientDescent, GradientDescentConfig


QUADRATIC = '(x-1)^2+2*(y+2)^2+3*(z-0.5)^2'


def test_symbolic_numeric_gradient_and_protocol():
    f = MultivariableFunction.from_text(QUADRATIC)
    assert isinstance(f, DifferentiableObjective)
    x, y, z = f.symbols
    assert f.symbolic_gradient == (2*x-2, 4*y+8, 6*z-3.0)
    np.testing.assert_allclose(f.gradient([4, 4, 4]), [6, 24, 21])
    assert f.value([4, 4, 4]) == 117.75
    assert tuple(diff(f.symbolic_form, s) for s in f.symbols) == f.symbolic_gradient


@pytest.mark.parametrize('strategy', ['fixed', 'armijo', 'auto'])
def test_quadratic_convergence_and_first_step(strategy):
    f = MultivariableFunction.from_text(QUADRATIC)
    initial = np.array([4., 4., 4.])
    r = GradientDescent(GradientDescentConfig(step_strategy=strategy, alpha=.1)).minimize(f, initial)
    assert r.converged and r.status == 'gradient_tolerance'
    np.testing.assert_allclose(r.parameters, [1, -2, .5], atol=2e-7)
    np.testing.assert_array_equal(initial, [4, 4, 4])
    np.testing.assert_allclose(r.snapshots[1].parameters, [3.4, 1.6, 1.9])
    assert r.loss == pytest.approx(f.value(r.parameters))
    assert r.gradient_norm == pytest.approx(np.linalg.norm(f.gradient(r.parameters)))
    if strategy != 'fixed':
        assert np.all(np.diff([s.loss for s in r.snapshots]) <= 1e-13)


@pytest.mark.parametrize('expression', [QUADRATIC, 'sin(x)*cos(y)+exp(z)+log(x,2)',
                                         'sqrt(x)+tan(y)+x*y*z', 'x+y+z', '7'])
def test_multivariable_central_difference(expression):
    f = MultivariableFunction.from_text(expression)
    point = np.array([1.4, .3, -.2])
    h = 1e-6
    estimate = [(f.value(point + h*e) - f.value(point - h*e))/(2*h) for e in np.eye(3)]
    np.testing.assert_allclose(f.gradient(point), estimate, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize('expression', ['x.real', 'x[0]', '__import__("os")', 'sin(x,foo=2)',
                                         '(lambda x:x)(1)', '[x]', 't+x', 'foo(x)', 'True',
                                         'x if y else z', 'x//2', 'log(x,2,3)', '(a:=1)'])
def test_multivariable_restricted_parser(expression):
    with pytest.raises(ValueError):
        MultivariableFunction.from_text(expression)


def test_normalization_and_univariate_compatibility():
    f = MultivariableFunction.from_text(' sen(x)^2 + cos(x)^2 + ln(y) + √(z) + π ')
    assert f.value([0, 1, 4]) == pytest.approx(3 + np.pi)
    assert parse_scalar('u**2', 'u') == parse_scalar_variables('u**2', ('u',))
    with pytest.raises(ValueError):
        parse_scalar('x+y', 'x')
    with pytest.raises(ValueError):
        parse_scalar_variables('x', ('x', 'x'))


@pytest.mark.parametrize('point', [[1, 2], [[1, 2, 3]], [1, np.inf, 2], [1, np.nan, 2],
                                   [1j, 2, 3], ['1', '2', '3'], [True, False, True]])
def test_invalid_points(point):
    with pytest.raises(ValueError):
        MultivariableFunction('x+y+z').value_and_grad(point)


def test_initial_domain_and_stationary_maximum_or_saddle():
    with pytest.raises(ValueError, match='Initial'):
        GradientDescent().minimize(MultivariableFunction('log(x)+y^2+z^2'), [-1, 0, 0])
    for expression in (QUADRATIC, '-x^2-y^2-z^2', 'x^2-y^2+z^2'):
        start = [1, -2, .5] if expression == QUADRATIC else [0, 0, 0]
        result = GradientDescent().minimize(MultivariableFunction(expression), start)
        assert result.converged and result.n_iter == 0
        assert result.snapshots[0].step is None


def test_compilation_does_not_happen_in_loop(monkeypatch):
    f = MultivariableFunction(QUADRATIC)
    def forbidden(*args, **kwargs):
        pytest.fail('lambdify called during optimization')
    monkeypatch.setattr('mlvectoranalysis.domain.multivariable_function.lambdify', forbidden)
    assert GradientDescent().minimize(f, [4, 4, 4]).converged
