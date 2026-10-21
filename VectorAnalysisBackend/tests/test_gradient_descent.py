import json

import numpy as np
import pytest

from mlvectoranalysis.domain.multivariable_function import MultivariableFunction
from mlvectoranalysis.optimization import GradientDescent, GradientDescentConfig, ObjectiveEvaluationError


class Quadratic:
    dimension = 2

    def value(self, p):
        return float(p@p/2)

    def value_and_grad(self, p):
        return self.value(p), p.copy()


@pytest.mark.parametrize('kwargs', [
    {'max_iter': -1}, {'max_iter': True}, {'max_iter': 1.2}, {'max_backtracks': -1},
    {'gradient_atol': -1}, {'gradient_rtol': np.inf}, {'step_rtol': np.nan},
    {'loss_rtol': '0'}, {'alpha': 0}, {'alpha': -1}, {'alpha': True}, {'alpha': np.inf},
    {'c1': 0}, {'c1': 1}, {'beta': 0}, {'beta': 1}, {'min_step': 0},
    {'max_points': 1}, {'max_points': 2.5}, {'history_mode': 'all'},
    {'step_strategy': 'adam'}, {'history_indices': (0,)}, {'history_indices': (0, 0)},
    {'history_indices': (-1, 2)}, {'history_indices': (False, 1)},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        GradientDescentConfig(**kwargs)


def test_zero_budget_and_max_iter_are_not_convergence():
    for count in (0, 3):
        r = GradientDescent(GradientDescentConfig(max_iter=count, step_strategy='fixed', alpha=.1)).minimize(Quadratic(), [1, 2])
        assert r.n_iter == count and r.status == 'max_iter' and not r.converged
        assert r.n_function_evals == r.n_gradient_evals == count+1
    r = GradientDescent(GradientDescentConfig(max_iter=0)).minimize(Quadratic(), [0, 0])
    assert r.converged


def test_callback_is_independent_and_can_cancel_initial_or_later():
    for cancel_at in (0, 3):
        seen = []
        def callback(snapshot):
            seen.append(snapshot.iteration)
            snapshot.parameters[:] = np.nan
            return snapshot.iteration < cancel_at
        r = GradientDescent(GradientDescentConfig(step_strategy='fixed', alpha=.1)).minimize(Quadratic(), [1, 2], callback)
        assert r.status == 'cancelled' and not r.converged and r.n_iter == cancel_at
        assert seen == list(range(cancel_at+1))
        assert np.isfinite(r.parameters).all()
        json.dumps(r.to_dict(), allow_nan=False)


def test_armijo_rejected_trials_do_not_evaluate_gradient():
    r = GradientDescent(GradientDescentConfig(max_iter=1, alpha=16)).minimize(Quadratic(), [1, 2])
    assert r.converged and r.n_iter == 1
    assert r.n_gradient_evals == 2
    assert r.n_function_evals == 7  # initial + 5 losses (16,8,4,2,1) + accepted pair
    assert r.snapshots[-1].step == 1


@pytest.mark.parametrize('kwargs', [{'max_backtracks': 0}, {'min_step': 3}])
def test_line_search_failure_preserves_state(kwargs):
    r = GradientDescent(GradientDescentConfig(alpha=4, **kwargs)).minimize(Quadratic(), [1, 2])
    assert r.status == 'line_search_failed' and r.n_iter == 0 and not r.converged
    np.testing.assert_array_equal(r.parameters, [1, 2])
    assert r.loss == 2.5


def test_domain_rejection_can_recover_with_smaller_step():
    f = MultivariableFunction('x-log(x)+y^2+z^2')
    r = GradientDescent(GradientDescentConfig(alpha=10)).minimize(f, [2, 0, 0])
    assert r.converged
    np.testing.assert_allclose(r.parameters, [1, 0, 0], atol=1e-7)
    assert r.n_function_evals > 2*r.n_iter+1


def test_domain_failure_at_gradient_keeps_valid_state():
    f = MultivariableFunction('sqrt(x)+y^2+z^2')
    r = GradientDescent(GradientDescentConfig(alpha=4, step_strategy='fixed')).minimize(f, [1, 0, 0])
    assert r.status == 'non_finite' and r.n_iter == 0
    json.dumps(r.to_dict(), allow_nan=False)


def test_fixed_excessive_step_need_not_be_monotonic():
    r = GradientDescent(GradientDescentConfig(max_iter=3, alpha=3, step_strategy='fixed')).minimize(Quadratic(), [1, 2])
    assert r.loss > r.snapshots[0].loss and not r.converged
    assert r.status == 'max_iter'


def test_overflow_and_nonfinite_gradient_preserve_last_finite_state():
    r = GradientDescent(GradientDescentConfig(alpha=1e200, step_strategy='fixed')).minimize(Quadratic(), [1, 2])
    assert r.status == 'non_finite' and r.n_iter == 0
    json.dumps(r.to_dict(), allow_nan=False)
    class BadGradient(Quadratic):
        def value_and_grad(self, p):
            return self.value(p), p.copy() if p[0] == 1 else np.full(2, np.nan)
    r = GradientDescent(GradientDescentConfig(step_strategy='fixed', alpha=.1)).minimize(BadGradient(), [1, 2])
    assert r.status == 'non_finite' and r.n_iter == 0


def test_zero_and_relative_steps_are_stalled_not_converged():
    class Linear:
        dimension = 2
        def value(self, p): return float(p[0])
        def value_and_grad(self, p): return self.value(p), np.array([1., 0.])
    for alpha in (1e-30, 1e-8):
        r = GradientDescent(GradientDescentConfig(alpha=alpha, step_strategy='fixed',
            step_rtol=1e-6, loss_rtol=1e-6)).minimize(Linear(), [1, 0])
        assert r.status == 'stalled' and not r.converged


@pytest.mark.parametrize('curvature', [0, -1, np.inf, np.nan, 1e-300])
def test_invalid_exact_curvature_and_non_descent(curvature):
    class BadCurvature(Quadratic):
        def directional_curvature(self, p): return curvature
    r = GradientDescent().minimize(BadCurvature(), [1, 2])
    assert r.status == 'non_finite' and not r.converged
    assert r.n_iter == 0


def test_exact_rejects_unsupported_objective():
    with pytest.raises(ValueError, match='directional_curvature'):
        GradientDescent(GradientDescentConfig(step_strategy='exact_quadratic')).minimize(Quadratic(), [1, 2])


def test_initial_errors_and_programming_errors_are_not_hidden():
    for initial in ([1], [1, np.inf], [1j, 2]):
        with pytest.raises(ValueError):
            GradientDescent().minimize(Quadratic(), initial)
    class Bug(Quadratic):
        def value(self, p): raise RuntimeError('real bug')
    with pytest.raises(RuntimeError, match='real bug'):
        GradientDescent().minimize(Bug(), [1, 2])
    class Domain(Quadratic):
        def value(self, p): raise ObjectiveEvaluationError('domain')
    with pytest.raises(ValueError, match='Initial'):
        GradientDescent().minimize(Domain(), [1, 2])
    with pytest.raises(ValueError):
        GradientDescent(GradientDescentConfig(history_indices=(0, 2))).minimize(Quadratic(), [1, 2])


def test_relative_gradient_tolerance_uses_initial_gradient():
    r = GradientDescent(GradientDescentConfig(alpha=.5, step_strategy='fixed',
        gradient_atol=0, gradient_rtol=.3)).minimize(Quadratic(), [100, 0])
    assert r.n_iter == 2 and r.gradient_norm == 25 and r.converged


def test_failure_after_valid_updates_preserves_matching_final_metrics():
    class Limited(Quadratic):
        def value_and_grad(self, p):
            if p[0] < .8:
                raise ObjectiveEvaluationError('Limited domain')
            return super().value_and_grad(p)
    result = GradientDescent(GradientDescentConfig(step_strategy='fixed', alpha=.1)).minimize(Limited(), [1, 2])
    assert result.n_iter == 2 and result.status == 'non_finite' and not result.converged
    np.testing.assert_allclose(result.parameters, [.81, 1.62])
    assert result.snapshots[-1].iteration == 2
    assert result.loss == result.snapshots[-1].loss == pytest.approx(result.parameters@result.parameters/2)
    assert result.gradient_norm == pytest.approx(np.linalg.norm(result.parameters))
    json.dumps(result.to_dict(), allow_nan=False)


def test_exact_step_rejects_numerical_loss_increase():
    class WrongCurvature(Quadratic):
        def directional_curvature(self, direction): return float(direction@direction)/4
    result = GradientDescent().minimize(WrongCurvature(), [1, 2])
    assert result.status == 'non_finite' and result.n_iter == 0 and not result.converged
    assert result.n_curvature_evals == 1
    np.testing.assert_array_equal(result.parameters, [1, 2])


def test_callback_arithmetic_bug_is_not_hidden_as_objective_failure():
    def callback(snapshot):
        if snapshot.iteration == 1:
            raise ZeroDivisionError('callback bug')
    with pytest.raises(ZeroDivisionError, match='callback bug'):
        GradientDescent().minimize(Quadratic(), [1, 2], callback)
