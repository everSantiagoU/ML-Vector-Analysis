import numpy as np
import pytest

from mlvectoranalysis.optimization import GradientDescent, GradientDescentConfig, LinearLeastSquaresObjective


@pytest.mark.parametrize('strategy,expected_step,expected_weight', [('fixed', .1, .5), ('exact_quadratic', .4, 2)])
def test_hand_calculated_update(strategy, expected_step, expected_weight):
    obj = LinearLeastSquaresObjective([[1], [2]], [2, 4], fit_intercept=False)
    value, grad = obj.value_and_grad([0])
    assert value == 5
    np.testing.assert_array_equal(grad, [-5])
    result = GradientDescent(GradientDescentConfig(max_iter=1, step_strategy=strategy, alpha=.1)).minimize(obj, [0])
    assert result.n_iter == 1
    np.testing.assert_allclose(result.parameters, [expected_weight])
    assert result.snapshots[-1].step == pytest.approx(expected_step)
    assert result.n_curvature_evals == (strategy == 'exact_quadratic')


@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('block_size', [None, 4])
def test_gradient_and_curvature_against_explicit_operator(fit_intercept, standardize, block_size):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(11, 3))*[1, 2, 5]+[1, 2, -1]
    targets = rng.normal(size=11)
    obj = LinearLeastSquaresObjective(X, targets, fit_intercept=fit_intercept,
                                      standardize=standardize, block_size=block_size)
    p = rng.normal(size=obj.dimension)
    Z = (X-obj.mean)/obj.scale
    residual = Z@p[:3] + (p[-1] if fit_intercept else 0) - targets
    value, gradient = obj.value_and_grad(p)
    assert value == pytest.approx(residual@residual/22)
    np.testing.assert_allclose(gradient[:3], Z.T@residual/11, atol=1e-14)
    if fit_intercept:
        assert gradient[-1] == pytest.approx(residual.mean())
    h = 1e-6
    numeric = [(obj.value(p+h*e)-obj.value(p-h*e))/(2*h) for e in np.eye(obj.dimension)]
    np.testing.assert_allclose(gradient, numeric, rtol=2e-6, atol=2e-8)
    u = Z@gradient[:3] + (gradient[-1] if fit_intercept else 0)
    assert obj.directional_curvature(gradient) == pytest.approx(u@u/11)


@pytest.mark.parametrize('strategy', ['fixed', 'exact_quadratic'])
@pytest.mark.parametrize('standardize', [True, False])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_full_block_equivalence_one_update_per_pass(strategy, standardize, fit_intercept):
    rng = np.random.default_rng(23)
    X, targets = rng.normal(size=(19, 4)), rng.normal(size=19)
    config = GradientDescentConfig(max_iter=7, step_strategy=strategy, alpha=.07,
                                   gradient_atol=0, gradient_rtol=0, step_rtol=0, loss_rtol=0)
    results = []
    for block_size in (None, 1, 6, 100):
        obj = LinearLeastSquaresObjective(X, targets, fit_intercept=fit_intercept,
                                          standardize=standardize, block_size=block_size)
        result = GradientDescent(config).minimize(obj, np.zeros(obj.dimension))
        assert result.n_iter == 7
        assert result.n_gradient_evals == 8
        assert len(result.snapshots) == 8
        results.append(result)
    for result in results[1:]:
        np.testing.assert_allclose(result.parameters, results[0].parameters, atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose([s.loss for s in result.snapshots],
                                   [s.loss for s in results[0].snapshots], atol=1e-14)


def test_arrays_and_gradients_are_not_mutated_or_reallocated():
    X = np.arange(36., dtype=float).reshape(6, 6)[:, ::2]
    targets = np.arange(6.)
    original_X, original_targets = X.copy(), targets.copy()
    obj = LinearLeastSquaresObjective(X, targets, block_size=2)
    assert np.shares_memory(obj.X, X) and np.shares_memory(obj.targets, targets)
    work_ids = id(obj._residual), id(obj._work_gradient)
    p = np.zeros(4)
    _, gradient = obj.value_and_grad(p)
    saved_gradient = gradient.copy()
    GradientDescent().minimize(obj, p)
    np.testing.assert_array_equal(gradient, saved_gradient)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(targets, original_targets)
    np.testing.assert_array_equal(p, np.zeros(4))
    assert work_ids == (id(obj._residual), id(obj._work_gradient))
    converted = LinearLeastSquaresObjective(X.astype(np.float32), targets)
    assert converted.X.dtype == np.float64


@pytest.mark.parametrize('X,targets', [([], []), ([1, 2], [1, 2]), ([[]], [1]),
    ([[1], [2]], [[1], [2]]), ([[1], [2]], [1]), ([[np.inf]], [1]), ([[1]], [np.nan]),
    ([[1j]], [1]), ([[1]], [1j]), ([['1']], [1]), ([[1]], ['1']),
    (np.array([[1]], dtype=object), [1]), ([[True]], [1]), ([[1]], [True]),
    ([[1], [2, 3]], [1, 2])])
def test_invalid_data(X, targets):
    with pytest.raises(ValueError):
        LinearLeastSquaresObjective(X, targets)


@pytest.mark.parametrize('kwargs', [{'block_size': 0}, {'block_size': True}, {'block_size': 1.5},
                                   {'fit_intercept': 1}, {'standardize': 'yes'}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        LinearLeastSquaresObjective([[1]], [1], **kwargs)


def test_stable_statistics_large_offset_constant_and_no_centering():
    X = np.column_stack([1e12 + np.arange(21.), np.full(21, 9.)])
    obj = LinearLeastSquaresObjective(X, np.zeros(21), standardize=True, block_size=4)
    np.testing.assert_allclose(obj.mean, X.mean(axis=0), rtol=0, atol=1e-4)
    np.testing.assert_allclose(obj.scale, [np.std(np.arange(21.)), 1], rtol=1e-12)
    obj2 = LinearLeastSquaresObjective(X, np.zeros(21), standardize=True, fit_intercept=False, block_size=4)
    np.testing.assert_array_equal(obj2.mean, [0, 0])
    np.testing.assert_allclose(obj2.scale, obj.scale)


def test_exact_monotonicity_and_initial_zero_gradient():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(70, 8))
    obj = LinearLeastSquaresObjective(X, X@np.arange(8.)+3)
    result = GradientDescent().minimize(obj, np.zeros(9))
    assert result.converged
    assert np.all(np.diff([s.loss for s in result.snapshots]) <= 1e-13)
    zero = LinearLeastSquaresObjective(np.zeros((3, 2)), np.zeros(3), fit_intercept=False)
    result = GradientDescent().minimize(zero, [0, 0])
    assert result.n_iter == result.n_curvature_evals == 0
    assert result.converged
