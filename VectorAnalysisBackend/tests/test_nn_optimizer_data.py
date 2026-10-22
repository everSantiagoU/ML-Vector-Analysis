import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from mlvectoranalysis.nn import Parameter, SGDOptimizer, MiniBatchIterator, NumericalError


def test_parameter_owned_buffers_sgd_and_zero_grad():
    original = np.array([1., 2.])
    parameter = Parameter(original, 'weight')
    original[:] = 9
    assert_array_equal(parameter.data, [1, 2])
    data, grad = parameter.data, parameter.grad
    optimizer = SGDOptimizer([parameter], .2)
    parameter.grad[:] = [2, -3]
    optimizer.step()
    assert_allclose(parameter.data, [.6, 2.6])
    assert parameter.data is data and parameter.grad is grad
    assert parameter.data.dtype == parameter.grad.dtype == np.float64
    optimizer.zero_grad()
    assert_array_equal(parameter.grad, [0, 0])
    parameter.assign([3, 4])
    assert parameter.data is data
    assert optimizer.workspace_bytes == data.nbytes


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_gradient_step_is_atomic(bad):
    first, last = Parameter([1., 2.]), Parameter([3.])
    first.grad[:] = 1
    last.grad[:] = bad
    with pytest.raises(NumericalError):
        SGDOptimizer([first, last], .1).step()
    assert_array_equal(first.data, [1, 2])
    assert_array_equal(last.data, [3])
    assert first.version == last.version == 0


def test_overflow_step_is_atomic():
    first, last = Parameter([1.]), Parameter([-1e308])
    first.grad[:] = 1
    last.grad[:] = 1e308
    with pytest.raises(NumericalError):
        SGDOptimizer([first, last], 1.).step()
    assert_array_equal(first.data, [1])
    assert_array_equal(last.data, [-1e308])
    assert first.version == last.version == 0


@pytest.mark.parametrize('rate', [True, 0, -1, np.nan, np.inf, '1', 1j])
def test_invalid_learning_rates(rate):
    with pytest.raises(ValueError):
        SGDOptimizer([Parameter([1])], rate)


@pytest.mark.parametrize('data', [[1j], np.array([1], dtype=object), ['1'], [True], [np.nan], []])
def test_invalid_parameter_data(data):
    with pytest.raises(ValueError):
        Parameter(data)


def test_duplicate_and_empty_parameters():
    p = Parameter([1])
    for parameters in ([p, p], [], [1]):
        with pytest.raises(ValueError):
            SGDOptimizer(parameters, .1)


@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('batch_size', [1, 4, 20])
def test_minibatches_pairing_coverage_last_and_slices(shuffle, batch_size):
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X[:, 0] * 3
    batches = list(MiniBatchIterator(X, y, batch_size, shuffle=shuffle, seed=12))
    assert len(batches) == int(np.ceil(10 / batch_size))
    assert len(batches[-1][0]) == (10 % batch_size or min(batch_size, 10))
    seen = np.concatenate([x[:, 0] for x, _ in batches])
    assert_array_equal(np.sort(seen), X[:, 0])
    for x, target in batches:
        assert_array_equal(target, x[:, 0] * 3)
        assert np.shares_memory(x, X) == (not shuffle)


def test_minibatches_rng_advances_reproduces_and_preserves_global_rng():
    before = np.random.get_state()
    X, y = np.arange(40.).reshape(20, 2), np.arange(20)
    a, b = (MiniBatchIterator(X, y, 3, seed=4) for _ in range(2))
    order = lambda iterator: np.concatenate([labels for _, labels in iterator])
    first, second = order(a), order(a)
    assert_array_equal(first, order(b))
    assert_array_equal(second, order(b))
    assert not np.array_equal(first, second)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    assert_array_equal(before[1], after[1])


@pytest.mark.parametrize('size', [True, 0, -1, 1.5])
def test_invalid_batch_size(size):
    with pytest.raises(ValueError):
        MiniBatchIterator([[1]], [0], size)


@pytest.mark.parametrize('X,y', [([], []), ([[1]], [0, 1]), ([[1]], [[np.nan]]), ([[1]], [True])])
def test_invalid_datasets(X, y):
    with pytest.raises(ValueError):
        MiniBatchIterator(X, y, 1)
