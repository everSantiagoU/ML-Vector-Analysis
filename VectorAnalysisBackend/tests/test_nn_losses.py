import numpy as np
import pytest
from numpy.testing import assert_allclose
from mlvectoranalysis.nn import HalfMSELoss, CrossEntropyLoss, NumericalError


def test_half_mse_scalar_and_multioutput():
    loss = HalfMSELoss()
    value, gradient = loss.value_and_grad([[1.], [3.]], [[0.], [1.]])
    assert value == pytest.approx(1.25)
    assert_allclose(gradient, [[.5], [1.]])
    p = np.array([[1., 3.], [2., -1.]])
    y = np.zeros_like(p)
    value, gradient = loss.value_and_grad(p, y)
    assert value == pytest.approx(np.sum(p**2) / 8)
    assert loss.value(p, y) == value
    assert_allclose(gradient, p / 4)


@pytest.mark.parametrize('p,y', [([[1]], [1]), ([], []), ([[]], [[]]), ([[np.inf]], [[1]]),
    ([[1]], [[np.nan]]), ([[1, 2]], [[1]]), ([[1j]], [[0]])])
def test_mse_rejects_invalid(p, y):
    with pytest.raises(ValueError):
        HalfMSELoss().value_and_grad(p, y)


def test_cross_entropy_uniform():
    loss, grad = CrossEntropyLoss().value_and_grad(np.zeros((2, 3)), [0, 2])
    assert loss == pytest.approx(np.log(3))
    assert_allclose(grad, [[-1/3, 1/6, 1/6], [1/6, 1/6, -1/3]])


def test_cross_entropy_stability_shift_invariance_and_zero_sum():
    logits = np.array([[1000., -1000., 999.], [-1000., 1000., 999.]])
    labels = np.array([1, 2])
    loss = CrossEntropyLoss()
    value, grad = loss.value_and_grad(logits, labels)
    shifted_value, shifted_grad = loss.value_and_grad(logits + [[123.], [-432.]], labels)
    assert np.isfinite(value) and value > 1000
    assert value == loss.value(logits, labels) == shifted_value
    assert_allclose(grad, shifted_grad)
    assert_allclose(grad.sum(axis=1), 0, atol=1e-16)
    with pytest.raises(NumericalError, match='float64'):
        loss.value([[1e308, -1e308]], [1])


@pytest.mark.parametrize('targets', [[0., 1.], [True, False], [-1, 0], [0, 3], [[0], [1]], [0], ['0', '1']])
def test_cross_entropy_invalid_targets(targets):
    with pytest.raises(ValueError):
        CrossEntropyLoss().value(np.zeros((2, 3)), targets)


@pytest.mark.parametrize('logits', [np.zeros((2, 1)), np.zeros((0, 3)), [[np.nan, 0], [0, 1]], [1, 2]])
def test_cross_entropy_invalid_logits(logits):
    with pytest.raises(ValueError):
        CrossEntropyLoss().value(logits, [0, 1])


def test_value_does_not_delegate_to_value_and_grad():
    for loss, predictions, targets in [(HalfMSELoss(), [[1]], [[0]]), (CrossEntropyLoss(), [[1, 0]], [0])]:
        loss.value_and_grad = lambda *args: pytest.fail('Validation must not request a gradient.')
        assert np.isfinite(loss.value(predictions, targets))
