import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from mlvectoranalysis.nn import LinearLayer, Tanh, ReLU, MLP, SGDOptimizer, Parameter


def linear():
    layer = LinearLayer(2, 2, rng=np.random.default_rng(3))
    layer.weight.assign([[1, 2], [3, 4]])
    layer.bias.assign([.5, -.5])
    return layer


def test_linear_analytical_and_no_second_batch_division():
    layer = linear()
    X = np.array([[1., 2.], [-1., 3.]])
    upstream = np.array([[.2, -.1], [.3, .4]])
    out, cache = layer.forward(X, retain_cache=True)
    assert_allclose(out, [[7.5, 9.5], [8.5, 9.5]])
    dx = layer.backward(upstream, cache)
    assert_allclose(dx, upstream @ layer.weight.data.T)
    assert_allclose(layer.weight.grad, X.T @ upstream)
    assert_allclose(layer.bias.grad, [.5, .3])
    _, cache = layer.forward(X, retain_cache=True)
    layer.backward(upstream, cache)
    assert_allclose(layer.bias.grad, [.5, .3])  # overwrite, not accumulation


def test_linear_all_vjps_central_differences():
    layer = linear()
    X = np.array([[.3, -.6], [.7, .2]])
    upstream = np.array([[.8, -.2], [-.1, .4]])
    _, cache = layer.forward(X, retain_cache=True)
    dx = layer.backward(upstream, cache)
    for data, analytical in [(X, dx)] + [(p.data, p.grad.copy()) for p in layer.parameters()]:
        numerical = np.zeros_like(data)
        for index in np.ndindex(data.shape):
            original = data[index]
            try:
                data[index] = original + 1e-6
                plus = np.sum(layer.forward(X)[0] * upstream)
                data[index] = original - 1e-6
                minus = np.sum(layer.forward(X)[0] * upstream)
                numerical[index] = (plus - minus) / 2e-6
            finally:
                data[index] = original
        assert_allclose(analytical, numerical, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize('activation', [Tanh, ReLU])
def test_activation_values_gradient_and_nonmutation(activation):
    layer = activation()
    X = np.array([[-1., 0., .7]])
    upstream = np.array([[2., -3., .4]])
    original_X, original_upstream = X.copy(), upstream.copy()
    out, cache = layer.forward(X, retain_cache=True)
    expected = np.tanh(X) if activation is Tanh else np.maximum(X, 0)
    assert_allclose(out, expected)
    dx = layer.backward(upstream, cache)
    assert_allclose(dx, upstream * (1 - expected**2 if activation is Tanh else X > 0))
    assert_array_equal(X, original_X)
    assert_array_equal(upstream, original_upstream)
    for index in (0, 2):  # ReLU finite differences away from the kink
        right, left = X.copy(), X.copy()
        right[0, index] += 1e-6
        left[0, index] -= 1e-6
        numerical = np.sum((layer.forward(right)[0] - layer.forward(left)[0]) * upstream) / 2e-6
        assert_allclose(dx[0, index], numerical, atol=1e-9)
    with pytest.raises(ValueError, match='consumed'):
        layer.backward(upstream, cache)


@pytest.mark.parametrize('factory', [linear, Tanh, ReLU, lambda: MLP([2, 3, 2], seed=4)])
def test_context_ownership_shapes_and_interleaved_inference(factory):
    layer, other = factory(), factory()
    X = np.ones((3, 2))
    out, cache = layer.forward(X, retain_cache=True)
    assert layer.forward(X * 2)[1] is None
    with pytest.raises(ValueError):
        layer.backward(np.ones_like(out), None)
    with pytest.raises(ValueError):
        layer.backward(np.ones_like(out), other.forward(X, retain_cache=True)[1])
    with pytest.raises(ValueError):
        layer.backward(np.ones((1, 1)), cache)
    dx = layer.backward(np.ones_like(out), cache)
    reference_out, reference_cache = other.forward(X, retain_cache=True)
    assert_allclose(dx, other.backward(np.ones_like(reference_out), reference_cache))
    with pytest.raises(ValueError):
        layer.backward(np.ones_like(out), cache)


@pytest.mark.parametrize('update', ['assign', 'sgd'])
@pytest.mark.parametrize('factory', [linear, lambda: MLP([2, 3, 2], seed=5)])
def test_parameter_changes_invalidate_context(factory, update):
    model = factory()
    out, context = model.forward(np.ones((2, 2)), retain_cache=True)
    if update == 'assign':
        model.parameters()[0].assign(model.parameters()[0].data + 1)
    else:
        SGDOptimizer(model.parameters(), .1).step()
    with pytest.raises(ValueError, match='changed'):
        model.backward(np.ones_like(out), context)


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_invalid_widths(value):
    with pytest.raises(ValueError):
        LinearLayer(value, 2, rng=np.random.default_rng(1))


@pytest.mark.parametrize('X', [[], [[np.nan, 0]], [[np.inf, 0]], [1, 2], [[1]], [[1j, 2]], [['1', '2']]])
def test_invalid_layer_inputs(X):
    with pytest.raises(ValueError):
        linear().forward(X)


def test_no_sharing_and_initialization():
    layer = linear()
    with pytest.raises(ValueError, match='shared'):
        MLP.from_layers([layer, layer])
    other = linear()
    other.weight = layer.weight
    with pytest.raises(ValueError, match='duplicate'):
        MLP.from_layers([layer, other])
    model = MLP([200, 300, 200], activation='relu', seed=9)
    assert_allclose(np.var(model.layers[0].weight.data), 2 / 200, rtol=.03)
    assert_allclose(np.var(model.layers[-1].weight.data), 2 / 500, rtol=.03)
    assert np.count_nonzero(model.layers[0].bias.data) == 0
    assert not np.array_equal(model.layers[0].weight.data, model.layers[-1].weight.data.T)


def test_prediction_by_blocks_and_public_outputs():
    model = MLP([2, 4, 3], seed=8)
    X = np.random.default_rng(4).normal(size=(11, 2))
    direct = model.forward(X)[0]
    assert_allclose(model.predict(X, 3), direct)
    proba = model.predict_proba(X, 4)
    assert_allclose(proba.sum(axis=1), 1)
    assert_array_equal(model.predict_classes(X, 2), direct.argmax(axis=1))
    assert_allclose(proba, np.exp(direct) / np.exp(direct).sum(axis=1, keepdims=True))
    for method in ('predict_proba', 'predict_classes'):
        with pytest.raises(ValueError):
            getattr(MLP([2, 1]), method)(X)


def test_two_training_contexts_are_independent():
    layer = linear()
    _, first = layer.forward(np.ones((3, 2)), retain_cache=True)
    _, second = layer.forward(np.full((3, 2), 2.), retain_cache=True)
    layer.backward(np.ones((3, 2)), first)
    assert_allclose(layer.weight.grad, np.full((2, 2), 3.))
    layer.backward(np.ones((3, 2)), second)
    assert_allclose(layer.weight.grad, np.full((2, 2), 6.))


def test_explicit_context_release_rejects_backward():
    model = MLP([2, 3, 2], seed=42)
    _, context = model.forward(np.ones((2, 2)), retain_cache=True)
    model.release_context(context)
    model.release_context(context)
    with pytest.raises(ValueError, match='consumed'):
        model.backward(np.ones((2, 2)), context)


def test_initialization_matches_one_persistent_generator():
    seed = 91
    model = MLP([2, 3, 2], seed=seed)
    rng = np.random.default_rng(seed)
    assert_allclose(model.layers[0].weight.data, rng.normal(0, np.sqrt(2/5), (2, 3)))
    assert_allclose(model.layers[-1].weight.data, rng.normal(0, np.sqrt(2/5), (3, 2)))


def test_model_construction_does_not_touch_global_rng():
    before = np.random.get_state()
    MLP([2, 8, 2], seed=3)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    assert_array_equal(before[1], after[1])
