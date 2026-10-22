import numpy as np
import pytest
from numpy.testing import assert_allclose
from mlvectoranalysis.nn import MLP, HalfMSELoss, CrossEntropyLoss, SGDOptimizer, Trainer, TrainingConfig
from mlvectoranalysis.optimization import LinearLeastSquaresObjective


@pytest.mark.parametrize('loss,targets', [(HalfMSELoss(), np.array([[.3, -.4], [.8, .1], [-.2, .6]])),
                                         (CrossEntropyLoss(), np.array([0, 1, 1]))])
def test_all_network_parameters_and_inputs_against_central_differences(loss, targets):
    model = MLP([2, 3, 2], seed=16)
    X = np.array([[.2, -.5], [.8, .3], [-.4, .7]])
    predictions, context = model.forward(X, retain_cache=True)
    _, dout = loss.value_and_grad(predictions, targets)
    dx = model.backward(dout, context)
    # Every element of every weight/bias and every input is checked.
    for data, analytical in [(p.data, p.grad.copy()) for p in model.parameters()] + [(X, dx)]:
        numerical = np.empty_like(data)
        for index in np.ndindex(data.shape):
            original = data[index]
            try:
                data[index] = original + 1e-6
                plus = loss.value(model.forward(X)[0], targets)
                data[index] = original - 1e-6
                minus = loss.value(model.forward(X)[0], targets)
                numerical[index] = (plus - minus) / 2e-6
            finally:
                data[index] = original
        assert_allclose(analytical, numerical, rtol=2e-5, atol=3e-10)


def test_linear_regression_parity_full_batch_step():
    X = np.array([[1., 2.], [-1., 3.], [.5, -.8]])
    y = np.array([.7, -.2, .4])
    model = MLP([2, 1], seed=3)
    model.parameters()[0].assign([[.2], [-.3]])
    model.parameters()[1].assign([.1])
    initial = np.array([.2, -.3, .1])
    objective = LinearLeastSquaresObjective(X, y, standardize=False)
    expected_loss, expected_gradient = objective.value_and_grad(initial)
    loss = HalfMSELoss()
    prediction, context = model.forward(X, retain_cache=True)
    actual_loss, dout = loss.value_and_grad(prediction, y[:, None])
    model.backward(dout, context)
    assert actual_loss == pytest.approx(expected_loss)
    assert_allclose(model.parameters()[0].grad[:, 0], expected_gradient[:2])
    assert_allclose(model.parameters()[1].grad, expected_gradient[2:])
    SGDOptimizer(model.parameters(), .05).step()
    expected = initial - .05 * expected_gradient
    assert_allclose(model.parameters()[0].data[:, 0], expected[:2])
    assert_allclose(model.parameters()[1].data, expected[2:])
    assert loss.value(model.predict(X), y[:, None]) == pytest.approx(objective.value(expected))


def test_xor_controlled_overfit():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 1, 1, 0])
    model = MLP([2, 8, 2], seed=42)
    trainer = Trainer(CrossEntropyLoss(), SGDOptimizer(model.parameters(), .2),
                      TrainingConfig(epochs=1200, batch_size=4, seed=42, eval_every=100))
    result = trainer.fit(model, X, y)
    assert result.history[-1].train_accuracy == 1
    assert result.history[-1].train_loss < .1
    assert result.optimizer_steps == 1200


def test_sine_independent_validation_fast_integration():
    train_seed, val_seed = np.random.SeedSequence(42).spawn(2)
    X = np.random.default_rng(train_seed).uniform(-np.pi, np.pi, (128, 1))
    V = np.random.default_rng(val_seed).uniform(-np.pi, np.pi, (256, 1))
    mean, scale = X.mean(axis=0), X.std(axis=0)
    model = MLP([1, 24, 24, 1], seed=42)
    trainer = Trainer(HalfMSELoss(), SGDOptimizer(model.parameters(), .03),
                      TrainingConfig(epochs=600, batch_size=32, seed=42, eval_every=100))
    result = trainer.fit(model, (X - mean) / scale, np.sin(X),
                         validation_data=((V - mean) / scale, np.sin(V)))
    assert result.status == 'max_epochs'
    assert np.sqrt(2 * result.history[-1].val_loss) < .15
    assert result.history[-1].val_loss < result.history[0].val_loss / 10
