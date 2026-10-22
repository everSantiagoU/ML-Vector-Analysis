from dataclasses import FrozenInstanceError
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from mlvectoranalysis.nn import (MLP, HalfMSELoss, CrossEntropyLoss, Trainer, TrainingConfig,
                                SGDOptimizer, NumericalError)


def setup(**kwargs):
    model = MLP([1, 3, 1], seed=8)
    trainer = Trainer(HalfMSELoss(), SGDOptimizer(model.parameters(), .03), TrainingConfig(**kwargs))
    return model, trainer


X = np.array([[.1], [.2], [.4], [.7], [1.]])
y = X**2


def test_counts_weighted_metrics_final_evaluation_and_json():
    model, trainer = setup(epochs=7, batch_size=2, eval_every=3, history_limit=20)
    result = trainer.fit(model, X, y, validation_data=(X + .1, y - .1))
    assert result.epochs_completed == 7 and result.optimizer_steps == 21
    assert [s.epoch for s in result.history] == [0, 3, 6, 7]
    assert result.history[0].optimizer_steps == 0
    assert result.status == 'max_epochs'
    final = result.history[-1]
    assert final.train_loss == pytest.approx(HalfMSELoss().value(model.predict(X), y))
    assert final.val_loss == pytest.approx(HalfMSELoss().value(model.predict(X + .1), y - .1))
    json.dumps(result.to_dict(), allow_nan=False)
    assert result.to_dict()['config']['seed'] == 42


def test_evaluate_is_read_only_and_unequal_batch_accuracy():
    model = MLP([1, 2], seed=4)
    trainer = Trainer(CrossEntropyLoss(), SGDOptimizer(model.parameters(), .1), TrainingConfig(batch_size=2))
    for p in model.parameters():
        p.grad[:] = 17
    before = [(p.data.copy(), p.grad.copy(), p.version) for p in model.parameters()]
    labels = np.array([0, 1, 0, 1, 1])
    outputs, context = model.forward(X, retain_cache=True)
    metric = trainer.evaluate(model, X, labels)
    assert metric.loss == pytest.approx(CrossEntropyLoss().value(outputs, labels))
    assert metric.accuracy == np.mean(outputs.argmax(axis=1) == labels)
    for p, (data, grad, version) in zip(model.parameters(), before):
        assert_array_equal(p.data, data)
        assert_array_equal(p.grad, grad)
        assert p.version == version
    model.backward(np.ones_like(outputs), context)


def test_validation_does_not_influence_training_or_shuffle():
    a, trainer_a = setup(epochs=5, batch_size=2)
    b, trainer_b = setup(epochs=5, batch_size=2)
    trainer_a.fit(a, X, y, validation_data=(X * 5, y + 100))
    trainer_b.fit(b, X, y)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert_array_equal(pa.data, pb.data)


@pytest.mark.parametrize('limit', [2, 3, 7, 200])
def test_bounded_history_retains_initial_latest(limit):
    model, trainer = setup(epochs=35, batch_size=3, history_limit=limit)
    result = trainer.fit(model, X, y)
    assert len(result.history) <= limit
    assert result.history[0].epoch == 0
    assert result.history[-1].epoch == 35
    assert result.history[-1].optimizer_steps == 70


@pytest.mark.parametrize('cancel_epoch', [0, 2])
def test_cancellation_immutable_scalar_callback(cancel_epoch):
    model, trainer = setup(epochs=7, batch_size=2)
    seen = []
    def callback(snapshot):
        with pytest.raises(FrozenInstanceError):
            snapshot.train_loss = 3
        seen.append(snapshot)
        return snapshot.epoch < cancel_epoch
    result = trainer.fit(model, X, y, callback=callback)
    assert result.status == 'cancelled'
    assert result.epochs_completed == cancel_epoch
    assert result.optimizer_steps == cancel_epoch * 3
    assert result.history[-1] == seen[-1]


@pytest.mark.parametrize('error', [RuntimeError('callback bug'), NumericalError('callback bug')])
def test_callback_programming_exceptions_propagate(error):
    model, trainer = setup(epochs=2)
    def callback(snapshot):
        if snapshot.epoch == 1:
            raise error
    with pytest.raises(type(error), match='callback bug'):
        trainer.fit(model, X, y, callback=callback)


def test_nonfinite_training_status_and_last_valid_update():
    model = MLP([1, 1], seed=4)
    model.parameters()[0].assign([[1]])
    model.parameters()[1].assign([0])
    trainer = Trainer(HalfMSELoss(), SGDOptimizer(model.parameters(), 1e308),
                      TrainingConfig(epochs=2, batch_size=1, shuffle=False))
    result = trainer.fit(model, [[1], [1]], [[0], [0]])
    assert result.status == 'non_finite'
    assert result.optimizer_steps == 1 and result.epochs_completed == 0
    assert result.history[-1].optimizer_steps == 1
    assert result.history[-1].train_loss is None
    assert_allclose(model.parameters()[0].data, [[-1e308]])
    assert_allclose(model.parameters()[1].data, [-1e308])
    json.dumps(result.to_dict(), allow_nan=False)


def test_invalid_initial_state_fails_before_updates():
    model, trainer = setup()
    before = [p.data.copy() for p in model.parameters()]
    with pytest.raises(NumericalError):
        trainer.fit(model, X, np.full_like(y, 1e308))
    for p, data in zip(model.parameters(), before):
        assert_array_equal(p.data, data)


def test_optimizer_ownership_and_invalid_validation_before_updates():
    model, trainer = setup()
    other, _ = setup()
    with pytest.raises(ValueError, match='exactly'):
        trainer.fit(other, X, y)
    before = [p.data.copy() for p in model.parameters()]
    with pytest.raises(ValueError):
        trainer.fit(model, X, y, validation_data=(X, y[:, 0]))
    for p, data in zip(model.parameters(), before):
        assert_array_equal(p.data, data)


def test_second_fit_continues_weights_with_fresh_counts():
    model, trainer = setup(epochs=3, batch_size=2)
    first = trainer.fit(model, X, y)
    second = trainer.fit(model, X, y)
    assert first.history[-1].train_loss == second.history[0].train_loss
    assert second.history[0].optimizer_steps == second.history[0].epoch == 0
    assert second.optimizer_steps == first.optimizer_steps == 9


@pytest.mark.parametrize('config', [dict(epochs=0), dict(epochs=True), dict(batch_size=1.5),
    dict(eval_every=0), dict(history_limit=1), dict(shuffle=1), dict(seed=-1), dict(seed=True)])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        TrainingConfig(**config)
