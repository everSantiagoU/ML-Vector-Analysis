"""Training is distinct from evaluation. SGD alone owns the learning rate."""
from dataclasses import dataclass
import numpy as np
from ._validation import array, finite, integer, NumericalError, numeric
from .data import MiniBatchIterator
from .losses import HalfMSELoss, CrossEntropyLoss
from .optimizers import SGDOptimizer
from .training_result import EpochMetrics, EvaluationMetrics, TrainingResult, _History


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    shuffle: bool = True
    seed: int | None = 42
    eval_every: int = 1
    history_limit: int = 200

    def __post_init__(self):
        for field, minimum in (('epochs', 1), ('batch_size', 1), ('eval_every', 1), ('history_limit', 2)):
            object.__setattr__(self, field, integer(getattr(self, field), field, minimum))
        if type(self.shuffle) is not bool:
            raise ValueError('shuffle must be bool.')
        if self.seed is not None:
            object.__setattr__(self, 'seed', integer(self.seed, 'seed', 0))


class Trainer:
    def __init__(self, loss, optimizer, config=None):
        if not isinstance(loss, (HalfMSELoss, CrossEntropyLoss)) or not isinstance(optimizer, SGDOptimizer):
            raise ValueError('Trainer requires a supported loss and SGDOptimizer.')
        self.loss, self.optimizer = loss, optimizer
        self.config = config if config is not None else TrainingConfig()
        if not isinstance(self.config, TrainingConfig):
            raise ValueError('config must be TrainingConfig.')

    def _prepare(self, model, X, targets):
        X = array(X, 'X', 2)
        if X.shape[1] != model.input_width:
            raise ValueError('Dataset width does not match model.')
        targets = self.loss.validate_targets(targets, len(X), model.output_width)
        return X, targets

    @numeric
    def _evaluate_prepared(self, model, X, targets, batch_size):
        total_loss, correct = 0.0, 0
        classification = isinstance(self.loss, CrossEntropyLoss)
        for start in range(0, len(X), batch_size):
            rows = X[start:start + batch_size]
            labels = targets[start:start + batch_size]
            predictions, _ = model.forward(rows)
            total_loss += self.loss.value(predictions, labels) * (len(rows) / len(X))
            if classification:
                correct += int(np.count_nonzero(predictions.argmax(axis=1) == labels))
        return EvaluationMetrics(float(finite(total_loss)), correct / len(X) if classification else None)

    def evaluate(self, model, X, targets, *, batch_size=None):
        """Read-only evaluation: no optimizer, gradient or training RNG changes."""
        X, targets = self._prepare(model, X, targets)
        size = self.config.batch_size if batch_size is None else integer(batch_size, 'batch_size')
        return self._evaluate_prepared(model, X, targets, size)

    def fit(self, model, X_train, targets_train, *, validation_data=None, callback=None):
        """Continue current weights; fresh counts/history and configured RNG per fit.

        Invalid input/initial state raises before updates. Subsequent arithmetic
        failure returns non_finite and leaves the last successfully updated model.
        Callback exceptions propagate, including FloatingPointError.
        """
        if tuple(model.parameters()) != self.optimizer.parameters:
            raise ValueError('Optimizer must reference exactly the ordered model parameters.')
        if callback is not None and not callable(callback):
            raise ValueError('callback must be callable.')
        train = self._prepare(model, X_train, targets_train)
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError('validation_data must be a separate (X, targets) pair.')
            validation = self._prepare(model, *validation_data)
        else:
            validation = None
        config = self.config
        history = _History(config.history_limit)
        epochs_completed = steps = 0

        def snapshot():
            train_metric = self._evaluate_prepared(model, *train, config.batch_size)
            val_metric = (self._evaluate_prepared(model, *validation, config.batch_size)
                          if validation is not None else None)
            return EpochMetrics(epochs_completed, steps, train_metric.loss,
                                val_metric.loss if val_metric else None, train_metric.accuracy,
                                val_metric.accuracy if val_metric else None)

        def result(status, message=''):
            return TrainingResult(status, epochs_completed, steps, history.snapshots,
                                  config, self.optimizer.learning_rate, config.seed, message)

        initial = snapshot()  # Invalid initial evaluation must not return apparent success.
        history.record(initial)
        if callback is not None and callback(initial) is False:
            return result('cancelled')
        batches = MiniBatchIterator(*train, config.batch_size, shuffle=config.shuffle, seed=config.seed)
        for epoch in range(1, config.epochs + 1):
            try:
                for X_batch, y_batch in batches:
                    context = None
                    try:
                        self.optimizer.zero_grad()
                        predictions, context = model.forward(X_batch, retain_cache=True)
                        _, gradient = self.loss.value_and_grad(predictions, y_batch)
                        model.backward(gradient, context)
                        self.optimizer.step()
                        steps += 1
                    finally:
                        if context is not None:
                            model.release_context(context)
                epochs_completed = epoch
                current = snapshot() if epoch % config.eval_every == 0 or epoch == config.epochs else None
            except NumericalError as error:
                # A failed evaluation is represented by nulls, never stale losses
                # attached to newer parameter counters or invalid JSON numbers.
                try:
                    current = snapshot()
                except NumericalError:
                    current = EpochMetrics(epochs_completed, steps, None)
                history.record(current)
                return result('non_finite', str(error))
            if current is not None:
                history.record(current)
                if callback is not None and callback(current) is False:
                    return result('cancelled')
        return result('max_epochs', 'Epoch budget completed; no convergence claim.')
