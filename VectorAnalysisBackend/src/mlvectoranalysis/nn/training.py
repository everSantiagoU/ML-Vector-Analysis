"""Training is distinct from evaluation and depends on an optimizer protocol."""
from dataclasses import asdict
from dataclasses import dataclass
import numpy as np
from ._validation import array, finite, integer, NumericalError, numeric
from .data import MiniBatchIterator
from .losses import HalfMSELoss, CrossEntropyLoss
from .optimizers import Optimizer
from .checkpoint import TrainingState
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
    def __init__(self, loss, optimizer: Optimizer, config=None):
        required = ('parameters', 'learning_rate', 'zero_grad', 'step', 'state_dict', 'load_state_dict')
        if not isinstance(loss, (HalfMSELoss, CrossEntropyLoss)) or any(not hasattr(optimizer, name) for name in required):
            raise ValueError('Trainer requires a supported loss and optimizer protocol implementation.')
        if not isinstance(optimizer.parameters, tuple) or not isinstance(optimizer.learning_rate, float):
            raise ValueError('Optimizer protocol has invalid parameters or learning_rate.')
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

    def fit(self, model, X_train, targets_train, *, validation_data=None, callback=None,
            resume_state=None, checkpoint_callback=None, additional_epochs=None):
        """Train for ``config.epochs``, or ``additional_epochs`` after a checkpoint.

        A resumed fit is exact only at completed epoch boundaries.  The caller
        loads model/optimizer first, then passes the matching TrainingState;
        its local shuffle-generator state is restored before the next epoch.
        """
        if tuple(model.parameters()) != self.optimizer.parameters:
            raise ValueError('Optimizer must reference exactly the ordered model parameters.')
        if callback is not None and not callable(callback):
            raise ValueError('callback must be callable.')
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise ValueError('checkpoint_callback must be callable.')
        if resume_state is not None and not isinstance(resume_state, TrainingState):
            raise ValueError('resume_state must be TrainingState.')
        if resume_state is None and additional_epochs is not None:
            raise ValueError('additional_epochs is only valid when resuming.')
        if resume_state is not None:
            epochs_to_run = integer(additional_epochs, 'additional_epochs')
            saved = dict(resume_state.config)
            required_config = {'epochs', 'batch_size', 'shuffle', 'seed', 'eval_every', 'history_limit'}
            if set(saved) != required_config or any(saved[key] != getattr(self.config, key)
                                                    for key in ('batch_size', 'shuffle', 'seed', 'eval_every', 'history_limit')):
                raise ValueError('Resume TrainingState is incompatible with TrainingConfig.')
            if resume_state.history_state is None:
                raise ValueError('Resume TrainingState lacks bounded history internals.')
        else:
            epochs_to_run = self.config.epochs
        train = self._prepare(model, X_train, targets_train)
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError('validation_data must be a separate (X, targets) pair.')
            validation = self._prepare(model, *validation_data)
        else:
            validation = None
        config = self.config
        if resume_state is None:
            history = _History(config.history_limit)
            epochs_completed = steps = 0
        else:
            history = _History.from_state_dict(resume_state.to_dict()['history_state'], config.history_limit)
            epochs_completed, steps = resume_state.epoch_completed, resume_state.optimizer_steps
        end_epoch = epochs_completed + epochs_to_run

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

        batches = MiniBatchIterator(*train, config.batch_size, shuffle=config.shuffle, seed=config.seed)
        if resume_state is not None:
            try:
                batches.rng.bit_generator.state = resume_state.to_dict()['rng_state']
            except (TypeError, ValueError) as error:
                raise ValueError('Resume TrainingState has an invalid shuffle RNG state.') from error

        def state():
            return TrainingState(epochs_completed, steps, batches.rng.bit_generator.state, history.snapshots,
                                 asdict(config), history.state_dict())

        if resume_state is None:
            initial = snapshot()  # Invalid initial evaluation must not return apparent success.
            history.record(initial)
            if callback is not None and callback(initial) is False:
                return result('cancelled')
        for epoch in range(epochs_completed + 1, end_epoch + 1):
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
                current = snapshot() if (epoch % config.eval_every == 0 or epoch == end_epoch
                                         or checkpoint_callback is not None) else None
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
                if checkpoint_callback is not None and checkpoint_callback(state()) is False:
                    return result('cancelled')
        return result('max_epochs', 'Epoch budget completed; no convergence claim.')
