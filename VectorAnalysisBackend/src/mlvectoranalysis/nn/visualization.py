"""Plotting-neutral adapter for training/checkpoint summaries.

Applications may pass this JSON-ready payload to Matplotlib, a web UI, or a
report.  The neural-network core itself never imports a plotting package.
"""
from collections.abc import Mapping


def training_visualization(result, *, checkpoint_epochs=(), optimizer_state=None):
    """Return scalar series and labels from a TrainingResult or manifest data."""
    optimizer_state = optimizer_state or {}
    return dict(epochs=[point.epoch for point in result.history],
                train_loss=[point.train_loss for point in result.history],
                validation_loss=[point.val_loss for point in result.history],
                train_accuracy=[point.train_accuracy for point in result.history],
                validation_accuracy=[point.val_accuracy for point in result.history],
                optimizer_steps=result.optimizer_steps, status=result.status,
                optimizer_type=optimizer_state.get('optimizer_type'),
                optimizer_hyperparameters={key: value for key, value in optimizer_state.items()
                                           if key in {'learning_rate', 'beta1', 'beta2', 'epsilon'}},
                checkpoint_epochs=list(checkpoint_epochs))


def manifest_visualization(manifest):
    """Extract checkpoint labels without reading arrays or importing Matplotlib."""
    if not isinstance(manifest, Mapping):
        raise ValueError('manifest must be a mapping.')
    state, optimizer = manifest.get('training_state'), manifest.get('optimizer')
    if not isinstance(state, Mapping) or not isinstance(optimizer, Mapping):
        raise ValueError('manifest lacks training_state or optimizer.')
    return dict(epoch_completed=state.get('epoch_completed'), optimizer_steps=state.get('optimizer_steps'),
                optimizer_type=optimizer.get('optimizer_type'),
                optimizer_hyperparameters={key: optimizer[key] for key in ('learning_rate', 'beta1', 'beta2', 'epsilon') if key in optimizer},
                metadata=manifest.get('metadata', {}))
