"""Stateless losses: reduction belongs here, never in a layer or SGD."""
import numpy as np
from ._validation import array, finite, numeric


class HalfMSELoss:
    def validate_targets(self, targets, n, width):
        targets = array(targets, 'targets', 2)
        if targets.shape != (n, width):
            raise ValueError('Regression targets must match predictions (B,K), without broadcasting.')
        return targets

    @numeric
    def _compute(self, predictions, targets, gradient):
        predictions = array(predictions, 'predictions', 2)
        targets = self.validate_targets(targets, *predictions.shape)
        residual = finite(predictions - targets)
        loss = float(finite(np.sum((residual / np.sqrt(residual.size))**2) / 2))
        return loss, finite(residual / residual.size) if gradient else None

    def value(self, predictions, targets):
        return self._compute(predictions, targets, False)[0]

    def value_and_grad(self, predictions, targets):
        return self._compute(predictions, targets, True)


@numeric
def _softmax_parts(logits):
    shifted = finite(logits - logits.max(axis=1, keepdims=True))
    exponentials = np.exp(shifted)
    denominator = exponentials.sum(axis=1, keepdims=True)
    return shifted, exponentials, denominator


def softmax(logits):
    logits = array(logits, 'logits', 2)
    if logits.shape[1] < 2:
        raise ValueError('Classification requires at least two logits.')
    _, exponentials, denominator = _softmax_parts(logits)
    return exponentials / denominator


class CrossEntropyLoss:
    def validate_targets(self, targets, n, width):
        targets = np.asarray(targets)
        if width < 2 or targets.shape != (n,) or targets.dtype.kind not in 'iu':
            raise ValueError('Classification needs C>=2 and integer targets of shape (B,).')
        if np.any(targets < 0) or np.any(targets >= width):
            raise ValueError('Class indices must be in [0,C).')
        return targets

    @numeric
    def _compute(self, predictions, targets, gradient):
        logits = array(predictions, 'logits', 2)
        targets = self.validate_targets(targets, *logits.shape)
        shifted, exponentials, denominator = _softmax_parts(logits)
        n = len(logits)
        terms = finite(np.log(denominator[:, 0]) - shifted[np.arange(n), targets])
        loss = float(finite(np.sum(terms / n)))
        if not gradient:
            return loss, None
        probabilities = exponentials / denominator
        probabilities[np.arange(n), targets] -= 1
        probabilities /= n
        return loss, probabilities

    def value(self, predictions, targets):
        return self._compute(predictions, targets, False)[0]

    def value_and_grad(self, predictions, targets):
        return self._compute(predictions, targets, True)
