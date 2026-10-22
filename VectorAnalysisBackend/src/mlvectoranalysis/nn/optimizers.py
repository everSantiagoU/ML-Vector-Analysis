"""Basic SGD with atomic numeric preflight using one reusable workspace."""
import numpy as np
from ._validation import DTYPE, finite, numeric, positive_real
from .parameters import unique_parameters


class SGDOptimizer:
    def __init__(self, parameters, learning_rate):
        self._parameters = unique_parameters(parameters)
        self._learning_rate = positive_real(learning_rate, 'learning_rate')
        self._workspace = np.empty(max(p.data.size for p in self.parameters), dtype=DTYPE)

    @property
    def parameters(self):
        return self._parameters

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def workspace_bytes(self):
        return self._workspace.nbytes

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def _candidate(self, parameter):
        buffer = self._workspace[:parameter.data.size].reshape(parameter.data.shape)
        np.multiply(parameter.grad, self.learning_rate, out=buffer)
        np.subtract(parameter.data, buffer, out=buffer)
        return finite(buffer)

    @numeric
    def step(self):
        for parameter in self.parameters:
            if (parameter.grad.shape != parameter.data.shape or parameter.grad.dtype != DTYPE
                    or parameter.data.dtype != DTYPE or not parameter.data.flags.writeable):
                raise ValueError('Parameter buffers must remain writable float64 with matching shapes.')
            finite(parameter.grad)
            finite(parameter.data)
        # No parameter mutates until every candidate has passed validation.
        for parameter in self.parameters:
            self._candidate(parameter)
        for parameter in self.parameters:
            parameter.assign(self._candidate(parameter))
