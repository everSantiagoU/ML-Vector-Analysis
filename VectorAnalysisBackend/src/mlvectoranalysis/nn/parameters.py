"""Owned, stable parameter buffers; assign/SGD version live execution contexts."""
import numpy as np
from ._validation import array


class Parameter:
    def __init__(self, data, name=None):
        if name is not None and not isinstance(name, str):
            raise ValueError('Parameter name must be a string or None.')
        self._data = array(data, 'parameter').copy()
        self._grad = np.zeros_like(self._data)
        self.name = name
        self._version = 0

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return self._grad

    @property
    def version(self):
        return self._version

    def zero_grad(self):
        self._grad.fill(0)

    def assign(self, values):
        """Copy new finite values in place and invalidate retained contexts.

        Direct ndarray writes are supported for setup/gradient checks only when
        no context is live. Use assign() for tracked parameter changes.
        """
        values = array(values, 'parameter values')
        if values.shape != self.data.shape:
            raise ValueError('Parameter shape cannot change.')
        np.copyto(self._data, values)
        self._version += 1


def unique_parameters(parameters):
    result = tuple(parameters)
    if not result or any(not isinstance(p, Parameter) for p in result):
        raise ValueError('Expected a nonempty collection of Parameter objects.')
    if len({id(p) for p in result}) != len(result):
        raise ValueError('Shared/duplicate parameters are not supported.')
    return result
