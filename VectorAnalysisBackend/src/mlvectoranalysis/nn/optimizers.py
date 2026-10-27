"""Small, atomic float64 optimizers used by the NumPy MLP."""
from collections.abc import Mapping
from typing import Protocol
import numpy as np
from ._validation import DTYPE, finite, numeric, positive_real
from .parameters import Parameter, unique_parameters


class Optimizer(Protocol):
    """Structural contract: ordered parameters and copied serializable state."""
    @property
    def parameters(self) -> tuple[Parameter, ...]: ...
    @property
    def learning_rate(self) -> float: ...
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: Mapping) -> None: ...


def _layout(parameters):
    return [dict(name=p.name, shape=list(p.data.shape), dtype=p.data.dtype.name) for p in parameters]


def _validate_mapping(state, expected):
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError('Optimizer state has missing or unexpected fields.')


def _validate_layout(parameters, layout):
    if not isinstance(layout, (list, tuple)) or len(layout) != len(parameters):
        raise ValueError('Optimizer state parameter count is incompatible.')
    for parameter, item in zip(parameters, layout):
        if not isinstance(item, Mapping) or set(item) != {'name', 'shape', 'dtype'}:
            raise ValueError('Optimizer state parameter layout is invalid.')
        if item['name'] != parameter.name or tuple(item['shape']) != parameter.data.shape or item['dtype'] != np.dtype(DTYPE).name:
            raise ValueError('Optimizer state parameter order, shape, name, or dtype is incompatible.')


def _validate_parameter_buffers(parameters):
    for parameter in parameters:
        if (parameter.grad.shape != parameter.data.shape or parameter.grad.dtype != DTYPE
                or parameter.data.dtype != DTYPE or not parameter.data.flags.writeable):
            raise ValueError('Parameter buffers must remain writable float64 with matching shapes.')
        finite(parameter.grad)
        finite(parameter.data)


class SGDOptimizer:
    """Classic SGD; O(P) time and one O(max parameter size) workspace."""
    _STATE_VERSION = 1
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
        _validate_parameter_buffers(self.parameters)
        # No parameter mutates until every candidate has passed validation.
        for parameter in self.parameters:
            self._candidate(parameter)
        for parameter in self.parameters:
            parameter.assign(self._candidate(parameter))

    def state_dict(self):
        return dict(format_version=self._STATE_VERSION, optimizer_type='sgd',
                    learning_rate=float(self.learning_rate), parameters=_layout(self.parameters))

    def _validated_state(self, state):
        _validate_mapping(state, {'format_version', 'optimizer_type', 'learning_rate', 'parameters'})
        if state['format_version'] != self._STATE_VERSION or state['optimizer_type'] != 'sgd':
            raise ValueError('Incompatible SGD state format.')
        rate = positive_real(state['learning_rate'], 'learning_rate')
        _validate_layout(self.parameters, state['parameters'])
        return rate

    def load_state_dict(self, state):
        self._learning_rate = self._validated_state(state)


class AdamOptimizer:
    """Classic Adam: O(P) update, O(2P) moments, atomic failed steps."""
    _STATE_VERSION = 1

    def __init__(self, parameters, learning_rate, *, beta1=.9, beta2=.999, epsilon=1e-8):
        self._parameters = unique_parameters(parameters)
        self._learning_rate = positive_real(learning_rate, 'learning_rate')
        self._beta1 = self._beta(beta1, 'beta1')
        self._beta2 = self._beta(beta2, 'beta2')
        self._epsilon = positive_real(epsilon, 'epsilon')
        self._t = 0
        self._m = tuple(np.zeros_like(p.data) for p in self.parameters)
        self._v = tuple(np.zeros_like(p.data) for p in self.parameters)
        maximum = max(p.data.size for p in self.parameters)
        self._m_work, self._v_work, self._sqrt_work, self._candidate_work = (
            np.empty(maximum, dtype=DTYPE) for _ in range(4))

    @staticmethod
    def _beta(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f'{name} must be a finite float64 in [0, 1).')
        value = float(value)
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f'{name} must be a finite float64 in [0, 1).')
        return value

    @property
    def parameters(self): return self._parameters
    @property
    def learning_rate(self): return self._learning_rate
    @property
    def beta1(self): return self._beta1
    @property
    def beta2(self): return self._beta2
    @property
    def epsilon(self): return self._epsilon
    @property
    def t(self): return self._t
    @property
    def workspace_bytes(self): return sum(x.nbytes for x in (self._m_work, self._v_work, self._sqrt_work, self._candidate_work))
    @property
    def state_bytes(self): return sum(x.nbytes for x in self._m) + sum(x.nbytes for x in self._v)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def _views(self, parameter):
        shape, size = parameter.data.shape, parameter.data.size
        return tuple(x[:size].reshape(shape) for x in (self._m_work, self._v_work, self._sqrt_work, self._candidate_work))

    def _candidate(self, parameter, m, v, new_t):
        mw, vw, sqrtw, candidate = self._views(parameter)
        np.multiply(m, self.beta1, out=mw)
        np.multiply(parameter.grad, 1. - self.beta1, out=candidate)
        np.add(mw, candidate, out=mw)
        np.multiply(v, self.beta2, out=vw)
        np.multiply(parameter.grad, parameter.grad, out=candidate)
        np.multiply(candidate, 1. - self.beta2, out=candidate)
        np.add(vw, candidate, out=vw)
        np.sqrt(vw, out=sqrtw)
        np.divide(sqrtw, np.sqrt(1. - self.beta2 ** new_t), out=sqrtw)
        np.add(sqrtw, self.epsilon, out=sqrtw)
        np.divide(mw, 1. - self.beta1 ** new_t, out=candidate)
        np.divide(candidate, sqrtw, out=candidate)
        np.multiply(candidate, self.learning_rate, out=candidate)
        np.subtract(parameter.data, candidate, out=candidate)
        finite(mw); finite(vw); finite(candidate)
        return mw, vw, candidate

    @numeric
    def step(self):
        _validate_parameter_buffers(self.parameters)
        for m, v, parameter in zip(self._m, self._v, self.parameters):
            if m.shape != parameter.data.shape or v.shape != parameter.data.shape or m.dtype != DTYPE or v.dtype != DTYPE:
                raise ValueError('Adam moment buffers are incompatible with parameters.')
            finite(m); finite(v)
        new_t = self._t + 1
        for parameter, m, v in zip(self.parameters, self._m, self._v):
            self._candidate(parameter, m, v, new_t)
        for parameter, m, v in zip(self.parameters, self._m, self._v):
            mw, vw, candidate = self._candidate(parameter, m, v, new_t)
            parameter.assign(candidate)
            np.copyto(m, mw); np.copyto(v, vw)
        self._t = new_t

    def state_dict(self):
        return dict(format_version=self._STATE_VERSION, optimizer_type='adam', learning_rate=float(self.learning_rate),
                    beta1=float(self.beta1), beta2=float(self.beta2), epsilon=float(self.epsilon), t=int(self.t),
                    parameters=_layout(self.parameters), m=[x.copy() for x in self._m], v=[x.copy() for x in self._v])

    def _validated_state(self, state):
        expected = {'format_version', 'optimizer_type', 'learning_rate', 'beta1', 'beta2', 'epsilon', 't', 'parameters', 'm', 'v'}
        _validate_mapping(state, expected)
        if state['format_version'] != self._STATE_VERSION or state['optimizer_type'] != 'adam':
            raise ValueError('Incompatible Adam state format.')
        values = (positive_real(state['learning_rate'], 'learning_rate'), self._beta(state['beta1'], 'beta1'),
                  self._beta(state['beta2'], 'beta2'), positive_real(state['epsilon'], 'epsilon'))
        if isinstance(state['t'], (bool, np.bool_)) or not isinstance(state['t'], (int, np.integer)) or state['t'] < 0:
            raise ValueError('Adam t must be a nonnegative integer.')
        _validate_layout(self.parameters, state['parameters'])
        if not isinstance(state['m'], (list, tuple)) or not isinstance(state['v'], (list, tuple)) or len(state['m']) != len(self._m) or len(state['v']) != len(self._v):
            raise ValueError('Adam moment count is incompatible.')
        moments = []
        for candidate_m, candidate_v, parameter in zip(state['m'], state['v'], self.parameters):
            if not isinstance(candidate_m, np.ndarray) or not isinstance(candidate_v, np.ndarray):
                raise ValueError('Adam moments must be ndarrays.')
            if candidate_m.shape != parameter.data.shape or candidate_v.shape != parameter.data.shape or candidate_m.dtype != DTYPE or candidate_v.dtype != DTYPE:
                raise ValueError('Adam moment shape or dtype is incompatible.')
            finite(candidate_m); finite(candidate_v)
            moments.append((candidate_m.copy(), candidate_v.copy()))
        return values, int(state['t']), moments

    def load_state_dict(self, state):
        values, t, moments = self._validated_state(state)
        self._learning_rate, self._beta1, self._beta2, self._epsilon = values
        for target_m, target_v, (source_m, source_v) in zip(self._m, self._v, moments):
            np.copyto(target_m, source_m); np.copyto(target_v, source_v)
        self._t = t
