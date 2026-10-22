"""Small layer protocol and per-execution contexts; no global last_input."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from ._validation import array, finite, integer, numeric
from .parameters import Parameter


class Layer(Protocol):
    def forward(self, inputs, *, retain_cache=False): ...
    def backward(self, upstream_gradient, cache): ...
    def parameters(self) -> tuple[Parameter, ...]: ...


@dataclass(eq=False)
class _Cache:
    owner: object
    shape: tuple
    payload: object
    versions: tuple = ()
    consumed: bool = False

    def validate(self, owner, shape):
        if self.owner is not owner or self.consumed:
            raise ValueError('Context belongs to another layer/model or was consumed.')
        if tuple(shape) != self.shape:
            raise ValueError('Upstream gradient shape does not match context.')
        if self.versions != tuple(p.version for p in owner.parameters()):
            raise ValueError('Parameters changed since forward; context is stale.')

    def release(self):
        self.payload = None
        self.consumed = True


def checked_cache(cache, owner, shape):
    if not isinstance(cache, _Cache):
        raise ValueError('Backward requires a retained forward context.')
    cache.validate(owner, shape)
    return cache


class LinearLayer:
    def __init__(self, in_features, out_features, *, rng, initialization='xavier', name='linear'):
        self.in_features = integer(in_features, 'in_features')
        self.out_features = integer(out_features, 'out_features')
        if not isinstance(rng, np.random.Generator):
            raise ValueError('rng must be a numpy.random.Generator.')
        if initialization not in ('xavier', 'he'):
            raise ValueError('initialization must be xavier or he.')
        scale = np.sqrt(2 / (in_features if initialization == 'he' else in_features + out_features))
        self.weight = Parameter(rng.normal(0, scale, (in_features, out_features)), f'{name}.weight')
        self.bias = Parameter(np.zeros(out_features), f'{name}.bias')

    def parameters(self):
        return self.weight, self.bias

    @numeric
    def forward(self, inputs, *, retain_cache=False):
        inputs = array(inputs, 'inputs', 2)
        if inputs.shape[1] != self.in_features:
            raise ValueError('Incorrect input width.')
        outputs = finite(inputs @ self.weight.data + self.bias.data)
        # Float64 inputs are referenced, never copied: do not mutate until backward.
        cache = _Cache(self, outputs.shape, inputs, tuple(p.version for p in self.parameters())) if retain_cache else None
        return outputs, cache

    @numeric
    def backward(self, upstream_gradient, cache):
        upstream = array(upstream_gradient, 'upstream_gradient', 2)
        cache = checked_cache(cache, self, upstream.shape)
        try:
            dx = finite(upstream @ self.weight.data.T)
            dw = finite(cache.payload.T @ upstream)
            db = finite(upstream.sum(axis=0))
            np.copyto(self.weight.grad, dw)
            np.copyto(self.bias.grad, db)
            return dx
        finally:
            cache.release()
