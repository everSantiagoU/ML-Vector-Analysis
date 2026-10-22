"""Elementwise VJPs. Only Tanh output or a ReLU boolean mask is retained."""
import numpy as np
from ._validation import array, finite, numeric
from .layers import _Cache, checked_cache


class Tanh:
    def parameters(self):
        return ()

    @numeric
    def forward(self, inputs, *, retain_cache=False):
        outputs = np.tanh(array(inputs, 'inputs', 2))
        return outputs, _Cache(self, outputs.shape, outputs) if retain_cache else None

    @numeric
    def backward(self, upstream_gradient, cache):
        upstream = array(upstream_gradient, 'upstream_gradient', 2)
        cache = checked_cache(cache, self, upstream.shape)
        try:
            return finite(upstream * (1 - cache.payload**2))
        finally:
            cache.release()


class ReLU:
    def parameters(self):
        return ()

    @numeric
    def forward(self, inputs, *, retain_cache=False):
        inputs = array(inputs, 'inputs', 2)
        outputs = np.maximum(inputs, 0)
        return outputs, _Cache(self, outputs.shape, inputs > 0) if retain_cache else None

    @numeric
    def backward(self, upstream_gradient, cache):
        upstream = array(upstream_gradient, 'upstream_gradient', 2)
        cache = checked_cache(cache, self, upstream.shape)
        try:
            return finite(upstream * cache.payload)
        finally:
            cache.release()
