"""Sequential dense network, with a linear output for either supported task."""
import numpy as np
from ._validation import DTYPE, array, integer
from .layers import LinearLayer, _Cache, checked_cache
from .activations import Tanh, ReLU
from .losses import softmax
from .parameters import unique_parameters


class MLP:
    def __init__(self, sizes, *, activation='tanh', seed=None):
        sizes = tuple(integer(size, 'width') for size in sizes)
        if len(sizes) < 2 or activation not in ('tanh', 'relu'):
            raise ValueError('At least two widths and activation tanh/relu are required.')
        if seed is not None:
            seed = integer(seed, 'seed', 0)
        rng = np.random.default_rng(seed)
        layers = []
        for i, (left, right) in enumerate(zip(sizes, sizes[1:])):
            hidden = i < len(sizes) - 2
            layers.append(LinearLayer(left, right, rng=rng,
                initialization='he' if hidden and activation == 'relu' else 'xavier', name=f'linear_{i}'))
            if hidden:
                layers.append(Tanh() if activation == 'tanh' else ReLU())
        self._initialize(layers)

    @classmethod
    def from_layers(cls, layers):
        """Compose compatible layers without allowing shared layers/parameters."""
        model = cls.__new__(cls)
        model._initialize(tuple(layers))
        return model

    def _initialize(self, layers):
        self._layers = tuple(layers)
        if not self.layers or len({id(layer) for layer in self.layers}) != len(self.layers):
            raise ValueError('Empty networks and shared layers are not supported.')
        if not isinstance(self.layers[0], LinearLayer) or not isinstance(self.layers[-1], LinearLayer):
            raise ValueError('First and last layers must be linear.')
        width = self.layers[0].in_features
        for layer in self.layers:
            if not isinstance(layer, (LinearLayer, Tanh, ReLU)):
                raise ValueError('Unsupported layer.')
            if isinstance(layer, LinearLayer):
                if layer.in_features != width:
                    raise ValueError('Incompatible layer widths.')
                width = layer.out_features
        self._parameters = unique_parameters(p for layer in self.layers for p in layer.parameters())
        self.input_width = self.layers[0].in_features
        self.output_width = width

    @property
    def layers(self):
        return self._layers

    def parameters(self):
        return self._parameters

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, X, *, retain_cache=False):
        caches = []
        try:
            for layer in self.layers:
                X, cache = layer.forward(X, retain_cache=retain_cache)
                if retain_cache:
                    caches.append(cache)
            context = _Cache(self, X.shape, caches, tuple(p.version for p in self.parameters())) if retain_cache else None
            return X, context
        except Exception:
            for cache in caches:
                cache.release()
            raise

    def release_context(self, context):
        """Release activations on cancellation/error; repeated release is harmless."""
        if not isinstance(context, _Cache) or context.owner is not self:
            raise ValueError('Context belongs to another model.')
        if context.payload is not None:
            for cache in context.payload:
                cache.release()
        context.release()

    def backward(self, d_outputs, context):
        gradient = array(d_outputs, 'd_outputs', 2)
        context = checked_cache(context, self, gradient.shape)
        try:
            for layer, cache in zip(self.layers, context.payload):
                cache.validate(layer, cache.shape)
            for layer, cache in zip(reversed(self.layers), reversed(context.payload)):
                gradient = layer.backward(gradient, cache)
            return gradient
        finally:
            self.release_context(context)

    def _predict(self, X, batch_size, transform, width, dtype):
        X = array(X, 'X', 2)
        batch_size = integer(batch_size, 'batch_size')
        if X.shape[1] != self.input_width:
            raise ValueError('Incorrect input width.')
        shape = (len(X), width) if width is not None else (len(X),)
        result = np.empty(shape, dtype=dtype)
        for start in range(0, len(X), batch_size):
            output, _ = self.forward(X[start:start + batch_size])
            result[start:start + batch_size] = transform(output)
        return result

    def predict(self, X, batch_size=256):
        """Raw outputs; final O(N * output_width) array is retained."""
        return self._predict(X, batch_size, lambda x: x, self.output_width, DTYPE)

    def predict_proba(self, X, batch_size=256):
        """Classification only: stable softmax per block."""
        return self._predict(X, batch_size, softmax, self.output_width, DTYPE)

    def predict_classes(self, X, batch_size=256):
        """Classification only: class argmax per block, final O(N) indices."""
        if self.output_width < 2:
            raise ValueError('Classification requires at least two logits.')
        return self._predict(X, batch_size, lambda x: x.argmax(axis=1), None, np.int64)
