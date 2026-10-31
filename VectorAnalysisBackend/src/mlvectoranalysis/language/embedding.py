"""Trainable float64 lookup; scatter-add handles repeated IDs correctly."""
import numpy as np
from ..nn.parameters import Parameter
from ..nn.layers import _Cache, checked_cache
from ..nn._validation import array, finite, integer, numeric
from .text import token_ids


class Embedding:
    def __init__(self, vocab_size, embedding_dim, *, seed=None):
        self.vocab_size = integer(vocab_size, 'vocab_size', 2)
        self.embedding_dim = integer(embedding_dim, 'embedding_dim')
        seed = None if seed is None else integer(seed, 'seed', 0)
        self.weight = Parameter(np.random.default_rng(seed).normal(
            0, 1 / np.sqrt(embedding_dim), (vocab_size, embedding_dim)), 'embedding.weight')

    def parameters(self):
        return (self.weight,)

    def forward(self, ids, *, retain_cache=False):
        ids = token_ids(ids, self.vocab_size, 2)
        outputs = finite(self.weight.data[ids])
        cache = _Cache(self, outputs.shape, ids.copy(), (self.weight.version,)) if retain_cache else None
        return outputs, cache

    @numeric
    def backward(self, upstream_gradient, cache):
        upstream = array(upstream_gradient, 'upstream_gradient', 3)
        cache = checked_cache(cache, self, upstream.shape)
        try:
            self.weight.grad.fill(0)
            np.add.at(self.weight.grad, cache.payload.ravel(), upstream.reshape(-1, self.embedding_dim))
            finite(self.weight.grad)
            # Integer IDs are discrete: no derivative with respect to IDs.
            return None
        finally:
            cache.release()
