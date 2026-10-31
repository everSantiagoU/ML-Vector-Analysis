"""Fixed context embeddings + dense MLP, with an explicit Trainer adapter."""
import numpy as np
from ..nn import MLP
from ..nn.layers import _Cache, checked_cache
from ..nn._validation import array, integer
from ..nn.losses import softmax
from .embedding import Embedding
from .text import token_ids
from .data import ContextDataset


class ContextLanguageModel:
    def __init__(self, vocab_size, context_length, embedding_dim=8, hidden_dim=24, *, seed=42):
        self.context_length = integer(context_length, 'context_length')
        self.embedding = Embedding(vocab_size, embedding_dim, seed=seed)
        self.mlp = MLP([context_length * embedding_dim, hidden_dim, vocab_size], seed=seed)
        self.input_width, self.output_width = context_length, vocab_size
        self._parameters = self.embedding.parameters() + self.mlp.parameters()

    def parameters(self):
        return self._parameters

    def forward(self, ids, *, retain_cache=False):
        ids = token_ids(ids, self.output_width, 2)
        if ids.shape[1] != self.context_length:
            raise ValueError('Incorrect context width.')
        embeddings, ec = self.embedding.forward(ids, retain_cache=retain_cache)
        try:
            logits, mc = self.mlp.forward(embeddings.reshape(len(ids), -1), retain_cache=retain_cache)
        except Exception:
            if ec is not None:
                ec.release()
            raise
        context = (_Cache(self, logits.shape, (ec, mc), tuple(p.version for p in self.parameters()))
                   if retain_cache else None)
        return logits, context

    def release_context(self, context):
        if not isinstance(context, _Cache) or context.owner is not self:
            raise ValueError('Context belongs to another model.')
        if context.payload is not None:
            ec, mc = context.payload
            ec.release()
            self.mlp.release_context(mc)
        context.release()

    def backward(self, upstream_gradient, context):
        upstream = array(upstream_gradient, 'upstream_gradient', 2)
        context = checked_cache(context, self, upstream.shape)
        try:
            ec, mc = context.payload
            gradient = self.mlp.backward(upstream, mc)
            self.embedding.backward(gradient.reshape(ec.shape), ec)
        finally:
            self.release_context(context)

    def predict_logits(self, ids):
        return self.forward(ids)[0]

    def predict_proba(self, ids):
        return softmax(self.predict_logits(ids))

    def predict_classes(self, ids):
        return self.predict_logits(ids).argmax(axis=1)


class WindowModelAdapter:
    """Trainer inputs are example indices, never floating-point token IDs.

    Pair this adapter ONLY with dataset.training_arrays(). Inference on actual
    token contexts uses ContextLanguageModel directly.
    """
    input_width = 1

    def __init__(self, model, dataset):
        if not isinstance(model, ContextLanguageModel) or not isinstance(dataset, ContextDataset):
            raise ValueError('Expected ContextLanguageModel and ContextDataset.')
        if model.context_length != dataset.context_length or model.output_width != dataset.vocab_size:
            raise ValueError('Model and dataset dimensions must match.')
        self.model, self.dataset = model, dataset
        self.output_width = model.output_width

    def parameters(self):
        return self.model.parameters()

    def forward(self, indices, *, retain_cache=False):
        indices = array(indices, 'example indices', 2)
        if (indices.shape[1] != 1 or np.any(indices < 0) or np.any(indices >= len(self.dataset))
                or np.any(indices != np.floor(indices))):
            raise ValueError('Expected valid integer-valued example indices (B,1).')
        contexts, _ = self.dataset.batch(indices[:, 0].astype(np.int64))
        return self.model.forward(contexts, retain_cache=retain_cache)

    def backward(self, gradient, context):
        return self.model.backward(gradient, context)

    def release_context(self, context):
        self.model.release_context(context)
