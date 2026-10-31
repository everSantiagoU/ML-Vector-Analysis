"""Store tokens once; represent valid windows by their start offsets."""
import numpy as np
from ..nn._validation import integer
from .text import token_ids


class ContextDataset:
    def __init__(self, documents, context_length, vocab_size):
        self.context_length = integer(context_length, 'context_length')
        self.vocab_size = integer(vocab_size, 'vocab_size', 2)
        if isinstance(documents, (str, bytes)):
            raise ValueError('Expected documents of integer IDs.')
        try:
            docs = [token_ids(doc, self.vocab_size) for doc in documents]
        except TypeError as error:
            raise ValueError('Expected documents of integer IDs.') from error
        if not docs:
            raise ValueError('Corpus must not be empty.')
        starts, offset = [], 0
        for doc in docs:
            starts.append(np.arange(offset, offset + max(0, len(doc) - self.context_length), dtype=np.int64))
            offset += len(doc)
        self.starts = np.concatenate(starts)
        if not len(self.starts):
            raise ValueError('Corpus has no complete context/target examples.')
        self.ids = np.concatenate(docs, dtype=np.int64)
        self.ids.flags.writeable = self.starts.flags.writeable = False

    def __len__(self):
        return len(self.starts)

    def batch(self, indices):
        indices = token_ids(indices, len(self))
        starts = self.starts[indices]
        contexts = self.ids[starts[:, None] + np.arange(self.context_length)]
        return contexts, self.ids[starts + self.context_length]

    def training_arrays(self):
        """Explicit float64 example indices for the existing dense Trainer."""
        return np.arange(len(self), dtype=np.float64)[:, None], self.ids[self.starts + self.context_length]
