"""Resident dense datasets: one O(N) permutation and copies of selected batches."""
import numpy as np
from ._validation import array, integer


class MiniBatchIterator:
    def __init__(self, X, targets, batch_size, *, shuffle=True, seed=None):
        self.X = array(X, 'X', 2)
        self.targets = np.asarray(targets)
        if self.targets.ndim not in (1, 2) or len(self.targets) != len(self.X):
            raise ValueError('X and targets must have the same positive row count.')
        if self.targets.dtype.kind not in 'iuf' or 0 in self.targets.shape or not np.isfinite(self.targets).all():
            raise ValueError('Targets must be nonempty finite real numbers.')
        self.batch_size = integer(batch_size, 'batch_size')
        if type(shuffle) is not bool:
            raise ValueError('shuffle must be bool.')
        self.shuffle = shuffle
        self.rng = np.random.default_rng(None if seed is None else integer(seed, 'seed', 0))

    def __iter__(self):
        indices = self.rng.permutation(len(self.X)) if self.shuffle else None
        for start in range(0, len(self.X), self.batch_size):
            selection = slice(start, start + self.batch_size) if indices is None else indices[start:start + self.batch_size]
            yield self.X[selection], self.targets[selection]
