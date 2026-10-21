"""Bounded progressive decimation: retain 0, multiples of stride, and latest."""
import numpy as np

from ._validation import integer
from .result import IterationSnapshot


class TrajectoryRecorder:
    def __init__(self, mode='sampled', max_points=200, indices=None):
        if mode not in ('none', 'metrics', 'sampled'):
            raise ValueError('history mode must be none, metrics or sampled.')
        self.max_points = integer(max_points, 'max_points', 2)
        self.mode = mode
        self.indices = None if indices is None else tuple(integer(i, 'axis') for i in indices)
        if self.indices is not None and (len(self.indices) not in (2, 3) or len(set(self.indices)) != len(self.indices)):
            raise ValueError('Select two or three distinct history indices.')
        self._points = []
        self._stride = 1

    def record(self, iteration, parameters, loss, gradient_norm, step):
        if self.mode == 'none':
            return
        if self._points and iteration <= self._points[-1].iteration:
            raise ValueError('Snapshots must be recorded in strictly increasing order.')
        coordinates = None
        if self.mode == 'sampled':
            coordinates = (parameters.copy() if self.indices is None
                           else np.array(parameters[list(self.indices)], copy=True))
        snapshot = IterationSnapshot(iteration, coordinates, loss, gradient_norm, step)
        if self._points and self._points[-1].iteration % self._stride != 0:
            self._points.pop()  # The previous transient endpoint is no longer needed.
        self._points.append(snapshot)
        while len(self._points) > self.max_points:
            self._stride *= 2
            self._points = [point for point in self._points[:-1]
                            if point.iteration % self._stride == 0] + [snapshot]

    @property
    def snapshots(self):
        return tuple(self._points)
