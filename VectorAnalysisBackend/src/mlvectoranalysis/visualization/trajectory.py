"""Iteration-index interpolation of retained samples, not an ODE solution."""
import numpy as np

from ..optimization._validation import integer
from ..optimization.result import OptimizationResult


class OptimizationTrajectory:
    def __init__(self, result: OptimizationResult, axes=None):
        self.result = result
        available = result.history_indices
        if axes is None:
            if len(result.parameters) > 3 and available:
                raise ValueError('Select two or three original parameter indices for a projection.')
            axes = available if len(available) in (2, 3) else ()
        self.axes = tuple(integer(axis, 'axis') for axis in axes)
        if self.axes and (len(self.axes) not in (2, 3) or len(set(self.axes)) != len(self.axes)):
            raise ValueError('A trajectory requires two or three distinct axes.')
        if any(axis not in available for axis in self.axes):
            raise ValueError('Selected axes were not captured in this history.')
        self._positions = [available.index(axis) for axis in self.axes]

    def to_dict(self):
        snapshots = self.result.snapshots
        return dict(iterations=[s.iteration for s in snapshots], losses=[s.loss for s in snapshots],
                    gradient_norms=[s.gradient_norm for s in snapshots],
                    points=[s.parameters[self._positions].tolist() for s in snapshots] if self.axes else [],
                    axes=[self.result.parameter_names[axis] for axis in self.axes],
                    original_indices=list(self.axes), coordinate_system=self.result.coordinate_system,
                    is_projection=bool(self.axes) and len(self.axes) < len(self.result.parameters),
                    interpolation='linear_between_retained_iterations',
                    sampled=len(snapshots) < self.result.n_iter + 1)

    def interpolate(self, iteration):
        if not self.axes or not self.result.snapshots:
            raise ValueError('Interpolation requires captured parameter coordinates.')
        if (isinstance(iteration, (bool, np.bool_)) or not np.isscalar(iteration)
                or not isinstance(iteration, (int, float, np.integer, np.floating)) or not np.isfinite(iteration)):
            raise ValueError('iteration must be a finite real number.')
        points = self.result.snapshots
        if iteration < points[0].iteration or iteration > points[-1].iteration:
            raise ValueError('Interpolation outside the retained iteration interval is not allowed.')
        index = int(np.searchsorted([s.iteration for s in points], iteration))
        right = points[index]
        if right.iteration == iteration:
            return right.parameters[self._positions].copy()
        left = points[index - 1]
        fraction = (iteration - left.iteration) / (right.iteration - left.iteration)
        return ((1 - fraction) * left.parameters[self._positions]
                + fraction * right.parameters[self._positions])
