"""Dense, float64, full-batch least squares with an implicit scaled operator."""
import numpy as np

from ._validation import integer, parameters, real_array, finite_scalar
from .objective import ObjectiveEvaluationError


class LinearLeastSquaresObjective:
    """L = ||Xw+b-targets||²/(2n). Parameters are [weights, optional intercept].

    Float64 inputs (including strided layouts) are shared, never written. Other
    real numeric dtypes/lists require a one-time conversion. Callers must not
    mutate shared inputs during optimization. Workspaces make this object
    non-reentrant; returned gradients own their storage.
    """

    def __init__(self, X, targets, *, fit_intercept=True, standardize=False, block_size=None):
        if type(fit_intercept) is not bool or type(standardize) is not bool:
            raise ValueError('fit_intercept and standardize must be bool.')
        if block_size is not None:
            block_size = integer(block_size, 'block_size', 1)
        validation_chunk = block_size or 8192
        self.X = real_array(X, 'X', 2, validation_chunk)
        self.targets = real_array(targets, 'targets', 1, validation_chunk)
        self.n, self.d = self.X.shape
        if self.n == 0 or self.d == 0 or self.targets.shape != (self.n,):
            raise ValueError('X must have n>0, d>0 and targets must have shape (n,).')
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.dimension = self.d + int(fit_intercept)
        self.block_size = min(block_size or self.n, self.n)
        self.parameter_names = tuple(f'weight_{i}' for i in range(self.d)) + (('intercept',) if fit_intercept else ())
        self.coordinate_system = 'standardized_parameters' if standardize else 'original_parameters'
        self.mean = np.zeros(self.d)
        self.scale = np.ones(self.d)
        if standardize:
            self._statistics()
        self._residual = np.empty(self.block_size)
        self._work_gradient = np.empty(self.d)

    def _statistics(self):
        # Stable shifted means and centered deviations, merged with Chan's rule.
        # A feature loop is confined to one-time statistics, never the gradient.
        # Only O(B) temporaries, even for statistics; no standardized B x d array.
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                for column in range(self.d):
                    count, mean, std = 0, 0.0, 0.0
                    for start in range(0, self.n, self.block_size):
                        values = self.X[start:start + self.block_size, column]
                        size = len(values)
                        block_mean = float(values[0] + np.mean(values - values[0]))
                        deviations = values - block_mean
                        largest = float(np.max(np.abs(deviations)))
                        block_std = (largest * float(np.linalg.norm(deviations / largest)) / np.sqrt(size)
                                     if largest else 0.0)
                        total = count + size
                        fraction = size / total
                        delta = block_mean - mean
                        if count:
                            std = float(np.hypot(np.hypot(std * np.sqrt(count / total),
                                                         block_std * np.sqrt(fraction)),
                                                 delta * np.sqrt(count / total * fraction)))
                            mean += delta * fraction
                        else:
                            mean, std = block_mean, block_std
                        count = total
                    self.mean[column] = mean if self.fit_intercept else 0.0
                    self.scale[column] = std if std > 0 else 1.0
                if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
                    raise ValueError('Standardization statistics exceed float64 range.')
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('Standardization statistics exceed float64 range.') from error

    def _evaluate(self, point, with_gradient):
        point = parameters(point, self.dimension)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                weights = point[:self.d] / self.scale
                offset = (point[-1] if self.fit_intercept else 0.0) - np.dot(self.mean, weights)
                loss = 0.0
                gradient = np.zeros(self.dimension) if with_gradient else None
                for start in range(0, self.n, self.block_size):
                    rows = self.X[start:start + self.block_size]
                    residual = self._residual[:len(rows)]
                    np.dot(rows, weights, out=residual)
                    residual += offset
                    residual -= self.targets[start:start + len(rows)]
                    loss += float(np.dot(residual, residual) / (2.0 * self.n))
                    if with_gradient:
                        residual_sum = np.sum(residual)
                        np.dot(rows.T, residual, out=self._work_gradient)
                        self._work_gradient -= self.mean * residual_sum
                        self._work_gradient /= self.scale
                        gradient[:self.d] += self._work_gradient / self.n
                        if self.fit_intercept:
                            gradient[-1] += residual_sum / self.n
                loss = finite_scalar(loss)
                if with_gradient and not np.isfinite(gradient).all():
                    raise ObjectiveEvaluationError('Least-squares gradient is non-finite.')
                return loss, gradient
        except (FloatingPointError, OverflowError) as error:
            raise ObjectiveEvaluationError('Least-squares evaluation exceeds float64 range.') from error

    def value(self, parameters):
        return self._evaluate(parameters, False)[0]

    def value_and_grad(self, parameters):
        return self._evaluate(parameters, True)

    def directional_curvature(self, direction):
        direction = parameters(direction, self.dimension)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                weights = direction[:self.d] / self.scale
                offset = (direction[-1] if self.fit_intercept else 0.0) - np.dot(self.mean, weights)
                curvature = 0.0
                for start in range(0, self.n, self.block_size):
                    rows = self.X[start:start + self.block_size]
                    values = self._residual[:len(rows)]
                    np.dot(rows, weights, out=values)
                    values += offset
                    curvature += float(np.dot(values, values) / self.n)
                return finite_scalar(curvature)
        except (FloatingPointError, OverflowError) as error:
            raise ObjectiveEvaluationError('Directional curvature exceeds float64 range.') from error
