"""Generic descent engine; max_iter counts accepted updates, starting at zero."""
from dataclasses import dataclass

import numpy as np

from ._validation import finite_scalar, integer, norm, parameters, real
from .history import TrajectoryRecorder
from .objective import DifferentiableObjective, ObjectiveEvaluationError, QuadraticObjective
from .result import IterationSnapshot, OptimizationResult


@dataclass(frozen=True)
class GradientDescentConfig:
    max_iter: int = 1000
    gradient_atol: float = 1e-8
    gradient_rtol: float = 1e-8
    step_rtol: float = 1e-14
    loss_rtol: float = 1e-15
    step_strategy: str = 'auto'
    alpha: float = 1.0
    c1: float = 1e-4
    beta: float = 0.5
    max_backtracks: int = 40
    min_step: float = 1e-16
    history_mode: str = 'sampled'
    max_points: int = 200
    history_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        integer(self.max_iter, 'max_iter')
        integer(self.max_backtracks, 'max_backtracks')
        integer(self.max_points, 'max_points', 2)
        for name in ('gradient_atol', 'gradient_rtol', 'step_rtol', 'loss_rtol'):
            real(getattr(self, name), name)
        for name in ('alpha', 'min_step', 'c1', 'beta'):
            real(getattr(self, name), name, strict=True)
        if self.c1 >= 1 or self.beta >= 1:
            raise ValueError('c1 and beta must be in (0,1).')
        if self.step_strategy not in ('auto', 'fixed', 'armijo', 'exact_quadratic'):
            raise ValueError('Unknown step_strategy.')
        recorder = TrajectoryRecorder(self.history_mode, self.max_points, self.history_indices)
        object.__setattr__(self, 'history_indices', recorder.indices)


_EVALUATION_ERRORS = (ObjectiveEvaluationError, FloatingPointError, OverflowError, ZeroDivisionError)
_MESSAGES = {
    'gradient_tolerance': 'Gradient satisfies absolute + initial-relative tolerance (stationarity only).',
    'max_iter': 'Maximum accepted updates reached without stationarity.',
    'stalled': 'Numerically zero or relatively small progress with gradient above tolerance.',
    'line_search_failed': 'Armijo search exhausted its backtracks or minimum step.',
    'non_finite': 'Numerical/domain failure; last finite accepted state preserved.',
    'cancelled': 'Callback cancelled optimization.',
}


class GradientDescent:
    def __init__(self, config: GradientDescentConfig | None = None):
        self.config = config if config is not None else GradientDescentConfig()
        if not isinstance(self.config, GradientDescentConfig):
            raise ValueError('config must be GradientDescentConfig.')

    def minimize(self, objective: DifferentiableObjective, initial_parameters, callback=None) -> OptimizationResult:
        """Callback runs at iteration 0 and after every accepted update.

        It receives a full copied parameter vector (O(p) per call), independent
        of projected history. Returning False cancels, including at iteration 0.
        Function/gradient counts include attempted evaluations that fail.
        """
        c = self.config
        dimension = integer(objective.dimension, 'objective.dimension', 1)
        point = parameters(initial_parameters, dimension).copy()
        if callback is not None and not callable(callback):
            raise ValueError('callback must be callable.')
        if c.history_indices is not None and max(c.history_indices) >= dimension:
            raise ValueError('History index exceeds objective dimension.')
        strategy = c.step_strategy
        quadratic = isinstance(objective, QuadraticObjective)
        if strategy == 'auto':
            strategy = 'exact_quadratic' if quadratic else 'armijo'
        if strategy == 'exact_quadratic' and not quadratic:
            raise ValueError('exact_quadratic requires directional_curvature capability.')
        recorder = TrajectoryRecorder(c.history_mode, c.max_points, c.history_indices)
        n_function = n_gradient = n_curvature = 0

        def evaluate(at):
            nonlocal n_function, n_gradient
            n_function += 1
            n_gradient += 1
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                loss, gradient = objective.value_and_grad(at)
                loss = finite_scalar(loss)
                raw = np.asarray(gradient)
                if raw.shape != (dimension,) or raw.dtype.kind not in 'iuf':
                    raise ValueError('Objective gradient violates the shape/dtype contract.')
                gradient = np.array(raw, dtype=np.float64, copy=True)
                if not np.isfinite(gradient).all():
                    raise ObjectiveEvaluationError('Non-finite objective gradient.')
                return loss, gradient, norm(gradient)

        try:
            loss, gradient, gradient_norm = evaluate(point)
            tolerance = c.gradient_atol + c.gradient_rtol * gradient_norm
            if not np.isfinite(tolerance):
                raise ValueError('Gradient tolerance exceeds float64 range.')
        except _EVALUATION_ERRORS as error:
            raise ValueError('Initial objective value/gradient must be real and finite.') from error
        iteration, step = 0, None
        recorder.record(0, point, loss, gradient_norm, None)

        def cancelled():
            return callback is not None and callback(
                IterationSnapshot(iteration, point.copy(), loss, gradient_norm, step)) is False

        status = 'max_iter'
        if cancelled():
            status = 'cancelled'
        elif gradient_norm <= tolerance:
            status = 'gradient_tolerance'
        else:
            for _ in range(c.max_iter):
                try:
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        step = c.alpha
                        if strategy in ('armijo', 'exact_quadratic'):
                            squared_norm = finite_scalar(np.dot(gradient, gradient))
                        if strategy == 'exact_quadratic':
                            n_curvature += 1
                            curvature = finite_scalar(objective.directional_curvature(gradient))
                            if curvature <= 0:
                                raise ObjectiveEvaluationError('Non-positive curvature above gradient tolerance.')
                            step = finite_scalar(squared_norm / curvature)
                            if step <= 0:
                                raise ObjectiveEvaluationError('Non-positive exact step.')

                        accepted = False
                        attempts = c.max_backtracks + 1 if strategy == 'armijo' else 1
                        for attempt in range(attempts):
                            if strategy == 'armijo' and step < c.min_step:
                                break
                            try:
                                trial = point - step * gradient
                                if not np.isfinite(trial).all():
                                    raise ObjectiveEvaluationError('Non-finite candidate.')
                                if np.array_equal(trial, point):
                                    status = 'stalled'
                                    break
                                if strategy == 'armijo':
                                    n_function += 1
                                    trial_loss = finite_scalar(objective.value(trial))
                                    if trial_loss > loss - c.c1 * step * squared_norm:
                                        step *= c.beta
                                        continue
                                trial_loss, trial_gradient, trial_norm = evaluate(trial)
                                if strategy != 'fixed':
                                    allowance = 64 * np.finfo(float).eps * max(1.0, abs(loss))
                                    if trial_loss > loss + allowance:
                                        raise ObjectiveEvaluationError('Supposed descent increased the objective.')
                                accepted = True
                                break
                            except _EVALUATION_ERRORS:
                                if strategy != 'armijo':
                                    raise
                                step *= c.beta
                        if not accepted:
                            if status != 'stalled':
                                status = 'line_search_failed'
                            break

                        # All reporting below refers to this same, fully evaluated state.
                        relative_step = norm(trial - point) / max(1.0, norm(point))
                        # Scaling first avoids overflow in subtracting large finite losses.
                        loss_scale = max(1.0, abs(loss), abs(trial_loss))
                        relative_loss = abs(trial_loss / loss_scale - loss / loss_scale)
                except _EVALUATION_ERRORS:
                    status = 'non_finite'
                    break
                point, loss, gradient, gradient_norm = trial, trial_loss, trial_gradient, trial_norm
                iteration += 1
                recorder.record(iteration, point, loss, gradient_norm, step)
                # User callback exceptions are not numerical objective failures.
                if cancelled():
                    status = 'cancelled'
                    break
                if gradient_norm <= tolerance:
                    status = 'gradient_tolerance'
                    break
                if relative_step <= c.step_rtol and relative_loss <= c.loss_rtol:
                    status = 'stalled'
                    break

        indices = (() if c.history_mode != 'sampled' else
                   c.history_indices if c.history_indices is not None else tuple(range(dimension)))
        return OptimizationResult(
            point.copy(), loss, gradient_norm, iteration, status == 'gradient_tolerance',
            status, _MESSAGES[status], n_function, n_gradient, n_curvature, recorder.snapshots,
            indices, tuple(getattr(objective, 'parameter_names', tuple(f'parameter_{i}' for i in range(dimension)))),
            getattr(objective, 'coordinate_system', 'parameters'), strategy)
