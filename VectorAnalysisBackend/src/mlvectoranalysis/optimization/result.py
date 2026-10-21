"""Owned snapshots and strict JSON exports; no training data in results."""
from dataclasses import dataclass
import json

import numpy as np


@dataclass(frozen=True)
class IterationSnapshot:
    iteration: int
    parameters: np.ndarray | None
    loss: float
    gradient_norm: float
    step: float | None

    def to_dict(self):
        result = dict(iteration=int(self.iteration),
                      parameters=None if self.parameters is None else self.parameters.tolist(),
                      loss=float(self.loss), gradient_norm=float(self.gradient_norm),
                      step=None if self.step is None else float(self.step))
        json.dumps(result, allow_nan=False)
        return result


@dataclass(frozen=True)
class OptimizationResult:
    parameters: np.ndarray
    loss: float
    gradient_norm: float
    n_iter: int
    converged: bool
    status: str
    message: str
    n_function_evals: int
    n_gradient_evals: int
    n_curvature_evals: int
    snapshots: tuple[IterationSnapshot, ...]
    history_indices: tuple[int, ...]
    parameter_names: tuple[str, ...]
    coordinate_system: str
    step_strategy: str

    def to_dict(self):
        result = dict(parameters=self.parameters.tolist(), loss=float(self.loss),
                      gradient_norm=float(self.gradient_norm), n_iter=int(self.n_iter),
                      converged=bool(self.converged), status=self.status, message=self.message,
                      n_function_evals=int(self.n_function_evals),
                      n_gradient_evals=int(self.n_gradient_evals),
                      n_curvature_evals=int(self.n_curvature_evals),
                      snapshots=[snapshot.to_dict() for snapshot in self.snapshots],
                      history_indices=list(self.history_indices), parameter_names=list(self.parameter_names),
                      coordinate_system=self.coordinate_system, step_strategy=self.step_strategy)
        json.dumps(result, allow_nan=False)
        return result
