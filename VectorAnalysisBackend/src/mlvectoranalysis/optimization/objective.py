"""Structural objective contracts; no concrete inheritance is required."""
from typing import Protocol, runtime_checkable

import numpy as np


class ObjectiveEvaluationError(ValueError):
    """An otherwise valid parameter vector leaves the real finite domain."""


@runtime_checkable
class DifferentiableObjective(Protocol):
    """Float64 vectors of shape (dimension,); inputs must not be mutated.

    A returned gradient must remain valid across subsequent evaluations.
    Implementations may reuse private workspaces, but must return owned gradients.
    Only ObjectiveEvaluationError/arithmetic exceptions signal a rejected trial;
    programming errors are not swallowed by the optimizer.
    """

    @property
    def dimension(self) -> int: ...

    def value(self, parameters: np.ndarray) -> float: ...

    def value_and_grad(self, parameters: np.ndarray) -> tuple[float, np.ndarray]: ...


@runtime_checkable
class QuadraticObjective(DifferentiableObjective, Protocol):
    """Optional capability: g^T H g, without materializing H."""

    def directional_curvature(self, direction: np.ndarray) -> float: ...
