"""Immutable scalar metrics and bounded history; no dataset or model snapshots."""
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    accuracy: float | None = None


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    optimizer_steps: int
    train_loss: float | None
    val_loss: float | None = None
    train_accuracy: float | None = None
    val_accuracy: float | None = None


@dataclass(frozen=True)
class TrainingResult:
    status: str
    epochs_completed: int
    optimizer_steps: int
    history: tuple[EpochMetrics, ...]
    config: object
    learning_rate: float
    seed: int | None
    message: str = ''

    def to_dict(self):
        result = asdict(self)
        # Fail loudly if a caller manually constructs an invalid DTO.
        json.dumps(result, allow_nan=False)
        return result


class _History:
    """Progressive decimation of record ordinals, retaining initial and latest."""
    def __init__(self, limit):
        self.limit, self.stride, self.ordinal = limit, 1, 0
        self.points = []

    def record(self, snapshot):
        if self.points and self.points[-1][1] == snapshot:
            return
        if self.points and self.points[-1][0] % self.stride:
            self.points.pop()
        self.points.append((self.ordinal, snapshot))
        self.ordinal += 1
        while len(self.points) > self.limit:
            self.stride *= 2
            self.points = [p for p in self.points[:-1] if p[0] % self.stride == 0] + [self.points[-1]]

    @property
    def snapshots(self):
        return tuple(snapshot for _, snapshot in self.points)
