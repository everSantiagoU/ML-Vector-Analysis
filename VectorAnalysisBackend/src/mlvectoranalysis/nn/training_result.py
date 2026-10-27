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

    def state_dict(self):
        return dict(limit=self.limit, stride=self.stride, ordinal=self.ordinal,
                    points=[dict(ordinal=ordinal, metric=asdict(metric)) for ordinal, metric in self.points])

    @classmethod
    def from_state_dict(cls, state, limit):
        if not isinstance(state, dict) or set(state) != {'limit', 'stride', 'ordinal', 'points'}:
            raise ValueError('History state is invalid.')
        if state['limit'] != limit or not isinstance(state['stride'], int) or state['stride'] < 1 or state['stride'] & (state['stride'] - 1):
            raise ValueError('History state is incompatible.')
        if not isinstance(state['ordinal'], int) or state['ordinal'] < 0 or not isinstance(state['points'], list):
            raise ValueError('History state is invalid.')
        result = cls(limit)
        result.stride, result.ordinal = state['stride'], state['ordinal']
        try:
            result.points = [(item['ordinal'], EpochMetrics(**item['metric'])) for item in state['points']]
        except (KeyError, TypeError) as error:
            raise ValueError('History points are invalid.') from error
        if len(result.points) > limit or any(not isinstance(i, int) or i < 0 for i, _ in result.points):
            raise ValueError('History state exceeds its limit.')
        return result
