"""Small structural ports; implementations must validate inputs and return DTOs."""
from typing import Callable, Protocol, runtime_checkable
from .dto import Answer, Chunk, Config, Dataset, EvaluationCase, EvaluationReport, Snapshot, Source

Progress = Callable[[int], bool | None]  # False requests cooperative cancellation.


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    def answer(self, question: str, context: tuple[Source, ...]) -> Answer: ...


@runtime_checkable
class DatasetAdapter(Protocol):
    extensions: tuple[str, ...]

    def parse(self, name: str, content: bytes, config: Config) -> Dataset: ...


@runtime_checkable
class RetrievalIndex(Protocol):
    name: str

    def build(self, chunks: tuple[Chunk, ...]) -> None: ...
    def search(self, question: str, top_k: int) -> tuple[Source, ...]: ...


@runtime_checkable
class Task(Protocol):
    def run(self, question: str) -> Answer: ...


@runtime_checkable
class Trainer(Protocol):
    """MVP job executor. No weights or optimizer implied by this port."""
    def execute(self, task: Task, question: str, progress: Progress | None = None) -> Answer: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, task: Task, cases: tuple[EvaluationCase, ...],
                 progress: Progress | None = None) -> EvaluationReport: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def save(self, snapshot: Snapshot) -> None: ...
    def load(self, project_id: str) -> Snapshot: ...
    def list_ids(self) -> tuple[str, ...]: ...
    def export(self, project_id: str) -> str: ...
