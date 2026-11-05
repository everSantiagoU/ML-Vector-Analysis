"""Local project workflows, independent of educational NumPy models and any UI."""
from .contracts import ArtifactStore, DatasetAdapter, Evaluator, ModelProvider, RetrievalIndex, Task, Trainer
from .data import EvaluationAdapter, TextAdapter
from .dto import (Answer, CaseResult, Chunk, Config, Dataset, Document, ErrorDTO, EvaluationCase,
                  EvaluationReport, FileMetadata, Project, ProjectError, Snapshot, Source, State, Strategy)
from .evaluation import LocalEvaluator
from .rag import ExtractiveDemoProvider, LexicalIndex, RagTask, SyncTrainer
from .service import ProjectService
from .storage import LocalArtifactStore

__all__ = [
    'ArtifactStore', 'DatasetAdapter', 'Evaluator', 'ModelProvider', 'RetrievalIndex', 'Task', 'Trainer',
    'EvaluationAdapter', 'TextAdapter', 'Answer', 'CaseResult', 'Chunk', 'Config', 'Dataset', 'Document',
    'ErrorDTO', 'EvaluationCase', 'EvaluationReport', 'FileMetadata', 'Project', 'ProjectError',
    'Snapshot', 'Source', 'State', 'Strategy', 'LocalEvaluator', 'ExtractiveDemoProvider',
    'LexicalIndex', 'RagTask', 'SyncTrainer', 'ProjectService', 'LocalArtifactStore',
]
