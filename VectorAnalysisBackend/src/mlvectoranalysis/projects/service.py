"""Application use cases; DTOs only at the boundary, no HTTP or UI dependencies."""
from dataclasses import replace
from pathlib import PurePath
from typing import Callable
from uuid import uuid4

from .contracts import ArtifactStore, DatasetAdapter, Evaluator, ModelProvider, Progress, RetrievalIndex, Trainer
from .data import EvaluationAdapter, TextAdapter, preview
from .dto import (Config, Dataset, ErrorDTO, EvaluationCase, Project, ProjectError, Snapshot, State,
                  require, text)
from .evaluation import LocalEvaluator
from .rag import ExtractiveDemoProvider, LexicalIndex, RagTask, SyncTrainer, segment
from .strategy import recommend


class ProjectService:
    def __init__(self, store: ArtifactStore, *, provider: ModelProvider | None = None,
                 adapters: tuple[DatasetAdapter, ...] | None = None,
                 index_factory: Callable[[], RetrievalIndex] = LexicalIndex,
                 trainer: Trainer | None = None, evaluator: Evaluator | None = None):
        self.store = store
        self.provider = provider if provider is not None else ExtractiveDemoProvider()
        self.adapters = adapters if adapters is not None else (TextAdapter(), EvaluationAdapter())
        self.index_factory = index_factory
        self.trainer = trainer if trainer is not None else SyncTrainer()
        self.evaluator = evaluator if evaluator is not None else LocalEvaluator()
        require(isinstance(store, ArtifactStore) and isinstance(self.provider, ModelProvider)
                and isinstance(self.trainer, Trainer) and isinstance(self.evaluator, Evaluator)
                and callable(index_factory) and isinstance(self.adapters, tuple)
                and all(isinstance(a, DatasetAdapter) for a in self.adapters), 'Contrato inválido.')
        extensions = [ext for adapter in self.adapters for ext in adapter.extensions]
        require(len(extensions) == len(set(extensions)), 'Adaptadores ambiguos.')

    def create_project(self, objective: str, *, task_type: str = 'rag', config: Config | None = None) -> Project:
        project = Project(uuid4().hex, objective, task_type, config if config is not None else Config())
        self.store.save(Snapshot(project))
        return project

    def get_project(self, project_id: str) -> Project:
        return self.store.load(project_id).project

    def list_projects(self) -> tuple[Project, ...]:
        return tuple(self.get_project(identifier) for identifier in self.store.list_ids())

    def import_files(self, project_id: str, files: tuple[tuple[str, bytes], ...]) -> Project:
        snapshot = self.store.load(project_id)
        project, config = snapshot.project, snapshot.project.config
        require(isinstance(files, tuple) and bool(files), 'Se requiere una lista de archivos.')
        require(len(files) + len(project.files) <= config.max_files, 'Demasiados archivos.', 'file_limit')
        names = {f.name.casefold() for f in project.files}
        total = sum(f.size_bytes for f in project.files)
        datasets = []
        for item in files:
            require(isinstance(item, tuple) and len(item) == 2, 'Archivo inválido.')
            name, content = item
            require(isinstance(name, str) and type(content) is bytes, 'Nombre o contenido inválido.')
            require(name.casefold() not in names, 'Nombre de archivo duplicado.')
            names.add(name.casefold())
            total += len(content)
            require(total <= config.max_total_bytes, 'Límite total de bytes excedido.', 'file_limit')
            adapter = next((a for a in self.adapters if PurePath(name).suffix.lower() in a.extensions), None)
            require(adapter is not None, 'Formato no soportado.', 'unsupported_format')
            dataset = adapter.parse(name, content, config)
            require(isinstance(dataset, Dataset) and dataset.source.name == name
                    and dataset.source.size_bytes == len(content), 'Resultado de adaptador inválido.')
            datasets.append(dataset)
        updated = project.transition(State.FILES, files=project.files + tuple(d.source for d in datasets),
                                     strategy=None, result=None, evaluation=None, error=None, progress=0)
        # All-or-nothing batch; no writes if parsing or cross-file validation fails.
        combined = Snapshot(updated, snapshot.documents + tuple(doc for d in datasets for doc in d.documents),
                            snapshot.cases + tuple(case for d in datasets for case in d.cases))
        self.store.save(combined)
        return updated

    def preview_and_validate(self, project_id: str) -> dict:
        snapshot = self.store.load(project_id)
        project = snapshot.project
        require(project.state in (State.FILES, State.VALIDATED), 'Importar archivos primero.', 'invalid_state')
        chunks = segment(snapshot.documents, project.config)  # Validate aggregate chunk budget now.
        updated = project if project.state == State.VALIDATED else project.transition(State.VALIDATED)
        self.store.save(replace(snapshot, project=updated))
        return {'project': updated.to_dict(), 'documents': [preview(d, project.config) for d in snapshot.documents],
                'case_count': len(snapshot.cases), 'chunk_count': len(chunks), 'valid': True}

    def recommend_strategy(self, project_id: str) -> Project:
        snapshot = self.store.load(project_id)
        require(snapshot.project.state in (State.VALIDATED, State.READY, State.FAILED, State.CANCELLED),
                'Validar archivos antes de recomendar.', 'invalid_state')
        strategy = recommend(snapshot.project, len(snapshot.cases))
        project = (snapshot.project.transition(State.READY, strategy=strategy, error=None, progress=0)
                   if snapshot.project.state != State.READY else replace(snapshot.project, strategy=strategy))
        self.store.save(replace(snapshot, project=project))
        return project

    def create_evaluation(self, project_id: str, cases: tuple[EvaluationCase, ...]) -> Project:
        snapshot = self.store.load(project_id)
        require(snapshot.project.state != State.RUNNING, 'Proyecto ocupado.', 'invalid_state')
        require(isinstance(cases, tuple) and bool(cases) and all(isinstance(c, EvaluationCase) for c in cases),
                'Casos inválidos.')
        if snapshot.project.state == State.EVALUATED:
            state = State.COMPLETED if snapshot.project.result else State.READY
            project = snapshot.project.transition(state, evaluation=None)
        else:
            project = replace(snapshot.project, evaluation=None)
        if project.strategy is not None:
            project = replace(project, strategy=recommend(project, len(cases)))
        updated = Snapshot(project, snapshot.documents, cases)
        self.store.save(updated)
        return project

    def _task(self, snapshot: Snapshot, index: RetrievalIndex) -> RagTask:
        require(isinstance(index, RetrievalIndex), 'Índice inválido.')
        index.build(segment(snapshot.documents, snapshot.project.config))
        return RagTask(index, self.provider, snapshot.project.config)

    def _execute(self, project_id: str, question: str | None, progress: Progress | None) -> Project:
        snapshot = self.store.load(project_id)
        project = snapshot.project
        require(project.strategy is not None and project.strategy.available,
                'Estrategia no disponible.', 'unavailable')
        require(progress is None or callable(progress), 'Callback inválido.')
        if question is not None:
            text(question, project.config.question_chars)
        else:
            require(bool(snapshot.cases), 'No hay casos de evaluación.')
        index = self.index_factory()
        require(isinstance(index, RetrievalIndex), 'Índice inválido.')
        project = project.transition(State.RUNNING, result=None if question is not None else project.result,
                                     evaluation=None, error=None, progress=0,
                                     index_version=index.name, provider_version=self.provider.name)
        current = replace(snapshot, project=project)
        self.store.save(current)

        def observe(percent: int):
            nonlocal current
            current = replace(current, project=replace(current.project, progress=percent))
            self.store.save(current)
            return progress(percent) if progress is not None else None

        try:
            task = self._task(current, index)
            if question is not None:
                result = self.trainer.execute(task, question, observe)
                require(result.question == question, 'Resultado de ejecución incompatible.')
                project = current.project.transition(State.COMPLETED, result=result, progress=100)
            else:
                evaluation = self.evaluator.evaluate(task, current.cases, observe)
                require(tuple(r.case for r in evaluation.results) == current.cases,
                        'Informe de evaluación incompatible.')
                project = current.project.transition(State.EVALUATED, evaluation=evaluation, progress=100)
        except Exception as error:
            if isinstance(error, ProjectError) and error.code == 'cancelled':
                project = current.project.transition(State.CANCELLED)
            else:
                # Never persist provider exception text, document content, paths, or secrets.
                known_codes = {'validation', 'chunk_limit', 'provider_contract', 'storage_io', 'artifact_limit'}
                code = error.code if isinstance(error, ProjectError) and error.code in known_codes else 'execution_failed'
                project = current.project.transition(State.FAILED,
                    error=ErrorDTO(code, 'La ejecución falló; revise contratos, límites y configuración.'))
        self.store.save(replace(current, project=project))
        return project

    def execute_rag(self, project_id: str, question: str, *, progress: Progress | None = None) -> Project:
        text(question, 100_000_000)
        return self._execute(project_id, question, progress)

    def run_evaluation(self, project_id: str, *, progress: Progress | None = None) -> Project:
        return self._execute(project_id, None, progress)

    def cancel(self, project_id: str) -> Project:
        snapshot = self.store.load(project_id)
        require(snapshot.project.state != State.RUNNING,
                'Cancelar ejecución síncrona mediante callback de progreso.', 'invalid_state')
        project = snapshot.project.transition(State.CANCELLED, error=None)
        self.store.save(replace(snapshot, project=project))
        return project

    def recover_interrupted(self, project_id: str) -> Project:
        """Call only after confirming the previous synchronous process has stopped."""
        snapshot = self.store.load(project_id)
        require(snapshot.project.state == State.RUNNING, 'No hay ejecución interrumpida.', 'invalid_state')
        project = snapshot.project.transition(State.FAILED,
            error=ErrorDTO('interrupted', 'Ejecución interrumpida; puede recomendar de nuevo y reintentar.'))
        self.store.save(replace(snapshot, project=project))
        return project

    def save_project(self, project_id: str) -> Project:
        snapshot = self.store.load(project_id)
        self.store.save(snapshot)
        return snapshot.project

    def export_project(self, project_id: str) -> dict:
        return {'project_id': project_id, 'path': self.store.export(project_id),
                'kind': 'configuration_and_normalized_data', 'trained_model': False}
