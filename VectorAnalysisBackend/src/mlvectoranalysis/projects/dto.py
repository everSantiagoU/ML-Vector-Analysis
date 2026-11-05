"""Versioned, strict JSON boundary types. No framework or numeric dependencies."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import json
import math
import re
import types
from typing import get_args, get_origin, get_type_hints


class ProjectError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def to_dict(self) -> dict:
        return {'code': self.code, 'message': str(self)}


def require(condition: bool, message: str, code: str = 'validation') -> None:
    if not condition:
        raise ProjectError(code, message)


def safe_name(value: str) -> str:
    require(isinstance(value, str) and 0 < len(value) <= 120
            and re.fullmatch(r'[\w][\w .-]*', value, re.UNICODE) is not None
            and '..' not in value and not value.endswith((' ', '.')),
            'Nombre inseguro.', 'unsafe_path')
    return value


def text(value: str, maximum: int = 4096) -> None:
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
            'Texto vacío, inválido o demasiado largo.')


def integer(value: int, low: int, high: int) -> None:
    require(type(value) is int and low <= value <= high, 'Entero fuera de límites.')


def dumps(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':'))
    except (ValueError, TypeError, RecursionError) as error:
        raise ProjectError('invalid_json', 'Se requieren valores JSON finitos.') from error


def loads(value: str) -> object:
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, 'Claves JSON duplicadas.', 'invalid_json')
            result[key] = item
        return result

    def constant(_):
        raise ProjectError('invalid_json', 'JSON no finito.')

    try:
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=constant)
        def finite(value):
            if isinstance(value, float):
                require(math.isfinite(value), 'JSON no finito.', 'invalid_json')
            elif isinstance(value, (list, dict)):
                for item in (value.values() if isinstance(value, dict) else value):
                    finite(item)
        finite(result)  # also catches 1e999 without encoding a second copy
        return result
    except (ValueError, TypeError, RecursionError) as error:
        raise ProjectError('invalid_json', 'JSON inválido.') from error


def _decode(kind, value):
    origin, args = get_origin(kind), get_args(kind)
    if origin is types.UnionType:
        for option in args:
            try:
                return _decode(option, value)
            except ProjectError:
                pass
        raise ProjectError('validation', 'Tipo incompatible.')
    if kind is type(None):
        require(value is None, 'Se esperaba null.')
    elif origin is tuple:
        require(isinstance(value, (list, tuple)), 'Se esperaba una lista.')
        return tuple(_decode(args[0], item) for item in value)
    elif isinstance(kind, type) and issubclass(kind, DTO):
        return value if isinstance(value, kind) else kind.from_dict(value)
    elif isinstance(kind, type) and issubclass(kind, Enum):
        try:
            return kind(value)
        except (ValueError, TypeError) as error:
            raise ProjectError('validation', 'Estado desconocido.') from error
    elif kind is float:
        require(type(value) in (int, float) and math.isfinite(value), 'Número no finito.')
        return float(value)
    else:
        require(type(value) is kind, 'Tipo incompatible.')
    return value


class DTO:
    def __post_init__(self):
        # Type checks also apply to direct Python constructors, not just JSON.
        for name, kind in get_type_hints(type(self)).items():
            value = getattr(self, name)
            object.__setattr__(self, name, _decode(kind, value))

    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, DTO):
                return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [convert(v) for v in value]
            require(value is None or type(value) in (str, bool, int, float), 'Valor no serializable.')
            require(not isinstance(value, float) or math.isfinite(value), 'Número no finito.')
            return value
        return convert(self)

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict), 'Se esperaba un objeto JSON.')
        known = {f.name for f in fields(cls)}
        require(set(value) <= known, 'Campos desconocidos.')
        hints = get_type_hints(cls)
        try:
            return cls(**{name: _decode(hints[name], item) for name, item in value.items()})
        except TypeError as error:
            raise ProjectError('validation', 'Campos requeridos ausentes.') from error


@dataclass(frozen=True)
class Config(DTO):
    max_files: int = 32
    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 4_000_000
    max_document_chars: int = 1_000_000
    chunk_chars: int = 800
    overlap_chars: int = 100
    max_chunks: int = 10_000
    context_chars: int = 4000
    question_chars: int = 2000
    top_k: int = 5
    max_cases: int = 100
    preview_chars: int = 300

    def __post_init__(self):
        super().__post_init__()
        for f in fields(self):
            integer(getattr(self, f.name), 0 if f.name == 'overlap_chars' else 1, 100_000_000)
        require(self.overlap_chars < self.chunk_chars, 'Solapamiento debe ser menor al fragmento.')
        require(self.max_file_bytes <= self.max_total_bytes, 'Límite por archivo supera el total.')
        require(self.top_k <= self.max_chunks, 'top_k supera max_chunks.')


@dataclass(frozen=True)
class FileMetadata(DTO):
    id: str
    name: str
    size_bytes: int
    sha256: str
    kind: str

    def __post_init__(self):
        super().__post_init__()
        safe_name(self.name)
        require(re.fullmatch('[a-f0-9]{64}', self.id) is not None
                and re.fullmatch('[a-f0-9]{64}', self.sha256) is not None, 'Hash inválido.')
        integer(self.size_bytes, 1, 100_000_000)
        require(self.kind in ('document', 'evaluation'), 'Tipo de archivo inválido.')


@dataclass(frozen=True)
class Document(DTO):
    source: FileMetadata
    content: str

    def __post_init__(self):
        super().__post_init__()
        text(self.content, 100_000_000)
        require(self.source.kind == 'document', 'Origen no es un documento.')
        require(len(self.content) <= self.source.size_bytes, 'Contenido excede bytes de origen.')


@dataclass(frozen=True)
class EvaluationCase(DTO):
    id: str
    question: str
    expected_answer: str | None = None
    expected_sources: tuple[str, ...] = ()
    criteria: tuple[str, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        safe_name(self.id)
        text(self.question)
        if self.expected_answer is not None:
            text(self.expected_answer, 100_000)
        require(len(set(self.expected_sources)) == len(self.expected_sources) <= 32, 'Fuentes duplicadas o excesivas.')
        for name in self.expected_sources:
            safe_name(name)
        require(len(self.criteria) <= 32, 'Demasiados criterios.')
        for criterion in self.criteria:
            text(criterion)


@dataclass(frozen=True)
class Dataset(DTO):
    source: FileMetadata
    documents: tuple[Document, ...] = ()
    cases: tuple[EvaluationCase, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if self.source.kind == 'document':
            require(len(self.documents) == 1 and self.documents[0].source == self.source and not self.cases,
                    'Dataset de documento inválido.')
        else:
            require(not self.documents and bool(self.cases), 'Dataset de evaluación inválido.')


@dataclass(frozen=True)
class Chunk(DTO):
    id: str
    document_id: str
    filename: str
    start: int
    end: int
    text: str

    def __post_init__(self):
        super().__post_init__()
        safe_name(self.id)
        safe_name(self.document_id)
        safe_name(self.filename)
        integer(self.start, 0, 100_000_000)
        integer(self.end, 1, 100_000_000)
        require(self.end - self.start == len(self.text) > 0, 'Offsets incompatibles.')


@dataclass(frozen=True)
class Source(DTO):
    chunk: Chunk
    rank: int
    score: float
    matched_tokens: tuple[str, ...]

    def __post_init__(self):
        super().__post_init__()
        integer(self.rank, 1, 100_000_000)
        require(self.score > 0, 'Puntaje debe ser positivo.')


@dataclass(frozen=True)
class Answer(DTO):
    question: str
    answer: str
    provider: str
    sources: tuple[Source, ...]
    used_chunk_ids: tuple[str, ...]
    supported: bool
    context_sources: tuple[Source, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        text(self.question, 100_000_000)
        text(self.answer, 100_000_000)
        text(self.provider, 120)
        require(len({s.chunk.id for s in self.sources}) == len(self.sources), 'Fuentes duplicadas.')
        require(set(self.used_chunk_ids) <= {s.chunk.id for s in self.sources}, 'Citas fuera de recuperación.')
        require(self.supported == bool(self.used_chunk_ids), 'Soporte y citas incompatibles.')
        retrieved = {s.chunk.id: s.chunk for s in self.sources}
        for source in self.context_sources:
            chunk = source.chunk
            original = retrieved.get(chunk.id)
            require(original is not None and chunk.document_id == original.document_id
                    and chunk.filename == original.filename and chunk.start == original.start
                    and chunk.end <= original.end and original.text.startswith(chunk.text),
                    'Contexto fuera de recuperación.')


@dataclass(frozen=True)
class CaseResult(DTO):
    case: EvaluationCase
    response: Answer
    exact_match: bool | None
    source_recall: float | None

    def __post_init__(self):
        super().__post_init__()
        require(self.response.question == self.case.question, 'Pregunta incompatible.')
        exact = self.response.answer == self.case.expected_answer if self.case.expected_answer is not None else None
        retrieved = {s.chunk.filename for s in self.response.sources}
        recall = (len(set(self.case.expected_sources) & retrieved) / len(self.case.expected_sources)
                  if self.case.expected_sources else None)
        require(self.exact_match == exact and self.source_recall == recall, 'Métricas incompatibles.')
        if self.source_recall is not None:
            require(0 <= self.source_recall <= 1, 'Recall fuera de rango.')


@dataclass(frozen=True)
class EvaluationReport(DTO):
    results: tuple[CaseResult, ...]
    exact_match_mean: float | None
    source_recall_mean: float | None
    criteria_policy: str = 'Registrados para revisión humana; no puntuados automáticamente.'

    def __post_init__(self):
        super().__post_init__()
        require(bool(self.results), 'Evaluación vacía.')
        require(len({r.case.id for r in self.results}) == len(self.results), 'Resultados duplicados.')
        for name, summary in (('exact_match', self.exact_match_mean), ('source_recall', self.source_recall_mean)):
            values = [float(getattr(r, name)) for r in self.results if getattr(r, name) is not None]
            require(summary == (sum(values) / len(values) if values else None), 'Resumen incompatible.')
        for value in (self.exact_match_mean, self.source_recall_mean):
            require(value is None or 0 <= value <= 1, 'Métrica fuera de rango.')


@dataclass(frozen=True)
class Strategy(DTO):
    capability: str
    available: bool
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    missing: tuple[str, ...]

    def __post_init__(self):
        super().__post_init__()
        require(self.capability in ('local_rag', 'evaluation', 'fine_tuning'), 'Capacidad desconocida.')
        require(bool(self.reasons) and bool(self.limitations), 'Estrategia sin explicación.')
        require(not (self.capability == 'fine_tuning' and self.available), 'Ajuste fino no disponible.')


class State(str, Enum):
    CREATED = 'created'
    FILES = 'files_imported'
    VALIDATED = 'validated'
    READY = 'ready'
    RUNNING = 'running'
    COMPLETED = 'completed'
    EVALUATED = 'evaluated'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


TRANSITIONS = {
    State.CREATED: {State.FILES, State.CANCELLED},
    State.FILES: {State.FILES, State.VALIDATED, State.CANCELLED},
    State.VALIDATED: {State.FILES, State.READY, State.CANCELLED},
    State.READY: {State.FILES, State.RUNNING, State.CANCELLED},
    State.RUNNING: {State.COMPLETED, State.EVALUATED, State.FAILED, State.CANCELLED},
    State.COMPLETED: {State.FILES, State.RUNNING, State.CANCELLED},
    State.EVALUATED: {State.FILES, State.RUNNING, State.CANCELLED, State.READY, State.COMPLETED},
    State.FAILED: {State.FILES, State.READY, State.CANCELLED},
    State.CANCELLED: {State.FILES, State.READY},
}


@dataclass(frozen=True)
class ErrorDTO(DTO):
    code: str
    message: str

    def __post_init__(self):
        super().__post_init__()
        text(self.code, 120)
        text(self.message)


@dataclass(frozen=True)
class Project(DTO):
    id: str
    objective: str
    task_type: str = 'rag'
    config: Config = Config()
    state: State = State.CREATED
    files: tuple[FileMetadata, ...] = ()
    strategy: Strategy | None = None
    result: Answer | None = None
    evaluation: EvaluationReport | None = None
    error: ErrorDTO | None = None
    progress: int = 0
    format_version: int = 1
    index_version: str = 'tfidf-cosine-v1'
    provider_version: str = 'extractive-demo-v1'

    def __post_init__(self):
        super().__post_init__()
        text(self.index_version, 120)
        text(self.provider_version, 120)
        require(re.fullmatch('[a-f0-9]{32}', self.id) is not None, 'ID de proyecto inválido.')
        text(self.objective)
        require(self.task_type in ('rag', 'evaluation', 'fine_tuning'), 'Tarea desconocida.')
        require(type(self.format_version) is int and self.format_version == 1, 'Versión no soportada.')
        integer(self.progress, 0, 100)
        require(len(self.files) <= self.config.max_files and len({f.name.casefold() for f in self.files}) == len(self.files),
                'Demasiados archivos o nombres duplicados.')
        require(sum(f.size_bytes for f in self.files) <= self.config.max_total_bytes
                and all(f.size_bytes <= self.config.max_file_bytes for f in self.files), 'Límite de bytes excedido.')
        if self.state in (State.READY, State.RUNNING, State.COMPLETED, State.EVALUATED):
            require(self.strategy is not None, 'Falta estrategia.')
        if self.state == State.COMPLETED:
            require(self.result is not None, 'Falta resultado.')
        if self.state == State.EVALUATED:
            require(self.evaluation is not None, 'Falta evaluación.')
        require((self.state == State.FAILED) == (self.error is not None), 'Estado y error incompatibles.')

    def transition(self, state: State, **changes) -> Project:
        from dataclasses import replace
        require(state in TRANSITIONS[self.state], 'Transición de estado inválida.', 'invalid_state')
        return replace(self, state=state, **changes)


@dataclass(frozen=True)
class Snapshot(DTO):
    project: Project
    documents: tuple[Document, ...] = ()
    cases: tuple[EvaluationCase, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        config = self.project.config
        require(len(self.cases) <= config.max_cases and len({c.id for c in self.cases}) == len(self.cases),
                'Casos duplicados o excesivos.')
        require(all(len(c.question) <= config.question_chars for c in self.cases), 'Pregunta excede límite.')
        require(tuple(d.source for d in self.documents) == tuple(f for f in self.project.files if f.kind == 'document'),
                'Documentos y metadatos incompatibles.')
        require(all(len(d.content) <= config.max_document_chars for d in self.documents), 'Documento excede límite.')
