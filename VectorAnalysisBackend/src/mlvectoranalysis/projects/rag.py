"""Deterministic lexical retrieval and explicitly extractive demo provider."""
from collections import Counter
from dataclasses import replace
import math
import re
import unicodedata

from .contracts import ModelProvider, Progress, RetrievalIndex, Task
from .dto import Answer, Chunk, Config, Document, ProjectError, Source, integer, require, text

# Fixed, versioned minimal stoplist. No semantic or language detection claim.
STOPWORDS = frozenset('el la los las un una unos unas de del a y o en es que qué cuál cuáles '
                      'cómo por para con se the a an of and or in is what which how to'.split())


def tokens(value: str) -> tuple[str, ...]:
    return tuple(t for t in re.findall(r'\w+', unicodedata.normalize('NFC', value).casefold())
                 if t not in STOPWORDS)


def segment(documents: tuple[Document, ...], config: Config) -> tuple[Chunk, ...]:
    require(isinstance(config, Config) and isinstance(documents, tuple)
            and all(isinstance(d, Document) for d in documents), 'Documentos inválidos.')
    require(len(documents) <= config.max_files, 'Demasiados documentos.')
    chunks = []
    for document in documents:
        require(len(document.content) <= config.max_document_chars, 'Documento demasiado largo.')
        for start in range(0, len(document.content), config.chunk_chars - config.overlap_chars):
            end = min(start + config.chunk_chars, len(document.content))
            require(len(chunks) < config.max_chunks, 'Demasiados fragmentos.', 'chunk_limit')
            chunks.append(Chunk(f'{document.source.id}-{start}', document.source.id,
                                document.source.name, start, end, document.content[start:end]))
            if end == len(document.content):
                break
    return tuple(chunks)


class LexicalIndex:
    """TF-IDF cosine: tf=count, idf=1+log((N+1)/(df+1)); positive matches only."""
    name = 'tfidf-cosine-v1'

    def __init__(self):
        self._chunks = ()
        self._vectors = ()
        self._idf = {}

    def build(self, chunks: tuple[Chunk, ...]) -> None:
        require(isinstance(chunks, tuple) and all(isinstance(c, Chunk) for c in chunks), 'Fragmentos inválidos.')
        require(len({c.id for c in chunks}) == len(chunks), 'Fragmentos duplicados.')
        counts = tuple(Counter(tokens(c.text)) for c in chunks)
        frequencies = Counter(t for row in counts for t in row)
        idf = {t: 1 + math.log((len(chunks) + 1) / (df + 1)) for t, df in frequencies.items()}
        vectors = []
        for row in counts:
            weights = {t: n * idf[t] for t, n in row.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1
            vectors.append({t: w / norm for t, w in weights.items()})
        self._chunks, self._vectors, self._idf = chunks, tuple(vectors), idf

    def search(self, question: str, top_k: int) -> tuple[Source, ...]:
        text(question, 100_000_000)
        integer(top_k, 1, 100_000_000)
        query = {t: n * self._idf[t] for t, n in Counter(tokens(question)).items() if t in self._idf}
        norm = math.sqrt(sum(w * w for w in query.values())) or 1
        ranked = []
        for chunk, row in zip(self._chunks, self._vectors):
            matches = tuple(sorted(query.keys() & row.keys()))
            score = sum(query[t] * row[t] / norm for t in matches)
            if score > 0:
                ranked.append((score, chunk, matches))
        ranked.sort(key=lambda item: (-item[0], item[1].filename, item[1].start, item[1].id))
        return tuple(Source(chunk, rank, score, matches)
                     for rank, (score, chunk, matches) in enumerate(ranked[:top_k], 1))


class ExtractiveDemoProvider:
    name = 'extractive-demo-v1'

    def answer(self, question: str, context: tuple[Source, ...]) -> Answer:
        text(question, 100_000_000)
        require(isinstance(context, tuple) and all(isinstance(s, Source) for s in context), 'Contexto inválido.')
        response = ('Extractos de demostración (sin LLM):\n' + '\n'.join(s.chunk.text for s in context)
                    if context else 'No se encontró soporte suficiente en los documentos.')
        return Answer(question, response, self.name, context, tuple(s.chunk.id for s in context), bool(context), context)


class RagTask:
    def __init__(self, index: RetrievalIndex, provider: ModelProvider, config: Config):
        require(isinstance(index, RetrievalIndex) and isinstance(provider, ModelProvider)
                and isinstance(config, Config), 'Contratos de RAG inválidos.')
        self.index, self.provider, self.config = index, provider, config

    def run(self, question: str) -> Answer:
        text(question, self.config.question_chars)
        retrieved = self.index.search(question, self.config.top_k)
        require(isinstance(retrieved, tuple) and len(retrieved) <= self.config.top_k
                and all(isinstance(s, Source) for s in retrieved), 'Resultado del índice inválido.')
        context = []
        remaining = self.config.context_chars
        for source in retrieved:
            if remaining <= 0:
                break
            excerpt = source.chunk.text[:remaining]
            # Do not pass a truncated excerpt that no longer contains any query evidence.
            if not set(tokens(question)).intersection(tokens(excerpt)):
                continue
            chunk = replace(source.chunk, end=source.chunk.start + len(excerpt), text=excerpt)
            context.append(replace(source, chunk=chunk))
            remaining -= len(excerpt)
        if not context:
            # A custom provider cannot hallucinate when retrieval supplies no evidence.
            return replace(ExtractiveDemoProvider().answer(question, ()), sources=retrieved)
        answer = self.provider.answer(question, tuple(context))
        require(isinstance(answer, Answer) and answer.question == question and answer.provider == self.provider.name
                and answer.sources == tuple(context)
                and len(answer.answer) <= self.config.context_chars + 1024,
                'Respuesta del proveedor inválida.', 'provider_contract')
        return replace(answer, sources=retrieved, context_sources=tuple(context))


class SyncTrainer:
    """Synchronous execution with cooperative cancellation at step boundaries."""
    def execute(self, task: Task, question: str, progress: Progress | None = None) -> Answer:
        require(isinstance(task, Task), 'Tarea inválida.')
        text(question, 100_000_000)
        checkpoint(progress, 0)
        answer = task.run(question)
        require(isinstance(answer, Answer), 'Resultado de tarea inválido.')
        checkpoint(progress, 100)
        return answer


def checkpoint(progress: Progress | None, percent: int) -> None:
    require(progress is None or callable(progress), 'Callback inválido.')
    if progress is not None and progress(percent) is False:
        raise ProjectError('cancelled', 'Ejecución cancelada en un límite de etapa.')
