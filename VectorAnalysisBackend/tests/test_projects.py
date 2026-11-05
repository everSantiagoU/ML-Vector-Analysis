"""Public workflow and port conformance, deterministic retrieval, safe persistence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import os

import pytest

from mlvectoranalysis.projects import (
    Answer, ArtifactStore, Config, DatasetAdapter, EvaluationAdapter, EvaluationCase,
    Evaluator, ExtractiveDemoProvider, LocalArtifactStore, LocalEvaluator, ModelProvider,
    Project, ProjectError, ProjectService, RagTask, RetrievalIndex, Snapshot, State,
    Task, TextAdapter, Trainer, SyncTrainer, LexicalIndex, ErrorDTO,
)
from mlvectoranalysis.projects.data import preview
from mlvectoranalysis.projects.dto import TRANSITIONS, dumps, loads
from mlvectoranalysis.projects.rag import segment
from mlvectoranalysis.projects.strategy import recommend


@pytest.fixture
def service(tmp_path):
    return ProjectService(LocalArtifactStore(tmp_path / 'data'))


def ready(service, files=None, **kwargs):
    project = service.create_project('Consultar manual local', **kwargs)
    service.import_files(project.id, files or (('manual.md', 'Python administra memoria.\nDatos locales.'.encode()),))
    service.preview_and_validate(project.id)
    return service.recommend_strategy(project.id)


def task(config=None):
    config = config or Config()
    document = TextAdapter().parse('manual.txt', b'Python administra memoria.', config).documents
    index = LexicalIndex()
    index.build(segment(document, config))
    return RagTask(index, ExtractiveDemoProvider(), config)


@pytest.mark.parametrize('implementation,contract', [
    (ExtractiveDemoProvider(), ModelProvider), (TextAdapter(), DatasetAdapter),
    (EvaluationAdapter(), DatasetAdapter), (LexicalIndex(), RetrievalIndex),
    (SyncTrainer(), Trainer), (LocalEvaluator(), Evaluator), (task(), Task),
])
def test_structural_contracts(implementation, contract):
    assert isinstance(implementation, contract)


def test_store_contract_and_dto_roundtrip(service):
    assert isinstance(service.store, ArtifactStore)
    project = ready(service)
    assert Project.from_dict(json.loads(json.dumps(project.to_dict(), allow_nan=False))) == project
    assert service.get_project(project.id) == project
    assert service.list_projects() == (project,)
    assert service.save_project(project.id) == project


@pytest.mark.parametrize('source', list(State))
@pytest.mark.parametrize('destination', list(State))
def test_transition_matrix(source, destination):
    base = Project('a' * 32, 'Consultar')
    strategy = recommend(base, 0)
    response = task().run('Python')
    report = LocalEvaluator().evaluate(task(), (EvaluationCase('case', 'Python'),))
    project = replace(base, state=source, strategy=strategy, result=response, evaluation=report,
                      error=ErrorDTO('test', 'Error') if source == State.FAILED else None)
    updates = {'error': ErrorDTO('test', 'Error') if destination == State.FAILED else None}
    if destination in TRANSITIONS[source]:
        assert project.transition(destination, **updates).state == destination
    else:
        with pytest.raises(ProjectError, match='Transición'):
            project.transition(destination, **updates)


@pytest.mark.parametrize('config', [dict(top_k=True), dict(chunk_chars=10, overlap_chars=10),
    dict(max_files=0), dict(max_file_bytes=20, max_total_bytes=10), dict(context_chars=float('nan')),
    dict(top_k=11, max_chunks=10), dict(question_chars='2')])
def test_invalid_config(config):
    with pytest.raises(ProjectError):
        Config(**config)


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', '1e999', '{"a":1,"a":2}', '{bad'])
def test_strict_json_read(value):
    with pytest.raises(ProjectError):
        loads(value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), object()])
def test_strict_json_write(value):
    with pytest.raises(ProjectError):
        dumps({'value': value})


@pytest.mark.parametrize('changes', [dict(id='../bad'), dict(objective=''), dict(task_type='unknown'),
    dict(format_version=2), dict(format_version=True), dict(progress=101), dict(state='wrong'),
    dict(state='completed'), dict(error=ErrorDTO('bad', 'error')), dict(extra=True)])
def test_invalid_project_dto(changes):
    value = Project('a' * 32, 'Objetivo').to_dict() | changes
    with pytest.raises(ProjectError):
        Project.from_dict(value)


@pytest.mark.parametrize('name', ['a.txt', 'a.md', 'a.TXT', 'guía local.md'])
def test_text_formats_and_normalization(name):
    value = TextAdapter().parse(name, '\ufeffCafe\u0301\r\nTexto\rFinal'.encode(), Config())
    assert value.documents[0].content == 'Café\nTexto\nFinal'
    assert value.source.name == name
    assert value.source.sha256 == hashlib.sha256('\ufeffCafe\u0301\r\nTexto\rFinal'.encode()).hexdigest()


@pytest.mark.parametrize('name', ['a.pdf', 'a.docx', 'a.csv', 'a.py', 'a.json'])
def test_reject_unsupported_text_formats(name):
    with pytest.raises(ProjectError):
        TextAdapter().parse(name, b'text', Config())


@pytest.mark.parametrize('name', ['../a.txt', '/a.txt', 'a/other.txt', 'a\\other.txt', '.hidden.txt',
    '..', 'a..txt', 'a.txt ', 'a\x00.txt', '<script>.txt', 'x' * 121])
def test_unsafe_file_names(name):
    with pytest.raises(ProjectError):
        TextAdapter().parse(name, b'text', Config())


@pytest.mark.parametrize('content', [b'\xff', b'text\x00', b'\x01text', b'', b'   '])
def test_invalid_encoding_or_empty(content):
    with pytest.raises(ProjectError):
        TextAdapter().parse('a.txt', content, Config())


def test_file_limits_preview_and_no_execution(service):
    config = Config(max_file_bytes=20, max_total_bytes=30, max_files=2, preview_chars=8)
    doc = TextAdapter().parse('a.md', b'<script>alert(1)', config).documents[0]
    result = preview(doc, config)
    assert result['text'] == '&lt;script&gt;'
    assert result['truncated'] and result['encoding'] == 'html_escaped_text'
    with pytest.raises(ProjectError):
        TextAdapter().parse('a.txt', b'a' * 21, config)
    with pytest.raises(ProjectError):
        TextAdapter().parse('a.txt', b'a' * 11, replace(config, max_document_chars=10))
    p = service.create_project('Consultar', config=config)
    service.import_files(p.id, (('a.txt', b'a' * 20),))
    before = service.get_project(p.id)
    for files in ((('b.txt', b'b' * 11),), (('b.txt', b'b'), ('c.txt', b'c')),
                  (('A.txt', b'duplicate'),), (('b.csv', b'bad'),)):
        with pytest.raises(ProjectError):
            service.import_files(p.id, files)
        assert service.get_project(p.id) == before


def test_import_batch_is_transactional(service):
    p = service.create_project('Consultar')
    with pytest.raises(ProjectError):
        service.import_files(p.id, (('good.txt', b'good'), ('bad.txt', b'\xff')))
    assert service.get_project(p.id) == p


@pytest.mark.parametrize('extension,content', [
    ('json', '[{"id":"one","question":"Python"}]'),
    ('jsonl', '{"id":"one","question":"Python"}\n'),
])
def test_evaluation_adapters(extension, content):
    dataset = EvaluationAdapter().parse(f'cases.{extension}', content.encode(), Config())
    assert dataset.cases == (EvaluationCase('one', 'Python'),)
    assert dataset.source.kind == 'evaluation' and not dataset.documents


@pytest.mark.parametrize('extension,content', [
    ('json', '{}'), ('json', '[]'), ('json', '[{"question":"x"}]'),
    ('json', '[{"id":"one","question":"x","unknown":1}]'),
    ('json', '[{"id":"one","question":"x","expected_answer":2}]'),
    ('json', '[{"id":"one","question":"x","expected_sources":["../x"]}]'),
    ('json', '[{"id":"one","question":"x","criteria":"not a list"}]'),
    ('json', '[{"id":"one","question":"x"},{"id":"one","question":"y"}]'),
    ('jsonl', '\n{"id":"one","question":"x"}'),
    ('jsonl', '{"id":"one","question":NaN}'), ('jsonl', '{"id":"one","question":"x","question":"y"}'),
    ('jsonl', '{"id":"one","question":"x"}\n\n'),
])
def test_bad_evaluation_inputs(extension, content):
    with pytest.raises(ProjectError):
        EvaluationAdapter().parse(f'cases.{extension}', content.encode(), Config())


def test_aggregate_case_limits(service):
    p = ready(service, config=Config(max_cases=1, question_chars=6))
    with pytest.raises(ProjectError):
        service.create_evaluation(p.id, (EvaluationCase('a', 'Python'), EvaluationCase('b', 'Python')))
    with pytest.raises(ProjectError):
        service.create_evaluation(p.id, (EvaluationCase('a', 'long question'),))
    with pytest.raises(ProjectError):
        EvaluationAdapter().parse('a.jsonl', b'{"id":"a","question":"x"}\n{"id":"b","question":"x"}', Config(max_cases=1))


def test_segment_offsets_and_chunk_limit():
    config = Config(chunk_chars=6, overlap_chars=2)
    documents = TextAdapter().parse('a.txt', 'Cafe\u0301\r\n0123456789'.encode(), config).documents
    chunks = segment(documents, config)
    assert [c.start for c in chunks] == [0, 4, 8, 12]
    for chunk in chunks:
        assert chunk.text == documents[0].content[chunk.start:chunk.end]
        assert chunk.filename == 'a.txt' and chunk.document_id == documents[0].source.id
    with pytest.raises(ProjectError, match='fragmentos'):
        segment(documents, replace(config, max_chunks=1, top_k=1))


def test_retrieval_order_scores_and_no_evidence():
    config = Config()
    docs = tuple(TextAdapter().parse(name, content, config).documents[0] for name, content in
                 [('z.txt', b'Python'), ('a.txt', b'Python'), ('b.txt', b'Python memoria'), ('c.txt', b'Rust')])
    index = LexicalIndex()
    index.build(segment(docs, config))
    sources = index.search('Python', 3)
    assert [s.chunk.filename for s in sources] == ['a.txt', 'z.txt', 'b.txt']
    assert [s.rank for s in sources] == [1, 2, 3]
    assert sources[0].score == sources[1].score > sources[2].score
    assert sources[0].matched_tokens == ('python',)
    assert index.search('Python', 3) == sources
    assert not index.search('banana', 3)
    assert not index.search('de la que', 3)
    with pytest.raises(ProjectError):
        index.search('Python', 0)
    assert not task().run('banana').supported


def test_context_budget_and_visible_sources():
    rag = task(Config(context_chars=8))
    response = rag.run('Python')
    assert response.supported
    assert sum(len(s.chunk.text) for s in response.context_sources) == 8
    assert response.context_sources[0].chunk.text == 'Python a'
    assert response.context_sources[0].chunk.end == 8
    assert response.sources[0].chunk.text == 'Python administra memoria.'
    assert set(response.used_chunk_ids) <= {s.chunk.id for s in response.sources}
    assert not task(Config(context_chars=2)).run('Python').supported


def test_end_to_end_isolated_reproducible_and_offline(service, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Network is forbidden')
    monkeypatch.setattr(socket, 'socket', forbidden)
    first = ready(service)
    second = ready(service, (('other.txt', b'Rust lenguaje.'),))
    p = service.execute_rag(first.id, 'Python')
    assert p.state == State.COMPLETED
    assert p.result == service.execute_rag(first.id, 'Python').result
    assert p.result == ProjectService(service.store).execute_rag(first.id, 'Python').result
    assert not service.execute_rag(second.id, 'Python').result.supported
    assert all(s.chunk.filename == 'manual.md' for s in p.result.sources)
    originals = {d.source.id: d.content for d in service.store.load(first.id).documents}
    for source in p.result.sources:
        c = source.chunk
        assert originals[c.document_id][c.start:c.end] == c.text
    assert p.result.provider == 'extractive-demo-v1'
    assert 'sin LLM' in p.result.answer


def test_evaluation_metrics_are_literal_and_retrieval_only(service):
    p = ready(service)
    answer = service.execute_rag(p.id, 'Python').result.answer
    cases = (EvaluationCase('one', 'Python', answer, ('manual.md', 'missing.txt'), ('Claridad',)),
             EvaluationCase('two', 'banana', 'Inventado', ('manual.md',)),
             EvaluationCase('three', 'Python'))
    service.create_evaluation(p.id, cases)
    observed = []
    project = service.run_evaluation(p.id, progress=lambda x: observed.append(x))
    report = project.evaluation
    assert observed == [0, 33, 66, 100]
    assert project.state == State.EVALUATED and project.result.answer == answer
    assert report.exact_match_mean == .5 and report.source_recall_mean == .25
    assert report.results[0].source_recall == .5
    assert report.results[2].exact_match is None and report.results[2].source_recall is None
    assert report.results[0].case.criteria == ('Claridad',)
    assert 'no puntuados' in report.criteria_policy
    json.dumps(report.to_dict(), allow_nan=False)
    assert service.run_evaluation(p.id).evaluation == report
    service.create_evaluation(p.id, (EvaluationCase('new', 'Python'),))
    assert service.get_project(p.id).evaluation is None


def test_strategy_is_explainable_and_fine_tuning_unavailable(service):
    p = ready(service)
    assert p.strategy.available and p.strategy.capability == 'local_rag'
    assert 'documentos TXT/MD' in p.strategy.reasons[1]
    eval_project = ready(service, task_type='evaluation')
    assert not eval_project.strategy.available and eval_project.strategy.missing
    service.create_evaluation(eval_project.id, (EvaluationCase('one', 'Python'),))
    assert service.get_project(eval_project.id).strategy.available
    future = ready(service, task_type='fine_tuning')
    assert future.strategy.capability == 'fine_tuning' and not future.strategy.available
    with pytest.raises(ProjectError, match='no disponible'):
        service.execute_rag(future.id, 'Python')
    assert not recommend(Project('a' * 32, 'Necesito ajuste fino'), 0).available
    assert recommend(Project('a' * 32, 'Evaluar', task_type='evaluation'), 0).missing


def test_progress_cancellation_failure_and_retry(service):
    p = ready(service)
    observed = []
    def observe(percent):
        observed.append((percent, service.get_project(p.id).state))
    assert service.execute_rag(p.id, 'Python', progress=observe).state == State.COMPLETED
    assert observed == [(0, State.RUNNING), (100, State.RUNNING)]
    assert service.execute_rag(p.id, 'Python', progress=lambda _: False).state == State.CANCELLED
    service.recommend_strategy(p.id)
    class Broken:
        name = 'broken'
        def answer(self, question, context):
            raise RuntimeError('SECRET and document text')
    faulty = ProjectService(service.store, provider=Broken())
    result = faulty.execute_rag(p.id, 'Python')
    assert result.state == State.FAILED and result.error.code == 'execution_failed'
    assert 'SECRET' not in dumps(result.to_dict())
    service.recommend_strategy(p.id)
    assert service.execute_rag(p.id, 'Python').state == State.COMPLETED


def test_bad_ports_are_rejected(service):
    with pytest.raises(ProjectError):
        ProjectService(object())
    with pytest.raises(ProjectError):
        ProjectService(service.store, adapters=(TextAdapter(), TextAdapter()))
    class InvalidProvider:
        name = 'invalid'
        def answer(self, question, context):
            return {'fake': 'answer'}
    p = ready(service)
    bad = ProjectService(service.store, provider=InvalidProvider()).execute_rag(p.id, 'Python')
    assert bad.state == State.FAILED and bad.error.code == 'provider_contract'
    with pytest.raises(ProjectError):
        SyncTrainer().execute(object(), 'Python')
    with pytest.raises(ProjectError):
        LocalEvaluator().evaluate(task(), ())


def test_unknown_and_invalid_states(service):
    p = service.create_project('Consultar')
    with pytest.raises(ProjectError):
        service.preview_and_validate(p.id)
    with pytest.raises(ProjectError):
        service.recommend_strategy(p.id)
    with pytest.raises(ProjectError):
        service.execute_rag(p.id, 'Python')
    assert service.cancel(p.id).state == State.CANCELLED
    with pytest.raises(ProjectError) as error:
        service.get_project('f' * 32)
    assert error.value.code == 'not_found'


@pytest.mark.parametrize('identifier', ['../bad', '/etc', 'a/b', '', 'z' * 32, 'a' * 33])
def test_store_rejects_unsafe_ids(service, identifier):
    with pytest.raises(ProjectError):
        service.store.load(identifier)


def test_store_rejects_symlinks(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectError):
        LocalArtifactStore(link)
    store = LocalArtifactStore(tmp_path / 'store')
    (store.root / ('a' * 32)).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectError):
        store.save(Snapshot(Project('a' * 32, 'Objetivo')))
    p = Project('b' * 32, 'Objetivo')
    store.save(Snapshot(p))
    target = store.root / p.id / 'export.json'
    target.symlink_to(outside / 'secret')
    with pytest.raises(ProjectError):
        store.export(p.id)
    assert not (outside / 'secret').exists()


def test_atomic_replace_failure_preserves_previous(service, monkeypatch):
    p = ready(service)
    snapshot = service.store.load(p.id)
    path = service.store.root / p.id / 'project.json'
    previous = path.read_bytes()
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr('mlvectoranalysis.projects.storage.os.replace', fail)
    with pytest.raises(ProjectError):
        service.store.save(replace(snapshot, project=replace(p, objective='Nuevo objetivo')))
    assert path.read_bytes() == previous
    assert not list(path.parent.glob('.pending-*'))


@pytest.mark.parametrize('corruption', ['json', 'hash', 'version', 'schema', 'missing', 'identity'])
def test_corrupted_artifact_rejected(service, corruption):
    p = ready(service)
    path = service.store.root / p.id / 'project.json'
    envelope = loads(path.read_text())
    if corruption == 'json':
        path.write_text('{broken')
    else:
        if corruption == 'hash':
            envelope['snapshot']['project']['objective'] = 'tampered'
        elif corruption == 'version':
            envelope['format_version'] = 2
        else:
            project = envelope['snapshot']['project']
            if corruption == 'schema':
                project['config']['top_k'] = 0
            elif corruption == 'missing':
                del project['format_version']
            else:
                project['id'] = 'f' * 32
            envelope['sha256'] = hashlib.sha256(dumps(envelope['snapshot']).encode()).hexdigest()
        path.write_text(dumps(envelope))
    with pytest.raises(ProjectError):
        service.get_project(p.id)


def test_artifact_limits(service):
    p = ready(service)
    small = LocalArtifactStore(service.store.root, max_artifact_bytes=10)
    with pytest.raises(ProjectError):
        small.load(p.id)
    with pytest.raises(ProjectError):
        small.save(service.store.load(p.id))
    assert service.get_project(p.id) == p


def test_export_reproducible_and_replayable(service, tmp_path):
    p = ready(service)
    answer = service.execute_rag(p.id, 'Python').result
    service.create_evaluation(p.id, (EvaluationCase('one', 'Python', expected_sources=('manual.md',)),))
    evaluated = service.run_evaluation(p.id)
    exported = service.export_project(p.id)
    path = Path(exported['path'])
    first = path.read_bytes()
    assert Path(service.export_project(p.id)['path']).read_bytes() == first
    manifest = loads(first.decode())
    assert not manifest['trained_model'] and manifest['export_kind'] == 'configuration_and_normalized_data'
    assert manifest['versions']['index'] == 'tfidf-cosine-v1'
    snapshot = Snapshot.from_dict(manifest['snapshot'])
    other = ProjectService(LocalArtifactStore(tmp_path / 'replay'))
    other.store.save(snapshot)
    assert other.run_evaluation(p.id).evaluation == evaluated.evaluation
    assert other.execute_rag(p.id, 'Python').result == answer
    assert snapshot.documents and snapshot.project.files[0].sha256


def test_new_import_invalidates_strategy_results(service):
    p = ready(service)
    service.execute_rag(p.id, 'Python')
    updated = service.import_files(p.id, (('extra.txt', b'Nueva evidencia'),))
    assert updated.state == State.FILES and updated.strategy is None and updated.result is None


def test_example_cli(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(backend / 'src'), PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run([sys.executable, str(backend / 'examples/local_project.py'),
                                '--data-dir', str(tmp_path / 'demo')],
                               cwd=backend, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = loads(completed.stdout)
    assert result['evaluation']['source_recall_mean'] == 1
    assert Path(result['export']['path']).is_file()


def test_recover_interrupted_run(service):
    p = ready(service)
    snapshot = service.store.load(p.id)
    service.store.save(replace(snapshot, project=p.transition(State.RUNNING)))
    assert service.get_project(p.id).state == State.RUNNING
    recovered = service.recover_interrupted(p.id)
    assert recovered.state == State.FAILED and recovered.error.code == 'interrupted'
    service.recommend_strategy(p.id)
    assert service.execute_rag(p.id, 'Python').state == State.COMPLETED
    with pytest.raises(ProjectError):
        service.recover_interrupted(p.id)


def test_injected_provider_and_index_keep_frontend_dto(service):
    class OtherProvider:
        name = 'custom-extractive-v1'
        def answer(self, question, context):
            return replace(ExtractiveDemoProvider().answer(question, context), provider=self.name)
    class OtherIndex(LexicalIndex):
        name = 'custom-tfidf-v1'
    custom = ProjectService(service.store, provider=OtherProvider(), index_factory=OtherIndex)
    p = ready(custom)
    result = custom.execute_rag(p.id, 'Python')
    assert result.result.provider == result.provider_version == 'custom-extractive-v1'
    manifest = loads(Path(custom.export_project(p.id)['path']).read_text())
    assert manifest['versions']['index'] == 'custom-tfidf-v1'
    assert manifest['versions']['provider'] == 'custom-extractive-v1'
    assert isinstance(result.to_dict()['result']['sources'], list)


def test_no_provider_call_without_evidence(service):
    class MustNotRun:
        name = 'unused'
        def answer(self, question, context):
            raise AssertionError('No evidence')
    custom = ProjectService(service.store, provider=MustNotRun())
    p = ready(custom)
    response = custom.execute_rag(p.id, 'banana')
    assert response.state == State.COMPLETED and not response.result.supported
    assert response.result.used_chunk_ids == ()


def test_evaluation_cancellation_does_not_store_partial_report(service):
    p = ready(service)
    service.create_evaluation(p.id, (EvaluationCase('one', 'Python'), EvaluationCase('two', 'Python')))
    result = service.run_evaluation(p.id, progress=lambda n: n < 50)
    assert result.state == State.CANCELLED and result.progress == 50
    assert result.evaluation is None


def test_preview_validates_total_chunk_budget(service):
    p = service.create_project('Consultar', config=Config(chunk_chars=4, overlap_chars=0, top_k=1, max_chunks=1))
    service.import_files(p.id, (('a.txt', b'12345678'),))
    with pytest.raises(ProjectError):
        service.preview_and_validate(p.id)
    assert service.get_project(p.id).state == State.FILES


def test_case_schema_does_not_accept_wrong_metrics(service):
    from mlvectoranalysis.projects import EvaluationReport
    report = LocalEvaluator().evaluate(task(), (EvaluationCase('one', 'Python', expected_sources=('manual.txt',)),))
    data = report.to_dict()
    data['source_recall_mean'] = .5
    with pytest.raises(ProjectError):
        EvaluationReport.from_dict(data)
    data = report.to_dict()
    data['results'][0]['source_recall'] = .5
    with pytest.raises(ProjectError):
        EvaluationReport.from_dict(data)
