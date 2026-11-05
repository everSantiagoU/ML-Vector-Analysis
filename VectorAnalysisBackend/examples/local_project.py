"""Offline project -> documents -> RAG -> user evaluation -> JSON export."""
import argparse
from pathlib import Path

from mlvectoranalysis.projects import (Config, EvaluationCase, LocalArtifactStore,
                                       ProjectService, State)
from mlvectoranalysis.projects.dto import dumps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('outputs/local-projects'))
    parser.add_argument('--document', type=Path, action='append', help='UTF-8 .txt/.md; repeatable')
    parser.add_argument('--cases', type=Path, help='Evaluation .json/.jsonl (optional)')
    parser.add_argument('--question', default='¿Cómo administra Python la memoria?')
    args = parser.parse_args()
    config = Config()
    service = ProjectService(LocalArtifactStore(args.data_dir))
    project = service.create_project('Consultar documentación local', config=config)
    if args.document:
        paths = args.document + ([args.cases] if args.cases else [])
        if len(paths) > config.max_files:
            parser.error('Demasiados archivos.')
        for path in paths:
            with path.open('rb') as handle:
                content = handle.read(config.max_file_bytes + 1)
            service.import_files(project.id, ((path.name, content),))
    else:
        service.import_files(project.id, (('manual.md',
            '# Python\nPython administra memoria mediante objetos y recolección de basura.\n'
            'Este manual se consulta localmente.\n'.encode('utf-8')),))
        if args.cases:
            with args.cases.open('rb') as handle:
                service.import_files(project.id, ((args.cases.name, handle.read(config.max_file_bytes + 1)),))
    preview = service.preview_and_validate(project.id)
    service.recommend_strategy(project.id)
    response = service.execute_rag(project.id, args.question)
    if response.state != State.COMPLETED:
        raise RuntimeError(dumps(response.to_dict()))
    if not args.cases:
        source = args.document[0].name if args.document else 'manual.md'
        service.create_evaluation(project.id, (EvaluationCase('consulta-1', args.question,
            expected_sources=(source,), criteria=('Revisar manualmente si los extractos resuelven la consulta.',)),))
    evaluated = service.run_evaluation(project.id)
    if evaluated.state != State.EVALUATED:
        raise RuntimeError(dumps(evaluated.to_dict()))
    print(dumps({'project_id': project.id, 'preview': preview['documents'],
                 'answer': response.result.to_dict(), 'evaluation': evaluated.evaluation.to_dict(),
                 'export': service.export_project(project.id)}))


if __name__ == '__main__':
    main()
