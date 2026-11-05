"""Literal answer matching and retrieval recall; no semantic quality metric."""
from .contracts import Progress, Task
from .dto import Answer, CaseResult, EvaluationCase, EvaluationReport, require
from .rag import checkpoint


def mean(values):
    observed = [float(v) for v in values if v is not None]
    return sum(observed) / len(observed) if observed else None


class LocalEvaluator:
    def evaluate(self, task: Task, cases: tuple[EvaluationCase, ...],
                 progress: Progress | None = None) -> EvaluationReport:
        require(isinstance(task, Task) and isinstance(cases, tuple) and bool(cases)
                and all(isinstance(c, EvaluationCase) for c in cases), 'Evaluación inválida.')
        require(len({c.id for c in cases}) == len(cases), 'Casos duplicados.')
        results = []
        checkpoint(progress, 0)
        for index, case in enumerate(cases):
            response = task.run(case.question)
            require(isinstance(response, Answer), 'Respuesta inválida.')
            retrieved_names = {s.chunk.filename for s in response.sources}
            recall = (len(set(case.expected_sources) & retrieved_names) / len(case.expected_sources)
                      if case.expected_sources else None)
            exact = response.answer == case.expected_answer if case.expected_answer is not None else None
            results.append(CaseResult(case, response, exact, recall))
            checkpoint(progress, (index + 1) * 100 // len(cases))
        return EvaluationReport(tuple(results), mean(r.exact_match for r in results),
                                mean(r.source_recall for r in results))
