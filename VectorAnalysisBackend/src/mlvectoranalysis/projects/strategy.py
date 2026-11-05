"""Explainable rules over available capabilities, without model training claims."""
from .dto import Project, Strategy, require


def recommend(project: Project, case_count: int) -> Strategy:
    require(isinstance(project, Project) and type(case_count) is int and case_count >= 0, 'Proyecto inválido.')
    documents = [f for f in project.files if f.kind == 'document']
    total = sum(f.size_bytes for f in documents)
    objective = project.objective.casefold()
    requested_training = project.task_type == 'fine_tuning' or any(
        term in objective for term in ('fine-tun', 'fine tun', 'ajuste fino', 'afinar pesos'))
    reasons = (f'Objetivo declarado: {project.objective}',
               f'{len(documents)} documentos TXT/MD, {total} bytes; {case_count} casos de evaluación.',
               f'Tipo de tarea: {project.task_type}.')
    if requested_training:
        return Strategy('fine_tuning', False, reasons,
                        ('No hay ajuste fino de modelos de lenguaje en este flujo.',
                         'Requiere proveedor entrenable, datos supervisados, licencia y presupuesto de RAM/VRAM según modelo.'),
                        ('Implementación y validación de entrenamiento de un modelo local.',))
    evaluation = project.task_type == 'evaluation' or any(term in objective for term in ('evaluar', 'evaluación'))
    missing = (() if documents else ('Importar documentos TXT/MD.',))
    if evaluation and not case_count:
        missing += ('Crear o importar casos de evaluación.',)
    return Strategy('evaluation' if evaluation else 'local_rag', not missing, reasons,
                    ('Recuperación léxica; coincidencias de palabras no garantizan relevancia semántica.',
                     'Proveedor de extractos, sin LLM; RAG no modifica pesos.',
                     'Ejecución síncrona local y corpus en memoria con límites configurables.'), missing)
