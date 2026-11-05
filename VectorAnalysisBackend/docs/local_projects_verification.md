# Verificación ejecutada: proyectos locales

Fecha: 2026-11-04. Entorno existente `VectorAnalysisBackend/.venv`, Python 3.13.
No se instalaron dependencias ni se modificaron pruebas anteriores.

## Suite

Antes de editar, desde `VectorAnalysisBackend`:

```sh
.venv/bin/python -m pytest -q
```

Resultado: **605 passed in 21.38s**.

Después de la implementación y revisión final:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
```

Resultado: **795 passed in 20.36s**, sin fallos ni skips: 605 pruebas anteriores
más 190 nuevas. Incluye pruebas existentes de ejemplos con gráficos opcionales.

Las pruebas nuevas cubren contratos inyectables, las 81 combinaciones de estados,
DTOs y JSON estricto, UTF-8 y formatos, validación de casos, límites agregados,
previews, offsets normalizados, orden TF-IDF, contexto acotado, falta de evidencia,
RAG sin red, procedencia de fuentes, evaluación literal/recall, estrategias,
progreso, cancelación, reintento, recuperación tras interrupción, aislamiento,
recarga, corrupción, rutas y symlinks, fallos de reemplazo atómico, exportación y
reproducción. Se ejecuta también el nuevo CLI como subprocess.

## Ejemplos y compatibilidad

Se ejecutaron con `PYTHONPATH=src` y `.venv/bin/python` estos comandos, todos con
código de salida 0; los JSON producidos se validaron sin NaN/Infinity:

```sh
PYTHONPATH=src .venv/bin/python examples/optimize_xyz.py --output outputs/local-backend-verification/xyz.json
PYTHONPATH=src .venv/bin/python examples/train_linear_regression.py --block-size 128 --output outputs/local-backend-verification/regression.json
PYTHONPATH=src .venv/bin/python examples/train_mlp_xor.py --output outputs/local-backend-verification/xor.json
PYTHONPATH=src .venv/bin/python examples/train_mlp_sine.py --output outputs/local-backend-verification/sine.json
PYTHONPATH=src .venv/bin/python examples/train_language_model.py --output outputs/local-backend-verification/language.json
PYTHONPATH=src .venv/bin/python examples/train_mlp_resume.py --optimizer adam --epochs 6 --checkpoint outputs/local-backend-verification/resume.npz --output outputs/local-backend-verification/resume-6.json
PYTHONPATH=src .venv/bin/python examples/train_mlp_resume.py --optimizer adam --resume --additional-epochs 4 --checkpoint outputs/local-backend-verification/resume.npz --output outputs/local-backend-verification/resume-10.json
PYTHONPATH=src .venv/bin/python examples/local_project.py --data-dir outputs/local-backend-verification/projects
```

También se verificó `--help` de los siete scripts. El nuevo ejemplo consiguió
`source_recall_mean=1.0` para su fuente esperada, creó un manifiesto exportado y
mantuvo el proveedor etiquetado como extractivo sin LLM.

Se crearon además fixtures locales `manual.md`, `notas.txt` y `casos.jsonl` en
`outputs/local-backend-verification` y se ejecutó:

```sh
PYTHONPATH=src .venv/bin/python examples/local_project.py \
  --data-dir outputs/local-backend-verification/uploads \
  --document outputs/local-backend-verification/manual.md \
  --document outputs/local-backend-verification/notas.txt \
  --cases outputs/local-backend-verification/casos.jsonl \
  --question 'Python memoria'
```

Resultado: salida 0, dos previews y recall de fuente 1.0. Esto prueba la importación
combinada desde archivos reales; no constituye medición semántica.

Un proceso Python separado verificó que importar `mlvectoranalysis.projects` no
carga `numpy`, `sympy` ni `matplotlib`. `git diff --check` no detectó problemas.
Los artefactos manuales quedaron en `outputs/`, ya ignorado por Git. No hubo
commit ni push.
