# Proyectos locales, RAG y evaluación — formato v1

Este módulo añade una biblioteca de casos de uso Python, sin servidor HTTP ni
frontend. `mlvectoranalysis.projects.ProjectService` es la frontera para una
API local futura. Funciona sin red y sin dependencias nuevas: biblioteca estándar
de Python 3.10+. No importa NumPy, SymPy, Matplotlib ni módulos educativos.

## Qué se encontró y qué se reutiliza

La inspección inicial encontró Git limpio, ningún `AGENTS.md`, ningún servidor
web y una carpeta frontend vacía. La suite existente tenía 605 pruebas y pasó
antes de editar. Las APIs existentes permanecen sin cambios.

| Componente existente | Implementación comprobada | Relación con esta fase |
| --- | --- | --- |
| `domain`, `input`, `parsing`, `analysis` | Validación restringida, funciones SymPy y DTO `to_dict()` | Se conserva el estilo de fachada y DTO; no se interpreta texto documental como expresiones |
| `optimization`, `ml` | Descenso de gradiente y regresión lineal | Permanecen independientes |
| `nn` | MLP, backpropagation, SGD/Adam, evaluación, checkpoints NPZ con metadatos JSON | No son un proveedor generativo; no se cargan pesos para RAG |
| `language` | Tokenizer sencillo, vocabulario, embeddings entrenables y modelo de contexto fijo | Implementación educativa real; no es un LLM ni un generador documental adecuado |
| `visualization`, `nn.visualization`, ejemplos | Datos de trayectoria y gráficos opcionales | Siguen disponibles; no se incorpora UI ni Matplotlib al nuevo módulo |

Se reutilizan convenciones de validación, dataclasses, `to_dict()`, composición,
JSON finito y reemplazo atómico. Los helpers privados de entrenamiento no se
importan: acoplarlos añadiría dependencias numéricas sin beneficio. El tokenizer
RAG usa palabras Unicode y una lista fija pequeña de palabras vacías; el tokenizer
educativo conserva puntuación y vocabulario entrenable. Son contratos diferentes.
La introducción de `language_model.md` describe un diseño histórico; sus módulos
sí existen. No se deduce de esa documentación soporte LLM, formatos arbitrarios
ni checkpoints del modelo compuesto.

## Arquitectura y puertos

| Módulo | Responsabilidad |
| --- | --- |
| `dto.py` | Tipos inmutables, validación, JSON estricto, configuración, estados y errores |
| `contracts.py` | Protocolos estructurales tipados y callback de progreso |
| `data.py` | Adaptadores TXT/MD y JSON/JSONL, normalización y previsualización |
| `rag.py` | Segmentación, índice TF-IDF, contexto, proveedor de extractos y ejecución síncrona |
| `evaluation.py` | Evaluación independiente del ejecutor/entrenamiento |
| `strategy.py` | Reglas de recomendación deterministas y explicables |
| `storage.py` | Persistencia y exportación JSON local atómica |
| `service.py` | Casos de uso; orquesta puertos y conserva estados |

Los contratos son deliberadamente pequeños:

- `DatasetAdapter.parse(name, content: bytes, config) -> Dataset`: extensiones
  explícitas y resultado común con origen, documentos o casos.
- `RetrievalIndex.build(chunks)` y `search(question, top_k) -> tuple[Source, ...]`:
  índice reemplazable; la fábrica crea uno nuevo por ejecución/proyecto.
- `ModelProvider.answer(question, context) -> Answer`: recibe exclusivamente los
  fragmentos que caben en contexto; debe conservarlos en `Answer.sources`, citar
  solo sus IDs y respetar el presupuesto de salida. `RagTask` valida el resultado.
- `Task.run(question) -> Answer`: ejecución de una consulta, independiente del
  proveedor y del evaluador.
- `Trainer.execute(task, question, progress) -> Answer`: ejecutor de trabajos del
  MVP. `SyncTrainer` **no entrena pesos** y es distinto de `nn.Trainer`.
- `Evaluator.evaluate(task, cases, progress) -> EvaluationReport`: evaluación
  separada; no llama optimizadores ni altera pesos.
- `ArtifactStore.save(snapshot)`, `load(id)`, `list_ids()`, `export(id)`: almacenamiento
  de snapshots tipados, no de objetos Python arbitrarios.

`ProjectService` permite inyectar adaptadores, proveedor, fábrica de índice,
ejecutor, evaluador y almacén. Los plugins son código local de confianza: el
backend no ejecuta el contenido importado ni instala plugins desde documentos.
Una implementación personalizada debe mantener los contratos y probar su propio
comportamiento. Exportar sus nombres/versiones no exporta su código ni garantiza
reproducibilidad de un proveedor externo.

## Flujo y estados

```text
created -> files_imported -> validated -> ready -> running -> completed
                                                   |           |
                                                   |           +-> running -> evaluated
                                                   +-> failed / cancelled
```

`ready` significa que existe una recomendación; hay que consultar
`strategy.available`, que puede ser `false` si falta información o se solicita
ajuste fino. La ejecución lo exige. `Project.transition()` valida la tabla
`TRANSITIONS`; la suite comprueba las 81 combinaciones de estados.

Se pueden añadir archivos antes o después de ejecutar: se regresa a
`files_imported` y se invalidan estrategia, resultado e informe. La importación
de un lote es todo-o-nada y rechaza nombres duplicados, incluso con diferencias
de mayúsculas. No hay borrado/reemplazo de archivos en esta versión.

Desde `completed`/`evaluated` se puede ejecutar otra consulta o evaluación.
Se conserva el último resultado RAG al evaluar; una nueva consulta invalida el
informe anterior. `create_evaluation` reemplaza el conjunto e invalida su informe.
Un error de entrada lanza `ProjectError(code, message)` y no cambia el proyecto.
Un fallo durante ejecución devuelve proyecto `failed` con `error` estructurado;
no persiste el texto de excepciones del proveedor. No hay logs con documentos.

El progreso se persiste antes del callback: 0/100 para consulta; 0 y porcentaje
por caso para evaluación. Devolver `False` cancela en ese límite y descarta el
resultado parcial. La construcción del índice y una llamada al proveedor no
pueden interrumpirse a mitad. `cancel()` opera fuera de `running`; no hay hilos,
cola ni API de cancelación concurrente. Tras `failed`/`cancelled`, recomendar de
nuevo habilita un reintento si la capacidad está disponible.

Si un proceso termina abruptamente, el snapshot puede quedar `running`.
`recover_interrupted(id)` lo marca `failed/interrupted`; usarlo únicamente después
de comprobar que el proceso anterior terminó. La lectura no recupera ejecuciones
automáticamente. Hay un solo escritor: no se soportan procesos concurrentes ni
mutaciones desde callbacks.

## Frontera prevista para TypeScript/React

Todos los DTOs ofrecen `to_dict()` y `from_dict()`, con enums como strings,
tuplas como listas y opcionales como `null`. Los campos desconocidos se rechazan;
los constructores validan tipos y rangos. Los metadatos guardados deben incluir
el esquema completo y `format_version=1`. JSON rechaza claves duplicadas,
NaN, Infinity y desbordamientos como `1e999`.

| Operación Python | Entrada | Salida serializable |
| --- | --- | --- |
| `create_project` | `objective`, `task_type='rag'`, `Config` opcional | `Project` |
| `list_projects` / `get_project` | ninguno / ID | lista de `Project` / `Project` |
| `import_files` | ID y tupla de `(nombre, bytes)` | `Project` |
| `preview_and_validate` | ID | proyecto, previews, conteos, `valid` |
| `recommend_strategy` | ID | `Project` con estrategia |
| `execute_rag` | ID, pregunta, callback opcional | `Project` con resultado o error |
| `create_evaluation` | ID y tupla de `EvaluationCase` | `Project` |
| `run_evaluation` | ID, callback opcional | `Project` con informe o error |
| `get_project` | ID | estado, progreso, resultado y error actuales |
| `cancel` / `recover_interrupted` | ID | `Project` |
| `save_project` | ID | `Project`; las mutaciones ya se guardan automáticamente |
| `export_project` | ID | ID, ruta, tipo de exportación y `trained_model=false` |

Los bytes de importación son la excepción a la frontera JSON: una futura API
local puede recibirlos mediante upload multipart o un canal binario limitado,
y pasar nombre/bytes al caso de uso. No debe aceptar rutas arbitrarias del cliente.
La biblioteca no tiene rutas HTTP, puerto abierto ni dependencia de React.
Una futura UI TypeScript necesita un puente/API Python; no puede importar estas
clases Python directamente.

Campos principales de `Project`: `id`, `objective`, `task_type`, `config`,
`state`, `files` (solo metadatos), `strategy`, `result`, `evaluation`, `error`,
`progress`, `format_version`, `index_version`, `provider_version`.
`Snapshot` agrega documentos normalizados y casos para persistencia; no se
necesita enviarlos al frontend al consultar estado.

`Answer` contiene `question`, `answer`, `provider`, `supported`, `sources`,
`context_sources` y `used_chunk_ids`. Cada `Source` incluye `rank`, `score`,
`matched_tokens` y `chunk={id, document_id, filename, start, end, text}`.
`sources` es el top-k recuperado completo; `context_sources` son sus extractos
acotados entregados al proveedor; `used_chunk_ids` indica los citados. Un fragmento
recuperado puede no caber en contexto. `supported` significa que se usaron
extractos con coincidencias léxicas, **no que la respuesta esté verificada
semánticamente**. Sin coincidencias utilizables se devuelve explícitamente
«No se encontró soporte suficiente en los documentos», sin invocar al proveedor.

Las previews están truncadas y escapadas como texto HTML (`encoding` explícito).
No se interpreta Markdown ni HTML. Las respuestas y fuentes conservan texto
original normalizado: mostrarlas como nodos de texto, nunca con
`dangerouslySetInnerHTML`. El frontend puede capturar `ProjectError` y serializar
`error.to_dict()`; no debe presentar todos los estados `ready` como ejecutables.

## Formatos admitidos y límites

Solo se admiten documentos **`.txt` y `.md` UTF-8**, BOM opcional. Markdown se
trata como texto. Se rechazan bytes UTF-8 inválidos, controles binarios y
archivos vacíos. No se soportan CSV, PDF, DOCX, imágenes, URL remotas ni archivos
comprimidos. Los offsets `[start, end)` cuentan caracteres Unicode del texto
normalizado NFC con finales LF; no son offsets en bytes, posiciones en el archivo
original ni páginas. Los fragmentos tienen tamaño fijo y solapamiento: pueden
cortar palabras o párrafos.

**`.json` y `.jsonl` son exclusivamente conjuntos de evaluación**, no documentos
RAG ni exportaciones del proyecto. JSON requiere lista de objetos. JSONL requiere
un objeto por línea; acepta un salto final y rechaza líneas vacías intermedias.
Ejemplo válido de JSON:

```json
[
  {
    "id": "memoria-1",
    "question": "¿Cómo administra Python la memoria?",
    "expected_answer": null,
    "expected_sources": ["manual.md"],
    "criteria": ["Revisar si el extracto permite contestar la consulta"]
  }
]
```

`id` y `question` son obligatorios; los demás son opcionales. IDs deben ser únicos.
Las fuentes esperadas son nombres exactos de archivo; pueden no existir para
medir un fallo de recuperación. No se interpreta `criteria` como instrucciones.
Los criterios se guardan para revisión humana. `exact_match` compara la respuesta
completa literalmente, incluidos etiqueta del demo, espacios y puntuación.
`source_recall` es `|fuentes esperadas ∩ nombres recuperados| / |fuentes esperadas|`.
Se mide sobre `sources`, no solo sobre contexto usado. Se registran ambos.
Las medias incluyen solo casos con la expectativa correspondiente y son `null`
si no hay ninguno. No se mide similitud semántica ni cumplimiento automático de
criterios. No hay fase de entrenamiento asociada a evaluación.

Valores predeterminados de `Config`:

| Límite | Valor |
| --- | ---: |
| Archivos por proyecto (documentos + evaluaciones) | 32 |
| Bytes por archivo / total de archivos importados | 1.000.000 / 4.000.000 |
| Caracteres normalizados por documento | 1.000.000 |
| Tamaño / solapamiento de fragmento | 800 / 100 caracteres |
| Fragmentos por ejecución | 10.000 |
| Contexto / pregunta | 4.000 / 2.000 caracteres |
| top-k / casos | 5 / 100 |
| Preview por documento | 300 caracteres antes de escape |
| Artefacto JSON (`LocalArtifactStore`) | 64.000.000 bytes |

Los límites son configurables y se validan. Los casos tienen además topes de
esquema: 4.096 caracteres por pregunta/criterio, 100.000 por respuesta esperada,
32 fuentes/criterios. Las preguntas RAG respetan `question_chars`; la respuesta
del proveedor debe caber en `context_chars + 1024`. Se eligen límites en caracteres
porque no hay tokenizer de un LLM. No se promete un presupuesto de tokens compatible
con modelos futuros. El backend valida límites antes de indexar; quien lea un
archivo de disco debe limitar también la lectura, como hace el ejemplo CLI.

## Recuperación, seguridad y coste

El índice `tfidf-cosine-v1` usa palabras Unicode con `casefold`, una lista fija
pequeña de palabras vacías ES/EN, TF por conteo e
`idf=1+log((N+1)/(df+1))`. Ordena similitud coseno descendente y desempata por
nombre, offset e ID. Solo admite puntajes positivos. `matched_tokens` explica la
coincidencia; no hay embeddings semánticos, stemming ni reranker. No hay red,
descargas, ejecución de archivos ni evaluación de expresiones del documento.

Para D caracteres, C fragmentos, V términos y E entradas término-fragmento:
normalización/segmentación son O(D + caracteres solapados); construir TF-IDF
es O(D + E) aproximadamente; una consulta examina C vectores y Q términos de
consulta, O(CQ + C log C) en esta implementación sencilla. La evaluación construye
el índice una sola vez y repite consultas; una nueva ejecución lo reconstruye.

Residen documentos normalizados, fragmentos (incluido solapamiento) y diccionarios
TF-IDF O(D + E), con sobrecoste de objetos Python. No hay matriz densa documento ×
vocabulario. Los documentos no se copian por caso. Los DTOs comparten strings
inmutables y el guardado codifica JSON por partes, con temporales por string, sin
materializar otra copia codificada de todo el corpus. La carga JSON sí es completa
y acotada por el tamaño del artefacto; evaluación conserva todos los resultados,
acotados por casos, top-k, fragmentos y contexto. No es procesamiento streaming
para corpus grandes. El snapshot completo se reescribe en cada progreso, por lo
que la E/S de evaluación crece con casos × tamaño del snapshot. Para escalar,
separar documentos inmutables de progreso/resultados sería el siguiente cambio.

## Persistencia, exportación y reproducción

El almacén crea `<data-dir>/<project-id>/project.json` y `export.json`.
IDs de proyecto son UUID hex de 32 caracteres; nombres importados son datos, nunca
rutas de escritura. Se rechazan traversal, separadores, nombres ocultos, controles,
nombres demasiado largos y enlaces simbólicos en directorios de proyecto/destinos.
El directorio configurado se canonicaliza una vez (incluidos alias del SO como
`/var` en macOS). No se usa pickle. Los archivos son JSON con checksum SHA-256,
versión y validación de esquema al cargar. El checksum detecta corrupción, **no
es una firma ni protege frente a alguien que pueda reescribirlo**.

Las escrituras usan temporal en el mismo directorio, `flush`, `fsync` y
`os.replace`; ante fallo previo al reemplazo se conserva el artefacto anterior.
La atomicidad es por archivo, no una transacción entre `project.json` y un export
previo. No hay garantías contra manipulación concurrente maliciosa del sistema
de archivos, múltiples escritores ni pérdida de alimentación del directorio.
El directorio de datos debe estar bajo control del usuario local.

La exportación es un **manifiesto JSON autocontenido de configuración y datos**:
estrategia, límites, versiones de formato/backend/Python/Unicode/índice/proveedor,
metadatos y hashes de archivos originales, texto normalizado, casos, última
respuesta e informe. Contiene los datos del usuario en claro y no guarda bytes
originales ni un historial ilimitado. Los IDs de documento dependen de nombre y
bytes originales; los offsets dependen del texto normalizado. El índice se
reconstruye. `trained_model=false`: no hay pesos ni un modelo fine-tuned.

Para inspeccionar o reproducir un export generado por esta versión:

```python
from pathlib import Path
import hashlib
from mlvectoranalysis.projects import LocalArtifactStore, ProjectService, Snapshot
from mlvectoranalysis.projects.dto import loads, dumps

# Leer únicamente un export dentro del presupuesto configurado.
with Path("outputs/local-projects/ID/export.json").open("rb") as handle:
    raw = handle.read(64_000_001)
if len(raw) > 64_000_000:
    raise ValueError("Export demasiado grande")
manifest = loads(raw.decode("utf-8"))
assert manifest["format_version"] == 1
assert manifest["snapshot_sha256"] == hashlib.sha256(
    dumps(manifest["snapshot"]).encode("utf-8")).hexdigest()
snapshot = Snapshot.from_dict(manifest["snapshot"])
store = LocalArtifactStore("outputs/replay")
store.save(snapshot)
service = ProjectService(store)
# Para un snapshot que ya pasó por ready/completed/evaluated:
result = service.execute_rag(snapshot.project.id, "Python memoria")
print(result.to_dict())
```

El manifiesto es reproducible para un snapshot y entorno iguales; los IDs de
proyecto nuevos son aleatorios. La suite verifica igualdad de respuestas e
informes tras recargar/exportar/restaurar, además de aislamiento entre proyectos.
No compara IDs de proyectos recién creados como si fueran deterministas.

## Uso manual y pruebas

Desde `VectorAnalysisBackend`, con las dependencias existentes:

```sh
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python examples/local_project.py --help
PYTHONPATH=src .venv/bin/python examples/local_project.py --data-dir outputs/local-projects
# Archivos propios; --document se puede repetir:
PYTHONPATH=src .venv/bin/python examples/local_project.py \
  --data-dir outputs/local-projects \
  --document /ruta/manual.md --document /ruta/notas.txt \
  --cases /ruta/casos.jsonl --question '¿Cómo se administra la memoria?'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_projects.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
```

El ejemplo sin argumentos externos crea un manual pequeño y un caso con fuente
esperada. Imprime preview, respuesta, evaluación y ruta de exportación; esa salida
es el resultado solicitado, no un log interno. Crea un proyecto nuevo en cada
invocación. Los ejemplos anteriores conservan sus comandos en README. Matplotlib
es opcional en `requirements-examples.txt`; no se añade ninguna dependencia.

Uso directo mínimo:

```python
from mlvectoranalysis.projects import (
    ProjectService, LocalArtifactStore, EvaluationCase,
)
service = ProjectService(LocalArtifactStore("outputs/projects"))
p = service.create_project("Consultar documentación")
service.import_files(p.id, (("manual.txt", b"Python administra memoria."),))
service.preview_and_validate(p.id)
service.recommend_strategy(p.id)
answer = service.execute_rag(p.id, "Python memoria")
service.create_evaluation(p.id, (
    EvaluationCase("caso-1", "Python memoria", expected_sources=("manual.txt",)),
))
report = service.run_evaluation(p.id)
print(answer.to_dict(), report.to_dict(), service.export_project(p.id))
```

## Límites y siguiente incremento

El proveedor es un **demo extractivo**, no generación de un LLM ni comprensión
semántica. RAG no modifica pesos. Ajuste fino se modela como no disponible, con
requisitos explícitos de datos, proveedor entrenable, licencia y RAM/VRAM según
modelo; el MLP educativo no se anuncia como sustituto de un modelo base.

El siguiente incremento recomendado es una API local fina sobre estos casos de
uso, con carga binaria limitada y esquema JSON/TypeScript generado, y una cola de
un solo escritor para ejecutar en background. Después, un proveedor generativo
local con tokenizer y presupuesto de contexto propios, probado con las mismas
fuentes y evaluación ampliada. Nuevos formatos deben incorporarse por adaptadores
separados y pruebas reales. No hay soporte universal de formatos o modelos.
