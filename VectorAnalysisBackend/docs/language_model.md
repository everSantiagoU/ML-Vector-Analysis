# Modelado local de lenguaje: diseño previo

Inspección del código: `nn.MLP` compone capas densas y activaciones con contextos
por ejecución; `Parameter` mantiene buffers float64 y versiones; `Trainer`
convierte X a float64 y usa `MiniBatchIterator`. No existen tokenizer,
embeddings ni atención en ese código. Se reutilizarán MLP, CrossEntropyLoss,
softmax, Trainer, Adam y los helpers opcionales de los ejemplos.

Diseño: `language.text` separa Vocabulary y SimpleTokenizer;
`language.data` conserva un vector de IDs y offsets de ventanas válidas;
`language.embedding` hace lookup y backward con `np.add.at`;
`language.model` concatena embeddings y delega al MLP. Un adaptador explícito
presenta índices de ejemplos `(N,1)` float64 al Trainer existente y recupera
contextos enteros por lote. Evita convertir/copiar todas las ventanas a float64
y evita cambiar contratos existentes. No ofrece checkpoints del modelo compuesto.
El ejemplo CLI reutilizará los helpers JSON/Matplotlib del proyecto.

Política prevista: Unicode NFC + lower, palabras Unicode y puntuación individual,
espacios ignorados; decode canónico con un espacio entre todos los tokens.
PAD=0, UNK=1; vocabulario ordenado lexicográficamente. Sin padding automático,
sin cruzar documentos: documentos con <= C tokens no aportan ejemplos; corpus
sin ejemplos falla. El modelo es de contexto fijo, no un LLM ni Transformer.

## Implementación y contratos

```text
texto → NFC/lower/tokenize → Vocabulary → int64 por documento
      → ContextDataset (IDs + offsets) → lote (B,C)
      → Embedding (B,C,D) → reshape (B,C*D)
      → MLP Tanh (B,H) → logits (B,V) → softmax / argmax
```

`SimpleTokenizer.from_corpus(iterable_de_textos)` construye vocabulario sobre
train solamente. `encode(text)` devuelve int64 `(T,)`; `decode(ids)` devuelve
texto canónico, no reconstruye espacios, mayúsculas ni formato original.
`Vocabulary.encode(tokens)` y `decode(ids)` operan sobre listas de tokens.
El orden es lexicográfico Unicode después de normalizar. `\w+` agrupa letras,
números y guion bajo; cada carácter no alfanumérico y no blanco es un token
separado (apóstrofos y guiones también). No se eliminan acentos. Textos vacíos,
corpus vacío, tokens vacíos/con espacios e IDs vacíos o inválidos fallan con
`ValueError`. PAD y UNK están reservados y aparecen en decode literalmente;
re-tokenizar esos marcadores no es una operación inversa. No es BPE ni es
compatible con vocabularios externos.

`ContextDataset(documents, context_length, vocab_size)` recibe un iterable de
vectores enteros no vacíos. Para cada documento de longitud n genera
`max(n-C,0)` ejemplos: `t[i:i+C] → t[i+C]`. Documentos cortos se omiten;
un documento vacío es error. No agrega BOS/EOS, no crea padding ni atraviesa
límites. Si no hay ejemplos falla antes de entrenar. PAD se reserva para una
futura política explícita, pero si se proporciona manualmente se trata como
cualquier ID entrenable (sin máscara ni fila congelada).

`batch(indices)` devuelve contextos int64 `(B,C)` y targets int64 `(B,)`.
`training_arrays()` devuelve índices de ejemplos float64 `(N,1)` y targets
enteros `(N,)`; se utiliza **solo con WindowModelAdapter(model, dataset)**.
El adaptador comprueba índices enteros en rango, materializa el lote y delega
forward/backward. Para validación externa con Trainer hay que construir otro
adaptador para ese dataset y usar `evaluate`; no pasar sus índices a un
adaptador ligado a train. Los índices float64 limitan esta interfaz a corpus
residentes pequeños (no está diseñada para índices >= 2**53).

`ContextLanguageModel(V,C,embedding_dim=8,hidden_dim=24,seed=42)` ofrece
`forward(ids, retain_cache=False) -> (logits, context)`,
`predict_logits(ids)`, `predict_proba(ids)` y `predict_classes(ids)`.
Entradas enteras `(B,C)`, logits/probabilidades float64 `(B,V)`, clases enteras
`(B,)`. Estas APIs procesan el lote completo; dividir inferencia grande en lotes.
`parameters()` incluye embedding y MLP en orden fijo. No reemplazar componentes
ni parámetros una vez construido el modelo.

Embedding usa `Parameter`, copia IDs del lote al retener contexto para evitar
mutaciones del llamador y acumula colisiones con `np.add.at`. Backward
sobrescribe gradientes entre lotes; no devuelve gradiente sobre IDs discretos.
Valida dimensiones, enteros, rangos y finitud; errores aritméticos siguen
`NumericalError`. Contextos comprueban propietario, versión y consumo; se
liberan al terminar backward o con `release_context`. Si se abandona un contexto
tras una entrada inválida, el llamador debe liberarlo (Trainer lo hace en finally).
No hay caché global ni activaciones retenidas en inferencia. Se reutilizan
las utilidades internas de validación/contexto de `nn` dentro del mismo paquete.

## Uso reproducible

Desde `VectorAnalysisBackend`:

```sh
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python examples/train_language_model.py --help
PYTHONPATH=src .venv/bin/python examples/train_language_model.py --seed 42 --epochs 120 --batch-size 8 --learning-rate .03 --output outputs/language_model.json
# Solo si se desea PNG:
.venv/bin/python -m pip install -r requirements-examples.txt
PYTHONPATH=src .venv/bin/python examples/train_language_model.py --plot outputs/language_model.png --output outputs/language_model.json
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_language.py
.venv/bin/python -m pytest -q
```

```python
from mlvectoranalysis.language import (
    SimpleTokenizer, ContextDataset, ContextLanguageModel, WindowModelAdapter,
)
from mlvectoranalysis.nn import Trainer, TrainingConfig, AdamOptimizer, CrossEntropyLoss

texts = ['el gato bebe agua .', 'el perro bebe agua .']
tokenizer = SimpleTokenizer.from_corpus(texts)
data = ContextDataset([tokenizer.encode(t) for t in texts], 2, len(tokenizer.vocabulary))
model = ContextLanguageModel(len(tokenizer.vocabulary), 2, seed=42)
trainer = Trainer(CrossEntropyLoss(), AdamOptimizer(model.parameters(), .03),
                  TrainingConfig(epochs=120, batch_size=8, seed=42))
result = trainer.fit(WindowModelAdapter(model, data), *data.training_arrays())
probabilities = model.predict_proba(tokenizer.encode('el gato')[None, :])
```

JSON estricto contiene `summary`, `result` (historial existente) y
`visualization`: vocabulario token/ID, documentos tokenizados, ventanas con
contexto/target/predicción y una distribución completa ordenada por ID. La
misma estructura alimenta la gráfica; permite añadir una interfaz futura sin
cambiar el entrenamiento. El primer panel muestra el diccionario, el segundo
la alineación supervisada y el tercero probabilidades que suman uno para un
contexto. Las barras no son frecuencias exactas ni garantías de calidad: son
estimaciones del modelo tras entrenamiento finito. Por ejemplo, «el gato» tiene
más de un sucesor observado; no debería interpretarse un argmax como certeza.
Matplotlib solo se importa al pedir `--plot`; usa Agg, guarda PNG y cierra figura.
No hay ventanas, cloud ni nuevas dependencias numéricas.

## Tiempo y memoria

L = caracteres, T = tokens, V = vocabulario, N = ejemplos, C = contexto,
D = dimensión embedding, H = ancho oculto, B = lote. Costes de cadenas dependen
además de longitud de token; ordenamiento lexicográfico O(V log V) comparaciones.

| Operación | Tiempo | Memoria adicional/residente |
| --- | --- | --- |
| Normalización/tokenize | O(L) | O(L) por documento |
| Vocabulario | O(T + V log V) esperado | O(V) tokens/tabla (más texto almacenado) |
| Encode/decode | O(L) / O(T + longitud de salida) | O(T) IDs / texto de salida |
| Construcción dataset | O(T + N) | O(T + N), sin matriz N×C |
| Materializar lote | O(BC) | O(BC) índices temporales y contextos |
| Lookup embedding | O(BCD) | O(BCD) salida + O(BC) IDs retenidos |
| Backward embedding | O(VD + BCD) | gradiente residente O(VD), sin one-hot |
| MLP forward/backward | O(B(CDH + HV)) | parámetros/gradientes O(CDH + HV), activaciones O(B(CD+H+V)) |
| Softmax / entropía cruzada | O(BV) | O(BV), sin one-hot |

La tabla embedding y su gradiente ocupan 8VD bytes cada uno; Adam añade dos
momentos por parámetro y sus buffers de trabajo según su contrato existente.
El cero inicial del gradiente cuesta O(VD) por backward; `np.add.at` añade todas
las ocurrencias, incluso repetidas, sin matrices de indicadores. Se prioriza
esta solución simple sobre ordenar/agrupar IDs con temporales adicionales.

Se concatenan los documentos una vez en un buffer int64 propio de 8T bytes,
convirtiendo dtype durante la copia. Durante construcción los buffers del
llamador siguen vivos; el dataset no los conserva. Los offsets usan 8N bytes;
la concatenación de offsets tiene temporal O(N). `training_arrays()` crea 8N
bytes de índices float64 y 8N de targets. Llamarlo una vez por fit. Trainer
agrega O(N) índices al mezclar; copia solo filas del lote. El ejemplo conserva
sus pequeños documentos codificados para mostrar sus IDs, una duplicación
O(T) deliberada de presentación. Un batch crea índices temporales y una copia
de contextos de 8BC bytes; reshape de embeddings es vista. No se retiene
el corpus en forma de N ventanas completas.

Medición local con Python 3.13.7 / NumPy 2.2.6, 30 repeticiones, B=64, C=16,
D=32, V=1000: mediana backward **0,221 ms**; tabla 256000 bytes, gradiente
256000, IDs retenidos 8192 y salida lookup 262144. Un one-hot float64 hipotético
ocuparía 8192000 bytes. El benchmark valida contra suma de referencia con
repeticiones; mide backward, no inicialización ni forward. Los tiempos son
orientativos, no umbrales de tests ni comparación universal entre algoritmos.
En el ejemplo: corpus 160 bytes, offsets 96 bytes y seis contextos 96 bytes.

## Límites y evolución

Modelo educativo de contexto fijo, vocabulario cerrado, orden preservado por
concatenación. No es LLM, Transformer, atención ni generación a gran escala.
El coste de la salida crece con V; todo reside en RAM. PAD/UNK pueden recibir
probabilidad y no se enmascaran. No se promete reconstrucción exacta de texto,
calibración, generalización ni reanudación/checkpoints del modelo compuesto.
La semilla fija hace repetible el ejemplo dentro del mismo entorno, no garantiza
identidad bit a bit entre BLAS/plataformas. La pérdida es entrenamiento sobre
un corpus diminuto, no validación externa. Una futura capa secuencial podría
consumir `(B,C,D)` en lugar del flatten sin cambiar tokenización/dataset;
esta fase no implementa esa capa ni abstracciones de atención.

## Verificación ejecutada (2026-10-31)

Desde `VectorAnalysisBackend`, `.venv/bin/python -m pytest -q` terminó con
**605 passed in 19.97s**, sin skips. Incluye toda la suite previa sin modificarla,
y pruebas nuevas de tokenización/errores, límites/alineación de ventanas,
lote final, gradientes repetidos, diferencias finitas de embeddings y todos
los parámetros del modelo compuesto, contextos, reproducibilidad, probabilidades,
JSON, CLI con/sin PNG y bloqueo explícito de importación de Matplotlib sin plot.
`git diff --check` terminó sin errores.

Se ejecutaron `--help`, el comando sin gráfica con seed=42/epochs=120 y el
comando con `--plot outputs/language_model.png`; ambos terminaron correctamente.
Pérdida inicial **2.5810111102668905**, final **0.2745859425588517**, 12 ejemplos,
estado `max_epochs`. Predicciones mostradas: come, pescado, ., come, pan, .
El PNG fue inspeccionado visualmente. El entorno avisó que el caché global de
Matplotlib no era escribible y utilizó automáticamente un directorio temporal;
no impidió crear la gráfica. `benchmarks/benchmark_language.py` también terminó
correctamente y verificó la acumulación numérica.
