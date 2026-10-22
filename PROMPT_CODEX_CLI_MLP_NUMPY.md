# Prompt para Codex CLI — MLP NumPy, backpropagation y entrenamiento por minibatches

Implementa esta fase del backend de MLVectorAnalysis: un perceptrón multicapa (MLP) con forward y backward manuales en NumPy, pérdidas diferenciables, SGD y entrenamiento por minibatches. Entrega implementación, pruebas ejecutadas, ejemplos visuales y mediciones. No te limites a un plan ni implementes un framework general de deep learning.

## 1. Estado inicial, alcance y compatibilidad

Trabaja desde la raíz del repositorio y dentro de `VectorAnalysisBackend`. Lee AGENTS.md aplicables, README, dependencias y módulos existentes antes de editar. Revisa Git y conserva cambios ajenos. No hagas commits, merges, pushes ni despliegues.

La base existente incluye análisis de curvas vectoriales, parser seguro, optimización multivariable, regresión lineal batch, historial acotado y visualización. La última ejecución observada tenía 432 pruebas pasando: comprueba la línea base actual sin asumir ese número como inmutable.

Dependencias verificadas al preparar el encargo: NumPy 2.2.6, SymPy 1.14.0, pytest 9.0.2 y Matplotlib 3.10.8 para ejemplos. Usa el entorno virtual existente. No instales PyTorch, TensorFlow, sklearn, Jupyter ni otro framework para implementar esta fase. NumPy es el motor; pytest verifica gradientes con casos analíticos y diferencias finitas. PyTorch queda como referencia opcional futura, no como requisito ni como tests obligatorios omitidos.

Mantén intactas las APIs de V1 y descenso de gradiente. `LinearLeastSquaresObjective` ya implementa media pérdida cuadrática y gradientes de regresión; no la reemplaces ni cambies su factor 1/2. No uses `VectorFunction` como representación de pesos neuronales ni obligues al MLP a tener tres dimensiones.

Alcance: MLP secuencial denso, Tanh/ReLU, salida lineal, regresión y clasificación multiclase, SGD básico, minibatches, evaluación, métricas acotadas y ejemplos. No implementes Adam, momentum, atención, embeddings, dropout, batch normalization, GPU, autograd genérico, redes recurrentes, entrenamiento distribuido ni checkpoints de reanudación en esta entrega. Diseña interfaces que permitan agregarlos después sin sobreingeniería.

## 2. Principios algorítmicos

La técnica principal es diferenciación en modo inverso mediante la regla de la cadena: recorrer capas en orden inverso y reutilizar las activaciones necesarias. No es un algoritmo codicioso. No uses derivación simbólica ni diferencias finitas para entrenar.

Utiliza productos vector-Jacobiano: propaga el gradiente de la pérdida sin construir Jacobianos ni Hessianas completos. NumPy/BLAS debe ejecutar los productos matriciales; los bucles Python solo recorren capas, minibatches y épocas, nunca muestras o neuronas para calcular el forward/backward.

Separa responsabilidades: capas calculan salidas/gradientes; pérdidas calculan escalares y derivadas respecto de predicciones; optimizador actualiza parámetros; trainer organiza el entrenamiento; visualización consume métricas y predicciones. Favorece composición y protocolos pequeños en vez de jerarquías profundas o servicios globales.

## 3. Estructura y contratos propuestos

Organiza el código nuevo en `src/mlvectoranalysis/nn/`. Usa nombres claros y exportaciones públicas en `__init__.py`. Puedes ajustar la distribución de archivos si mantienes los contratos y explicas el cambio.

### `parameters.py`: Parameter

Representa `data: ndarray` y `grad: ndarray` con igual forma y dtype, y nombre estable opcional. Soporta `zero_grad()`. Arrays de parámetros entrenables float64, finitos y mutables; rechaza complejos y object sin conversiones silenciosas peligrosas. Un parámetro mantiene su identidad y su buffer entre pasos del optimizador.

En esta fase no se admite compartir la misma capa/Parameter en varias posiciones de la red: rechaza ese caso explícitamente en vez de actualizar pesos compartidos dos veces. Documenta cómo podría ampliarse mediante acumulación y deduplicación en una versión futura.

### `layers.py`: contrato Layer y LinearLayer

Contrato orientativo:

    forward(inputs, *, retain_cache=False) -> (outputs, cache_or_none)
    backward(upstream_gradient, cache) -> input_gradient
    parameters() -> colección ordenada de Parameter

Utiliza caches explícitos por forward, no un único `last_input` global que una llamada de inferencia pueda sobrescribir. El contexto pertenece a esa ejecución y debe ser validable: backward sin contexto, forma incompatible, contexto de otra capa/modelo o reutilización de un contexto consumido produce error claro. No soportes backward sobre parámetros modificados desde ese forward; implementa una comprobación ligera para cambios hechos mediante las APIs públicas.

Para X de forma (B,d_in), W de forma (d_in,d_out), b de forma (d_out,):

    Z = X @ W + b
    dX = dZ @ W.T
    dW = X.T @ dZ
    db = sum(dZ, axis=0)

No dividas nuevamente por B en la capa: la pérdida ya incluye su reducción. Calcula dX con los pesos anteriores a la actualización. Backward sobrescribe los buffers de gradiente de esta ejecución; no acumula implícitamente entre minibatches. Explica esta convención y conserva `zero_grad()` para un ciclo explícito.

Valida tamaños positivos y formas 2D. Inicializa pesos aleatorios con un `numpy.random.Generator` proporcionado y sesgos en cero. Para capas ocultas con ReLU, He normal (varianza 2/d_in); con Tanh, Xavier normal (varianza 2/(d_in+d_out)); salida lineal Xavier. No inicialices todos los pesos a cero ni reseedes cada capa con la misma semilla.

Caches: una capa lineal necesita su entrada para dW. Define claramente cuándo conserva una referencia en vez de copia y exige no mutar inputs entre forward/backward. La inferencia no retiene caches. El trainer respeta ese contrato y libera referencias tras cada backward; no almacena contextos en el historial.

### `activations.py`: Tanh y ReLU

    tanh: dX = upstream * (1 - output**2)
    ReLU: dX = upstream * (input > 0)

Fija derivada de ReLU en cero igual a 0. Guarda solo salida Tanh o máscara booleana ReLU cuando se solicite cache. Nunca modifiques activaciones/gradientes del llamador por una optimización in-place no documentada.

### `mlp.py`: MLP

Construcción desde tamaños, por ejemplo `MLP([2, 8, 2], activation="tanh", seed=42)`. Cada capa oculta tiene activación y la última es lineal: logits para clasificación, valores para regresión. Composición secuencial, no herencia de las clases V1.

- `forward(X, retain_cache=False)` retorna salida y contexto por ejecución.
- `backward(d_outputs, context)` recorre caches en orden inverso y devuelve d_inputs.
- `parameters()` retorna parámetros una sola vez, en orden estable; `zero_grad()` los limpia.
- `predict(X, batch_size=...)` devuelve salidas crudas sin guardar caches; no transforma logits en clases silenciosamente.
- `predict_proba(X, batch_size=...)` aplica softmax estable a logits; `predict_classes` usa argmax. Documenta que estos dos métodos corresponden a redes usadas para clasificación.
- Predicción por bloques para limitar temporales; el array final completo sí ocupa O(N*d_out) y debe contarse.

Una evaluación de validación entre forward y backward no debe sobrescribir el contexto de entrenamiento. Evita clases de modelo diferentes para cada tarea si solo cambia la pérdida.

### `losses.py`: HalfMSELoss y CrossEntropyLoss

Contrato sin estado mutable compartido:

    value(predictions, targets) -> float
    value_and_grad(predictions, targets) -> (float, d_predictions)

`value` para validación no debe construir un gradiente que no se usará.

**HalfMSELoss:** predicciones y targets estrictamente de forma (B,K), B,K>0. Convención explícita de promedio sobre TODOS los elementos:

    L = sum((pred-target)**2) / (2*B*K)
    d_pred = (pred-target) / (B*K)

Para K=1 coincide con la pérdida actual de regresión. No permitas broadcasting entre (B,) y (B,1). Los ejemplos deben transformar targets de regresión a (B,1).

**CrossEntropyLoss:** logits de forma (B,C), C>=2; targets enteros de forma (B,) en [0,C). Rechaza floats, booleanos e índices inválidos. Sin pesos de clase ni label smoothing en esta fase.

    shifted = logits - max(logits, axis=1, keepdims=True)
    log_den = log(sum(exp(shifted), axis=1))
    L = mean(log_den - shifted[arange(B), targets])
    probabilities = exp(shifted) / sum(exp(shifted), axis=1, keepdims=True)
    d_logits = probabilities.copy()
    d_logits[arange(B), targets] -= 1
    d_logits /= B

Reutiliza exponentiales y sumas sin duplicación innecesaria dentro de la misma llamada. No materialices targets one-hot, ni calcules `log(softmax)` directamente, ni construyas un Jacobiano de softmax, ni apliques softmax antes de esta pérdida. No recortes probabilidades para ocultar un error de estabilidad. Valores extremos aún pueden exceder float64: detecta ese caso y falla con mensaje claro, no con pérdida cero ficticia.

### `optimizers.py`: SGDOptimizer

Recibe parámetros y learning_rate real positivo finito. `zero_grad()` y `step()`:

    parameter.data -= learning_rate * parameter.grad

No vuelve a dividir gradientes por batch_size. No calcula pérdidas, no ejecuta backward y no usa la búsqueda exacta cuadrática/Armijo del módulo anterior. El trainer hace una actualización por minibatch.

Valida todos los gradientes antes de mutar cualquier parámetro. Evita una actualización parcial si detectas no finitos o si el cálculo de un nuevo parámetro excede float64. Puedes prevalidar candidatos por parámetro usando un buffer temporal reutilizable y aplicar después, o justificar otra solución con memoria acotada; no dupliques todo el modelo en cada paso sin medirlo.

### `data.py`: MiniBatchIterator

Valida X y targets compatibles, n>0, batch_size entero positivo. Conserva filas emparejadas. Con shuffle=True, genera una permutación de índices por época usando RNG local persistente; no reinicies la misma semilla cada época. El último minibatch incompleto se incluye, sin duplicar muestras. batch_size>n entrega un lote de n filas.

No copies todo X para mezclarlo cada época. Conserva O(n) índices y crea solo el lote seleccionado; reconoce que indexación avanzada copia ese lote. Sin shuffle, usa slices cuando sea posible. No conviertas el iterador en una lista de todos los lotes. Entrenamiento y validación deben tener APIs diferenciadas para evitar mezclar sus muestras.

### `training.py`: Trainer y TrainingConfig

Config validada: epochs, batch_size, learning_rate o configuración del optimizador, shuffle, seed, eval_every, history_limit, tolerancias/criterios opcionales solo si se implementan explícitamente. Evita valores duplicados ambiguos de tasa de aprendizaje entre trainer y optimizador: elige una fuente de verdad y documéntala.

`fit(model, X_train, targets_train, *, validation_data=None, callback=None) -> TrainingResult` recibe modelo, pérdida y optimizador mediante composición/inyección coherente. Una opción es construir Trainer(loss, optimizer, config) y validar que el optimizador corresponde exactamente a los parámetros del modelo.

Ciclo por minibatch: zero_grad → forward con cache → loss.value_and_grad → backward → SGD.step. Libera el contexto incluso al terminar por error/cancelación. No normalices ni conviertas de dtype todo el dataset en cada época: valida y prepara una vez.

Separación de responsabilidades: fit entrena; evaluate recorre lotes sin actualizar pesos, gradientes ni RNG de entrenamiento. No ajustes normalización con validación. Si añades escalado en los ejemplos, aprende estadísticas solo de entrenamiento.

En cada época registrada, calcula train_loss y val_loss a parámetros fijos después de las actualizaciones, recorriendo datasets por lotes. Acumula pérdidas ponderadas por tamaño real del lote, no promedies medias de lotes con tamaños distintos. No llames a la media de pérdidas de minibatches durante actualizaciones `train_loss` final: representan estados distintos. Si expones esa métrica, dale un nombre diferente.

Una época completada y un optimizer step son contadores distintos. Registra estado inicial (epoch=0, steps=0), epochs_completed y optimizer_steps. `max_epochs` no implica convergencia matemática. No exijas que la pérdida disminuya en cada minibatch; SGD puede oscilar.

Admite callback con snapshot escalar/pequeño, sin acceso mutable a buffers internos; False cancela con estado `cancelled`. Excepciones de programación del callback se propagan. Distingue configuración inválida (ValueError) de fallo numérico durante entrenamiento (`non_finite` o excepción numérica documentada), sin reportar éxito ni datos mezclados. Mantén el último modelo actualizado de forma válida; si el estado inicial es inválido, falla antes de comenzar.

No implementes parada por gradiente global mediante el gradiente de un solo minibatch. Para esta entrega basta epochs limitadas, cancelación y detección de fallos. Documenta si una segunda llamada a fit continúa desde los pesos actuales; propuesta: sí, con contadores/historial nuevos y RNG reinicializado desde la configuración, sin afirmar reanudación exacta de checkpoint.

### `training_result.py`: TrainingResult y EpochMetrics

Métricas de entrenamiento/validación, estados, contadores, configuración y semilla. Para clasificación, accuracy además de loss; para regresión, usa HalfMSE y aclara que RMSE = sqrt(2*HalfMSE) con esta reducción.

Historial acotado que conserve inicio y último estado, inspirado en `TrajectoryRecorder` existente. Reutiliza su política si conviene, pero no fuerces métricas neuronales dentro de un DTO que exige una trayectoria de parámetros. Extrae un helper común solo si es pequeño y conserva tests anteriores.

No guardes todos los pesos, gradientes, activaciones o predicciones por paso. Diagnósticos por capa son opcionales y acotados: normas de pesos/gradientes y porcentaje de activaciones ReLU cero. Si se mide el gradiente del último minibatch, etiquétalo como tal, no como gradiente global. `to_dict()` produce JSON estricto sin NaN/Infinity ni arrays de entrenamiento.

Relaciones: MLP compone capas; cada LinearLayer posee Parameters; activaciones satisfacen Layer sin parámetros; Trainer usa MLP, pérdida, SGD e iterador; SGD referencia los Parameters del modelo; TrainingResult agrega EpochMetrics; ejemplos/visualización consumen resultados. No hay dependencias de las capas hacia gráficos, trainer o almacenamiento.

## 4. Eficiencia y complejidad verificables

Define anchuras d_0,...,d_L, P=sum(d_(l-1)*d_l+d_l), B=batch_size, N=muestras y E=épocas.

- Forward/backward por lote: O(B*sum(d_(l-1)*d_l)), más activaciones y pérdida; clasificación añade O(B*C).
- SGD: O(P) por actualización. Por época, además del trabajo matricial O(N*sum(d_(l-1)*d_l)), cuenta O(ceil(N/B)*P) de actualizaciones y la evaluación adicional cuando corresponda.
- Memoria de pesos/gradientes: O(P). Activaciones/caches: O(B*sum(d_l)). Datos: O(N*d_0) más targets; mezcla exacta: O(N) índices; historial con H registros y métricas por L capas: O(H*L).
- No hay almacenamiento O(E*P) ni O(N*P) de gradientes por muestra. Tampoco Jacobianos de tamaño cuadrático.
- Matrices y kernels vectorizados; no `np.vectorize`, ni concatenar/flatten todos los parámetros cada paso para calcular una norma.
- Float64 inicial para facilitar gradient checking. No prometas soporte float32/mixed precision sin implementarlo y probarlo. La API debe permitir una futura política de dtype sin constantes dispersas.
- Prefiere estabilidad numérica, ownership claro y código legible a microoptimizaciones especulativas. Reutiliza buffers donde sea seguro; no hagas in-place sobre caches necesarios para backward.
- Memoria acotada no significa que datasets arbitrarios quepan en RAM: esta entrega usa datos densos residentes. Registra copias de conversión de entrada, los buffers temporales de SGD y la memoria de resultados de predict.

El caching de activaciones es un intercambio tiempo/espacio deliberado. No implementes activation checkpointing ahora; documenta como extensión para redes profundas. No afirmes que el tamaño de lote más grande siempre es más rápido ni que produce la misma trayectoria de entrenamiento.

## 5. Tests obligatorios y criterios de aceptación

Agrega archivos de tests por responsabilidad. Mantén la suite previa. Usa float64, semillas locales y tolerancias justificadas.

1. LinearLayer: forward manual, dX/dW/db analíticos y diferencias finitas centrales en un caso pequeño. Verifica sumas de sesgo y ausencia de segunda división por B.
2. Activaciones: valores, gradientes, ReLU en cero; para diferencias finitas de ReLU usa puntos alejados del quiebre. Sin mutación del input/upstream.
3. HalfMSE: caso pred=[[1],[3]], target=[[0],[1]] → loss=1.25 y grad=[[0.5],[1.0]]. Casos multisalida, forma incompatible, lote vacío y no finitos.
4. Entropía cruzada: logits cero con C=3 → loss=log(3); gradiente correcto. Prueba logits de magnitud 1000, invariancia ante sumar una constante por fila, suma de gradientes por fila cercana a cero y targets inválidos. Distingue extremos manejables de overflow inevitable.
5. MLP end-to-end: gradiente de TODOS los parámetros frente a diferencias finitas centrales en red pequeña con Tanh y ambas pérdidas. Usa h alrededor de 1e-6 y tolerancias acordes; restaura cada peso tras perturbación. No limites el test a que los gradientes tengan la forma correcta.
6. SGD: actualización manual, zero_grad, identidad de buffers, no acumulación entre lotes, rechazo de parámetros duplicados e inputs inválidos; fallo numérico no deja un conjunto parcialmente actualizado.
7. Minibatches: cada muestra una vez, emparejamiento X/targets, último lote, batch_size>n, reproducibilidad y avance del RNG entre épocas. RNG global no se modifica.
8. Contextos: inferencia intercalada no corrompe backward; ausencia/reutilización/contexto ajeno rechazados; actualización mediante API invalida contextos antiguos. Los tests no deben depender de la organización privada de buffers.
9. Trainer: conteos de épocas/steps, métricas ponderadas con lotes desiguales, evaluación no mutante, cancelación, límites de historial y JSON estricto. Ningún uso de validación para actualizar parámetros.
10. Paridad de una red sin capas ocultas y HalfMSE con la regresión lineal existente: misma inicialización, sin estandarización, un paso full-batch fijo y comparación de gradientes/pesos/pérdida. Atiende W=(d,1) frente a pesos (d,).
11. Sobreajuste controlado XOR: red [2,8,2], Tanh, 4 puntos y targets [0,1,1,0], seed fija. Debe llegar a 100% accuracy y cross-entropy <0.1 dentro de un presupuesto documentado razonable, inicialmente hasta 5000 épocas. Elige y verifica un learning_rate estable sin búsquedas masivas. Esto prueba capacidad, no generalización ni que toda semilla converja.
12. Regresión sin(x): experimento reproducible con train/validation independientes dentro del mismo intervalo, red pequeña Tanh y salida lineal; demuestra reducción sustancial de error y RMSE de validación <0.15 con configuración real medida. Mantén tests de integración rápidos; el ejemplo largo puede validarse fuera de la suite unitaria, pero registra el resultado real.

No instales PyTorch para cumplir estos tests. No reemplaces tests matemáticos por screenshots ni por una pérdida que baja. No uses límites estrictos de tiempo de reloj en pytest ni debilites aserciones para ocultar errores.

## 6. Ejemplos y visualización

Usa Matplotlib opcional solo en ejemplos, backend no interactivo para exportar PNG. Aprovecha convenciones actuales (`PYTHONPATH=src`, argparse, outputs ignorados, --help). Genera:

### `examples/train_mlp_xor.py`

Opciones seed, epochs, batch-size, learning-rate, output y plot. Reporta probabilidades/clases de los cuatro puntos, accuracy, loss, pasos y estado. Gráficas de pérdida y frontera de decisión 2D con puntos etiquetados. La frontera es probabilidad de clase 1 calculada en una malla moderada por bloques; no construyas millones de puntos. No llames validación independiente a los mismos cuatro puntos.

### `examples/train_mlp_sine.py`

Opciones anteriores y tamaño de train/validation. Aprende sin(x) en un intervalo documentado usando RNGs reproducibles y datos separados. Muestra curva real, puntos de entrenamiento, predicción inicial/final y pérdidas train/validation. Si normalizas x, calcula estadísticas solo con train y reutilízalas.

Para mostrar evolución, conserva predicciones de un número pequeño y acotado de épocas sobre una rejilla fija; no copies todo el modelo por paso. Describe exactamente qué se guarda. Los ejemplos deben poder ejecutarse sin gráficos y producir JSON estricto.

### `benchmarks/benchmark_mlp.py`

Compara algunos batch_size y al menos dos tamaños pequeños/medianos de red. Mantén fijo el trabajo declarado —dataset y épocas— y registra updates reales, ejemplos procesados, tiempo de preparación, entrenamiento, evaluación e instrumentación. Explica que cambiar batch_size cambia el número de updates y la trayectoria: mide rendimiento sin atribuirle automáticamente mejor aprendizaje.

Usa warmup separado con otro modelo, semillas y varias repeticiones razonables. Reporta mediana, configuración, dtype, versiones y BLAS. Estima memoria explícita de parámetros/gradientes/caches/historial y, si mides RSS o tracemalloc, indica sus limitaciones. Limita tamaños antes de asignar grandes arrays. No publiques mejoras porcentuales sin una comparación justa medida.

## 7. Documentación y entrega

Actualiza README y añade `docs/mlp_numpy.md`: arquitectura y relaciones, formas, fórmulas forward/backward, reducción de pérdidas, inicialización, cache lifecycle, minibatches vs batch por bloques, complejidades, interpretación de gráficos y límites. Incluye instrucciones manuales con comandos copiables en una línea para evitar errores de shell.

Ejecuta suite anterior+nueva, ambos ejemplos y benchmark moderado. Inspecciona los PNG para verificar ejes, leyendas y legibilidad; corrige fallos. Revisa git diff y no agregues caches ni outputs grandes.

Entrega un resumen con clases/archivos, comandos realmente ejecutados, cantidad de tests, resultados XOR/sin, mediciones y limitaciones. Si algo no se pudo verificar, indica el bloqueo sin declarar éxito. El objetivo es una base pequeña, correcta, modular y medible para aprendizaje de redes; no afirmar que este MLP ya es un LLM o que escala por sí solo al tamaño de uno.
