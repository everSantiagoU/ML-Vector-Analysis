# Prompt para Codex CLI — Adam, interfaz de optimizador y entrenamiento reanudable

Implementa esta fase en `VectorAnalysisBackend`: conserva SGD, incorpora Adam y permite guardar/cargar checkpoints locales para reanudar un entrenamiento exactamente en límites de época. Entrega código, pruebas, ejemplos visuales, benchmark y documentación. No hagas commits, merges, pushes ni despliegues.

## Punto de partida y límites

Lee `AGENTS.md` aplicables, estado de Git, README, `docs/mlp_numpy.md`, `docs/mlp_numpy_verification.md` y el módulo `src/mlvectoranalysis/nn/` antes de editar. Ejecuta la suite primero. En el estado esperado existen MLP NumPy float64, backpropagation manual, pérdidas, `SGDOptimizer`, minibatches, trainer, historial acotado, XOR y `sin(x)`; la última suite conocida fue de 549 pruebas, pero informa el resultado real.

No es una migración que elimine SGD: mantiene su API y comportamiento. Sustituye el acoplamiento de `Trainer` a `SGDOptimizer` por un contrato común mínimo. No agregues PyTorch, TensorFlow, sklearn, bases de datos, servicios cloud, serialización `pickle`, GPU, momentum-SGD, schedulers, mixed precision ni un gestor general de experimentos en esta entrega. NumPy y la biblioteca estándar bastan; Matplotlib ya es opcional para ejemplos. Si una dependencia adicional parece necesaria, explica el motivo y detente antes de incorporarla.

Esta fase prepara checkpoints para una futura sincronización a nube, pero solo escribe y lee rutas locales. El núcleo no importa herramientas de proveedor ni conoce AWS/Hugging Face.

## Diseño

### Optimizer protocol

Define un `typing.Protocol` estructural, por ejemplo `Optimizer`, y haz que el trainer dependa de él. Incluye como mínimo:

```python
parameters: tuple[Parameter, ...]
learning_rate: float
zero_grad() -> None
step() -> None
state_dict() -> dict
load_state_dict(state: Mapping) -> None
```

El contrato debe indicar que el orden e identidad de parámetros se conserva, `step()` verifica finitud antes de modificar y `state_dict()` devuelve copias sin referencias mutables internas. El trainer valida la colección ordenada de parámetros del modelo, pero no usa `isinstance(SGDOptimizer)`.

Mantén `SGDOptimizer` como implementación del contrato. Añade `state_dict/load_state_dict` con tipo, versión del formato, learning rate y la información necesaria para reanudar; valida tipos, shapes, orden y finitud antes de mutar. Si falla la carga, el optimizador no queda parcialmente modificado.

### AdamOptimizer

Implementa Adam clásico en `nn/optimizers.py` o módulo equivalente, sin optimizaciones desacopladas ni weight decay:

```text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t*g_t
m_hat = m_t / (1-beta1**t)
v_hat = v_t / (1-beta2**t)
theta_t = theta_(t-1) - learning_rate*m_hat/(sqrt(v_hat)+epsilon)
```

Usa valores por defecto beta1=0.9, beta2=0.999 y epsilon=1e-8; valida learning_rate y epsilon positivos/float64 finitos, 0<=beta1<1, 0<=beta2<1, parámetros únicos y gradientes/arrays válidos. Define el contador global `t`: comienza en cero y aumenta una vez por `step()` exitoso, nunca por parámetro. Un gradiente cero sigue siendo un paso válido y actualiza momentos/contador.

Para cada parámetro, Adam almacena `m` y `v` con igual shape/dtype; inicialízalos una vez, no por minibatch. No crees un diccionario o arrays nuevos por parámetro en cada `step()`. Reutiliza buffers temporales por tamaño máximo cuando sea seguro. Antes de mutar cualquier peso, calcula y valida todos los candidatos finitos: si hay overflow/NaN, ni pesos, ni momentos, ni contador cambian. Esta atomicidad abarca el estado completo del optimizador. Documenta el coste: SGD O(P) de estado auxiliar máximo; Adam O(2P) persistente más workspace O(max tamaño de parámetro), y ambos O(P) de tiempo por update, donde P es el número de parámetros.

`AdamOptimizer.state_dict()` incluye tipo de optimizador, versión, hiperparámetros, t, y copias ordenadas de m/v. `load_state_dict()` rechaza una configuración, número, forma, dtype o valores incompatibles antes de cambiar ningún buffer. No serialices objetos Parameter ni lambdas.

### Estado del modelo y checkpoint

Añade a `MLP` una representación explícita y validada de su arquitectura y métodos `state_dict()`/`load_state_dict()`. Incluye formato/version, tamaños, activación y valores de parámetros con nombres y shapes estables. Cargar en un modelo incompatible debe fallar antes de modificar pesos. No cambies la identidad de `Parameter.data` ni rompas contextos retenidos: cargar pesos usa `assign()` y exige que no haya contextos vivos, o documenta y comprueba una política segura equivalente.

Crea `nn/checkpoint.py`, una API local de alto nivel, por ejemplo:

```python
save_checkpoint(path, *, model, optimizer, training_state, metadata=None) -> CheckpointInfo
load_checkpoint(path, *, model, optimizer) -> TrainingState
```

`TrainingState` es una dataclass pequeña y serializable que contiene como mínimo epoch completada, optimizer_steps, estado del RNG de shuffle, historial acotado y configuración necesaria para continuar. No guarda `X`, targets, activaciones, contextos, gradientes ni callbacks. `metadata` contiene solo JSON y pequeñas etiquetas: versión de formato, timestamp UTC, versión de NumPy/Python, nombre de experimento, hash opcional del dataset y notas; no secretos.

Usa un formato seguro y portable: manifiesto JSON UTF-8 más arrays `.npz` de `numpy.savez_compressed` o un único `.npz` con un manifiesto JSON. Carga con `allow_pickle=False`. Valida el manifiesto, versiones, rutas/keys esperadas, tamaños y finitud antes de llamar a `model.load_state_dict()` y `optimizer.load_state_dict()`. Rechaza archivos corruptos, faltantes, incompatibles o con claves inesperadas sin modificar model/optimizer. No inventes éxito con carga parcial.

Guarda de modo atómico en el mismo sistema de archivos: escribe a un archivo temporal creado en el directorio destino, hace flush/fsync razonable, luego `os.replace`. Limpia el temporal ante error sin borrar un checkpoint existente. No escanees o borres directorios, no implementes retención automática todavía y no sobrescribas salvo que la ruta exacta se haya solicitado. Crea solo el directorio padre explícito de la ruta. La carga no debe escribir nada.

El checkpoint debe ser autocontenido para modelo y optimizador, pero la arquitectura se reconstruye mediante una factory explícita del llamador si hiciera falta; no ejecutes clases importadas desde un archivo externo. Para reanudar, recibe un MLP y optimizador compatibles ya construidos, o proporciona una función segura que solo acepte el tipo/activación/lista de tamaños que el propio proyecto produce.

### Trainer reanudable

Amplía `Trainer.fit` sin romper llamadas actuales. Propón una API clara, como `fit(..., resume_state=None, checkpoint_callback=None)` y/o `resume(...)`. La reanudación exacta se garantiza solo en límites de época; no prometas reanudar a mitad de minibatch.

Al iniciar un fit nuevo crea un RNG local persistente del shuffle. En cada época usa ese mismo generador, no resembrar. Al guardar, conserva `bit_generator.state` como JSON validable. Al cargar y reanudar, restaura el estado antes de generar la siguiente permutación. Conserva epoch completada y número de updates acumulados. Los resultados de una continuación deben incluir historia nueva o historia restaurada de forma explícita y sin duplicar la métrica de la frontera; conserva el límite de historial.

Define `epochs` con precisión: para un fit nuevo es cantidad de épocas a ejecutar; para una reanudación, usa `additional_epochs` o documenta sin ambigüedad si es total. Prefiere `additional_epochs` para evitar que un checkpoint de epoch 10 con `epochs=10` no haga trabajo por accidente.

Checkpoint al final de una época completamente evaluada: el callback recibe un snapshot inmutable y puede devolver False para cancelar. Nunca serialices un contexto de forward ni guardes durante un `step()`. Una excepción de callback se propaga y no crea checkpoint inesperado. Un error numérico no sobrescribe un checkpoint anterior; puede guardar explícitamente el último estado válido solo si una opción documentada lo solicita.

Mantén la evaluación sin mutar RNG, pesos, gradientes ni estado del optimizador. Entrenar N épocas seguidas debe coincidir, bajo la misma versión de NumPy/BLAS, con entrenar A épocas, guardar, crear proceso/objetos nuevos, cargar y entrenar N-A épocas: parámetros, estado Adam/SGD, contador, métricas y orden de minibatches deben ser equivalentes bit a bit si los inputs float64 y la plataforma son iguales. Documenta límites de reproducibilidad entre versiones/plataformas.

## Eficiencia y visualización

No aumentes la complejidad de forward/backward: sigue O(B*sum d[l-1]*d[l]) por minibatch. Adam añade O(P) tiempo y O(2P) persistente; checkpoint añade O(P) tiempo de I/O y O(P) espacio temporal solo en el momento de guardar, nunca por minibatch. No copies parámetros completos para snapshots de cada step ni construyas un historial O(epochs*P).

Por defecto guarda cada K épocas configurables y al final, con K>=1; el ejemplo usa pocos checkpoints. Guarda métricas JSON acotadas y pesos/estado binario separados. Reporta tamaño de checkpoint y tiempos de serialización/carga en benchmark, separándolos del tiempo de entrenamiento. No comuniques que la compresión siempre reduce tamaño/tiempo: mídelo.

Incluye un adaptador de visualización que pueda mostrar desde `TrainingResult` o el manifiesto: pérdidas train/validation, accuracy si aplica, época/steps, tipo de optimizador e hiperparámetros, y marcadores de checkpoint. El núcleo de `nn` no importa Matplotlib. No guardes una gráfica por época; ejemplos generan una PNG final y un JSON de resumen.

## Pruebas obligatorias

No borres ni relajes pruebas previas. Añade pruebas pequeñas y deterministas:

1. Contrato: Trainer funciona con SGD y Adam sin `isinstance` concreto; validación de parámetros incompatibles.
2. Adam: primer paso calculado a mano con un parámetro escalar; segundo paso, corrección de sesgo, gradiente cero, beta/epsilon/rate inválidos, gradientes no finitos, overflow y atomicidad completa de pesos/m/v/t.
3. `state_dict/load_state_dict` de SGD y Adam: copias independientes, round-trip, estados corruptos/incompatibles, shapes y órdenes distintos, sin mutación parcial al fallar.
4. Modelo: round-trip, arquitectura/activación y nombres incompatibles, pesos sin aliasing y preservación de identidad de buffers; contextos retenidos según la política documentada.
5. Checkpoints: round-trip en directorio temporal; JSON/NPZ sin pickle; checksum o detección de truncamiento/corrupción; archivo inexistente; claves extra; incompatibilidad; fallo durante guardado conserva checkpoint anterior; carga no escribe.
6. Reanudación exacta para SGD y Adam, con shuffle=True, último minibatch incompleto, validación e historial limitado: N épocas continuas frente a A+checkpoint+N-A. Compara todos los parámetros, m/v, t, epochs, steps, métricas y el próximo orden de minibatches. Ejecuta el tramo posterior con nuevos objetos para simular otro proceso.
7. Cancelación, `non_finite` y callback: ningún checkpoint automático inesperado; contextos liberados; estado válido sigue utilizable.
8. MLP con XOR y sin(x): Adam reduce pérdida de forma reproducible y conserva los criterios ya fijados. No asumas que debe superar a SGD; informa resultados medidos para ambas configuraciones.
9. Rendimiento: un benchmark moderado con SGD y Adam, warmup separado, repeticiones, N/arquitectura/batch/epochs fijos por comparación. Mide entrenamiento, evaluación, checkpoint save/load y bytes de estados. No impongas tiempos máximos en pytest.

## Ejemplos, documentación y entrega

Añade `examples/train_mlp_resume.py` con `--optimizer sgd|adam`, `--epochs`, `--additional-epochs`, `--checkpoint`, `--save-every`, `--resume`, `--output` y `--plot`. Debe demostrar una ejecución normal y otra reanudada; el checkpoint no se sube a Git. Añade una prueba de CLI que use una ruta temporal y no necesite red.

Actualiza README y crea `docs/adam_checkpoints.md`: fórmula y memoria de Adam; interfaz; esquema de checkpoint; límites de reanudación; comando de ejecución; guía para inspeccionar manifest/arrays; estrategia futura de caché y sincronización cloud. Documenta que el modelo sigue local y los checkpoints son artefactos que se podrán subir luego.

Actualiza `.gitignore` con rutas de checkpoints/outputs si faltan. No ignores tests, fuentes ni documentación.

Ejecuta suite completa, ejemplos SGD/Adam, reanudación y benchmark. Revisa `git diff --check`; no incluyas outputs, cachés o checkpoints generados. En la entrega final informa: archivos modificados, pruebas reales, resultados de comparación, bytes/timing de checkpoints, coste adicional de Adam, limitaciones y cualquier dependencia nueva. Si no instalaste nada, dilo explícitamente.
