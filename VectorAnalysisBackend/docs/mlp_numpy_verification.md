# Verificación ejecutada: MLP NumPy

## Corrección posterior a la auditoría

Se corrigió el ejemplo de sin(x) cuando el trainer retorna `non_finite` y el
RMSE final es `None`: el PNG muestra «RMSE validación = no disponible» e indica
el fallo numérico, sin intentar formatear `None` como float. El JSON conserva
el estado y sus valores null; no se sustituye una evaluación fallida por cero.
La gráfica de pérdidas conserva las métricas anteriores válidas de validación
aunque falte la última, muestra puntos aislados y usa escala lineal si hay
pérdidas cero.

Se añadieron cuatro casos en `tests/test_nn_examples.py`: ejecución real del CLI
con PNG/JSON tanto normal como con fallo numérico, y retención del historial
válido con pérdida positiva o cero. Las pruebas de gráficos requieren la
dependencia opcional Matplotlib y se omiten explícitamente si no está instalada.
En el entorno verificado estaba instalada: **549 pruebas aprobadas, ninguna
omitida**. Se repitieron los ejemplos completos: XOR obtuvo accuracy 1.0 y
loss 0.00442945145; sin(x), RMSE de validación 0.06831076491.

Caso de regresión ejecutado desde el backend y PNG inspeccionado visualmente:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python examples/train_mlp_sine.py --epochs 2 --learning-rate 1e308 --output /tmp/mlp-fix/sine-failure.json --plot /tmp/mlp-fix/sine-failure.png
```

El comando termina sin traceback y exporta ambos archivos; el entrenamiento
sigue identificado como fallido mediante `status: non_finite`. La tasa extrema
solo se utiliza para provocar y comprobar esa ruta de error.

Fecha: 2026-10-21. Entorno virtual existente, Python 3.13.7, NumPy 2.2.6,
SymPy 1.14.0, pytest 9.0.2, Matplotlib 3.10.8. macOS 27.0.1 arm64,
BLAS/LAPACK Accelerate; los controles de hilos no estaban definidos
(se usaron los valores predeterminados de la biblioteca).
No se instalaron dependencias ni se realizaron commits, merges, pushes o despliegues.

## Pruebas

La línea base ejecutada fue **432 passed in 1.36s**. La suite completa final
fue **545 passed in 1.75s**: 113 casos nuevos, sin modificar los tests anteriores.
Estos tiempos son observaciones, no límites exigidos por pytest.

Los cinco archivos `tests/test_nn_*.py` cubren capas, activaciones, pérdidas,
gradientes completos, optimizador, datos y entrenamiento. Incluyen derivadas
analíticas; diferencias finitas centrales de todos los parámetros y entradas
con Tanh y ambas pérdidas (`h=1e-6`, `rtol=2e-5`, `atol=3e-10`); ReLU lejos
del quiebre y derivada cero en el origen; overflow, tipos y formas inválidos;
contextos ajenos/consumidos/obsoletos; inferencia intercalada; identidad de
buffers y atomicidad de SGD; emparejamiento y RNG de minibatches; métricas
ponderadas; validación no mutante; cancelación/excepciones; JSON estricto;
historial acotado; continuidad de pesos en un segundo fit; paridad full-batch
con `LinearLeastSquaresObjective`; XOR y regresión sin(x).

La tolerancia absoluta en diferencias finitas permite el error de resta
centrada cerca de cero; el caso usa pesos/entradas pequeños float64 y comprueba
cada elemento. No se sustituyeron estas comprobaciones por una pérdida que
simplemente disminuye ni por gráficos. No hay pruebas omitidas de PyTorch.

## Ejemplos medidos

| Experimento | Configuración | Resultado |
| --- | --- | --- |
| XOR | [2,8,2], Tanh, seed=42, lr=0.2, B=4, 1200 épocas | accuracy=100%, cross-entropy=0.00442945145, 1200 pasos |
| sin(x) | [1,24,24,1], Tanh, seed=42, lr=0.03, B=32, 1200 épocas, train=128, val=256 | RMSE val=0.06831076491, 4800 pasos |
| sin(x), integración corta | Misma configuración, 600 épocas, evaluación cada 100 en pytest | RMSE val≈0.08797, menor que 0.15; pérdida menor que 1/10 de la inicial |

XOR comienza con pérdida 0.69614673334 y termina en `max_epochs` con clases
`[0,1,1,0]`. Probabilidades de clase 1, en orden `(0,0),(0,1),(1,0),(1,1)`:
`[0.002567662, 0.995566310, 0.995624972, 0.006298746]`.
Los cuatro puntos prueban capacidad de sobreajuste, no generalización. El
presupuesto verificado de 1200 épocas está por debajo de 5000; no se hizo
una búsqueda masiva de learning rates ni se afirma convergencia para toda semilla.

sin(x) comienza con RMSE de validación **0.76028002360**. Termina en
`max_epochs`, train HalfMSE **0.00203926639**, val HalfMSE **0.00233318030**.
Se usaron RNGs independientes en `[-π,π]` y estadísticas aprendidas únicamente
en train: media **0.12105946211**, desviación **1.87434264983**. Se conservaron
cinco series de 300 predicciones, épocas 0/300/600/900/1200. El criterio medido
es error en el mismo intervalo, no extrapolación ni convergencia matemática.

Ambos ejemplos se ejecutaron con PNG y sin gráficos. Se parsearon sus JSON
rechazando NaN/Infinity y se comprobó igualdad completa de resultados entre
las dos modalidades. Se inspeccionaron visualmente `outputs/mlp_xor.png` y
`outputs/mlp_sine.png`: ejes, leyendas, escala de pérdida, colorbar, contorno
0.5, etiquetas y curvas son legibles. Matplotlib usó un cache temporal al no
poder escribir en su ubicación habitual; la exportación terminó correctamente.
Los PNG y JSON permanecen en `outputs/`, ignorado por Git.

## Benchmark moderado

N=1024, E=8, seed=42, lr=0.01, float64, tres repeticiones por caso y warmup
separado de dos épocas sobre 64 muestras con otro modelo/semilla. Dataset y
pesos iniciales iguales por repetición. Cada ejecución procesó **8192 muestras
de entrenamiento** y **9216 de evaluación** (inicial + ocho épocas).

Tiempos medianos en **milisegundos**. Las medianas de columnas se calculan
independientemente, por lo que su suma puede no ser exactamente la mediana total.

| Red | B | Updates | Preparación ms | Entrenamiento ms | Evaluación ms | Serialización ms | Fit total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [8, 16, 1] | 16 | 512 | 0.158 | 60.814 | 17.605 | 0.069 | 78.420 |
| [8, 16, 1] | 64 | 128 | 0.126 | 17.176 | 5.848 | 0.060 | 23.062 |
| [8, 16, 1] | 256 | 32 | 0.157 | 5.276 | 2.007 | 0.061 | 7.283 |
| [8, 64, 32, 1] | 16 | 512 | 0.198 | 98.083 | 30.385 | 0.073 | 128.468 |
| [8, 64, 32, 1] | 64 | 128 | 0.152 | 29.552 | 11.140 | 0.060 | 40.663 |
| [8, 64, 32, 1] | 256 | 32 | 0.145 | 11.648 | 5.751 | 0.057 | 17.400 |

`training_seconds = fit_seconds - evaluation_seconds`: incluye validaciones,
orquestación, loss/backward, SGD y mantenimiento del historial. La columna de
instrumentación mide serialización del resultado; el coste de timers e
historial sigue incluido en fit. El warmup no está incluido. Las repeticiones
se hicieron secuencialmente en un proceso, sin RSS/tracemalloc. El JSON conserva
cada repetición, no solo las medianas, además de versiones y configuración BLAS.

Al aumentar B se reducen los updates de 512 a 128 y 32. Aunque los tiempos
bajan en estos casos pequeños, eso no demuestra una mejora porcentual general,
mejor aprendizaje ni equivalencia de trayectorias. No se fija un límite de
tiempo como criterio de aceptación.

Memoria explícita y límites:

| Red | Parámetros (bytes) | Gradientes (bytes) | Workspace SGD (bytes) |
| --- | ---: | ---: | ---: |
| [8,16,1] | 1288 | 1288 | 1024 |
| [8,64,32,1] | 21512 | 21512 | 16384 |

En ambas redes, datos/targets ocupan 73728 bytes y la permutación 8192 bytes.
Para la red [8,64,32,1] con B=64, la copia del lote suma 4608 bytes; el límite
conservador de caches es 103424 bytes y de temporales de backward, 114944 bytes.
Nueve métricas tienen hasta 504 bytes de payload escalar; el informe estima
por separado el overhead de objetos Python y el JSON. `predict` completo, si
se solicitara, agregaría 8192 bytes de salida; el benchmark evalúa por lotes
sin retener esa salida completa. La conversión de dtype ocupa cero bytes en
estos casos porque los datos ya son float64.

Son estimaciones explícitas, no un máximo total de memoria del proceso:
no incluyen íntegramente BLAS, allocator, objetos auxiliares ni temporales
booleanos de validación. Para otros dtypes hay una copia residente de entrada.
No se copia todo el modelo por paso: el workspace de SGD es el de la mayor
matriz de parámetros. Las gráficas tampoco guardan checkpoints de pesos.

## Comandos ejecutados

Desde `VectorAnalysisBackend` (la suite base se ejecutó primero sin la
variable `PYTHONDONTWRITEBYTECODE`; se restauró únicamente el bytecode
versionado que esa ejecución regeneró):

```sh
.venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python examples/train_mlp_xor.py --plot outputs/mlp_xor.png
PYTHONPATH=src .venv/bin/python examples/train_mlp_sine.py --plot outputs/mlp_sine.png
PYTHONPATH=src .venv/bin/python examples/train_mlp_sine.py --epochs 600 --output outputs/mlp_sine_600.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python examples/train_mlp_xor.py --output outputs/mlp_xor_no_plot.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python examples/train_mlp_sine.py --output outputs/mlp_sine_no_plot.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python benchmarks/benchmark_mlp.py --n 1024 --epochs 8 --batch-sizes 16 64 256 --repeats 3
PYTHONPATH=src .venv/bin/python examples/train_mlp_xor.py --help
PYTHONPATH=src .venv/bin/python examples/train_mlp_sine.py --help
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_mlp.py --help
```

También se revisaron `git diff --check`, diff del README y estado de Git.
Se preservaron los cambios ajenos iniciales (`.DS_Store`, eliminación del
prompt anterior y nuevo prompt sin seguimiento). No se agregaron outputs,
caches ni dependencias. Los límites de versionado de escrituras directas a
`Parameter.data`, datos residentes, concurrencia y extensiones fuera de alcance
se describen en [la guía](mlp_numpy.md).
