# Verificación ejecutada — 2026-10-20

## Entorno y regresiones

- Python 3.13.7, macOS 27.0.1 arm64, Apple M1, 8 CPU lógicas.
- NumPy 2.2.6, BLAS/LAPACK Accelerate (versión no informada por NumPy).
- SymPy 1.14.0 y pytest 9.0.2, sin cambiar sus versiones fijadas.
- Matplotlib 3.10.8 para los ejemplos opcionales con backend Agg.
- Se creó `.venv` local: el Python disponible inicialmente no tenía NumPy.
- No se encontraron archivos AGENTS.md aplicables en los directorios revisados.

Línea base real antes de modificar código: **277 passed in 0.53s**.
Suite final: **432 passed in 0.99s** (277 V1 + 155 nuevas).
`pip check`: **No broken requirements found**. `git diff --check` sin errores.
Los tiempos de pytest son observaciones, no umbrales de aceptación.

Se probaron gradientes simbólicos/numéricos y diferencias finitas centrales,
primeros pasos manuales, monotonía Armijo/exacta, punto inicial estacionario,
intercepto opcional, estandarización, layouts con strides, no mutación,
gradientes y snapshots independientes, formas/tipos inválidos, dominio,
overflow, estancamiento, búsquedas agotadas, curvatura inválida, cancelación,
propagación de errores de callbacks, historial prolongado acotado, JSON estricto
e interpolación. Se compararon pérdidas y predicciones con `numpy.linalg.lstsq`
en problemas pequeños, incluidos rango deficiente, n<d y columnas constantes.
Las diferencias finitas usan h=1e-6 y tolerancias de aproximadamente 2e-6
relativas/2e-8 absolutas por cancelación y error de truncamiento; los casos
analíticos y full/block usan tolerancias cercanas al redondeo float64. Los
casos de recuperación usan tolerancias acordes con el criterio de gradiente.

## Comandos principales ejecutados

Desde `VectorAnalysisBackend`, salvo `git diff --check` desde la raíz:

```sh
# Línea base, antes de crear el entorno virtual:
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt numpy==2.2.6 matplotlib==3.10.8
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m pip check

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/mlvectoranalysis-mpl .venv/bin/python examples/optimize_xyz.py --output /tmp/mlvectoranalysis-gd/xyz.json --plot /tmp/mlvectoranalysis-gd/xyz.png
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/mlvectoranalysis-mpl .venv/bin/python examples/train_linear_regression.py --output /tmp/mlvectoranalysis-gd/regression.json --plot /tmp/mlvectoranalysis-gd/regression.png
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python examples/train_linear_regression.py --noise 0 --block-size 128 --output /tmp/mlvectoranalysis-gd/regression-noiseless-block.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python examples/train_linear_regression.py --n 10 --d 1 --no-fit-intercept --output /tmp/mlvectoranalysis-gd/regression-one-parameter.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python examples/optimize_xyz.py --strategy fixed --alpha 0.1 --max-iter 1 --output /tmp/mlvectoranalysis-gd/first-step.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python examples/optimize_xyz.py --initial 1 -2 0.5 --output /tmp/mlvectoranalysis-gd/stationary.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python benchmarks/benchmark_gradient_descent.py --n 2000 --d 20 --steps 40 --block-size 256 --output /tmp/mlvectoranalysis-gd/benchmark-final.json
```

También se ejecutó `--help` de los tres programas. Los PNG se inspeccionaron
visualmente y se regeneraron desde sus JSON con el helper final de gráficos
para ajustar márgenes y evitar etiquetas recortadas. Los outputs y cachés de
Matplotlib quedaron en `/tmp`, sin incorporarlos como entregables al repositorio.

La instalación inicial falló por red bloqueada dentro del sandbox; el reintento
autorizado instaló las dependencias correctamente. El primer benchmark calculó
los cuatro casos pero no terminó el reporte al bloquearse `sysctl`. Se repitió
con autorización y confirmó Apple M1. El script final registra ese fallo de
consulta como metadato opcional; luego completó el benchmark dentro del sandbox.
No quedaron pruebas o ejemplos principales pendientes de ejecución.

## Resultados numéricos

Cuadrática `(x-1)^2+2*(y+2)^2+3*(z-0.5)^2`, inicio `[4,4,4]`, auto/Armijo,
alpha inicial 1:

- Pérdida inicial: 117.75.
- Parámetros finales: `[1.0000000447034836, -2.0, 0.5000000521540642]`.
- Pérdida final: `1.0158540675320182e-14`.
- Norma de gradiente: `3.2544627290728025e-7`, dentro de la tolerancia
  absoluta + relativa al gradiente inicial configurada.
- 26 actualizaciones; 105 evaluaciones de función, 27 de gradiente;
  `gradient_tolerance`.
- Paso fijo 0.1, una actualización: `[3.4,1.5999999999999996,1.9]`, pérdida
  37.56 y estado `max_iter`, sin falso éxito.
- Inicio óptimo `[1,-2,0.5]`: cero actualizaciones, pérdida/gradiente cero,
  paso inicial null.

Regresión predeterminada, semilla 42, 500 filas de entrenamiento, 200 de
evaluación independiente, 5 features, ruido de desviación 0.05, intercepto
verdadero 2.5, estandarización implícita y auto/exact_quadratic:

```text
coef verdadero = [0.3047170798, -1.0399841062, 0.7504511958, 0.9405647164, -1.9510351887]
coef ajustado  = [0.3087632815, -1.0402542804, 0.7510300564, 0.9395122324, -1.9514847482]
intercepto ajustado = 2.4976161062201907
```

- Error L2 de coeficientes: `0.004253192763722629`.
- Pérdida de entrenamiento: `0.001278156161438138`.
- Pérdida de evaluación: `0.0011977076978769777`.
- RMSE de evaluación contra la señal sin ruido: `0.006388061888875599`.
- 9 actualizaciones, `gradient_tolerance`.

Mismo ejemplo sin ruido y bloques de 128: error L2 de coeficientes
`9.382181587793879e-9`, intercepto `2.5000000192366323`, pérdida de entrenamiento
`2.459921612875084e-16`, pérdida de evaluación `2.36889215368823e-16`;
9 actualizaciones, `gradient_tolerance`. Una feature sin intercepto también
se ejecutó: una actualización, intercepto exactamente cero y exportación
de pérdidas sin inventar un segundo eje.

## Benchmark moderado real

Última ejecución del script final, `n=2000`, `d=20`, float64, semilla 42,
estandarización activada, paso fijo alpha=0.01, 40 pasos solicitados y realizados
en **cada** caso. Tolerancias desactivadas para trabajo controlado; estado
`max_iter` esperado, sin afirmar convergencia. Cada caso hizo 41 evaluaciones
de función y 41 de gradiente, cero de curvatura. Calentamiento separado y
un proceso nuevo por caso. Hilos configurados por entorno:
OMP/OPENBLAS/MKL/VECLIB=1, BLIS sin configurar; no se midió el número efectivo.

| Caso | Preparación ms | Entrenamiento ms | ms/paso | Pico incremental tracemalloc bytes | Pico RSS del proceso MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full, sin historial | 1.845 | 6.343 | 0.159 | 5,415 | 70.641 |
| Full, con historial | 1.586 | 6.511 | 0.163 | 18,711 | 70.812 |
| Bloques 256, sin historial | 10.805 | 15.075 | 0.377 | 5,384 | 70.828 |
| Bloques 256, con historial | 10.762 | 15.541 | 0.389 | 18,680 | 70.734 |

En todos: X=320,000 bytes y targets=16,000 bytes. Buffers explícitos de residuos
y gradiente parcial: full=16,160 bytes, bloques=2,208 bytes. El historial on
retuvo 21 puntos de 2 coordenadas (336 bytes de arrays, además de objetos y
métricas), con límite 30 y vector final completo de 21 parámetros.
Pérdida final full `5.948453019684803`, bloques `5.9484530196848`.

La estimación previa de arrays fue 450,400 bytes, por debajo del límite de
128 MiB. El pico observado por tracemalloc durante preparación fue 374,490
bytes full y 369,832 bytes bloques. El pico incremental de la tabla excluye
los datos/workspaces ya existentes al iniciar entrenamiento. RSS incluye
intérprete, librerías, calentamiento y datos; no representa memoria incremental
del algoritmo. Tracemalloc no garantiza observar todos los buffers nativos.

Esta medición incluye instrumentación y una ejecución por variante; no es un
estudio estadístico de rendimiento. En este tamaño los bloques redujeron los
buffers explícitos y fueron más lentos. No se deduce una mejora universal de
tiempo o RSS. No se verificaron hardware/BLAS distintos, datasets masivos,
concurrencia, todas las funciones no convexas ni todos los rangos de float64.

## Revisión de entrega

Se mantuvieron los cambios previos en `.DS_Store`, cachés ya versionadas,
el prompt V1 eliminado y los diagramas movidos hacia `utils/`. No se hicieron
staging, commits, merges, pushes ni despliegues. Los cambios propios están en
el backend: implementación, pruebas, dependencias, ejemplos, benchmark y
documentación. La inspección del diff y de las rutas nuevas no encontró
outputs voluminosos ni una ruta de producción con scipy/sklearn, Gram,
Hessianas densas o gradientes por muestra.
