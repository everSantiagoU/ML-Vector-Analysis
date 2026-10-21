# Descenso de gradiente y regresión batch

## Alcance y relación con V1

El motor minimiza un objetivo diferenciable de dimensión `p` con NumPy float64.
Hay dos implementaciones: una expresión escalar `f(x,y,z)` y mínimos cuadrados
densos de una salida. No hay servidor HTTP ni aplicación web. Las nueve clases
V1 mantienen sus relaciones y APIs: una `VectorFunction` sigue describiendo
una curva univariable. No se transforma el gradiente simbólico en esa clase.

| Archivo bajo `src/mlvectoranalysis/` | Clase y responsabilidad |
| --- | --- |
| `optimization/objective.py` | `DifferentiableObjective`: protocolo estructural de dimensión, valor y gradiente; `QuadraticObjective`: capacidad opcional de curvatura direccional |
| `domain/multivariable_function.py` | `MultivariableFunction`: parser restringido, derivadas y compilación numérica de f(x,y,z) |
| `optimization/linear_least_squares.py` | `LinearLeastSquaresObjective`: pérdida, gradiente batch, estadísticas y operador implícito |
| `optimization/gradient_descent.py` | `GradientDescentConfig`: dataclass validada e inmutable; `GradientDescent`: motor común |
| `optimization/result.py` | `OptimizationResult`, `IterationSnapshot`: estados finitos y exportación estricta |
| `optimization/history.py` | `TrajectoryRecorder`: almacenamiento acotado y decimación |
| `ml/linear_regression.py` | `LinearRegressionGD`: composición del objetivo y motor; coeficientes originales y predicción |
| `visualization/trajectory.py` | `OptimizationTrajectory`: métricas, ejes, proyección e interpolación |

El motor depende del protocolo y agrega un recorder. Ambos objetivos satisfacen
el protocolo sin heredar una clase concreta. El resultado agrega snapshots;
el adaptador de visualización consume ese resultado. La fachada de regresión
compone objetivo y motor, sin heredar la fachada de texto ni `VectorFunction`.

## Fórmulas y convenciones

Para f escalar: `q=[x,y,z]`, `q_next=q-alpha*gradient(f)(q)`, `alpha>0`.
`-gradient(f)` es una dirección local de descenso cuando el gradiente no es cero.
Los `q_k` son iterados discretos; no son una fórmula cerrada `r(t)` ni la solución
exacta de una ecuación diferencial. No es un algoritmo codicioso clásico.
En general no se garantiza un óptimo global, ni siquiera que un punto
estacionario sea un mínimo: también puede ser máximo o punto silla.

Regresión: `X.shape=(n,d)`, `targets.shape=(n,)`, `weights.shape=(d,)`.
El intercepto escalar es opcional y separado; no se añade una columna de unos:

```text
residual = X @ weights + intercept - targets
loss = dot(residual, residual) / (2*n)
grad_weights = X.T @ residual / n
grad_intercept = sum(residual) / n
```

`targets` son observaciones, distintas de la coordenada `y` de f(x,y,z).
El objetivo recibe `[weights..., intercept]` si `fit_intercept=True` (`p=d+1`),
y solo pesos si es `False` (`p=d`); en ese caso no entrena un intercepto ficticio.
La pérdida es la mitad del error cuadrático medio, sin regularización. Solo se
admite un target. Mínimos cuadrados es convexo, pero con rango deficiente los
parámetros óptimos pueden no ser únicos; compare predicciones y pérdida.

Cada actualización usa **todas las filas con los mismos parámetros**. Con
`block_size=B`, el objetivo acumula pérdida y gradiente de todas las rebanadas
(incluida la última incompleta) antes de devolverlos al motor. No es minibatch,
SGD ni entrenamiento fuera de memoria: X y targets continúan almacenados.

## API y entrada

```python
from mlvectoranalysis.domain import MultivariableFunction
from mlvectoranalysis.optimization import GradientDescent, GradientDescentConfig
from mlvectoranalysis.visualization import OptimizationTrajectory

f = MultivariableFunction.from_text('(x-1)^2+2*(y+2)^2+3*(z-0.5)^2')
config = GradientDescentConfig(step_strategy='auto', max_iter=1000, max_points=100)
result = GradientDescent(config).minimize(f, [4, 4, 4])
print(f.symbolic_gradient, result.status, result.parameters)
trajectory = OptimizationTrajectory(result)
payload = trajectory.to_dict()
midpoint = trajectory.interpolate(0.5)
```

`MultivariableFunction` admite exactamente `(x,y,z)` en ese orden, incluidas
expresiones que no usan todos los símbolos. `symbols`, `symbolic_form` y
`symbolic_gradient` exponen la explicación simbólica. Las derivadas y los dos
evaluadores (`value` y valor+gradiente con subexpresiones comunes) se construyen
una vez. No hay SymPy en cada iteración ni compilación durante el entrenamiento.

`_scalar_expression.parse_scalar_variables(text, variables)` generaliza el AST
existente; `parse_scalar(text, variable)` conserva la API V1 y delega al mismo
parser. La clase aplica `InputNormalizer` antes del parser. Se admiten números,
`pi`, `E`, operadores `+ - * / **`, `sin cos tan sqrt exp log`, las aridades V1
y sus alias normalizados. Se rechazan nombres externos, atributos, índices,
colecciones y llamadas Python arbitrarias. `lambdify` recibe exclusivamente
objetos SymPy construidos tras validar el AST completo; no recibe texto externo.
Esta gramática **no** es un sandbox con cuotas de CPU/memoria: expresiones
enormes todavía pueden ser costosas. El constructor textual usa la misma ruta.

Puntos de f y parámetros: vectores 1D de dimensión exacta, reales y finitos.
Datos de regresión: X 2D con `n,d>0`, targets estrictamente 1D de longitud n;
no se acepta `(n,1)` ni broadcasting accidental. Se rechazan complejos,
object, strings, bool y valores no finitos antes de convertir a float64.
Listas y arrays numéricos de otro dtype se convierten una vez; float32 no
conserva su ahorro de memoria. Arrays float64, incluso con strides o en orden
Fortran, se comparten sin forzar copia de layout. El código no modifica los
datos; el llamador tampoco debe modificar los arrays compartidos durante fit.
La validación de finitud recorre bloques de elementos, sin una máscara n×d.
BLAS puede crear buffers internos por layout: `out=` no promete eliminarlos.

Un objetivo personalizado necesita `dimension`, `value(p)` y
`value_and_grad(p)->(float, gradient)` de forma `(dimension,)`. No debe mutar
sus entradas. El gradiente devuelto debe permanecer válido tras evaluaciones
posteriores; el motor además lo copia al recibirlo. Los errores de dominio
numérico se indican con `ObjectiveEvaluationError` (subclase de `ValueError`)
o errores aritméticos; otros errores de programación se propagan. Los
objetivos de mínimos cuadrados reutilizan workspaces privados y no son
reentrantes: use una instancia por ejecución concurrente.

## Estrategia de paso y terminación

`GradientDescentConfig` valida enteros (sin bool), finitud y límites. Opciones:

| Opción | Predeterminado | Contrato |
| --- | --- | --- |
| `max_iter` | 1000 | Máximo de actualizaciones aceptadas; permite 0 |
| `gradient_atol`, `gradient_rtol` | 1e-8, 1e-8 | Tolerancia absoluta y relativa al gradiente inicial |
| `step_rtol`, `loss_rtol` | 1e-14, 1e-15 | Umbrales conjuntos de estancamiento, no de éxito |
| `step_strategy` | `auto` | `fixed`, `armijo`, `exact_quadratic`, `auto` |
| `alpha` | 1 | Paso fijo o paso inicial de cada búsqueda Armijo; positivo finito |
| `c1`, `beta` | 1e-4, 0.5 | Ambos estrictamente entre 0 y 1 |
| `max_backtracks` | 40 | Máximo de reducciones; hasta 41 candidatos |
| `min_step` | 1e-16 | Mínimo candidato de Armijo; no limita fixed/exact |
| `history_mode`, `max_points` | `sampled`, 200 | `none`, `metrics`, `sampled`; límite >=2 |
| `history_indices` | `None` | Todos los parámetros o 2–3 índices distintos válidos |

`auto` usa la capacidad `directional_curvature` para elegir paso exacto en
regresión; en un objetivo general usa Armijo. `fixed` aplica alpha sin prometer
monotonía. Se detectan valores no finitos y ausencia de progreso numérico.

Armijo acepta `f(q-alpha*g) <= f(q)-c1*alpha*dot(g,g)` y reduce el paso por beta.
Los candidatos rechazados evalúan solo el valor. Una salida del dominio puede
reintentarse con un paso menor. El candidato que satisface la condición se
evalúa con valor+gradiente para guardar un estado consistente; esto repite el
valor una vez en el estado aceptado, pero no calcula gradientes rechazados por
Armijo. Si su gradiente no es válido, también se reduce el paso.

Paso exacto cuadrático: `u=X@g_weights+g_intercept`, `curvature=dot(u,u)/n`,
`alpha=dot(g,g)/curvature`; con estandarización se usa Z implícita. Requiere
un producto matriz-vector adicional por paso, con todas las filas. Es exacto
solo a lo largo de la dirección `-g`, no una solución global en una actualización.
Fuera de un objetivo con esa capacidad, se lanza `ValueError` antes del bucle.
Curvatura no positiva/no finita, paso no válido u overflow producen fallo
numérico. Se comprueba primero el gradiente inicial para evitar 0/0. Si una
actualización supuestamente descendente aumenta la pérdida más de
`64*eps*max(1,abs(loss))`, se reporta `non_finite` (categoría de fallo numérico);
no hay fallback silencioso.

Solo hay convergencia si `||g_k||₂ <= gradient_atol + gradient_rtol*||g_0||₂`.
Esto certifica estacionariedad numérica. El estado inicial es iteración 0 y
si cumple la condición no se calcula curvatura ni se actualizan parámetros.
La norma usa `hypot` para no desbordar al elevar componentes al cuadrado.

Estancamiento: candidato exactamente igual al anterior, o simultáneamente:

```text
||q_next-q||₂ / max(1, ||q||₂) <= step_rtol
|loss_next/s - loss/s| <= loss_rtol, s=max(1, |loss|, |loss_next|)
```

| `status` | `converged` | Significado |
| --- | --- | --- |
| `gradient_tolerance` | True | Estacionariedad, no certificación de mínimo general |
| `max_iter` | False | Presupuesto de actualizaciones agotado |
| `stalled` | False | Progreso numéricamente insuficiente con gradiente grande |
| `line_search_failed` | False | Armijo agotó reducciones o paso mínimo |
| `non_finite` | False | Dominio/overflow/curvatura/descenso numérico inválidos |
| `cancelled` | False | Callback devolvió exactamente `False` |

La entrada inicial inválida lanza `ValueError`. Un fallo posterior conserva
los últimos parámetros aceptados finitos, su pérdida y su norma de gradiente.
Los contadores `n_function_evals`/`n_gradient_evals` incluyen intentos fallidos;
una evaluación conjunta suma uno a ambos. `n_curvature_evals` cuenta el trabajo
adicional del paso exacto. El estado aceptado se reutiliza en el siguiente paso.

El callback se ejecuta en 0 y tras cada actualización aceptada, con un snapshot
de **todos** los parámetros copiados, incluso si el historial está proyectado.
Modificar esa copia no corrompe el motor; el costo es O(p) por llamada. La
cancelación tiene precedencia sobre la prueba de gradiente de ese estado.

## Estandarización y fachada de regresión

```python
from mlvectoranalysis.ml import LinearRegressionGD
from mlvectoranalysis.optimization import GradientDescentConfig

model = LinearRegressionGD(
    fit_intercept=True, standardize=True, block_size=256,
    config=GradientDescentConfig(history_indices=(0, 2), max_points=100),
).fit(X_train, targets_train)
predictions = model.predict(X_eval)
print(model.coef_, model.intercept_, model.optimization_result_.status)
```

Estadísticas de entrenamiento, una sola vez: medias desplazadas, desviaciones
centradas y combinación estable por bloques (Chan). No se usa
`E[X²]-E[X]²`. El pequeño bucle por feature está restringido a estadísticas;
gradientes y predicciones usan productos vectorizados. La desviación estándar
es poblacional (`ddof=0`); escalas cero pasan a 1. Sin intercepto se escala
**sin centrar**, para conservar la misma familia de modelos.

```text
Z @ w = X @ (w/scale) - dot(mean,w/scale)
Z.T @ residual = (X.T @ residual - mean*sum(residual))/scale
coef_original = w_scaled/scale
intercept_original = b_scaled-dot(mean,coef_original)
```

No se almacena Z ni una submatriz B×d estandarizada. `mean=0` cuando no se
centra. `coef_` e `intercept_` siempre están en unidades originales;
`mean_` y `scale_` conservan las estadísticas. Predict usa esos coeficientes
equivalentes, sin recalcular estadísticas ni necesitar X de entrenamiento.
El resultado/historial conserva las coordenadas **de optimización** y lo
indica con `standardized_parameters` u `original_parameters`.

`fit` retorna self, inicia en cero y reemplaza el estado. Si lanza una excepción,
queda sin ajustar, sin mezclar atributos antiguos/nuevos. Una terminación del
motor sin convergencia aún publica su último estado finito: inspeccione
`optimization_result_.status` antes de dar por satisfactorio el ajuste.
`predict` rechaza uso antes de fit y número de features incompatible. Permite
cero filas de evaluación y valida datos reales finitos.

## Historial, JSON y visualización

`none` no almacena snapshots. `metrics` conserva métricas acotadas sin parámetros.
`sampled` conserva parámetros completos o 2–3 ejes originales seleccionados.
Los tres modos mantienen siempre el vector final completo en el resultado.
No se incluyen X ni targets. `to_dict()` usa listas y números nativos y verifica
`json.dumps(..., allow_nan=False)`. El paso inicial es `None`.

Decimación determinista: inicio 0, múltiplos de un stride (inicialmente 1) y el
estado más reciente. Al exceder M se duplica el stride y se descartan los
intermedios que no coinciden, conservando inicio/final. Un extremo transitorio
se reemplaza al llegar el siguiente estado. Nunca se acumula el historial entero.
La lista expuesta tras cada captura tiene longitud <=M, orden estricto y sin
duplicados. Las métricas usan el mismo límite. Las copias de coordenadas cuestan
O(p) u O(k) por iteración; la decimación filtra referencias O(M) al duplicar el
stride, O(M log T) acumulado, sin copiar todos los parámetros en cada paso.
Los snapshots son dataclasses congeladas, con arrays propios modificables;
no comparten memoria con el motor ni entre estados.

`OptimizationTrajectory(result, axes=(i,j[,k]))` informa nombres de ejes,
índices originales, sistema de coordenadas, pérdidas e iteraciones y si es
una proyección. Para p>3 requiere ejes explícitos (que deben estar capturados).
`axes=()` produce solo métricas; es el valor automático para p=1 o historial
sin coordenadas. No inventa dimensiones ni calcula PCA.

La interpolación es lineal por **índice de iteración**, exclusivamente entre
los puntos conservados. Si hubo decimación no reconstruye los descartados;
`sampled` lo informa. Las consultas en extremos devuelven copias exactas,
entre extremos interpolan y fuera del intervalo se rechazan. Un único punto
estacionario admite solo su índice 0. No es tiempo físico ni solución de EDO.
Para f(x,y,z) se dibuja una trayectoria 3D con color de pérdida, no la gráfica
completa de f, que requeriría cuatro dimensiones. En regresión una proyección
no describe toda la superficie de pérdida. No se generan mallas densas.

## Costo y límites

Validación y estadísticas iniciales: O(nd), además de las copias de entrada
documentadas. En cada evaluación se usan `X @ w` y `X.T @ residual`; la
transpuesta es una vista, sin forzar copia. No se forma Gram, Hessiana, inversa,
matriz n×n ni gradientes individuales n×d. Los buffers de residuos y gradiente
parcial se reutilizan, mientras el gradiente devuelto tiene almacenamiento propio.

Paso fijo: O(nd) por iteración, O(Tnd) total. Exacto cuadrático: mismo orden,
con otro producto matriz-vector. Armijo añade evaluaciones de pérdida: con b
candidatos O(bnd) por actualización en regresión, además del gradiente aceptado;
b está limitado por max_backtracks/min_step. En f depende del costo de evaluar
la expresión y sus derivadas compiladas. No se oculta ese costo bajo una constante.

Memoria total, contando datos normalizados y targets: full
O(nd+n+d+Mp), con p=d o d+1 y M puntos; bloques con k ejes,
O(nd+n+B+d+Mk). Con historial completo en bloques, sustituya Mk por Mp.
Metrics cuesta O(M); none no conserva puntos. Los buffers adicionales del
objetivo son O(n+d) u O(B+d). Estadísticas usan O(B) temporales por columna;
validación usa chunks acotados de elementos. X float64 ocupa `8*n*d` bytes.
Copias por conversión, datos originales retenidos por el llamador, callbacks,
BLAS y objetos Python suman memoria fuera de esos arrays base.

El condicionamiento y las tolerancias determinan T. Escalar ayuda a menudo,
pero no elimina colinealidad ni garantiza convergencia rápida. Grandes offsets
con dispersión muy pequeña pueden sufrir cancelación en el operador implícito;
datos o operaciones extremos pueden exceder float64 aun con entradas finitas.
No se promete tolerar toda escala ni función no diferenciable. No hay GPU,
regularización, múltiples targets, Adam ni sustitución por scipy/sklearn.

Una alternativa futura sería precomputar Gram con O(nd²) de preparación y
O(d²) de memoria, útil solo en ciertos casos d pequeño/T grande; formar
ecuaciones normales empeora el condicionamiento. No se implementa aquí.
QR/SVD/LSQR o métodos acelerados pueden superar GD en otros regímenes; no
hay una estrategia universalmente más rápida.

## Ejemplos y pruebas manuales

Desde `VectorAnalysisBackend` (Python >=3.10; verificación en 3.13.7):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Opcional: gráficos PNG; el núcleo no importa Matplotlib.
.venv/bin/python -m pip install -r requirements-examples.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider

PYTHONPATH=src .venv/bin/python examples/optimize_xyz.py --help
PYTHONPATH=src .venv/bin/python examples/train_linear_regression.py --help
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_gradient_descent.py --help

PYTHONPATH=src .venv/bin/python examples/optimize_xyz.py --output outputs/xyz.json --plot outputs/xyz.png
PYTHONPATH=src .venv/bin/python examples/optimize_xyz.py --strategy fixed --alpha 0.1 --max-iter 1 --output outputs/first-step.json
PYTHONPATH=src .venv/bin/python examples/optimize_xyz.py --initial 1 -2 0.5 --output outputs/stationary.json
PYTHONPATH=src .venv/bin/python examples/train_linear_regression.py --n 500 --d 5 --seed 42 --block-size 128 --output outputs/regression.json --plot outputs/regression.png
PYTHONPATH=src .venv/bin/python examples/train_linear_regression.py --noise 0 --block-size 128 --output outputs/noiseless.json
PYTHONPATH=src .venv/bin/python examples/train_linear_regression.py --axes 0 2 5 --output outputs/projected.json --plot outputs/projected.png

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src .venv/bin/python benchmarks/benchmark_gradient_descent.py --n 2000 --d 20 --steps 40 --block-size 256 --output outputs/benchmark.json
```

Revise manualmente el primer punto `[3.4,1.6,1.9]`, el inicial óptimo con cero
actualizaciones y paso null, la exportación JSON sin NaN/Infinity, el límite
de puntos y los ejes de las proyecciones. El ejemplo de regresión genera un
conjunto de evaluación independiente; ajusta estadísticas únicamente con
entrenamiento. `--noise 0` permite verificar recuperación de parámetros. Un
solo parámetro sin intercepto exporta pérdidas, sin inventar una trayectoria 2D.

El benchmark aísla cada caso full/block × historial on/off en un proceso;
usa semilla fija y calentamiento separado. Desactiva tolerancias para medir
trabajo controlado, pero informa pasos/estado reales si se detiene. Reporta
preparación, tiempo por paso, evaluaciones, arrays, versiones, CPU, BLAS e
hilos configurados (no garantiza observar el número efectivo de hilos).
Si no puede consultar la CPU, lo registra en `cpu_probe_error` sin invalidar
las mediciones. Estima arrays antes de asignar y limita esa estimación a 128
MiB; el ejemplo de regresión tiene límite de 256 MiB.

RSS es el máximo del **proceso**, no memoria incremental viva. El crecimiento
de ese máximo durante entrenamiento tampoco equivale a asignaciones. Se
reporta por separado el pico incremental observado por tracemalloc; no se
garantiza que capture buffers nativos de BLAS. Los tiempos incluyen su
instrumentación y no son umbrales de pytest ni promesas de rendimiento.

## Referencias técnicas

- [Boyd y Vandenberghe, capítulo 9](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf): métodos de descenso y búsqueda de paso.
- [NumPy dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html): productos y contrato de `out`.
- [SymPy lambdify](https://docs.sympy.org/latest/modules/utilities/lambdify.html): compilación numérica; no se le entrega entrada textual sin validar.
- [NumPy lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html): referencia utilizada exclusivamente en pruebas pequeñas.
