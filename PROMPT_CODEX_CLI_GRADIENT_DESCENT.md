# Prompt para Codex CLI: optimización multivariable y regresión lineal batch

Implementa y prueba la siguiente etapa del backend MLVectorAnalysis. Entrega código funcional, pruebas ejecutadas, ejemplos reproducibles y documentación; no te detengas en un diseño. Trabaja dentro de `VectorAnalysisBackend`, conservando las APIs de V1.

## 1. Inspección y alcance

Lee las instrucciones AGENTS.md aplicables, el README, requirements, código y tests actuales. Ejecuta la suite antes de cambiar código y registra la línea base real. La última verificación anterior a este encargo tuvo 277 tests pasando, pero debes comprobar el estado actual.

Revisa `git status`: hay cambios ajenos, archivos de caché modificados y movimiento de diagramas hacia `utils/`. No reviertas, agregues a staging ni borres esos cambios. No hagas commits, merges, pushes o despliegues. No supongas que los diagramas todavía están en `diagrams/`.

La V1 contiene `ComponentFunction`, `VectorFunction`, parser AST restringido en `_scalar_expression.py`, normalizador, validador, clasificador, analizador, resultado y fachada. Hoy `VectorFunction` representa una curva de tres componentes de UNA variable, generalmente t. No cambies esa semántica.

Implementa dos casos con un motor común:

1. Minimización numérica de una función escalar diferenciable f(x,y,z), generando puntos de una trayectoria en R³.
2. Regresión lineal de una salida, con descenso de gradiente batch sobre X densa de forma (n,d), con d arbitrario y objetivo de forma (n,).

Incluye la preparación de datos para visualización, no una aplicación web ni un servicio HTTP. No añadas SGD, minibatch, Adam, GPU, entrenamiento distribuido ni otro algoritmo de optimización como sustituto del solicitado. No delegues el entrenamiento a sklearn o scipy.optimize.

## 2. Semántica matemática

Para una función multivariable:

    q = [x,y,z]
    q_next = q - alpha * gradient(f)(q)

El campo -gradient(f) da la dirección. Los iterados q_k son muestras discretas; no son automáticamente una fórmula cerrada r(t) ni la solución exacta de una ecuación diferencial. Ofrece interpolación lineal explícita de la trayectoria capturada, parametrizada por índice de iteración. Si hubo muestreo del historial, aclara que se interpolan solo los puntos conservados.

Para regresión, llama `targets` al vector de observaciones para no confundirlo con la coordenada y de f(x,y,z):

    residual = X @ weights + intercept - targets
    loss = dot(residual, residual) / (2*n)
    grad_weights = X.T @ residual / n
    grad_intercept = sum(residual) / n

El intercepto es opcional y se maneja por separado; no agregues una columna de unos a X. Sin intercepto no entrenes un parámetro ficticio. Define todas las convenciones de signo, factor 1/2, formas y dtype. Esta versión no incluye regularización ni múltiples targets.

Batch significa calcular el gradiente usando TODAS las filas con los mismos parámetros y hacer UNA actualización por iteración. Recorrer bloques de filas para ahorrar buffers sigue siendo batch solo si se acumula el gradiente completo antes de actualizar.

No denomines al método un algoritmo codicioso clásico ni prometas un óptimo global para f arbitraria. El descenso elige una dirección local; la regresión de mínimos cuadrados es convexa, pero puede tener parámetros óptimos no únicos si hay rango deficiente. Un gradiente pequeño en una función general certifica estacionariedad numérica, no que el punto sea un mínimo.

## 3. Arquitectura y relaciones

Propón nombres snake_case para archivos y estas clases públicas. Ajustes menores de organización son aceptables si documentas la correspondencia. Evita clases que solo envuelvan una llamada sin aportar un contrato.

### `optimization/objective.py`: `DifferentiableObjective`

Un `typing.Protocol` estructural con `dimension`, `value(parameters)` y `value_and_grad(parameters)`. Los métodos reciben vectores NumPy unidimensionales y devuelven un float y un gradiente de igual dimensión. No exige herencia concreta. Permite reutilizar cálculos, en especial los residuos. Documenta propiedad de buffers: los resultados no deben modificarse silenciosamente por una evaluación posterior mientras el motor los use.

### `domain/multivariable_function.py`: `MultivariableFunction`

Responsable de expresión simbólica, símbolos ordenados y funciones numéricas compiladas. Para la API de esta etapa admite variables exactamente `(x,y,z)`; el motor numérico sí es genérico en dimensión.

- `from_text(expression)` normaliza texto y lo procesa con la gramática restringida antes de construir SymPy.
- `value(point)`, `gradient(point)` y `value_and_grad(point)` satisfacen el protocolo.
- Calcula derivadas simbólicas UNA vez y compila evaluadores NumPy UNA vez, no dentro del bucle de entrenamiento.
- Expone el gradiente simbólico para explicación y documentación, sin convertirlo a `VectorFunction` univariable.

Generaliza el parser AST existente para un conjunto ordenado de variables sin romper `parse_scalar(text, variable)` ni los tests V1. Reutiliza sus operadores, funciones y normalización; no escribas un segundo evaluador inseguro. Rechaza identificadores externos y texto Python arbitrario. `lambdify` solo recibe expresiones y símbolos construidos por el parser validado; nunca texto sin validar. No prometas que el parser es un sandbox con límites de CPU/memoria.

Valida punto de forma (3,), números reales finitos y resultado real finito. Casos fuera del dominio inicial deben producir un error claro. Durante búsqueda de paso, un candidato fuera del dominio puede rechazarse y probar un paso menor. No captures excepciones indiscriminadamente para ocultar bugs.

### `optimization/linear_least_squares.py`: `LinearLeastSquaresObjective`

Implementa el protocolo para X, targets y fit_intercept. Es responsable de pérdida y gradientes vectorizados, acceso batch y curvatura direccional cuadrática opcional. Mantén referencias a arrays normalizados una vez, sin copias por iteración; no muta arrays del usuario. Documenta qué conversiones de dtype/layout requieren copia y cuándo se comparten datos.

Valida X 2D, targets estrictamente 1D, n>0, d>0, longitudes compatibles, datos numéricos reales finitos y sin dtype object. Rechaza complejos sin descartar silenciosamente su parte imaginaria. No permitas broadcasting accidental de targets (n,1). Valida una vez fuera del bucle, preferiblemente por bloques para no crear máscaras de tamaño n*d.

### `optimization/gradient_descent.py`: `GradientDescent`

Motor común que usa una referencia a `DifferentiableObjective` y configuración validada (`GradientDescentConfig`, dataclass). Método público `minimize(objective, initial_parameters, callback=None) -> OptimizationResult`.

Configura max_iter, tolerancias, estrategia de paso, límites de búsqueda e historial. Valida enteros, límites y finitud. No utilices diccionarios de opciones sin validación. Define max_iter como número máximo de actualizaciones aceptadas; el estado inicial es iteración 0. No muta initial_parameters.

### `optimization/result.py`: `OptimizationResult` y `IterationSnapshot`

Dataclasses con parámetros finales, pérdida final, norma de gradiente, n_iter, converged, status, mensaje, evaluaciones de función/gradiente y puntos capturados. Snapshot: iteración, parámetros o coordenadas proyectadas, pérdida, norma del gradiente y paso aceptado. En iteración 0 el paso es null/None.

`to_dict()` produce JSON estricto (`json.dumps(..., allow_nan=False)`), listas/números nativos y ningún SymPy/ndarray. Fallos después de iterados válidos conservan el último estado finito aceptado; no serialices NaN o Infinity como si fueran datos válidos. No incluyas X ni targets en el resultado.

### `optimization/history.py`: `TrajectoryRecorder`

Posee/agrega snapshots copiados, nunca vistas de buffers mutables. Es una dependencia del motor. Modos `none`, `metrics`, `sampled`; `sampled` tiene límite estricto max_points>=2, conserva inicio y último estado y usa una política determinista documentada que no acumula primero todo el historial. Puede realizar decimación progresiva; contabiliza su costo y evita copiar todo el historial en cada iteración.

Permite guardar todos los parámetros o solo 2–3 índices seleccionados. Con p parámetros y M puntos, memoria O(Mp) o O(Mk) para k coordenadas seleccionadas. El resultado final conserva siempre los p parámetros aunque se proyecte el historial. Las métricas también tienen límite, no listas ilimitadas ocultas.

### `ml/linear_regression.py`: `LinearRegressionGD`

Fachada con `fit(X, targets)`, `predict(X_new)`, `coef_`, `intercept_`, `optimization_result_`, opciones fit_intercept y standardize. Compone el objetivo, configuración y motor; no hereda de `VectorFunction` ni del handler de texto. `fit` retorna self; predict rechaza uso antes de fit y dimensiones incompatibles.

Publica coeficientes en unidades originales, aunque se optimice en coordenadas estandarizadas. Cada fit reemplaza el estado anterior; un fallo no debe dejar atributos de dos entrenamientos mezclados. La predicción no depende de conservar X de entrenamiento.

### `visualization/trajectory.py`: `OptimizationTrajectory`

Adaptador del resultado/historial a payload para gráficas de pérdida vs iteración y trayectoria 2D/3D. Informa nombres de ejes, índices originales y sistema de coordenadas. Interpolación lineal entre muestras retenidas; documenta extremos y rechaza consultas fuera del intervalo. Acepta el caso de un único punto inicial ya estacionario.

Para regresión de dimensión mayor que tres, exige seleccionar ejes y muestra que es una proyección: no la presentes como toda la superficie de pérdida ni como trayectoria física. No calcules PCA por defecto. Para f(x,y,z), muestra puntos 3D y pérdida mediante color o gráfico separado; su gráfica completa tiene cuatro dimensiones. No construyas una malla 3D densa de pérdidas automáticamente.

Relaciones: `MultivariableFunction` y `LinearLeastSquaresObjective` satisfacen el protocolo; `GradientDescent` depende del protocolo y agrega un recorder; produce `OptimizationResult`; el resultado agrega snapshots; `LinearRegressionGD` compone objetivo y motor; `OptimizationTrajectory` consume resultados. Las nueve clases V1 conservan sus relaciones y APIs.

## 4. Política de paso y convergencia

Implementa tres modos con un selector `auto`:

1. **fixed**: alpha positivo y finito configurado. No prometas descenso monótono para cualquier alpha. Detecta resultados no finitos y estancamiento; no declares convergencia al agotar iteraciones.
2. **armijo** para funciones generales: q_trial=q-alpha*g; acepta si f(q_trial)<=f(q)-c1*alpha*dot(g,g), con 0<c1<1, reducción 0<beta<1, max_backtracks y min_step. Evalúa pérdidas de candidatos sin recalcular gradientes rechazados. Conserva el último iterado aceptado si la búsqueda falla. Un paso numéricamente nulo con gradiente grande es estancamiento, no éxito.
3. **exact_quadratic** para mínimos cuadrados: sea g el gradiente completo, incluyendo intercepto cuando aplique; u=X@g_weights+g_intercept. La curvatura es dot(u,u)/n y alpha=dot(g,g)/curvatura. Esta es búsqueda exacta a lo largo de -g para una cuadrática, no solución global en una sola iteración. Usa el operador efectivo si hay estandarización. Protege denominador no positivo/no finito y overflow; comprueba primero si el gradiente ya satisface tolerancia. No conviertas 0/0 en un paso arbitrario. Fuera de un objetivo que exponga esta capacidad, rechaza el modo con error claro.

`auto` selecciona exact_quadratic para regresión y Armijo para f general. Documenta el producto matriz-vector adicional del paso exacto. Si un candidato supuestamente descendente aumenta significativamente la pérdida por error numérico, reporta fallo numérico o aplica un fallback Armijo explícito y registrado; no declares éxito silenciosamente.

Calcula pérdida/gradiente del mismo estado al reportar. Reutiliza el estado aceptado entre iteraciones. Define criterio de gradiente absoluto+relativo respecto del gradiente inicial, y criterios relativos de paso y cambio de pérdida para detectar estancamiento. Solo marca `converged=True` por el criterio de estacionariedad documentado. Estados mínimos: `gradient_tolerance`, `max_iter`, `stalled`, `line_search_failed`, `non_finite`, `cancelled`. Entrada inicial inválida lanza ValueError antes del bucle. Callback devuelve False para cancelar y recibe un snapshot que no pueda corromper el motor; documenta frecuencia y costo de copiar parámetros.

Prueba siempre el punto inicial: si ya es estacionario, n_iter=0 y no se divide por la curvatura. Distingue máximos/puntos silla de un mínimo en la documentación, aunque su estado sea stationary/converged.

## 5. Eficiencia en tiempo y espacio

Usa NumPy y BLAS para productos matriz-vector. Float64 es el dtype soportado inicialmente; no prometas ahorro float32 convirtiéndolo después silenciosamente. Agrega NumPy a dependencias con versión compatible realmente verificada. No añadas SciPy/sklearn como dependencias de producción si no hacen falta.

Prohibido en la ruta principal:

- Bucles Python por muestra o por feature para calcular el gradiente.
- `X.T @ X`, inversas, Hessianas densas, matrices n×n, `diag(residual)` o un tensor n×d de gradientes individuales.
- `np.vectorize` como falsa vectorización, `np.append`/`concatenate` del historial en cada paso, copias completas de X por iteración o conversión de X a listas para JSON.
- SymPy dentro del bucle de regresión o recompilar lambdify en cada iteración.

X.T normalmente es una vista; no fuerces una copia contigua de la transpuesta. Reutiliza buffers de residuos y gradiente cuando no rompa propiedad de datos ni snapshots; usa `out=` solo si formas, dtype y layout satisfacen la API. Mide las asignaciones, no supongas que todo `out` elimina copias internas de BLAS.

Incluye `block_size` opcional: procesa rebanadas de filas, suma pérdida y gradiente completos y actualiza al final. Evita crear submatrices transformadas de tamaño bloque*d cuando el álgebra permite productos y vectores. El modo completo cuesta O(n+d) de buffers adicionales; el modo por bloques O(B+d). Ambos siguen almacenando la X densa de O(nd). No lo anuncies como entrenamiento fuera de memoria ni como mini-batch.

Para `standardize=True`, calcula medias y escalas solo durante fit y reutilízalas en predict. Escalas cero se sustituyen por 1; controla cancelación numérica del cálculo de varianza usando un método estable, sin E[X²]-E[X]² ingenuo. Si fit_intercept=False, escala sin centrar, para no cambiar la familia de modelos.

Evita otra copia densa estandarizada usando el operador implícito:

    Z @ w = X @ (w/scale) - dot(mean, w/scale)
    Z.T @ residual = (X.T @ residual - mean*sum(residual)) / scale

La corrección de media es cero cuando no se centra. Recupera coef_original=w_scaled/scale e intercept_original=b_scaled-dot(mean,coef_original). Implementa también la curvatura del paso exacto con Z. El historial debe indicar si sus parámetros pertenecen al espacio estandarizado; nunca etiquetarlos como coeficientes originales sin transformarlos.

Analiza y documenta:

- Validación/estadísticas iniciales: O(nd), aparte de copias de entrada documentadas.
- Batch con paso fijo: O(nd) por iteración, O(Tnd) total.
- Paso exacto cuadrático: O(nd) por iteración con un producto adicional; no reduce el orden asintótico.
- Armijo: evaluaciones extra según rechazos; para regresión con b candidatos O(bnd) por iteración en implementación directa, y para f general depende del costo de su expresión. No ocultes la búsqueda de paso bajo una constante sin explicar sus límites.
- Memoria total con modo completo: O(nd+n+d+Mp), con p=d+1 o d y M snapshots; modo bloque O(nd+n+B+d+Mk) si se proyecta el historial, contando también targets. La X densa por sí sola ocupa aproximadamente 8*n*d bytes en float64.
- El número de iteraciones depende del condicionamiento y tolerancias. Escalar features ayuda frecuentemente, pero no elimina colinealidad ni garantiza convergencia rápida para cualquier problema.

No existe una estrategia universalmente más rápida. Explica como alternativa futura que precomputar Gram cuesta O(nd²) y O(d²) extra, útil solo en ciertos regímenes d pequeño/T grande, y empeora condicionamiento al formar ecuaciones normales. No implementes esa ruta ahora. QR/SVD/LSQR o métodos acelerados pueden superar GD en otros regímenes; no sustituyas el algoritmo solicitado ni atribuyas optimalidad universal a GD.

## 6. Pruebas de corrección y regresión

Mantén todos los tests V1. Añade tests deterministas con semillas y tolerancias justificadas:

1. f=(x-1)^2+2*(y+2)^2+3*(z-0.5)^2: gradiente simbólico y numérico correctos; descenso desde [4,4,4] hasta [1,-2,0.5] dentro de tolerancia. Con paso fijo 0.1, el primer punto debe ser [3.4,1.6,1.9]. Prueba Armijo y evaluación inicial ya óptima.
2. Gradientes de ambas clases objetivo frente a diferencias finitas centrales en puntos interiores válidos, incluyendo intercepto, sin intercepto y estandarización. Usa diferencias finitas solo para tests.
3. Regresión pequeña comprobable a mano: X=[[1],[2]], targets=[2,4], sin intercepto, weights inicial cero. loss=5, grad=[-5]; con alpha=0.1 primer peso=0.5. En paso exacto alpha=0.4 y el peso pasa a 2.
4. Datos sintéticos de rango completo y ruido cero: recuperación de coeficientes/intercepto. Con ruido: compara pérdida y predicciones con `numpy.linalg.lstsq`, usado solo como oráculo de tests. La columna de unos se admite en ese oráculo pequeño, no en producción.
5. Rango deficiente, columnas constantes, X cero, targets constantes, una fila, una feature y n<d. En rango deficiente compara pérdida/predicciones, no exijas los mismos coeficientes que lstsq. Documenta cuándo no hay solución única.
6. Equivalencia de full batch y acumulación por bloques —incluido último bloque incompleto— con paso fijo y exacto. Verifica una actualización por pasada; nunca una por bloque.
7. Invariantes: no mutación de X/targets/initial_parameters; snapshots independientes; predict antes de fit; fit repetido; coeficientes originales tras estandarización; validación de formas/tipos y valores no finitos.
8. max_iter, cancelación, gradiente cero, búsqueda fallida, paso mínimo, dominio inválido, overflow y alpha fijo excesivo. No aceptes un falso converged=True. No exijas monotonía en modo fijo arbitrario; sí verifica Armijo y cuadrática exacta con tolerancia de redondeo.
9. Historial acotado en ejecuciones largas: límite M respetado, inicio/final, índices ordenados sin duplicados, ninguna acumulación oculta ilimitada, proyección de ejes y JSON estricto. Interpolación en puntos, entre puntos y fuera del rango; trayectoria de un punto.
10. Parser multivariable: funciones permitidas, variables desconocidas, accesos a atributos, índices y llamadas no autorizadas. Regresiones del parser univariable y su normalización.

No pongas umbrales rígidos de milisegundos en pytest. No falsees benchmarks con tolerancias que hagan terminar un caso antes de realizar el trabajo declarado.

## 7. Ejemplos, medición y visualización

Entrega ejemplos ejecutables desde el backend con `PYTHONPATH=src`, con `--help` y parámetros documentados:

- `examples/optimize_xyz.py`: función cuadrática anterior; exportación JSON de historial 3D y pérdidas, con valores inicial/final/estado.
- `examples/train_linear_regression.py`: dataset reproducible, n,d, semilla, coeficientes e intercepto conocidos, entrenamiento y reporte de error/pérdida. Distingue datos de entrenamiento de evaluación y ajusta el escalador solo con entrenamiento.
- Visualización opcional con Matplotlib como dependencia de ejemplos separada: curva de pérdida y trayectoria 3D para f; para regresión selecciona coordenadas e indica proyección. Guarda imágenes y datos sin requerir abrir una ventana. El núcleo no importa Matplotlib. No calcules la pérdida en una malla de millones de puntos para dibujar una trayectoria.
- `benchmarks/benchmark_gradient_descent.py`: mide casos moderados seguros de n,d y número controlado de pasos; full/block e historial on/off. Informa tiempo de preparación, tiempo por paso, pasos/evaluaciones reales, tamaños de arrays, pico de memoria disponible, dtype, versiones, CPU/BLAS y configuración de hilos. Semilla fija y calentamiento separados. No uses grandes asignaciones sin estimar memoria. Si usas RSS, distingue pico del proceso y memoria incremental; `tracemalloc` por sí solo no es garantía de medir todos los buffers nativos.

Documenta la relación con V1, fórmulas, clases, complejidades, políticas de convergencia, ejemplos y limitaciones en README/documentación dedicada. Incluye una guía breve de pruebas manuales. No prometas que funciona perfectamente para toda escala o función; entrega evidencia y límites medibles.

## 8. Criterios de aceptación

Implementación de ambos flujos, API genérica del optimizador, regresión batch densa vectorizada, historial limitado y exportable, pruebas nuevas y V1 pasando. Ejecuta ejemplos y un benchmark moderado real. Inspecciona el diff y evita archivos de cache/outputs voluminosos en entregables. No cambies las dependencias existentes sin justificar compatibilidad; instala en entorno virtual si hace falta.

En tu entrega final informa archivos/clases, decisiones, número real de pruebas, comandos ejecutados, resultados numéricos de los ejemplos, mediciones con su entorno y limitaciones. Si algo no pudo ejecutarse, indícalo con precisión. No afirmes mejoras de tiempo o memoria sin medición.

## Referencias de consulta

Estas referencias son documentación técnica; las decisiones concretas del proyecto están en el encargo anterior.

- Boyd y Vandenberghe, *Convex Optimization*, capítulo 9, métodos de descenso y búsqueda de paso: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
- NumPy `dot`, BLAS y contrato de `out`: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
- SymPy `lambdify`, evaluadores numéricos y advertencia sobre entrada no validada: https://docs.sympy.org/latest/modules/utilities/lambdify.html
- NumPy `lstsq`, referencia numérica para mínimos cuadrados en tests: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
