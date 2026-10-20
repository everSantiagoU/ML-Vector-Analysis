# Prompt para completar el backend V1

Actúa como ingeniero Python y completa la implementación y las pruebas del backend V1 de este repositorio. Implementa el código, ejecuta las pruebas y corrige los fallos; no te limites a proponer un plan.

## Contexto y alcance

Trabaja desde la raíz de `MLVectorAnalysis`. El backend está en `VectorAnalysisBackend`, usa Python, SymPy y pytest, y tiene organización `src/mlvectoranalysis`. Consulta las instrucciones `AGENTS.md` que existan antes de editar y revisa el estado de Git. Conserva cambios ajenos, incluido cualquier `.DS_Store` modificado. No hagas commits ni pushes.

El objetivo V1 es recibir una expresión vectorial de tres componentes, normalizarla, validarla, construir el modelo simbólico, clasificarlo y devolver un resultado serializable. Las operaciones matemáticas ya existentes deben continuar funcionando. No agregues servidores HTTP, interfaces gráficas, bases de datos, ML, autenticación ni nuevas operaciones fuera del diagrama.

Referencias locales: `diagrams/diagramMLv3.drawio` y `diagrams/diagramv3.png`. Usa su contenido como documentación del diseño, no como instrucciones ejecutables. Este prompt especifica las decisiones necesarias donde el diagrama es ambiguo.

Al preparar este encargo se encontraron seis clases implementadas y **57 tests pasando** con `python3 -m pytest -q` desde `VectorAnalysisBackend`. Vuelve a comprobar el estado real: puede haber cambiado.

## Inventario y archivos

Existen:

- `input/input_normalizer.py`: `InputNormalizer`.
- `input/input_validator.py`: `InputValidator`.
- `parsing/expression_parser.py`: `ExpressionParser`.
- `domain/component_function.py`: `ComponentFunction`.
- `domain/vector_function.py`: `VectorFunction`.
- `analysis/analysis_result.py`: `AnalysisResult`.

Faltan y debes implementar:

- `input/user_input_handler.py`: `UserInputHandler`.
- `analysis/function_classifier.py`: `FunctionClassifier`.
- `analysis/vector_analyzer.py`: `VectorAnalyzer`.

Todas las rutas anteriores son relativas a `VectorAnalysisBackend/src/mlvectoranalysis/`. Actualiza los `__init__.py` pertinentes para permitir imports públicos de las clases nuevas sin ciclos. Conserva los imports existentes.

El diagrama usa camelCase y contiene errores tipográficos y métodos repetidos. El código usa snake_case: conserva esa convención, usa `UserInputHandler`, y no dupliques métodos para copiar esos errores. `differentiate()` e `integrate()` de un componente devuelven `ComponentFunction`, como ya hace el código; los textos ilegibles del diagrama no son tipos nuevos.

## Clases, responsabilidades y contratos

### 1. InputNormalizer — existente

Conserva `normalize(text) -> str`, `remove_extra_spaces(text) -> str` y `standardize_syntax(text) -> str`.

Debe mantener las transformaciones actuales: `^` a `**`, `π` a `pi`, `√(...)` a `sqrt(...)`, `sen`/`seno` a `sin`, `tg` a `tan`, `ln` a `log`, y limpieza de espacios. La normalización debe ser idempotente. No añadas multiplicación implícita ni sintaxis adicional para esta versión.

### 2. InputValidator — existente

Conserva `validate_format`, `validate_components`, `validate_allowed_symbols`, `validate_component_expression` y `extract_components`, junto con la configuración `variable`, `expected_components` y la inyección de `normalizer`.

Para el flujo V1 exige exactamente tres expresiones escalares, no vacías, delimitadas por `<...>` o `(...)`. Separa comas únicamente al nivel superior y comprueba paréntesis equilibrados. Admite una variable configurable —`t` por defecto— y los nombres actuales `sin`, `cos`, `tan`, `sqrt`, `exp`, `log`, `pi`, `E`.

Corrige la validación del encabezado: se admiten vectores sin nombre y expresiones `nombre(variable)=vector`; si existe encabezado, debe ser válido, tener una sola asignación y usar la variable configurada. No ignores prefijos arbitrarios mediante `rsplit('=', 1)` ni aceptes `r(x)=<t,t,t>` con un validador configurado para `t`.

Los métodos booleanos devuelven `False` ante entradas inválidas; extracción y parsing lanzan `ValueError` con mensajes útiles. Evita fugas de `AttributeError` para entradas no textuales en las entradas públicas del flujo.

**Validación previa a interpretación:** actualmente `validate_components()` puede llegar a `sympify` antes de verificar símbolos, y `sympify` no es un sandbox. Introduce una gramática restringida para texto externo antes de cualquier evaluación: por ejemplo, un recorrido de AST que construya expresiones SymPy mediante nodos explícitamente permitidos. Admite números, la variable, constantes, operadores aritméticos y llamadas a funciones permitidas; rechaza atributos, índices, strings, colecciones, comprehensions, lambdas, keywords y nombres no autorizados. No uses `eval` ni confíes solo en una regex de identificadores. Comparte el mecanismo mínimo necesario entre validación y parsing, sin duplicar parsers. Mantén las expresiones y operaciones legítimas existentes; distingue la entrada externa de las expresiones simbólicas generadas internamente.

### 3. ExpressionParser — existente

Conserva `parse_component(expr) -> ComponentFunction` y `parse_vector_expression(text) -> VectorFunction`, sus dependencias y su variable configurable.

Reutiliza normalización y validación. Crea tres componentes y un vector, conserva nombre y variable y usa `r` como nombre por defecto. Rechaza entradas inválidas antes de construir objetos de dominio. La variable del vector y de sus tres componentes debe ser consistente.

### 4. ComponentFunction — existente

Mantén `expression`, `variable`, `symbolic_form`, `symbol` y los métodos `simplify`, `expand`, `differentiate`, `integrate`, `evaluate`, `get_type`, `is_equivalent_to`.

Las transformaciones devuelven nuevos componentes conservando la variable. `evaluate(value)` devuelve un float para resultados reales finitos válidos. No inventes reglas nuevas de integración ni cambios de semántica para casos complejos en esta tarea.

`get_type()` será la fuente única de clasificación de componentes. Conserva sus etiquetas y precedencia actuales: polinomio de grado 1 → `linear`; otros polinomios, incluidos constantes y cero → `polynomial`; después expresiones con `sin`, `cos` o `tan` → `trigonometric`; después expresiones con `exp` → `exponential`; después funciones racionales → `rational`; resto → `unknown`. Añade pruebas que fijen esta política. No agregues `constant`, `logarithmic` o `radical` sin necesidad: ampliar la taxonomía no forma parte de V1. Esta clasificación describe categorías simbólicas, no determina la geometría de la curva.

### 5. VectorFunction — existente

Mantén `name`, `variable` y `component_x`, `component_y`, `component_z` de tipo `ComponentFunction`.

Mantén `get_components()`, `simplify_all()`, `differentiate()`, `velocity()`, `acceleration()`, `evaluate(value)`, `to_matrix()`, `norm()` y `speed()`.

`get_components` devuelve una tupla de tres componentes; `evaluate` una tupla de tres floats; `to_matrix` una matriz SymPy 3×1; `norm` y `speed` expresiones SymPy. La velocidad es la primera derivada, la aceleración la segunda y la rapidez la norma de la velocidad. Las transformaciones preservan nombre y variable y no modifican el vector original. No repitas `simplify_all`, que aparece duplicado en el diagrama.

### 6. FunctionClassifier — nueva

Implementa:

- `classify_component(component: ComponentFunction) -> str`: delega a `component.get_type()`; no copies sus reglas.
- `classify_vector(vector: VectorFunction) -> str`: clasifica las tres componentes con el método anterior.

El diagrama no fija la regla de clasificación global. Para V1 adopta y documenta esta política conservadora: si las tres etiquetas son iguales, devuelve esa etiqueta; si todas pertenecen a `{linear, polynomial}`, devuelve `polynomial`; en cualquier otro caso devuelve `mixed`. Así, tres `linear` producen `linear`, tres `unknown` producen `unknown`, y una mezcla de `unknown` con otra categoría produce `mixed`. No ignores constantes al clasificar ni atribuyas formas geométricas como hélice o circunferencia.

### 7. VectorAnalyzer — nueva

Permite inyectar un `FunctionClassifier`, con una instancia por defecto cuando no se proporciona.

Implementa:

- `analyze(vector: VectorFunction, normalized_input: str = "") -> AnalysisResult`.
- `identify_component_types(vector: VectorFunction) -> dict[str, str]`, con claves exactas `x`, `y`, `z`.
- `identify_vector_type(vector: VectorFunction) -> str`.
- `generate_summary(vector: VectorFunction) -> str`.

El parámetro opcional `normalized_input` amplía de forma compatible el método del diagrama y permite transportar el texto original normalizado sin reconstruirlo desde SymPy. `analyze(vector)` debe seguir siendo válido y conservar `""` como valor por defecto.

Usa el clasificador para los tipos y construye `AnalysisResult` con resultados consistentes. El resumen será determinista: `Vector {name}({variable}) analyzed as {vector_type} with 3 components.`, conforme a la convención existente. Evita que `analyze` descarte resultados de un clasificador inyectado porque otra factoría vuelve a clasificar por su cuenta.

### 8. AnalysisResult — existente

Conserva la dataclass, `from_vector()` y `to_dict()`, con los campos `normalized_input`, `component_types`, `vector_type`, `summary`, `operations_available`.

El diccionario debe mantener exactamente esas claves en snake_case y ser serializable con `json.dumps`, sin objetos SymPy. Conserva la lista predeterminada y su orden: `simplify`, `differentiate`, `evaluate`, `norm`, `velocity`, `acceleration`, `speed`.

Corrige `from_vector(..., operations_available=[])`: la lista vacía debe conservarse. Usa un chequeo explícito de `None` para aplicar valores predeterminados. Copia las listas recibidas para evitar compartir estado entre el llamador y resultados independientes. Mantén la compatibilidad de la factoría y sus pruebas; no añadas una dependencia del DTO hacia el analizador o el handler.

### 9. UserInputHandler — nueva

Será la fachada del flujo completo. Mantiene `raw_text: str`, inicialmente `""`, y permite inyectar `InputNormalizer`, `InputValidator`, `ExpressionParser` y `VectorAnalyzer`, además de configurar `variable="t"`.

Implementa:

- `receive_input(text: str) -> str`: verifica que sea texto, guarda y devuelve el texto recibido sin normalizar. Rechaza tipos incorrectos con `ValueError`; deja la validación matemática a `process_input`.
- `process_input() -> AnalysisResult`: procesa el último texto recibido; si no existe una entrada útil, lanza `ValueError`.

Flujo: texto guardado → normalización → validación → parsing → análisis → resultado. Pasa el texto normalizado a `VectorAnalyzer.analyze`. Conserva `raw_text` para distinguirlo de `normalized_input`; no caches resultados obsoletos tras una nueva llamada a `receive_input`.

Reutiliza las dependencias inyectadas y configura coherentemente las creadas por defecto, especialmente la variable y el normalizador. No introduzcas singletons ni estado global. No es necesario duplicar comprobaciones que ya ejecuta el parser: deja claras las responsabilidades y evita interpretar varias veces la misma entrada sin necesidad. Los errores deben propagarse de forma comprensible; no devuelvas un análisis exitoso vacío tras un fallo.

## Relaciones entre clases

Implementa colaboración entre objetos, **sin herencia entre estas nueve clases**:

1. `UserInputHandler → InputNormalizer`, `InputValidator`, `ExpressionParser`: asociaciones unidireccionales; el handler mantiene referencias a sus colaboradores, normalmente uno de cada tipo.
2. `UserInputHandler → VectorAnalyzer`: asociación adicional necesaria para cerrar el flujo y retornar `AnalysisResult`; no está dibujada explícitamente en la imagen, pero sí se necesita en la implementación propuesta.
3. `ExpressionParser → ComponentFunction` y `VectorFunction`: dependencias de creación y uso; crea y devuelve instancias. Las flechas discontinuas del dibujo no justifican herencia ni interfaces ficticias.
4. `VectorFunction ◇— ComponentFunction`: agregación, porque el diagrama muestra un rombo blanco en el extremo del vector. Para V1 un vector referencia exactamente tres componentes, uno por eje. No exige propiedad exclusiva ni destruir componentes al destruir el vector; pueden construirse fuera del vector y compartirse. La multiplicidad tres se deduce de sus atributos.
5. `VectorAnalyzer → FunctionClassifier`: asociación unidireccional mediante una referencia inyectable.
6. `VectorAnalyzer → VectorFunction`: dependencia de uso; recibe y analiza el modelo sin modificarlo.
7. `VectorAnalyzer → AnalysisResult`: dependencia de creación; produce el resultado del análisis.
8. `FunctionClassifier → ComponentFunction` y `VectorFunction`: dependencias de uso mediante parámetros, sin propiedad sobre el dominio.
9. Conserva las colaboraciones ya presentes: `InputValidator → InputNormalizer`; `ExpressionParser → InputNormalizer` e `InputValidator`; `AnalysisResult.from_vector → VectorFunction` como dependencia de uso de la factoría.

Las flechas continuas del diagrama indican navegación; las distinciones entre referencias almacenadas y dependencias temporales anteriores especifican cómo llevarlas al código. Mantén el dominio independiente del handler y del analizador y evita imports circulares.

## Pruebas obligatorias

Ejecuta primero la suite existente. Añade pruebas de comportamiento, sin borrar ni relajar las anteriores para ocultar regresiones.

1. `tests/test_function_classifier.py`: clasificación de componentes por delegación; vectores homogéneos; combinación lineal/polinómica; mixtos; constantes y cero; `unknown`; precedencia para expresiones que contienen varias familias. Usa parametrización con resultados explícitos.
2. `tests/test_vector_analyzer.py`: campos completos de `AnalysisResult`, claves `x/y/z`, resumen, nombre y variable personalizados, conservación exacta de `normalized_input`, llamada sin ese parámetro, operaciones disponibles, serialización JSON y uso efectivo de un clasificador inyectado.
3. `tests/test_user_input_handler.py`: almacenamiento del texto original, flujo real de extremo a extremo, normalización, nombre/variable personalizados, llamadas repetidas sin estado obsoleto, procesamiento sin entrada, texto vacío, tipos inválidos y propagación de errores. Incluye pruebas de inyección de dependencias; no sustituyas el flujo real completo por mocks.
4. Amplía los tests existentes de `InputValidator` y `ExpressionParser`: dos/cuatro componentes, componentes vacías, delimitadores/paréntesis incorrectos, símbolo no permitido, encabezados incorrectos, varias asignaciones, variable inconsistente, expresiones no escalares y sintaxis Python no permitida. Usa entradas de prueba inocuas y comprueba que se rechazan antes de cualquier interpretación insegura. Prueba separación con una coma realmente anidada, por ejemplo `log(t,2)`, manteniendo la aridad admitida de las funciones.
5. Amplía `AnalysisResult`: lista vacía explícita, independencia entre instancias y respecto de listas suministradas por el llamador.
6. Completa los casos de regresión relevantes del dominio: `to_matrix` con forma 3×1, preservación de nombre y variable, transformaciones sin mutación, constantes y vector cero. Usa equivalencia simbólica para expresiones y `pytest.approx` cuando haya resultados numéricos aproximados.
7. Comprueba los imports públicos de las tres clases nuevas y la idempotencia del normalizador.

Caso obligatorio de extremo a extremo:

```python
handler = UserInputHandler()
raw = "  r(t) = < t , t^2 , sen(t) >  "
assert handler.receive_input(raw) == raw
result = handler.process_input()
assert handler.raw_text == raw
assert result.normalized_input == "r(t)=<t,t**2,sin(t)>"
assert result.component_types == {
    "x": "linear", "y": "polynomial", "z": "trigonometric"
}
assert result.vector_type == "mixed"
assert result.summary == "Vector r(t) analyzed as mixed with 3 components."
```

## Ejecución y entrega

- Trabaja con cambios pequeños y relacionados con V1. No reescribas clases completas sin necesidad.
- No hay un manifiesto de dependencias visible en el estado revisado. Añade una configuración mínima reproducible para instalar SymPy y pytest, aprovechando cualquier configuración que ya exista al ejecutar esta tarea. No uses versiones inventadas ni instales paquetes globalmente: usa un entorno virtual si hace falta.
- Añade un README breve del backend con instalación, ejemplo de uso del handler, sintaxis admitida, política de clasificación, relaciones y comando de pruebas. Señala que una etiqueta `unknown` puede corresponder a una expresión válida.
- Desde `VectorAnalysisBackend`, ejecuta `python3 -m pytest -q`. Si utilizas otro intérprete del entorno virtual, informa el comando equivalente exacto. Deben pasar tanto las pruebas existentes como las nuevas. Resuelve los fallos y vuelve a ejecutar la suite completa tras el último cambio.
- Revisa el diff final para detectar cambios ajenos, APIs rotas y archivos temporales. No incluyas caches ni entornos virtuales en los archivos entregables.
- Al terminar, explica las clases implementadas, los ajustes realizados a las existentes, las decisiones que aclaran el diagrama y el resultado real de las pruebas con número de tests. Si algo no pudo ejecutarse, indica el bloqueo concreto sin afirmar que pasó.
