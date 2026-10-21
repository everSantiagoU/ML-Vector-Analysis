# Backend de MLVectorAnalysis

La etapa de optimización agrega descenso de gradiente para funciones escalares
`f(x,y,z)` y regresión lineal batch densa con dimensión arbitraria. Conserva
las APIs de V1: `VectorFunction` sigue siendo una curva de tres componentes
de **una** variable, no un campo gradiente multivariable.

Consulta [la guía de optimización](docs/gradient_descent.md) para fórmulas,
clases, convergencia, historial, complejidades y ejemplos; y [la verificación
ejecutada](docs/gradient_descent_verification.md) para resultados y mediciones.
NumPy 2.2.6 es la única nueva dependencia del núcleo; Matplotlib 3.10.8 es
opcional en `requirements-examples.txt`. SymPy y pytest conservan sus versiones.

```sh
# Desde VectorAnalysisBackend, tras instalar requirements.txt en .venv:
PYTHONPATH=src .venv/bin/python examples/optimize_xyz.py
PYTHONPATH=src .venv/bin/python examples/train_linear_regression.py --block-size 128
```

Los ejemplos guardan JSON en `outputs/` (ignorado por Git). Aceptan `--help`,
`--output` y `--plot ruta.png`; los gráficos se guardan sin abrir ventanas.

## Flujo V1

Flujo Python/SymPy: texto → normalización → validación restringida → modelo
simbólico de tres componentes → clasificación → resultado serializable.
Requiere Python 3.10 o posterior. Las versiones fijadas corresponden al entorno
con el que se ejecutaron las pruebas.

## Instalación y pruebas

Desde `VectorAnalysisBackend`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

`pytest.ini` añade `src` al path de las pruebas. Para usar la biblioteca sin
instalarla, ejecuta Python con `PYTHONPATH=src`:

```sh
PYTHONPATH=src python3 - <<'PY'
import json
from mlvectoranalysis.input import UserInputHandler

handler = UserInputHandler()
handler.receive_input("  r(t) = < t , t^2 , sen(t) >  ")
result = handler.process_input()
print(json.dumps(result.to_dict(), indent=2))
PY
```

El resultado conserva `normalized_input="r(t)=<t,t**2,sin(t)>"`, los tipos
`linear`, `polynomial`, `trigonometric` por eje y el tipo global `mixed`.
El texto recibido sigue disponible sin modificar en `handler.raw_text`.

## Sintaxis y validación

Se admiten `<a,b,c>`, `(a,b,c)` y `nombre(t)=<a,b,c>`; hay exactamente tres
expresiones escalares. `variable="x"` configura otra variable tanto en el
encabezado como en las componentes. El nombre por defecto es `r`.
La variable debe ser un identificador que no sea palabra reservada ni un nombre
de función o constante admitida.

- Números reales, variable configurada, `pi`, `E`, paréntesis y `+`, `-`, `*`,
  `/`, `**` (incluidos signos unarios).
- `sin`, `cos`, `tan`, `sqrt`, `exp` con un argumento; `log` con uno o dos,
  como `log(t,2)`. Las comas anidadas no separan componentes.
- Normalización idempotente de espacios, `^`, `π`, `√(...)`, `sen`/`seno`,
  `tg` y `ln`. No se admite multiplicación implícita.

El AST completo de cada componente se valida antes de construir expresiones
SymPy usando exclusivamente operadores y funciones permitidos. No se ejecuta
`eval` ni `sympify` sobre texto externo. Atributos, índices, colecciones,
strings, comprehensions, lambdas, argumentos keyword y nombres desconocidos
se rechazan. Los métodos booleanos del validador retornan `False`; extracción,
parsing y la fachada lanzan `ValueError` ante entradas inválidas.

El constructor directo de `ComponentFunction` y las cadenas pasadas a
`is_equivalent_to` conservan su API para **texto de dominio de confianza**.
La entrada externa debe pasar por `UserInputHandler` o `ExpressionParser`.
Las transformaciones internas transportan objetos SymPy; así se conservan
resultados legítimos como `erf` generados por integración, aunque no pertenezcan
a la gramática de entrada. La validez sintáctica no garantiza que la expresión
sea real y finita en todos los valores de la variable.

## Clasificación y relaciones

`ComponentFunction.get_type()` es la única fuente de reglas, en este orden:
polinomio de grado 1 → `linear`; otros polinomios (incluidos constantes y cero)
→ `polynomial`; presencia de `sin`, `cos` o `tan` → `trigonometric`; presencia
de `exp` → `exponential`; función racional → `rational`; resto → `unknown`.
Una expresión válida, como `log(t)` o `sqrt(t)`, puede ser `unknown`.

`FunctionClassifier` delega la clasificación de componentes. Tres etiquetas
iguales conservan esa etiqueta; una combinación de `linear` y `polynomial`
produce `polynomial`; las demás combinaciones producen `mixed`. Las constantes
participan y no se deducen formas geométricas.

Las nueve clases colaboran sin herencia. `UserInputHandler` mantiene normalizador,
validador, parser y analizador inyectables. El parser reutiliza el validador para
validar y construir cada expresión una sola vez; la fachada no repite esa
validación. Los colaboradores creados por defecto comparten variable y
normalizador. Si se inyecta un parser, se reutilizan sus colaboradores; las
dependencias explícitas incompatibles se rechazan sin modificarlas.

El parser crea componentes y vectores. `VectorFunction` agrega exactamente tres
componentes que pueden compartirse entre vectores; sus transformaciones devuelven
nuevos objetos y conservan nombre y variable. El dominio no depende de la fachada
ni del analizador. `VectorAnalyzer` usa un clasificador inyectable y construye
`AnalysisResult` sin descartar sus etiquetas ni volver a clasificarlas mediante
la factoría del DTO. `AnalysisResult.from_vector()` conserva su API independiente.

El resumen es `Vector {name}({variable}) analyzed as {vector_type} with 3 components.`
`analyze(vector)` admite omitir el texto normalizado, con valor `""`.
`to_dict()` mantiene las cinco claves `normalized_input`, `component_types`,
`vector_type`, `summary`, `operations_available`; sus valores son serializables
con `json.dumps`. Las operaciones predeterminadas son, en orden: `simplify`,
`differentiate`, `evaluate`, `norm`, `velocity`, `acceleration`, `speed`.

Estas decisiones concretan las ambigüedades del diagrama: métodos en snake_case,
sin duplicados, agregación de componentes, dependencias de creación sin herencia
y asociación adicional de la fachada al analizador para completar el flujo V1.
