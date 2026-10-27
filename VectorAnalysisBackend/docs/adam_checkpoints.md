# Adam, checkpoints locales y reanudación por época

`mlvectoranalysis.nn` conserva `SGDOptimizer` y añade `AdamOptimizer`. Ambos implementan el protocolo estructural `Optimizer`: tupla ordenada de `Parameter`, `learning_rate`, `zero_grad()`, `step()`, `state_dict()` y `load_state_dict()`. El trainer no depende de una clase concreta, pero exige que los parámetros sean exactamente los del MLP, en el mismo orden.

## Adam y coste

Para cada gradiente `g`, Adam calcula:

```text
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g*g
theta = theta - lr * (m/(1-beta1^t)) / (sqrt(v/(1-beta2^t)) + epsilon)
```

Los valores por defecto son `beta1=0.9`, `beta2=0.999` y `epsilon=1e-8`. `t` empieza en cero y avanza una vez por actualización exitosa, incluso con gradiente cero. Antes de escribir, Adam valida todos los candidatos: un fallo numérico deja intactos pesos, momentos y `t`.

SGD tarda `O(P)` por update y usa como máximo un workspace `O(max parámetro)`. Adam también tarda `O(P)`, conserva `2P` valores (`m`, `v`) y usa workspace temporal `O(max parámetro)`. No hay weight decay, momentum-SGD, scheduler ni precisión mixta.

## Estado y formato

`MLP.state_dict()` contiene versión, tamaños, activación, nombres/formas/dtype estables y copias de pesos. `load_state_dict()` valida todo antes de copiar con `Parameter.assign()`: no reemplaza buffers. Se rechaza una carga si hay un contexto de `forward(..., retain_cache=True)` vivo; libérelo o complete `backward` primero.

`save_checkpoint(path, model=..., optimizer=..., training_state=...)` guarda un único NPZ comprimido con manifiesto JSON UTF-8 y arrays float64. No usa pickle (`load_checkpoint` siempre usa `allow_pickle=False`). El manifiesto incluye versión, metadatos JSON pequeños, arquitectura, layout del optimizador, época/steps, estado del RNG de shuffle e historial acotado. Adam añade momentos y `t`; datasets, targets, gradientes, contextos, callbacks y secretos no se serializan.

La escritura usa un temporal en el directorio de destino, `fsync` y `os.replace`; si falla, se limpia solo el temporal y el checkpoint anterior permanece. La carga es solo lectura, detecta CRC/truncamiento de NPZ, claves extra, incompatibilidades, dtype/shape no finitos y valida modelo y optimizador antes de mutar cualquiera. Inspección segura:

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import json, numpy as np
with np.load('outputs/checkpoints/mlp_resume.npz', allow_pickle=False) as f:
    print(f.files)
    print(json.loads(f['__mlvectoranalysis_manifest_utf8__'].tobytes()))
PY
```

## Reanudación y ejemplo

La equivalencia bit a bit se promete solo entre límites de época, con misma versión de NumPy/BLAS, plataforma e inputs float64. `TrainingState` restaura el `Generator` local antes de la siguiente permutación; cubre shuffle y el último lote incompleto. `epochs` es un presupuesto para una ejecución nueva; una continuación requiere `additional_epochs`. La evaluación no toca RNG, pesos, gradientes ni estado de optimizador.

```sh
PYTHONPATH=src .venv/bin/python examples/train_mlp_resume.py --optimizer adam --epochs 6 --save-every 2 --checkpoint /tmp/mlp.npz --output /tmp/first.json
PYTHONPATH=src .venv/bin/python examples/train_mlp_resume.py --optimizer adam --resume --additional-epochs 4 --checkpoint /tmp/mlp.npz --output /tmp/resumed.json
```

`checkpoint_callback` recibe un `TrainingState` inmutable tras evaluar una época completa y puede devolver `False` para cancelar. Si lanza, la excepción se propaga y no hay guardado implícito. El ejemplo guarda cada `K` épocas (`K>=1`) y al final; produce JSON resumen y, con `--plot`, una PNG final. `training_visualization()` y `manifest_visualization()` exponen pérdidas, accuracy, steps, hiperparámetros y marcadores para interfaces sin que el núcleo importe Matplotlib.

Estos artefactos son estrictamente locales. Una futura caché/sincronización cloud debe tratar el NPZ validado como blob inmutable y conservar el manifiesto para compatibilidad; el núcleo no conoce AWS, Hugging Face ni proveedores.

## Mediciones locales

En el entorno del proyecto (Python 3.13, NumPy 2.2.6, macOS arm64), con las arquitecturas, datos y 1200 épocas de los ejemplos existentes, SGD/Adam completaron el presupuesto y redujeron la pérdida reproduciblemente. No se interpreta esto como superioridad general de Adam:

| Caso | SGD | Adam |
| --- | --- | --- |
| XOR `[2,8,2]`, B=4 | cross-entropy `0.00442945`, accuracy `1.0` | cross-entropy `0.00002311`, accuracy `1.0` (`lr=0.03`) |
| sin(x) `[1,24,24,1]`, B=32 | RMSE val `0.06831076` | RMSE val `0.01216763` (`lr=0.01`) |

El benchmark moderado se ejecuta sin red y separa warmup, entrenamiento, evaluación, checkpoint y carga:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_mlp.py --n 1024 --epochs 8 --batch-sizes 16 64 256 --repeats 3
```

Una corrida pequeña (`N=64`, 2 épocas, B=16, una repetición) midió, para `[8,16,1]`, SGD: checkpoint `3,084` bytes, guardado `2.105 ms`, carga `0.680 ms`; Adam: `7,385` bytes, `1.218 ms`, `1.046 ms`. Para `[8,64,32,1]`: SGD `22,955` bytes y Adam `66,712` bytes. La compresión y los tiempos dependen de datos/entorno; no se afirma una reducción ni ventaja universal.
