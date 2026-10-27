"""Fixed dataset/epoch work, separate warmup, repeated measurements and estimates."""
import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
from mlvectoranalysis.nn import (AdamOptimizer, HalfMSELoss, MLP, SGDOptimizer, Trainer, TrainingConfig,
                                 load_checkpoint, save_checkpoint)


class TimedTrainer(Trainer):
    evaluation_seconds = 0.
    evaluation_examples = 0

    def _evaluate_prepared(self, model, X, targets, batch_size):
        start = perf_counter()
        result = super()._evaluate_prepared(model, X, targets, batch_size)
        self.evaluation_seconds += perf_counter() - start
        self.evaluation_examples += len(X)
        return result


def trial(n, widths, batch_size, epochs, seed, optimizer_name='sgd'):
    start = perf_counter()
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, widths[0]))
    targets = np.sin(X[:, :widths[-1]])
    model = MLP(widths, seed=seed)
    optimizer = SGDOptimizer(model.parameters(), .01) if optimizer_name == 'sgd' else AdamOptimizer(model.parameters(), .01)
    trainer = TimedTrainer(HalfMSELoss(), optimizer,
                           TrainingConfig(epochs=epochs, batch_size=batch_size, seed=seed, history_limit=20))
    preparation = perf_counter() - start
    start = perf_counter()
    final_states = []
    result = trainer.fit(model, X, targets, checkpoint_callback=lambda state: final_states.append(state))
    fit_seconds = perf_counter() - start
    start = perf_counter()
    serialized = json.dumps(result.to_dict(), allow_nan=False)
    instrumentation = perf_counter() - start
    if result.status != 'max_epochs':
        raise RuntimeError(f'Benchmark did not complete declared work: {result.status}')
    B = min(n, batch_size)
    params = model.parameters()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'benchmark.npz'
        start = perf_counter()
        info = save_checkpoint(checkpoint, model=model, optimizer=optimizer, training_state=final_states[-1],
                               metadata={'experiment_name': 'benchmark_mlp'})
        checkpoint_save_seconds = perf_counter() - start
        restored = MLP(widths, seed=seed + 1000)
        restored_optimizer = SGDOptimizer(restored.parameters(), .01) if optimizer_name == 'sgd' else AdamOptimizer(restored.parameters(), .01)
        start = perf_counter()
        load_checkpoint(checkpoint, model=restored, optimizer=restored_optimizer)
        checkpoint_load_seconds = perf_counter() - start
    memory = dict(parameter_bytes=sum(p.data.nbytes for p in params),
                  gradient_bytes=sum(p.grad.nbytes for p in params),
                  optimizer_workspace_bytes=optimizer.workspace_bytes,
                  optimizer_state_bytes=getattr(optimizer, 'state_bytes', 0),
                  retained_cache_upper_bound_bytes=8 * B * (widths[0] + 2 * sum(widths[1:])),
                  backward_temporaries_upper_bound_bytes=8 * (B * 3 * max(widths) + max(a*b+b for a,b in zip(widths, widths[1:]))),
                  dataset_bytes=X.nbytes + targets.nbytes, input_dtype_conversion_bytes=0,
                  shuffle_index_bytes=n * np.dtype(np.intp).itemsize,
                  selected_batch_bytes=8 * B * (widths[0] + widths[-1]),
                  history_scalar_payload_upper_bound_bytes=len(result.history) * 7 * 8,
                  history_python_object_upper_bound_bytes=sys.getsizeof(result.history) + sum(
                      sys.getsizeof(m) + sys.getsizeof(vars(m)) + sum(sys.getsizeof(v) for v in vars(m).values())
                      for m in result.history),
                  serialized_result_bytes=len(serialized.encode()),
                  full_predict_result_bytes_if_requested=n * widths[-1] * 8)
    return dict(optimizer=optimizer_name, preparation_seconds=preparation, fit_seconds=fit_seconds,
                training_seconds=fit_seconds - trainer.evaluation_seconds,
                evaluation_seconds=trainer.evaluation_seconds, instrumentation_seconds=instrumentation,
                checkpoint_save_seconds=checkpoint_save_seconds, checkpoint_load_seconds=checkpoint_load_seconds,
                checkpoint_bytes=info.bytes_written,
                updates=result.optimizer_steps, examples_processed=epochs*n,
                evaluation_examples=trainer.evaluation_examples, epochs_completed=result.epochs_completed,
                final_train_loss=result.history[-1].train_loss, status=result.status, memory_estimates=memory)


def main():
    cli = argparse.ArgumentParser(description='MLP NumPy: dos redes × tamaños de lote, mediana de repeticiones.')
    cli.add_argument('--n', type=int, default=1024)
    cli.add_argument('--epochs', type=int, default=8)
    cli.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 64, 256])
    cli.add_argument('--repeats', type=int, default=3)
    cli.add_argument('--seed', type=int, default=42)
    cli.add_argument('--optimizers', choices=('sgd', 'adam'), nargs='+', default=['sgd', 'adam'])
    cli.add_argument('--output', type=Path, default=Path('outputs/benchmark_mlp.json'))
    args = cli.parse_args()
    networks = [[8, 16, 1], [8, 64, 32, 1]]
    if min(args.n, args.epochs, args.repeats, *args.batch_sizes) < 1 or args.seed < 0:
        cli.error('Tamaños, épocas y repeticiones positivos; seed no negativa.')
    # Cap before datasets, permutations, or large batch arrays are allocated.
    estimated = 8 * args.n * 12 + 8 * min(args.n, max(args.batch_sizes)) * 600 + 200_000
    if estimated > 128 * 1024**2 or args.epochs > 1000 or args.repeats > 20:
        cli.error('Límite: arrays estimados <=128 MiB, épocas <=1000 y repeticiones <=20.')
    cases = []
    for widths in networks:
        for batch in args.batch_sizes:
            for optimizer_name in args.optimizers:
                trial(64, widths, min(batch, 64), 2, args.seed + 1, optimizer_name)  # separate discarded warmup
                runs = [trial(args.n, widths, batch, args.epochs, args.seed, optimizer_name) for _ in range(args.repeats)]
                timings = {key: statistics.median(run[key] for run in runs) for key in (
                    'preparation_seconds', 'fit_seconds', 'training_seconds', 'evaluation_seconds', 'instrumentation_seconds',
                    'checkpoint_save_seconds', 'checkpoint_load_seconds', 'checkpoint_bytes')}
                cases.append(dict(widths=widths, batch_size=batch, optimizer=optimizer_name, medians=timings, runs=runs))
    payload = dict(config=vars(args) | {'output': str(args.output)}, dtype='float64',
        versions=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
        blas=np.__config__.show(mode='dicts'),
        thread_environment={key: os.environ.get(key, 'library default') for key in
                            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        notes=['Fixed N and E, same generated data/initial weights per repetition; batch size changes updates and trajectory.',
               'Training time = fit wall time minus timed evaluation; includes validation, loss/backward, SGD and orchestration.',
               'Evaluation includes initial and every completed epoch, fixed weights, no optimizer updates.',
               'Instrumentation time measures result serialization only; timer/history overhead remains in fit.',
               'Warmup is a different model/dataset and excluded; sequential repetitions in one process.',
               'Memory counts explicit arrays and conservative cache/backward bounds, not measured RSS or tracemalloc.',
               'Python/allocator/BLAS internals and transient validation boolean arrays are not fully counted; estimates are not total RSS.',
               'Float64 input conversion copies are zero here; other numeric dtypes require one resident copy.',
               'predict output is hypothetical; evaluation retains only one batch. No speedup or learning superiority claim.'],
        cases=cases)
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + '\n', encoding='utf-8')
    for case in cases:
        print(json.dumps(dict(widths=case['widths'], batch_size=case['batch_size'], **case['medians'],
                              optimizer=case['optimizer'], updates=case['runs'][0]['updates'],
                              examples_processed=case['runs'][0]['examples_processed']), allow_nan=False))


if __name__ == '__main__':
    main()
