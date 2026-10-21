"""Moderate benchmark; each case gets a fresh process and separate warmup."""
import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np
import sympy

from mlvectoranalysis.optimization import GradientDescent, GradientDescentConfig, LinearLeastSquaresObjective


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def worker(args):
    rng = np.random.default_rng(args.seed)
    warm_X = rng.normal(size=(128, 8))
    warm = LinearLeastSquaresObjective(warm_X, rng.normal(size=128))
    GradientDescent(GradientDescentConfig(max_iter=3, alpha=.01, step_strategy='fixed',
                                         history_mode='none')).minimize(warm, np.zeros(9))
    del warm, warm_X
    gc.collect()
    tracemalloc.start()
    preparation_start = perf_counter()
    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, args.d))
    targets = X@rng.normal(size=args.d)+2+rng.normal(size=args.n)*.1
    block = args.block_size if args.worker.startswith('block') else None
    objective = LinearLeastSquaresObjective(X, targets, block_size=block, standardize=args.standardize)
    preparation_seconds = perf_counter()-preparation_start
    _, prep_peak = tracemalloc.get_traced_memory()
    history = 'sampled' if args.worker.endswith('on') else 'none'
    # Disable tolerance-based early stopping for this controlled-work benchmark.
    # Actual counts/status are still reported; numerical stagnation can stop it.
    config = GradientDescentConfig(max_iter=args.steps, step_strategy=args.strategy, alpha=args.alpha,
        gradient_atol=0, gradient_rtol=0, step_rtol=0, loss_rtol=0,
        history_mode=history, history_indices=(0, 1), max_points=args.max_points)
    initial = np.zeros(objective.dimension)
    rss_before = peak_rss_bytes()
    traced_before, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    start = perf_counter()
    result = GradientDescent(config).minimize(objective, initial)
    seconds = perf_counter()-start
    _, training_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_peak = peak_rss_bytes()
    snapshot_bytes = sum(s.parameters.nbytes for s in result.snapshots if s.parameters is not None)
    return dict(case=args.worker, seed=args.seed, n=args.n, d=args.d, block_size=block,
                standardize=args.standardize, strategy=result.step_strategy, alpha=args.alpha,
                requested_steps=args.steps, actual_steps=result.n_iter, status=result.status,
                function_evaluations=result.n_function_evals, gradient_evaluations=result.n_gradient_evals,
                curvature_evaluations=result.n_curvature_evals,
                preparation_seconds=preparation_seconds, training_seconds=seconds,
                seconds_per_accepted_step=seconds/result.n_iter if result.n_iter else None,
                loss=result.loss, gradient_norm=result.gradient_norm,
                dtype=str(X.dtype), X_shape=list(X.shape), X_bytes=X.nbytes, targets_bytes=targets.nbytes,
                objective_workspace_bytes=objective._residual.nbytes+objective._work_gradient.nbytes,
                snapshots=len(result.snapshots), snapshot_coordinate_bytes=snapshot_bytes,
                tracemalloc_preparation_peak_bytes=prep_peak,
                tracemalloc_training_peak_increment_bytes=max(0, training_peak-traced_before),
                process_peak_rss_bytes=rss_peak, process_highwater_growth_training_bytes=max(0, rss_peak-rss_before))


def main():
    parser = argparse.ArgumentParser(description='Benchmark moderado full/block × historial on/off, aislado por proceso.')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--max-points', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--strategy', choices=['fixed', 'exact_quadratic', 'armijo'], default='fixed')
    parser.add_argument('--standardize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=Path, help='JSON opcional; stdout siempre incluye el reporte.')
    parser.add_argument('--worker', choices=['full_off', 'full_on', 'block_off', 'block_on'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.n, args.d, args.steps, args.block_size) <= 0:
        parser.error('n, d, steps y block-size deben ser positivos.')
    estimated = 8*(args.n*args.d + 8*args.n + 12*args.d + 2*args.max_points)
    if estimated > 128*1024**2:
        parser.error(f'Estimación de arrays {estimated} bytes > límite seguro 128 MiB; reduzca n o d.')
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    cases = []
    for name in ('full_off', 'full_on', 'block_off', 'block_on'):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', name,
                   '--n', str(args.n), '--d', str(args.d), '--steps', str(args.steps),
                   '--block-size', str(args.block_size), '--max-points', str(args.max_points),
                   '--seed', str(args.seed), '--strategy', args.strategy, '--alpha', str(args.alpha),
                   '--standardize' if args.standardize else '--no-standardize']
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        cases.append(json.loads(completed.stdout))
    cpu = platform.processor() or platform.machine()
    cpu_probe_error = None
    if sys.platform == 'darwin':
        probe = subprocess.run(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True)
        if probe.returncode == 0:
            cpu = probe.stdout.strip()
        else:
            cpu_probe_error = probe.stderr.strip()
    report = dict(environment=dict(python=sys.version, numpy=np.__version__, sympy=sympy.__version__,
                                   platform=platform.platform(), cpu=cpu, logical_cpus=os.cpu_count(),
                                   cpu_probe_error=cpu_probe_error,
                                   blas_configuration=np.__config__.show(mode='dicts'),
                                   thread_environment={name: os.environ.get(name, 'unset (library default)') for name in
                                       ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                        'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}),
                  estimated_array_bytes=estimated,
                  measurement_notes=['One fresh worker process per case, warmup excluded.',
                      'Timings include tracemalloc instrumentation and one run; no universal speed claims.',
                      'RSS is process high-water, not incremental live allocation; growth may be zero.',
                      'tracemalloc observes registered allocations, not necessarily all native BLAS buffers.',
                      'Targets and dense X stay in memory in block mode.',
                      'History on stores two projected coordinates; final parameters remain complete.'], cases=cases)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
