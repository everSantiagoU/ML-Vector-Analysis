"""Reproducible training/evaluation split; scaler is fitted on training only."""
import argparse
import json
from pathlib import Path

import numpy as np

from mlvectoranalysis.ml import LinearRegressionGD
from mlvectoranalysis.optimization import GradientDescentConfig
from mlvectoranalysis.visualization import OptimizationTrajectory


def main():
    parser = argparse.ArgumentParser(description='Regresión lineal batch reproducible con evaluación independiente.')
    parser.add_argument('--n', type=int, default=500, help='Filas de entrenamiento.')
    parser.add_argument('--d', type=int, default=5, help='Número de features.')
    parser.add_argument('--n-eval', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=float, default=.05, help='Desviación estándar del ruido.')
    parser.add_argument('--block-size', type=int)
    parser.add_argument('--standardize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fit-intercept', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--strategy', choices=['auto', 'fixed', 'armijo', 'exact_quadratic'], default='auto')
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--max-iter', type=int, default=2000)
    parser.add_argument('--max-points', type=int, default=200)
    parser.add_argument('--axes', type=int, nargs='+', help='2–3 índices de parámetros; intercepto = d. Por defecto: 0 1.')
    parser.add_argument('--output', type=Path, default=Path('outputs/regression.json'))
    parser.add_argument('--plot', type=Path, help='PNG opcional; muestra una proyección de los parámetros optimizados.')
    args = parser.parse_args()
    if min(args.n, args.d, args.n_eval) <= 0 or not np.isfinite(args.noise) or args.noise < 0:
        parser.error('n, d y n-eval deben ser positivos; noise finito y no negativo.')
    estimated = 8*((args.n+args.n_eval)*args.d + 4*(args.n+args.n_eval)+10*args.d)
    if estimated > 256*1024**2:
        parser.error('Este ejemplo limita las asignaciones estimadas a 256 MiB; reduzca n o d.')
    dimension = args.d + int(args.fit_intercept)
    if args.axes is None:
        args.axes = [0, 1] if dimension >= 2 else []
    if args.axes and (len(args.axes) not in (2, 3) or len(set(args.axes)) != len(args.axes)
                      or min(args.axes) < 0 or max(args.axes) >= dimension):
        parser.error('Seleccione 2–3 índices distintos dentro de los parámetros.')
    if args.plot and not args.axes:
        parser.error('Un modelo de un parámetro exporta pérdidas; la gráfica de trayectoria requiere al menos dos.')
    rng = np.random.default_rng(args.seed)
    true_coef = rng.normal(size=args.d)
    true_intercept = 2.5 if args.fit_intercept else 0.
    feature_scale = np.linspace(1, 4, args.d)
    X = rng.normal(size=(args.n, args.d))
    X *= feature_scale
    X += 2
    targets = X@true_coef + true_intercept + rng.normal(scale=args.noise, size=args.n)
    X_eval = rng.normal(size=(args.n_eval, args.d))
    X_eval *= feature_scale
    X_eval += 2
    eval_signal = X_eval@true_coef + true_intercept
    eval_targets = eval_signal + rng.normal(scale=args.noise, size=args.n_eval)
    config = GradientDescentConfig(max_iter=args.max_iter, alpha=args.alpha, step_strategy=args.strategy,
                                   max_points=args.max_points, history_indices=tuple(args.axes) if args.axes else None)
    model = LinearRegressionGD(fit_intercept=args.fit_intercept, standardize=args.standardize,
                               block_size=args.block_size, config=config).fit(X, targets)
    result = model.optimization_result_
    prediction = model.predict(X_eval)
    summary = dict(n_train=args.n, n_eval=args.n_eval, d=args.d, seed=args.seed,
                   true_coef=true_coef.tolist(), coef=model.coef_.tolist(), true_intercept=true_intercept,
                   intercept=model.intercept_, coefficient_error=float(np.linalg.norm(model.coef_-true_coef)),
                   train_loss=result.loss, evaluation_loss=float(np.mean((prediction-eval_targets)**2)/2),
                   evaluation_signal_rmse=float(np.sqrt(np.mean((prediction-eval_signal)**2))),
                   n_iter=result.n_iter, status=result.status, dtype=str(X.dtype),
                   feature_mean=model.mean_.tolist(), feature_scale=model.scale_.tolist())
    trajectory = OptimizationTrajectory(result, axes=args.axes).to_dict()
    report = dict(summary=summary, result=result.to_dict(), trajectory=trajectory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))
    if args.plot:
        from _plotting import save_trajectory_plot
        save_trajectory_plot(trajectory, args.plot, 'Regresión batch: parámetros en coordenadas de optimización')


if __name__ == '__main__':
    main()
