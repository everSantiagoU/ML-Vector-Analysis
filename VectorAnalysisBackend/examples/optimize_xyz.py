"""Run from the backend: PYTHONPATH=src python examples/optimize_xyz.py --help."""
import argparse
import json
from pathlib import Path

from mlvectoranalysis.domain import MultivariableFunction
from mlvectoranalysis.optimization import GradientDescent, GradientDescentConfig
from mlvectoranalysis.visualization import OptimizationTrajectory


def main():
    parser = argparse.ArgumentParser(description='Descenso de f(x,y,z), JSON y gráfico opcional sin ventana.')
    parser.add_argument('--expression', default='(x-1)^2+2*(y+2)^2+3*(z-0.5)^2')
    parser.add_argument('--initial', type=float, nargs=3, default=[4, 4, 4], metavar=('X', 'Y', 'Z'))
    parser.add_argument('--strategy', choices=['auto', 'fixed', 'armijo'], default='auto')
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--max-points', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('outputs/xyz.json'))
    parser.add_argument('--plot', type=Path, help='PNG opcional; requiere requirements-examples.txt.')
    args = parser.parse_args()
    objective = MultivariableFunction.from_text(args.expression)
    config = GradientDescentConfig(max_iter=args.max_iter, step_strategy=args.strategy,
                                   alpha=args.alpha, max_points=args.max_points)
    result = GradientDescent(config).minimize(objective, args.initial)
    trajectory = OptimizationTrajectory(result).to_dict()
    report = dict(expression=objective.expression, symbolic_gradient=[str(g) for g in objective.symbolic_gradient],
                  initial=args.initial, initial_loss=objective.value(args.initial),
                  result=result.to_dict(), trajectory=trajectory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(initial_loss=report['initial_loss'], final_parameters=result.parameters.tolist(),
                          final_loss=result.loss, gradient_norm=result.gradient_norm, n_iter=result.n_iter,
                          status=result.status, function_evaluations=result.n_function_evals,
                          gradient_evaluations=result.n_gradient_evals, output=str(args.output)), indent=2, allow_nan=False))
    if args.plot:
        from _plotting import save_trajectory_plot
        save_trajectory_plot(trajectory, args.plot, 'Descenso de una función escalar f(x,y,z)')


if __name__ == '__main__':
    main()
