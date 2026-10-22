"""Optional presentation helpers; the neural-network core never imports plots."""
import argparse
import json
from pathlib import Path
from mlvectoranalysis.nn import TrainingConfig


def parser(description, name, batch_size, learning_rate):
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--epochs', type=int, default=1200)
    result.add_argument('--batch-size', type=int, default=batch_size)
    result.add_argument('--learning-rate', type=float, default=learning_rate)
    result.add_argument('--output', type=Path, default=Path(f'outputs/{name}.json'))
    result.add_argument('--plot', type=Path, help='PNG opcional, sin ventanas.')
    return result


def config(args):
    return TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, history_limit=150)


def report(args, summary, result, **extra):
    payload = dict(summary=summary, result=result.to_dict(), **extra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')


def loss_plot(ax, result):
    import math

    losses = []
    for field, label in (('train_loss', 'Entrenamiento'), ('val_loss', 'Validación independiente')):
        points = [(m.epoch, getattr(m, field)) for m in result.history
                  if getattr(m, field) is not None and math.isfinite(getattr(m, field))]
        if points:
            epochs, values = zip(*points)
            ax.plot(epochs, values, marker='.', markersize=3, label=label)
            losses.extend(values)
    # Preserve earlier validation metrics even if the final evaluation failed.
    # A zero loss is valid, but cannot be displayed on a logarithmic axis.
    scale = 'log' if losses and min(losses) > 0 else 'linear'
    ax.set(xlabel='Época', ylabel='Pérdida', yscale=scale, title='Pérdida con pesos fijos al evaluar')
    ax.grid(alpha=.2)
    if losses:
        ax.legend()
