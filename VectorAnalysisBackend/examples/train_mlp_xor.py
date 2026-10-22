"""Capacity experiment on four XOR points, not an independent validation set."""
import numpy as np
from mlvectoranalysis.nn import MLP, Trainer, SGDOptimizer, CrossEntropyLoss
from _mlp_common import parser, config, report, plotting, save, loss_plot


def main():
    args = parser('XOR: MLP [2,8,2] con Tanh y cross-entropy; capacidad, no generalización.',
                  'mlp_xor', 4, .2).parse_args()
    configuration = config(args)
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = np.array([0, 1, 1, 0])
    model = MLP([2, 8, 2], activation='tanh', seed=args.seed)
    trainer = Trainer(CrossEntropyLoss(), SGDOptimizer(model.parameters(), args.learning_rate), configuration)
    result = trainer.fit(model, X, targets)
    summary = dict(seed=args.seed, architecture=[2, 8, 2], status=result.status,
                   loss=result.history[-1].train_loss, accuracy=result.history[-1].train_accuracy,
                   epochs_completed=result.epochs_completed, optimizer_steps=result.optimizer_steps,
                   points=X.tolist(), targets=targets.tolist(),
                   probabilities=model.predict_proba(X).tolist(), classes=model.predict_classes(X).tolist())
    report(args, summary, result)
    if args.plot:
        plt = plotting()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        loss_plot(axes[0], result)
        axis = np.linspace(-.3, 1.3, 160)
        xx, yy = np.meshgrid(axis, axis)
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        probability = model.predict_proba(grid, batch_size=256)[:, 1].reshape(xx.shape)
        field = axes[1].contourf(xx, yy, probability, levels=np.linspace(0, 1, 21), cmap='coolwarm', vmin=0, vmax=1)
        axes[1].contour(xx, yy, probability, levels=[.5], colors='black', linewidths=1)
        axes[1].scatter(X[:, 0], X[:, 1], c=targets, cmap='coolwarm', vmin=0, vmax=1,
                        s=100, edgecolors='black', label='4 puntos de entrenamiento', zorder=3)
        for point, target in zip(X, targets):
            axes[1].annotate(f'clase {target}', point + [.035, .045], fontsize=9)
        axes[1].set(xlabel='x₁', ylabel='x₂', title='XOR: P(clase 1), frontera = 0,5', aspect='equal')
        axes[1].legend(loc='lower center', fontsize=8)
        fig.colorbar(field, ax=axes[1], label='Probabilidad de clase 1')
        save(fig, args.plot)
        plt.close(fig)


if __name__ == '__main__':
    main()
