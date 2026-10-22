"""Independent sin(x) samples on [-pi, pi]; training-only input statistics."""
import numpy as np
from mlvectoranalysis.nn import MLP, Trainer, SGDOptimizer, HalfMSELoss
from _mlp_common import parser, config, report, plotting, save, loss_plot


def main():
    cli = parser('Aprende sin(x) en [-π,π] con train/validation independientes.', 'mlp_sine', 32, .03)
    cli.add_argument('--n-train', type=int, default=128)
    cli.add_argument('--n-validation', type=int, default=256)
    args = cli.parse_args()
    configuration = config(args)
    if min(args.n_train, args.n_validation) < 2 or 64 * (args.n_train + args.n_validation) > 128 * 1024**2:
        cli.error('Use al menos 2 muestras por conjunto y una estimación de arrays <=128 MiB.')
    train_seed, validation_seed = np.random.SeedSequence(args.seed).spawn(2)
    X = np.random.default_rng(train_seed).uniform(-np.pi, np.pi, (args.n_train, 1))
    validation_X = np.random.default_rng(validation_seed).uniform(-np.pi, np.pi, (args.n_validation, 1))
    y, validation_y = np.sin(X), np.sin(validation_X)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    train_input, validation_input = (X - mean) / scale, (validation_X - mean) / scale
    grid = np.linspace(-np.pi, np.pi, 300)[:, None]
    model = MLP([1, 24, 24, 1], activation='tanh', seed=args.seed)
    trainer = Trainer(HalfMSELoss(), SGDOptimizer(model.parameters(), args.learning_rate), configuration)
    # At most 5 arrays of 300 predictions; no parameter/activation checkpoints.
    selected_epochs = set(np.linspace(0, args.epochs, 5, dtype=int).tolist())
    evolution = []
    def capture(snapshot):
        if snapshot.epoch in selected_epochs:
            evolution.append(dict(epoch=snapshot.epoch, predictions=model.predict((grid - mean) / scale).ravel().tolist()))
    result = trainer.fit(model, train_input, y, validation_data=(validation_input, validation_y), callback=capture)
    final = result.history[-1]
    summary = dict(seed=args.seed, architecture=[1, 24, 24, 1], interval=[-float(np.pi), float(np.pi)],
                   n_train=args.n_train, n_validation=args.n_validation, status=result.status,
                   train_loss=final.train_loss, validation_loss=final.val_loss,
                   initial_validation_rmse=float(np.sqrt(2 * result.history[0].val_loss)),
                   validation_rmse=float(np.sqrt(2 * final.val_loss)) if final.val_loss is not None else None,
                   epochs_completed=result.epochs_completed, optimizer_steps=result.optimizer_steps,
                   train_mean=mean.tolist(), train_scale=scale.tolist(), retained_prediction_arrays=len(evolution))
    report(args, summary, result, grid=grid.ravel().tolist(), evolution=evolution)
    if args.plot:
        plt = plotting()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        axes[0].plot(grid[:, 0], np.sin(grid[:, 0]), color='black', lw=2, label='sin(x) real')
        axes[0].scatter(X[:, 0], y[:, 0], s=12, color='gray', alpha=.4, label='Entrenamiento')
        for i, entry in enumerate(evolution):
            endpoint = i in (0, len(evolution) - 1)
            axes[0].plot(grid[:, 0], entry['predictions'], lw=1.7 if endpoint else 1,
                         alpha=1 if endpoint else .45, linestyle='--' if i == 0 else '-',
                         label=f'Predicción, época {entry["epoch"]}')
        rmse = summary['validation_rmse']
        rmse_label = f'{rmse:.4f}' if rmse is not None else 'no disponible'
        axes[0].set(xlabel='x (radianes)', ylabel='y', title=f'sin(x): RMSE validación = {rmse_label}')
        if result.status == 'non_finite':
            fig.suptitle('Entrenamiento detenido por un fallo numérico · solo se muestran datos válidos')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=.2)
        loss_plot(axes[1], result)
        save(fig, args.plot)
        plt.close(fig)


if __name__ == '__main__':
    main()
