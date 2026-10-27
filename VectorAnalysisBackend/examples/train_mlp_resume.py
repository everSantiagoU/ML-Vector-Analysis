"""Train or resume a small sin(x) MLP with safe local checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np

from mlvectoranalysis.nn import (AdamOptimizer, HalfMSELoss, MLP, SGDOptimizer, Trainer,
                                 TrainingConfig, load_checkpoint, save_checkpoint)
from mlvectoranalysis.nn.visualization import training_visualization


def arguments():
    parser = argparse.ArgumentParser(description='MLP NumPy resumible en límites de época (sin red).')
    parser.add_argument('--optimizer', choices=('sgd', 'adam'), default='adam')
    parser.add_argument('--epochs', type=int, default=40, help='Épocas de una ejecución nueva.')
    parser.add_argument('--additional-epochs', type=int, help='Épocas adicionales tras --resume.')
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/checkpoints/mlp_resume.npz'))
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/mlp_resume.json'))
    parser.add_argument('--plot', type=Path, help='PNG final opcional; requiere Matplotlib.')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = arguments()
    if args.epochs < 1 or args.save_every < 1 or (args.resume and (args.additional_epochs is None or args.additional_epochs < 1)):
        raise SystemExit('--epochs/--save-every deben ser positivos; --resume requiere --additional-epochs positivo.')
    rng = np.random.default_rng(args.seed)
    X = np.linspace(-np.pi, np.pi, 96, dtype=np.float64).reshape(-1, 1)
    y = np.sin(X)
    model = MLP([1, 16, 16, 1], activation='tanh', seed=args.seed)
    optimizer = (SGDOptimizer(model.parameters(), .03) if args.optimizer == 'sgd'
                 else AdamOptimizer(model.parameters(), .01))
    config = TrainingConfig(epochs=args.epochs, batch_size=16, shuffle=True, seed=args.seed, history_limit=100)
    trainer = Trainer(HalfMSELoss(), optimizer, config)
    resume_state = load_checkpoint(args.checkpoint, model=model, optimizer=optimizer) if args.resume else None
    target_epoch = (resume_state.epoch_completed + args.additional_epochs) if args.resume else args.epochs
    checkpoint_epochs = []

    def checkpoint_callback(state):
        if state.epoch_completed % args.save_every == 0 or state.epoch_completed == target_epoch:
            info = save_checkpoint(args.checkpoint, model=model, optimizer=optimizer, training_state=state,
                                   metadata={'experiment_name': 'train_mlp_resume', 'notes': 'local artifact; suitable for later upload'})
            checkpoint_epochs.append(state.epoch_completed)
            print(f'checkpoint epoch={state.epoch_completed} bytes={info.bytes_written}')

    result = trainer.fit(model, X, y, resume_state=resume_state, additional_epochs=args.additional_epochs,
                         checkpoint_callback=checkpoint_callback) if args.resume else trainer.fit(
                             model, X, y, checkpoint_callback=checkpoint_callback)
    payload = dict(result=result.to_dict(), visualization=training_visualization(
        result, checkpoint_epochs=checkpoint_epochs, optimizer_state=optimizer.state_dict()), checkpoint=str(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result.status, epochs_completed=result.epochs_completed,
                          optimizer_steps=result.optimizer_steps, final_loss=result.history[-1].train_loss), allow_nan=False))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        vis = payload['visualization']
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.plot(vis['epochs'], vis['train_loss'], marker='.', label='train')
        for epoch in checkpoint_epochs:
            axis.axvline(epoch, color='tab:green', alpha=.25)
        axis.set(xlabel='Época', ylabel='Half MSE', title=f"sin(x) — {args.optimizer}")
        axis.legend(); axis.grid(alpha=.2)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)


if __name__ == '__main__':
    main()
