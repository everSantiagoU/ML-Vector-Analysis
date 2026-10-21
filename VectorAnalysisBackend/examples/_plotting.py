"""Optional, headless plotting for examples; the core never imports Matplotlib."""
from pathlib import Path


def save_trajectory_plot(payload, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    figure = plt.figure(figsize=(12, 5))
    figure.subplots_adjust(left=.08, right=.92, bottom=.14, top=.82, wspace=.30)
    loss_axis = figure.add_subplot(1, 2, 1)
    loss_axis.plot(payload['iterations'], payload['losses'], marker='.', markersize=3)
    loss_axis.set(xlabel='Iteración', ylabel='Pérdida', title='Pérdida en muestras retenidas')
    loss_axis.grid(alpha=.2)
    points = np.asarray(payload['points'])
    axes = payload['axes']
    trajectory_axis = figure.add_subplot(1, 2, 2, projection='3d' if len(axes) == 3 else None)
    coordinates = [points[:, i] for i in range(len(axes))]
    trajectory_axis.plot(*coordinates, color='0.65', linewidth=1)
    colored = trajectory_axis.scatter(*coordinates, c=payload['losses'], cmap='viridis', s=20)
    trajectory_axis.set_xlabel(axes[0])
    trajectory_axis.set_ylabel(axes[1])
    if len(axes) == 3:
        trajectory_axis.set_zlabel(axes[2])
    label = 'Proyección' if payload['is_projection'] else 'Trayectoria discreta'
    system = {'standardized_parameters': 'parámetros estandarizados',
              'original_parameters': 'parámetros originales'}.get(payload['coordinate_system'], payload['coordinate_system'])
    trajectory_axis.set_title(f'{label} · {system}', fontsize=11)
    figure.colorbar(colored, ax=trajectory_axis, label='Pérdida', shrink=.65, pad=.12)
    figure.suptitle(title)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
