"""Integration checks for optional plotting, including numeric training failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('rate,status', [('0.03', 'max_epochs'), ('1e308', 'non_finite')])
def test_sine_cli_exports_plot_and_strict_json_on_success_or_failure(tmp_path, rate, status):
    pytest.importorskip('matplotlib', reason='PNG examples require requirements-examples.txt')
    report = tmp_path / 'sine.json'
    plot = tmp_path / 'sine.png'
    env = dict(os.environ, PYTHONPATH=str(BACKEND / 'src'), PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run(
        [sys.executable, str(BACKEND / 'examples/train_mlp_sine.py'), '--epochs', '2',
         '--learning-rate', rate, '--output', str(report), '--plot', str(plot)],
        cwd=BACKEND, env=env, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'Traceback' not in completed.stderr
    payload = json.loads(report.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert payload['result']['status'] == status
    assert json.loads(completed.stdout) == payload['summary']
    assert plot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    if status == 'non_finite':
        assert payload['summary']['validation_rmse'] is None
        assert payload['result']['history'][-1]['val_loss'] is None
        assert payload['result']['history'][0]['val_loss'] is not None
    else:
        assert payload['summary']['validation_rmse'] is not None


@pytest.mark.parametrize('initial_loss', [0.0, 0.5])
def test_loss_plot_keeps_valid_history_when_final_metrics_are_missing(monkeypatch, initial_loss):
    pytest.importorskip('matplotlib', reason='PNG examples require requirements-examples.txt')
    monkeypatch.syspath_prepend(str(BACKEND / 'examples'))
    from _mlp_common import loss_plot, plotting

    plt = plotting()
    fig, ax = plt.subplots()
    history = [SimpleNamespace(epoch=0, train_loss=initial_loss, val_loss=initial_loss),
               SimpleNamespace(epoch=0, train_loss=None, val_loss=None)]
    try:
        loss_plot(ax, SimpleNamespace(history=history))
        assert [line.get_label() for line in ax.lines] == ['Entrenamiento', 'Validación independiente']
        for line in ax.lines:
            assert list(line.get_xdata()) == [0]
            assert list(line.get_ydata()) == [initial_loss]
        assert ax.get_yscale() == ('linear' if initial_loss == 0 else 'log')
    finally:
        plt.close(fig)
