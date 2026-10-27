"""Deterministic coverage for Adam, portable checkpoints and epoch resumes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mlvectoranalysis.nn import (AdamOptimizer, HalfMSELoss, MLP, Parameter, SGDOptimizer,
                                 Trainer, TrainingConfig, load_checkpoint, save_checkpoint)


def test_adam_two_steps_and_failed_step_are_atomic():
    parameter = Parameter([1.])
    optimizer = AdamOptimizer([parameter], .1)
    parameter.grad[:] = 2.
    optimizer.step()
    assert_allclose(parameter.data, [.9000000005])
    parameter.grad[:] = 0.
    optimizer.step()
    assert optimizer.t == 2
    before = (parameter.data.copy(), optimizer.state_dict())
    parameter.grad[:] = np.inf
    with pytest.raises(FloatingPointError):
        optimizer.step()
    assert_array_equal(parameter.data, before[0])
    after = optimizer.state_dict()
    assert after['t'] == before[1]['t']
    assert_array_equal(after['m'][0], before[1]['m'][0])
    assert_array_equal(after['v'][0], before[1]['v'][0])


@pytest.mark.parametrize('optimizer_type', [SGDOptimizer, AdamOptimizer])
def test_checkpoint_roundtrip_and_exact_epoch_resume(tmp_path, optimizer_type):
    X = np.array([[.1], [.2], [.4], [.7], [1.]])
    y = X ** 2
    full_config = TrainingConfig(epochs=7, batch_size=2, seed=13, history_limit=20)
    direct_model = MLP([1, 3, 1], seed=8)
    direct_optimizer = optimizer_type(direct_model.parameters(), .03)
    direct = Trainer(HalfMSELoss(), direct_optimizer, full_config).fit(direct_model, X, y)

    first_model = MLP([1, 3, 1], seed=8)
    first_optimizer = optimizer_type(first_model.parameters(), .03)
    saved = []
    first = Trainer(HalfMSELoss(), first_optimizer, TrainingConfig(epochs=3, batch_size=2, seed=13, history_limit=20)).fit(
        first_model, X, y, checkpoint_callback=saved.append)
    checkpoint = tmp_path / 'resume.npz'
    info = save_checkpoint(checkpoint, model=first_model, optimizer=first_optimizer, training_state=saved[-1])
    assert info.bytes_written == checkpoint.stat().st_size > 0
    with np.load(checkpoint, allow_pickle=False) as archive:
        manifest = json.loads(archive['__mlvectoranalysis_manifest_utf8__'].tobytes())
        assert manifest['optimizer']['optimizer_type'] in {'sgd', 'adam'}

    resumed_model = MLP([1, 3, 1], seed=99)
    resumed_optimizer = optimizer_type(resumed_model.parameters(), .99)
    state = load_checkpoint(checkpoint, model=resumed_model, optimizer=resumed_optimizer)
    resumed = Trainer(HalfMSELoss(), resumed_optimizer, full_config).fit(
        resumed_model, X, y, resume_state=state, additional_epochs=4)
    assert resumed.history == direct.history
    assert (resumed.epochs_completed, resumed.optimizer_steps) == (direct.epochs_completed, direct.optimizer_steps)
    for left, right in zip(direct_model.parameters(), resumed_model.parameters()):
        assert_array_equal(left.data, right.data)
    if optimizer_type is AdamOptimizer:
        assert resumed_optimizer.t == direct_optimizer.t
        assert_array_equal(resumed_optimizer.state_dict()['m'][0], direct_optimizer.state_dict()['m'][0])
    assert first.optimizer_steps == 9


def test_checkpoint_rejection_does_not_modify_model_or_optimizer(tmp_path):
    model = MLP([1, 2, 1], seed=4)
    optimizer = AdamOptimizer(model.parameters(), .01)
    trainer = Trainer(HalfMSELoss(), optimizer, TrainingConfig(epochs=1, batch_size=1, shuffle=False))
    saved = []
    trainer.fit(model, [[1.]], [[0.]], checkpoint_callback=saved.append)
    path = tmp_path / 'bad.npz'
    save_checkpoint(path, model=model, optimizer=optimizer, training_state=saved[-1])
    original = [p.data.copy() for p in model.parameters()]
    path.write_bytes(b'not an npz')
    with pytest.raises(ValueError):
        load_checkpoint(path, model=model, optimizer=optimizer)
    for parameter, values in zip(model.parameters(), original):
        assert_array_equal(parameter.data, values)


def test_resume_cli_uses_only_temporary_local_paths(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'examples' / 'train_mlp_resume.py'
    checkpoint, first, resumed = (tmp_path / name for name in ('checkpoint.npz', 'first.json', 'resumed.json'))
    environment = os.environ | {'PYTHONPATH': str(root / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    common = [sys.executable, str(script), '--optimizer', 'sgd', '--checkpoint', str(checkpoint), '--save-every', '1']
    subprocess.run(common + ['--epochs', '2', '--output', str(first)], cwd=root, env=environment, check=True, capture_output=True, text=True)
    subprocess.run(common + ['--resume', '--epochs', '2', '--additional-epochs', '2', '--output', str(resumed)],
                   cwd=root, env=environment, check=True, capture_output=True, text=True)
    assert checkpoint.is_file()
    assert json.loads(resumed.read_text())['result']['epochs_completed'] == 4
