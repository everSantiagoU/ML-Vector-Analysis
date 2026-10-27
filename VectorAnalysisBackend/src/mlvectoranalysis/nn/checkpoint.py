"""Safe local MLP checkpoints: UTF-8 JSON manifest plus ``.npz`` arrays.

The module intentionally has no cloud/provider imports.  Files are written to
one temporary file in the destination directory and atomically replaced only
after NumPy has finished writing it.
"""
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from datetime import datetime, timezone
import json
import os
import platform
import tempfile
import zipfile
import numpy as np

from ._validation import DTYPE, finite, integer
from .training_result import EpochMetrics

_FORMAT_VERSION = 1
_MANIFEST_KEY = '__mlvectoranalysis_manifest_utf8__'


def _json_copy(value, name='value'):
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must contain only finite JSON values.') from error
    return json.loads(encoded)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class TrainingState:
    """Only epoch-boundary training data; never datasets, gradients or caches."""
    epoch_completed: int
    optimizer_steps: int
    rng_state: Mapping
    history: tuple[EpochMetrics, ...]
    config: Mapping
    history_state: Mapping | None = None

    def __post_init__(self):
        object.__setattr__(self, 'epoch_completed', integer(self.epoch_completed, 'epoch_completed', 0))
        object.__setattr__(self, 'optimizer_steps', integer(self.optimizer_steps, 'optimizer_steps', 0))
        if not isinstance(self.history, tuple) or not all(isinstance(item, EpochMetrics) for item in self.history):
            raise ValueError('history must be a tuple of EpochMetrics.')
        rng = _json_copy(_thaw(self.rng_state), 'rng_state')
        config = _json_copy(_thaw(self.config), 'config')
        if not isinstance(rng, dict) or not isinstance(config, dict):
            raise ValueError('rng_state and config must be JSON objects.')
        history_state = None if self.history_state is None else _json_copy(_thaw(self.history_state), 'history_state')
        object.__setattr__(self, 'rng_state', _freeze(rng))
        object.__setattr__(self, 'config', _freeze(config))
        object.__setattr__(self, 'history_state', None if history_state is None else _freeze(history_state))

    def to_dict(self):
        return dict(epoch_completed=self.epoch_completed, optimizer_steps=self.optimizer_steps,
                    rng_state=_thaw(self.rng_state), history=[asdict(item) for item in self.history],
                    config=_thaw(self.config), history_state=None if self.history_state is None else _thaw(self.history_state))

    @classmethod
    def from_dict(cls, value):
        expected = {'epoch_completed', 'optimizer_steps', 'rng_state', 'history', 'config', 'history_state'}
        if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value['history'], list):
            raise ValueError('Training state is invalid.')
        try:
            history = tuple(EpochMetrics(**item) for item in value['history'])
        except (TypeError, ValueError) as error:
            raise ValueError('Training history is invalid.') from error
        return cls(value['epoch_completed'], value['optimizer_steps'], value['rng_state'], history,
                   value['config'], value['history_state'])


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    bytes_written: int
    manifest: Mapping


def _record(parameter, key):
    return dict(name=parameter.name, shape=list(parameter.data.shape), dtype=parameter.data.dtype.name, key=key)


def _manifest_and_arrays(model, optimizer, training_state, metadata):
    if not isinstance(training_state, TrainingState):
        raise ValueError('training_state must be TrainingState.')
    supplied = {} if metadata is None else _json_copy(metadata, 'metadata')
    if not isinstance(supplied, dict):
        raise ValueError('metadata must be a JSON object.')
    supplied.setdefault('timestamp_utc', datetime.now(timezone.utc).isoformat())
    supplied.setdefault('python_version', platform.python_version())
    supplied.setdefault('numpy_version', np.__version__)
    model_state, optimizer_state, arrays = model.state_dict(), optimizer.state_dict(), {}
    model_records = []
    for index, (parameter, entry) in enumerate(zip(model.parameters(), model_state['parameters'])):
        key = f'model.parameter.{index}'
        arrays[key] = entry['data']
        model_records.append(_record(parameter, key))
    model_manifest = {key: value for key, value in model_state.items() if key != 'parameters'} | {'parameters': model_records}
    optimizer_manifest = {key: value for key, value in optimizer_state.items() if key not in {'m', 'v'}}
    if optimizer_state['optimizer_type'] == 'adam':
        m_keys, v_keys = [], []
        for index, (m, v) in enumerate(zip(optimizer_state['m'], optimizer_state['v'])):
            m_key, v_key = f'optimizer.m.{index}', f'optimizer.v.{index}'
            arrays[m_key], arrays[v_key] = m, v
            m_keys.append(m_key); v_keys.append(v_key)
        optimizer_manifest['m_keys'], optimizer_manifest['v_keys'] = m_keys, v_keys
    manifest = dict(format_version=_FORMAT_VERSION, metadata=supplied, training_state=training_state.to_dict(),
                    model=model_manifest, optimizer=optimizer_manifest)
    # This also guarantees no accidental object arrays enter the archive.
    _json_copy(manifest, 'manifest')
    return manifest, arrays


def save_checkpoint(path, *, model, optimizer, training_state, metadata=None):
    """Write a portable checkpoint atomically; only the exact target is replaced."""
    destination = Path(path)
    if not destination.name:
        raise ValueError('Checkpoint path must name a file.')
    manifest, arrays = _manifest_and_arrays(model, optimizer, training_state, metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', dir=destination.parent, prefix=f'.{destination.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            manifest_bytes = json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True).encode('utf-8')
            np.savez_compressed(handle, **{_MANIFEST_KEY: np.frombuffer(manifest_bytes, dtype=np.uint8)}, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        return CheckpointInfo(destination, destination.stat().st_size, MappingProxyType(_json_copy(manifest)))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _read_array(archive, record):
    if not isinstance(record, Mapping) or set(record) != {'name', 'shape', 'dtype', 'key'} or not isinstance(record['key'], str):
        raise ValueError('Checkpoint parameter record is invalid.')
    data = archive[record['key']]
    if data.dtype != DTYPE or tuple(record['shape']) != data.shape or record['dtype'] != np.dtype(DTYPE).name:
        raise ValueError('Checkpoint array shape or dtype is invalid.')
    finite(data)
    return data.copy()


def _states_from_manifest(archive, manifest):
    expected = {'format_version', 'metadata', 'training_state', 'model', 'optimizer'}
    if not isinstance(manifest, dict) or set(manifest) != expected or manifest['format_version'] != _FORMAT_VERSION:
        raise ValueError('Checkpoint manifest is incompatible.')
    if not isinstance(manifest['metadata'], dict):
        raise ValueError('Checkpoint metadata is invalid.')
    training_state = TrainingState.from_dict(manifest['training_state'])
    model_manifest, optimizer_manifest = manifest['model'], manifest['optimizer']
    if not isinstance(model_manifest, dict) or not isinstance(optimizer_manifest, dict):
        raise ValueError('Checkpoint model or optimizer section is invalid.')
    model_state = {key: value for key, value in model_manifest.items() if key != 'parameters'}
    if not isinstance(model_manifest.get('parameters'), list):
        raise ValueError('Checkpoint model parameters are invalid.')
    if any(not isinstance(record, Mapping) or set(record) != {'name', 'shape', 'dtype', 'key'}
           for record in model_manifest['parameters']):
        raise ValueError('Checkpoint model parameter records are invalid.')
    model_state['parameters'] = [dict(name=record['name'], shape=record['shape'], dtype=record['dtype'], data=_read_array(archive, record))
                               for record in model_manifest['parameters']]
    optimizer_state = {key: value for key, value in optimizer_manifest.items() if key not in {'m_keys', 'v_keys'}}
    if optimizer_state.get('optimizer_type') == 'adam':
        m_keys, v_keys = optimizer_manifest.get('m_keys'), optimizer_manifest.get('v_keys')
        parameters = optimizer_state.get('parameters')
        if (not isinstance(m_keys, list) or not isinstance(v_keys, list) or not isinstance(parameters, list)
                or len(m_keys) != len(parameters) or len(v_keys) != len(parameters)
                or any(not isinstance(record, Mapping) or set(record) != {'name', 'shape', 'dtype'} for record in parameters)):
            raise ValueError('Checkpoint Adam moment keys are invalid.')
        optimizer_state['m'] = [_read_array(archive, dict(name=record['name'], shape=record['shape'], dtype=record['dtype'], key=key))
                                for record, key in zip(parameters, m_keys)]
        optimizer_state['v'] = [_read_array(archive, dict(name=record['name'], shape=record['shape'], dtype=record['dtype'], key=key))
                                for record, key in zip(parameters, v_keys)]
    elif 'm_keys' in optimizer_manifest or 'v_keys' in optimizer_manifest:
        raise ValueError('Unexpected optimizer arrays in checkpoint.')
    expected_keys = {_MANIFEST_KEY}
    expected_keys.update(record['key'] for record in model_manifest['parameters'])
    expected_keys.update(optimizer_manifest.get('m_keys', ()))
    expected_keys.update(optimizer_manifest.get('v_keys', ()))
    if set(archive.files) != expected_keys:
        raise ValueError('Checkpoint contains missing or unexpected array keys.')
    return training_state, model_state, optimizer_state


def load_checkpoint(path, *, model, optimizer):
    """Validate an entire archive before changing the supplied model/optimizer."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f'Checkpoint does not exist: {source}')
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) and _MANIFEST_KEY not in archive.files:
                raise ValueError('Checkpoint manifest is missing.')
            raw = archive[_MANIFEST_KEY]
            if raw.dtype != np.uint8 or raw.ndim != 1:
                raise ValueError('Checkpoint manifest is invalid.')
            manifest = json.loads(raw.tobytes().decode('utf-8'))
            state, model_state, optimizer_state = _states_from_manifest(archive, manifest)
    except (OSError, ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f'Unable to load checkpoint: {error}') from error
    # Both validators allocate/copy and check compatibility before either object mutates.
    model._validated_state(model_state)
    optimizer._validated_state(optimizer_state)
    if getattr(model, '_live_contexts', ()):
        raise ValueError('Cannot load checkpoint while retained contexts are live.')
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return state
