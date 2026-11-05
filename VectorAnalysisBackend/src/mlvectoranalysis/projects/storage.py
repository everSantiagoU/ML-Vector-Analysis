"""Local single-writer JSON store. No pickle, network, archives, or user paths."""
from pathlib import Path
import hashlib
import json
import unicodedata
import os
import platform
import re
import tempfile

from .dto import ProjectError, Snapshot, integer, loads, require, safe_name


def _parts(payload: dict):
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':'))
    for part in encoder.iterencode(payload):
        yield part.encode('utf-8')


def _digest(payload: dict) -> str:
    digest = hashlib.sha256()
    for part in _parts(payload):
        digest.update(part)
    return digest.hexdigest()


class LocalArtifactStore:
    def __init__(self, directory: str | Path, *, max_artifact_bytes: int = 64_000_000):
        integer(max_artifact_bytes, 1, 1_000_000_000)
        configured = Path(directory).absolute()
        require(not configured.is_symlink(), 'Directorio simbólico no permitido.', 'unsafe_path')
        # Canonicalize OS aliases such as macOS /var -> /private/var once.
        self.root = configured.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_artifact_bytes = max_artifact_bytes

    def _path(self, project_id: str, name: str = 'project.json') -> Path:
        require(isinstance(project_id, str) and re.fullmatch('[a-f0-9]{32}', project_id) is not None,
                'ID de proyecto inseguro.', 'unsafe_path')
        safe_name(name)
        folder = self.root / project_id
        destination = folder / name
        require(self.root.resolve() == self.root and not self.root.is_symlink() and not folder.is_symlink() and not destination.is_symlink(),
                'Enlace simbólico no permitido.', 'unsafe_path')
        require(destination.resolve().is_relative_to(self.root.resolve()), 'Ruta fuera del almacén.', 'unsafe_path')
        return destination

    def _write(self, path: Path, payload: dict) -> None:
        temporary = None
        try:
            path.parent.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as handle:
                temporary = Path(handle.name)
                written = 0
                for part in _parts(payload):
                    written += len(part)
                    require(written <= self.max_artifact_bytes, 'Artefacto excede límite.', 'artifact_limit')
                    handle.write(part)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError as error:
            raise ProjectError('storage_io', 'No se pudo guardar el artefacto.') from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save(self, snapshot: Snapshot) -> None:
        require(isinstance(snapshot, Snapshot), 'Snapshot inválido.')
        payload = snapshot.to_dict()
        self._write(self._path(snapshot.project.id), {
            'format_version': 1, 'sha256': _digest(payload),
            'snapshot': payload,
        })

    def load(self, project_id: str) -> Snapshot:
        path = self._path(project_id)
        try:
            with path.open('rb') as handle:
                content = handle.read(self.max_artifact_bytes + 1)
            require(len(content) <= self.max_artifact_bytes, 'Artefacto excede límite.', 'artifact_limit')
            envelope = loads(content.decode('utf-8'))
            require(isinstance(envelope, dict) and set(envelope) == {'format_version', 'sha256', 'snapshot'}
                    and type(envelope['format_version']) is int and envelope['format_version'] == 1,
                    'Formato de artefacto inválido.', 'corrupt_artifact')
            digest = _digest(envelope['snapshot'])
            require(envelope['sha256'] == digest, 'Integridad del artefacto inválida.', 'corrupt_artifact')
            snapshot = Snapshot.from_dict(envelope['snapshot'])
            require(snapshot.to_dict() == envelope['snapshot'], 'Metadatos incompletos.', 'corrupt_artifact')
            require(snapshot.project.id == project_id, 'Proyecto incompatible.', 'corrupt_artifact')
            return snapshot
        except FileNotFoundError as error:
            raise ProjectError('not_found', 'Proyecto no encontrado.') from error
        except (UnicodeError, OSError) as error:
            raise ProjectError('storage_io', 'No se pudo leer el artefacto.') from error

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(p.name for p in self.root.iterdir()
                            if re.fullmatch('[a-f0-9]{32}', p.name) and self._path(p.name).is_file()))

    def export(self, project_id: str) -> str:
        snapshot = self.load(project_id)
        payload = snapshot.to_dict()
        manifest = {
            'format_version': 1,
            'export_kind': 'configuration_and_normalized_data',
            'trained_model': False,
            'versions': {'backend': 'local-projects-v1', 'python': platform.python_version(),
                         'normalization': 'NFC-LF-v1', 'unicode': unicodedata.unidata_version, 'index': snapshot.project.index_version,
                         'provider': snapshot.project.provider_version},
            'snapshot_sha256': _digest(payload),
            'snapshot': payload,
        }
        path = self._path(project_id, 'export.json')
        self._write(path, manifest)
        return str(path)
