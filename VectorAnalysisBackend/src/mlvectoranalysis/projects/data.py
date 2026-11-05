"""Bounded UTF-8 adapters. Imported content is never evaluated or rendered."""
import hashlib
import html
from pathlib import PurePath
import unicodedata

from .dto import (Config, Dataset, Document, EvaluationCase, FileMetadata,
                  ProjectError, loads, require, safe_name)


def decode(name: str, content: bytes, config: Config, extensions: tuple[str, ...]) -> tuple[FileMetadata, str]:
    safe_name(name)
    require(isinstance(config, Config), 'Configuración inválida.')
    require(PurePath(name).suffix.lower() in extensions, 'Formato no soportado.', 'unsupported_format')
    require(type(content) is bytes and 0 < len(content) <= config.max_file_bytes,
            'Archivo vacío o excede límite de bytes.', 'file_limit')
    try:
        value = content.decode('utf-8-sig', errors='strict')
    except UnicodeDecodeError as error:
        raise ProjectError('encoding', 'Se requiere UTF-8 válido.') from error
    require(not any(unicodedata.category(c) == 'Cc' and c not in '\n\r\t' for c in value),
            'Contenido con controles binarios.', 'encoding')
    digest = hashlib.sha256(content).hexdigest()
    identifier = hashlib.sha256(name.encode('utf-8') + b'\0' + content).hexdigest()
    kind = 'document' if PurePath(name).suffix.lower() in ('.txt', '.md') else 'evaluation'
    return FileMetadata(identifier, name, len(content), digest, kind), value


class TextAdapter:
    extensions = ('.txt', '.md')

    def parse(self, name: str, content: bytes, config: Config) -> Dataset:
        source, value = decode(name, content, config, self.extensions)
        # Offsets refer to this canonical text, not to UTF-8 bytes in the upload.
        normalized = unicodedata.normalize('NFC', value.replace('\r\n', '\n').replace('\r', '\n'))
        require(len(normalized) <= config.max_document_chars, 'Límite de caracteres excedido.', 'file_limit')
        return Dataset(source, (Document(source, normalized),))


class EvaluationAdapter:
    extensions = ('.json', '.jsonl')

    def parse(self, name: str, content: bytes, config: Config) -> Dataset:
        source, value = decode(name, content, config, self.extensions)
        if PurePath(name).suffix.lower() == '.json':
            rows = loads(value)
            require(isinstance(rows, list), 'JSON de evaluación debe ser una lista.')
        else:
            rows = []
            for line in value.splitlines():
                require(bool(line.strip()), 'JSONL contiene una línea vacía.')
                require(len(rows) < config.max_cases, 'Demasiados casos.', 'case_limit')
                rows.append(loads(line))
        require(0 < len(rows) <= config.max_cases, 'Cantidad de casos inválida.', 'case_limit')
        cases = tuple(EvaluationCase.from_dict(row) for row in rows)
        require(len({c.id for c in cases}) == len(cases), 'IDs de caso duplicados.')
        require(all(len(c.question) <= config.question_chars for c in cases), 'Pregunta excede límite.')
        return Dataset(source, cases=cases)


def preview(document: Document, config: Config) -> dict:
    # Escape after truncation, never parse Markdown or return active HTML.
    return {'source': document.source.to_dict(), 'text': html.escape(document.content[:config.preview_chars]),
            'encoding': 'html_escaped_text', 'truncated': len(document.content) > config.preview_chars,
            'characters': len(document.content), 'offset_basis': 'normalized_unicode_characters'}
