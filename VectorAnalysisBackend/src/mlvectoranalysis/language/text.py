"""Deterministic local word/punctuation tokenization, without external models."""
import re
import unicodedata
import numpy as np
from ..nn._validation import integer


def token_ids(values, size, ndim=1):
    result = np.asarray(values)
    if result.ndim != ndim or result.dtype.kind not in 'iu' or 0 in result.shape:
        raise ValueError(f'IDs must be a nonempty {ndim}D integer array.')
    if np.any(result < 0) or np.any(result >= size):
        raise ValueError('IDs outside vocabulary range.')
    return result


def _token(token):
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise ValueError('Tokens must be nonempty strings without whitespace.')
    return token


class Vocabulary:
    PAD, UNK = '<PAD>', '<UNK>'
    pad_id, unk_id = 0, 1

    def __init__(self, tokens):
        if isinstance(tokens, (str, bytes)):
            raise ValueError('Expected an iterable of tokens, not text.')
        try:
            unique = {_token(t) for t in tokens}
        except TypeError as error:
            raise ValueError('Expected an iterable of tokens.') from error
        if not unique or unique.intersection((self.PAD, self.UNK)):
            raise ValueError('Vocabulary needs ordinary tokens; special tokens are reserved.')
        self.tokens = (self.PAD, self.UNK, *sorted(unique))
        self._ids = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def token_to_id(self, token):
        return self._ids.get(_token(token), self.unk_id)

    def id_to_token(self, index):
        index = integer(index, 'token ID', 0)
        if index >= len(self):
            raise ValueError('ID outside vocabulary range.')
        return self.tokens[index]

    def encode(self, tokens):
        if isinstance(tokens, (str, bytes)):
            raise ValueError('Expected tokens, not text.')
        try:
            result = np.array([self.token_to_id(t) for t in tokens], dtype=np.int64)
        except TypeError as error:
            raise ValueError('Expected an iterable of tokens.') from error
        return token_ids(result, len(self))

    def decode(self, ids):
        return [self.tokens[i] for i in token_ids(ids, len(self))]


class SimpleTokenizer:
    def __init__(self, vocabulary):
        if not isinstance(vocabulary, Vocabulary):
            raise ValueError('vocabulary must be Vocabulary.')
        self.vocabulary = vocabulary

    @staticmethod
    def tokenize(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Text must be a nonempty string.')
        return re.findall(r'\w+|[^\w\s]', unicodedata.normalize('NFC', text).lower())

    @classmethod
    def from_corpus(cls, documents):
        if isinstance(documents, (str, bytes)):
            raise ValueError('Corpus must be an iterable of documents.')
        try:
            return cls(Vocabulary(t for doc in documents for t in cls.tokenize(doc)))
        except TypeError as error:
            raise ValueError('Corpus must be an iterable of documents.') from error

    def encode(self, text):
        return self.vocabulary.encode(self.tokenize(text))

    def decode(self, ids):
        return ' '.join(self.vocabulary.decode(ids))
