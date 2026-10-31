import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from mlvectoranalysis.language import Vocabulary, SimpleTokenizer, ContextDataset, Embedding, ContextLanguageModel, WindowModelAdapter
from mlvectoranalysis.nn import Trainer, TrainingConfig, AdamOptimizer, CrossEntropyLoss
from mlvectoranalysis.nn.losses import softmax


def test_tokenizer_vocabulary():
    tok = SimpleTokenizer.from_corpus(['¡HOLA, café! hola', 'adiós'])
    assert tok.tokenize('  HOLA,\t cafe\u0301!\n') == ['hola', ',', 'café', '!']
    assert tok.decode(tok.encode('HOLA, café!')) == 'hola , café !'
    assert tok.encode('desconocido').tolist() == [1]
    assert tok.decode([0, 1]) == '<PAD> <UNK>'
    assert tok.vocabulary.tokens == SimpleTokenizer.from_corpus(['adiós', '¡HOLA, café! hola']).vocabulary.tokens
    assert tok.vocabulary.tokens[:2] == ('<PAD>', '<UNK>')
    assert tok.vocabulary.token_to_id('ausente') == 1
    assert_array_equal(tok.encode(tok.decode(tok.encode('hola!'))), tok.encode('hola!'))


@pytest.mark.parametrize('bad', ['', '  ', None, 3, []])
def test_bad_text(bad):
    with pytest.raises(ValueError):
        SimpleTokenizer.tokenize(bad)


@pytest.mark.parametrize('bad', [[], [''], ['a b'], ['<PAD>'], 'abc', None, [5]])
def test_bad_vocabulary(bad):
    with pytest.raises(ValueError):
        Vocabulary(bad)


@pytest.mark.parametrize('bad', [[], [''], 'abc', None])
def test_bad_corpus(bad):
    with pytest.raises(ValueError):
        SimpleTokenizer.from_corpus(bad)


@pytest.mark.parametrize('bad', [[], [-1], [99], [True], [1.0], [[1]], [np.nan]])
def test_bad_decode(bad):
    with pytest.raises(ValueError):
        Vocabulary(['a']).decode(bad)


def test_windows_boundaries_memory_and_final_batch():
    data = ContextDataset([[2, 3, 4, 5], [6], [5, 4, 3]], 2, 7)
    contexts, targets = data.batch([0, 1, 2])
    assert_array_equal(contexts, [[2, 3], [3, 4], [5, 4]])
    assert_array_equal(targets, [4, 5, 3])
    assert data.ids.nbytes == 8 * 8
    assert data.starts.nbytes == 3 * 8
    model = ContextLanguageModel(7, 2, seed=1)
    trainer = Trainer(CrossEntropyLoss(), AdamOptimizer(model.parameters(), .01),
                      TrainingConfig(epochs=1, batch_size=2))
    result = trainer.fit(WindowModelAdapter(model, data), *data.training_arrays())
    assert result.optimizer_steps == 2
    long = ContextDataset([np.arange(1000) % 7], 20, 7)
    assert len(long) == 980
    assert_array_equal(long.batch([979])[0][0], np.arange(979, 999) % 7)


@pytest.mark.parametrize('docs,c,size', [([], 2, 5), ([[1, 2]], 2, 5), ([[]], 1, 5),
                                      ([[1, 9]], 1, 5), ([[1., 2.]], 1, 5),
                                      ([[1, 2]], 0, 5), ([[True, False]], 1, 5)])
def test_bad_windows(docs, c, size):
    with pytest.raises(ValueError):
        ContextDataset(docs, c, size)


def test_embedding_repeats_and_finite_difference():
    embedding = Embedding(4, 2, seed=5)
    assert_array_equal(embedding.weight.data, Embedding(4, 2, seed=5).weight.data)
    ids = np.array([[1, 1], [2, 1]])
    output, context = embedding.forward(ids, retain_cache=True)
    assert output.shape == (2, 2, 2) and output.dtype == np.float64
    upstream = np.arange(8, dtype=float).reshape(2, 2, 2)
    embedding.backward(upstream, context)
    expected = np.zeros((4, 2))
    expected[1] = upstream[0, 0] + upstream[0, 1] + upstream[1, 1]
    expected[2] = upstream[1, 0]
    assert_array_equal(embedding.weight.grad, expected)
    for index in np.ndindex(embedding.weight.data.shape):
        original = embedding.weight.data[index]
        embedding.weight.data[index] = original + 1e-6
        plus = np.sum(embedding.forward(ids)[0] * upstream)
        embedding.weight.data[index] = original - 1e-6
        minus = np.sum(embedding.forward(ids)[0] * upstream)
        embedding.weight.data[index] = original
        assert expected[index] == pytest.approx((plus-minus)/2e-6, abs=1e-8)
    with pytest.raises(ValueError):
        embedding.backward(upstream, context)


@pytest.mark.parametrize('ids', [[[-1]], [[4]], [[1.]], [[True]], [[np.nan]], [1], [[]]])
def test_bad_embedding_ids(ids):
    with pytest.raises(ValueError):
        Embedding(4, 2).forward(ids)


def test_embedding_validation_and_contexts():
    e = Embedding(4, 2)
    _, cache = e.forward([[1]], retain_cache=True)
    with pytest.raises(ValueError):
        e.backward(np.ones((1, 2, 2)), cache)
    with pytest.raises(ValueError):
        e.backward(np.full((1, 1, 2), np.inf), cache)
    e.weight.assign(e.weight.data)
    with pytest.raises(ValueError):
        e.backward(np.ones((1, 1, 2)), cache)
    cache.release()


def train():
    tokenizer = SimpleTokenizer.from_corpus(['a b c a b c a b c'])
    dataset = ContextDataset([tokenizer.encode('a b c a b c a b c')], 2, len(tokenizer.vocabulary))
    model = ContextLanguageModel(len(tokenizer.vocabulary), 2, embedding_dim=3, hidden_dim=6, seed=10)
    trainer = Trainer(CrossEntropyLoss(), AdamOptimizer(model.parameters(), .03),
                      TrainingConfig(epochs=60, batch_size=4, seed=10))
    result = trainer.fit(WindowModelAdapter(model, dataset), *dataset.training_arrays())
    return model, dataset, result


def test_training_reproducibility_and_probabilities():
    model, data, result = train()
    other, _, second = train()
    assert result.to_dict() == second.to_dict()
    for a, b in zip(model.parameters(), other.parameters()):
        assert_array_equal(a.data, b.data)
    assert result.history[-1].train_loss < result.history[0].train_loss / 10
    X, y = data.batch(np.arange(len(data)))
    logits = model.predict_logits(X)
    assert logits.shape == (len(data), data.vocab_size)
    assert np.isfinite(logits).all()
    probabilities = model.predict_proba(X)
    assert_allclose(probabilities.sum(axis=1), 1)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert_array_equal(model.predict_classes(X), y)
    assert_allclose(softmax([[1000., 999., -1000.]]).sum(), 1)
    json.dumps(result.to_dict(), allow_nan=False)


def test_composed_gradient_and_release():
    model = ContextLanguageModel(4, 2, 2, 3, seed=2)
    ids, targets = np.array([[1, 1], [2, 1]]), np.array([2, 3])
    loss = CrossEntropyLoss()
    logits, context = model.forward(ids, retain_cache=True)
    model.backward(loss.value_and_grad(logits, targets)[1], context)
    assert context.payload is None and not model.mlp._live_contexts
    for p in model.parameters():
        analytic = p.grad.copy()
        for index in np.ndindex(p.data.shape):
            original = p.data[index]
            p.data[index] = original + 1e-6
            plus = loss.value(model.predict_logits(ids), targets)
            p.data[index] = original - 1e-6
            minus = loss.value(model.predict_logits(ids), targets)
            p.data[index] = original
            assert analytic[index] == pytest.approx((plus-minus)/2e-6, abs=1e-8)
    _, cache = model.forward(ids, retain_cache=True)
    model.parameters()[0].assign(model.parameters()[0].data)
    with pytest.raises(ValueError):
        model.backward(np.ones((2, 4)), cache)
    model.release_context(cache)
    model.release_context(cache)
    with pytest.raises(ValueError):
        model.predict_logits([[1]])


@pytest.mark.parametrize('with_plot', [False, True])
def test_cli_json_visualization(tmp_path, with_plot):
    if with_plot:
        pytest.importorskip('matplotlib')
    backend = Path(__file__).resolve().parents[1]
    output, plot = tmp_path / 'result.json', tmp_path / 'result.png'
    command = [sys.executable, 'examples/train_language_model.py', '--epochs', '3', '--output', str(output)]
    if with_plot:
        command += ['--plot', str(plot)]
    completed = subprocess.run(command, cwd=backend, env=dict(os.environ, PYTHONPATH=str(backend / 'src')),
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(), parse_constant=lambda x: pytest.fail(x))
    assert json.loads(completed.stdout) == payload['summary']
    vis = payload['visualization']
    vocab = vis['vocabulary']
    for w in vis['windows']:
        assert w['context_tokens'] == [vocab[i]['token'] for i in w['context_ids']]
        assert w['target_token'] == vocab[w['target_id']]['token']
    for doc in vis['documents']:
        assert doc['tokens'] == [vocab[i]['token'] for i in doc['ids']]
    assert len(vis['distribution']['probabilities']) == len(vocab)
    assert_allclose(sum(vis['distribution']['probabilities']), 1)
    if with_plot:
        assert plot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


@pytest.mark.parametrize('indices', [[[-1]], [[1.5]], [[3]], [[np.inf]], [[0, 1]]])
def test_adapter_rejects_invalid_example_indices(indices):
    data = ContextDataset([[1, 2, 3]], 1, 4)
    adapter = WindowModelAdapter(ContextLanguageModel(4, 1), data)
    with pytest.raises(ValueError):
        adapter.forward(indices)


def test_cli_without_matplotlib_import(tmp_path, monkeypatch):
    backend = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(backend / 'examples'))
    import builtins
    import train_language_model
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == 'matplotlib' or name.startswith('matplotlib.'):
            raise AssertionError('Non-plot flow imported Matplotlib')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(sys, 'argv', ['train_language_model.py', '--epochs', '1',
                                    '--output', str(tmp_path / 'without_plot.json')])
    train_language_model.main()
    assert (tmp_path / 'without_plot.json').is_file()
