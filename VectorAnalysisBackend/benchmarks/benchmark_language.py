"""Informative scatter-add measurement; no timing thresholds in tests."""
import json
import platform
from time import perf_counter
import numpy as np
from mlvectoranalysis.language import Embedding


def main():
    rng = np.random.default_rng(42)
    batch, context, dimension, vocabulary = 64, 16, 32, 1000
    ids = rng.integers(0, vocabulary, (batch, context))
    upstream = rng.normal(size=(batch, context, dimension))
    embedding = Embedding(vocabulary, dimension, seed=42)
    expected = np.zeros_like(embedding.weight.grad)
    for index, gradient in zip(ids.ravel(), upstream.reshape(-1, dimension)):
        expected[index] += gradient
    times = []
    for _ in range(30):
        _, cache = embedding.forward(ids, retain_cache=True)
        start = perf_counter()
        embedding.backward(upstream, cache)
        times.append(perf_counter() - start)
    np.testing.assert_allclose(embedding.weight.grad, expected)
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                         shape=[batch, context, dimension], vocabulary=vocabulary,
                         median_backward_ms=1000 * float(np.median(times)),
                         parameter_bytes=embedding.weight.data.nbytes,
                         gradient_bytes=embedding.weight.grad.nbytes,
                         retained_ids_bytes=ids.nbytes, lookup_output_bytes=upstream.nbytes,
                         hypothetical_one_hot_bytes=batch*context*vocabulary*8), indent=2))


if __name__ == '__main__':
    main()
