"""Tiny local next-token experiment: memorization, not generalization."""
import numpy as np
from mlvectoranalysis.language import SimpleTokenizer, ContextDataset, ContextLanguageModel, WindowModelAdapter
from mlvectoranalysis.nn import Trainer, AdamOptimizer, CrossEntropyLoss
from _mlp_common import parser, config, report, plotting, save

CORPUS = ('el gato come pescado .', 'el perro come pan .',
          'el gato bebe agua .', 'el perro bebe agua .')


def run(args):
    tokenizer = SimpleTokenizer.from_corpus(CORPUS)
    vocab = tokenizer.vocabulary
    documents = [tokenizer.encode(text) for text in CORPUS]
    dataset = ContextDataset(documents, 2, len(vocab))
    model = ContextLanguageModel(len(vocab), 2, seed=args.seed)
    trainer = Trainer(CrossEntropyLoss(), AdamOptimizer(model.parameters(), args.learning_rate), config(args))
    result = trainer.fit(WindowModelAdapter(model, dataset), *dataset.training_arrays())
    contexts, targets = dataset.batch(np.arange(min(6, len(dataset))))
    probabilities = model.predict_proba(contexts)
    predictions = probabilities.argmax(axis=1)
    visualization = dict(
        vocabulary=[dict(token=token, id=i) for i, token in enumerate(vocab.tokens)],
        documents=[dict(text=text, tokens=vocab.decode(ids), ids=ids.tolist())
                   for text, ids in zip(CORPUS, documents)],
        windows=[dict(context_ids=ctx.tolist(), context_tokens=vocab.decode(ctx),
                      target_id=int(target), target_token=vocab.id_to_token(target),
                      predicted_id=int(pred), predicted_token=vocab.id_to_token(pred))
                 for ctx, target, pred in zip(contexts, targets, predictions)],
        distribution=dict(context_ids=contexts[0].tolist(), context_tokens=vocab.decode(contexts[0]),
                          token_ids=list(range(len(vocab))), tokens=list(vocab.tokens),
                          probabilities=probabilities[0].tolist()))
    summary = dict(seed=args.seed, initial_loss=result.history[0].train_loss,
                   final_loss=result.history[-1].train_loss, status=result.status,
                   predictions=[w['predicted_token'] for w in visualization['windows']],
                   examples=len(dataset), corpus_bytes=dataset.ids.nbytes,
                   window_offsets_bytes=dataset.starts.nbytes,
                   materialized_context_batch_bytes=contexts.nbytes)
    return summary, result, visualization


def plot(visualization, path):
    plt = plotting()
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].axis('off')
    axes[0].set_title('1. Vocabulario local: token → ID (PAD=0, UNK=1)')
    axes[0].text(.02, .6, '\n'.join(
        '    '.join(f"{v['token']} → {v['id']}" for v in visualization['vocabulary'][i:i+5])
        for i in range(0, len(visualization['vocabulary']), 5)), fontsize=11)
    axes[1].axis('off')
    axes[1].set_title('2. Ventanas reales: contexto → siguiente token observado')
    axes[1].text(.02, .9, '\n'.join(
        f"{' '.join(w['context_tokens'])}  {w['context_ids']} → {w['target_token']} ({w['target_id']})"
        for w in visualization['windows']), va='top', fontsize=11)
    dist = visualization['distribution']
    axes[2].bar(dist['tokens'], dist['probabilities'])
    axes[2].set(ylim=(0, 1), ylabel='Probabilidad', xlabel='Token candidato',
                title='3. Distribución aprendida después de: ' + ' '.join(dist['context_tokens']))
    axes[2].tick_params(axis='x', rotation=45)
    save(fig, path)
    plt.close(fig)


def main():
    argument_parser = parser('Predicción local del siguiente token con embeddings y MLP.', 'language_model', 8, .03)
    argument_parser.set_defaults(epochs=120)
    args = argument_parser.parse_args()
    summary, result, visualization = run(args)
    report(args, summary, result, visualization=visualization)
    if args.plot:
        plot(visualization, args.plot)


if __name__ == '__main__':
    main()
