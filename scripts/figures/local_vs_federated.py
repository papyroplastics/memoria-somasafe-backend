"""Each client trained alone on its own shard vs. the federated and centralized train.py
runs, all scored on the trainer's eval split: what a client gains by joining the
federation. Writes results/<model>/local_vs_federated[_<tag>]/."""

import argparse

from common.config import DATASETS_DIR, SEED
from ml.dataset_list import DATASETS
from ml.model_list import MODELS
from ml.sources.common import pool
from ml.training import normal_loop
from ..common.plots import bar_plot, client_name, metric_name
from ..common.reports import get_report_dir, read_run, write_metrics_csv, write_yaml


LABELS = {'federated': 'federado', 'centralized': 'centralizado'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to train')
    parser.add_argument('--dataset', choices=sorted(DATASETS), default=None,
                        help="Override the trainer's training dataset")
    parser.add_argument('--tag', default=None,
                        help='Tag of the normal/federated train.py runs to compare against')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per local-only model')
    args = parser.parse_args()

    spec = MODELS[args.model]
    kwargs = {'training_key': args.dataset} if args.dataset else {}

    def trainer():
        return spec.trainer_cls(spec.model_cls(), DATASETS_DIR, **kwargs)

    base = trainer()
    eval_sets = base.eval_datasets()
    if not eval_sets:
        raise SystemExit(f"{args.model} has no eval split to score local models on")
    eval_dataset = pool(eval_sets)
    metric = base.primary_metric

    rows = []
    for sid, dataset in zip(base.subject_ids(), base.subject_datasets()):
        history = normal_loop(trainer(), dataset, eval_dataset, args.epochs)
        value = history[-1][2][metric] if history else float('nan')
        rows.append({'client': sid, 'batches': len(dataset), metric: value})
        print(f"{sid}: {metric}={value:.4f}")

    references = {}
    for loop, label in (('federated', 'federated'), ('normal', 'centralized')):
        try:
            references[label] = read_run(args.model, loop, args.tag)['final'][metric]
        except SystemExit as e:
            print(f"no {label} reference: {e}")

    report_dir = get_report_dir(args.model, 'local_vs_federated' + (f'_{args.tag}' if args.tag else ''))
    bar_plot(report_dir / 'local_vs_federated.png', [client_name(r['client']) for r in rows],
             [r[metric] for r in rows], 'cliente entrenado por separado', metric_name(metric),
             f'{args.model} — solo local vs. federado',
             hlines=[(v, LABELS[label]) for label, v in references.items()])
    write_metrics_csv(rows, report_dir, 'local_vs_federated.csv')
    write_yaml(report_dir / 'local_vs_federated.yaml', {
        'model': args.model,
        'dataset': base.training_key,
        'eval_dataset': base.eval_key,
        'local_epochs': args.epochs,
        'references': references,
        'seed': SEED,
    })


if __name__ == "__main__":
    main()
