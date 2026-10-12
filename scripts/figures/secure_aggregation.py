"""Convergence and Byzantine robustness of the secure-aggregation round (clipped deltas,
session means, trimmed mean over sessions) against the dense trimmed mean over client
deltas and an undefended plain mean. Every run trains a fresh model with secure_loop; the
figures, a CSV of every round and a YAML of the configuration go to
results/<model>/secure_aggregation/<experiment>/."""

import argparse
import math
from dataclasses import replace

import tensorflow as tf

from common.config import DATASETS_DIR, SEED
from ml.model_list import MODELS
from ml.sources.common import pool
from ml.training import ATTACKS, Attack, SecureConfig, secure_loop
from ..common.plots import line_plot, metric_name
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml

PLAIN = SecureConfig(secure=False, trim=0.0, clip=math.inf)
DENSE = SecureConfig(secure=False)
SECURE = SecureConfig()

SWEEPS = {
    'session-size': ('session_size', [3, 5, 10, 20, 50]),
    'trim': ('trim', [0.0, 0.1, 0.2, 0.3, 0.4]),
    'clip': ('clip', [1.0, 0.1, 0.03, 0.01, 0.003, 0.001]),
    'participants': ('clients_per_round', [10, 20, 50, 100]),
}
EXPERIMENTS = ['convergence', *SWEEPS, 'byzantine']

LABELS = {'plain mean': 'media simple', 'dense trimmed mean': 'media recortada densa',
          'secure sessions': 'sesiones seguras', 'secure (random)': 'segura (aleatorio)',
          'secure (spread)': 'segura (dispersos)', 'secure (clustered)': 'segura (agrupados)'}
DATASETS = {'mnist-iid-100': 'MNIST IID, 100 clientes',
            'mnist-noniid-100': 'MNIST no-IID, 100 clientes'}
FIELDS = {'session_size': 'tamaño de sesión', 'trim': 'razón de recorte',
          'clip': 'cota de recorte B', 'clients_per_round': 'clientes por ronda'}
ATTACK_NAMES = {'sign-flip': 'inversión de signo', 'noise': 'ruido', 'push': 'empuje a ±B'}


def labeled(series: dict) -> dict:
    return {LABELS.get(k, DATASETS.get(k, k)): v for k, v in series.items()}


class Runner:

    def __init__(self, model: str, args: argparse.Namespace):
        self.model = model
        self.args = args
        self.rows: list[dict] = []
        self._data: dict[str, tuple[list, tf.data.Dataset]] = {}
        self._memo: dict[tuple, list[float]] = {}
        self.dense = replace(DENSE, clip=args.clip) if args.clip else DENSE
        self.secure = replace(SECURE, clip=args.clip) if args.clip else SECURE

    def data(self, dataset: str):
        if dataset not in self._data:
            trainer = self.trainer(dataset)
            self._data[dataset] = (trainer.subject_datasets(), pool(trainer.eval_datasets()))
        return self._data[dataset]

    def trainer(self, dataset: str):
        spec = MODELS[self.model]
        return spec.trainer_cls(spec.model_cls(), DATASETS_DIR, training_key=dataset)

    def run(self, experiment: str, dataset: str, label: str, config: SecureConfig,
            attack: Attack | None = None, **fields) -> list[float]:
        key = (dataset, config, attack)
        if key not in self._memo:
            clients, eval_dataset = self.data(dataset)
            trainer = self.trainer(dataset)
            print(f"\n=== {experiment} {dataset} {label} {config} {attack}", flush=True)
            history = secure_loop(trainer, clients, eval_dataset, self.args.local_epochs,
                                  self.args.rounds, config, attack, SEED)
            self._memo[key] = [h[2][trainer.primary_metric] for h in history]
        values = self._memo[key]
        for r, v in enumerate(values, start=1):
            self.rows.append({'experiment': experiment, 'dataset': dataset, 'label': label,
                              **fields, 'round': r, 'accuracy': v})
        return values


def run_convergence(runner: Runner, args, report_dir):
    rounds = list(range(1, args.rounds + 1))
    finals = {}
    for dataset in args.datasets:
        series = {
            'plain mean': runner.run('convergence', dataset, 'plain mean', PLAIN),
            'dense trimmed mean': runner.run('convergence', dataset, 'dense trimmed mean',
                                             runner.dense),
            'secure sessions': runner.run('convergence', dataset, 'secure sessions',
                                          runner.secure),
        }
        finals[dataset] = {label: values[-1] for label, values in series.items()}
        line_plot(report_dir / f'convergence_{dataset}.png', rounds, labeled(series),
                  'ronda global', metric_name('accuracy'),
                  f'{runner.model}, {DATASETS.get(dataset, dataset)} — reglas de agregación')
    return {'final_accuracy': finals}


def run_sweep(name: str):
    field, values = SWEEPS[name]

    def sweep(runner: Runner, args, report_dir):
        series, baselines = {}, {}
        for dataset in args.datasets:
            series[dataset] = [
                runner.run(name, dataset, f'{field}={v}', replace(runner.secure, **{field: v}),
                           **{field: v})[-1]
                for v in values]
            baselines[dataset] = runner.run('convergence', dataset, 'dense trimmed mean',
                                            runner.dense)[-1]
        line_plot(report_dir / f'{name}.png', values, labeled(series), FIELDS[field],
                  f'{metric_name("accuracy")} tras {args.rounds} rondas',
                  f'{runner.model} — agregación segura, {FIELDS[field]}', logx=(name == 'clip'))
        return {'swept': field, 'values': values, 'final_accuracy': series,
                'dense_trimmed_baseline': baselines}

    return sweep


def run_byzantine(runner: Runner, args, report_dir):
    fractions = [0.0, *args.fractions]
    summary = {}
    for dataset in args.datasets:
        for kind in args.attacks:
            aggregators = {
                'plain mean': PLAIN,
                'dense trimmed mean': runner.dense,
                **{f'secure ({p})': runner.secure for p in args.placements},
            }
            series = {}
            for label, config in aggregators.items():
                placement = label[len('secure ('):-1] if label.startswith('secure') else 'random'
                series[label] = [
                    runner.run('byzantine', dataset, label, config,
                               Attack(kind, f, placement, args.attack_scale) if f else None,
                               kind=kind, fraction=f)[-1]
                    for f in fractions]
            summary.setdefault(dataset, {})[kind] = series
            line_plot(report_dir / f'byzantine_{kind}_{dataset}.png', fractions, labeled(series),
                      'fracción de clientes maliciosos',
                      f'{metric_name("accuracy")} tras {args.rounds} rondas',
                      f'{runner.model}, {DATASETS.get(dataset, dataset)} — ataque de '
                      f'{ATTACK_NAMES.get(kind, kind)}, B = {runner.secure.clip:g}')
    return {'fractions': fractions, 'attack_scale': args.attack_scale,
            'final_accuracy': summary}


RUNNERS = {'convergence': run_convergence, **{name: run_sweep(name) for name in SWEEPS},
           'byzantine': run_byzantine}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to train')
    parser.add_argument('experiment', choices=EXPERIMENTS, help='Experiment to run')
    parser.add_argument('--datasets', nargs='+', default=['mnist-iid-100', 'mnist-noniid-100'],
                        help='Training dataset keys; each shard is one client')
    parser.add_argument('--rounds', type=int, default=20, help='Global rounds')
    parser.add_argument('--local-epochs', type=int, default=1, help='Local epochs per round')
    parser.add_argument('--attacks', nargs='+', choices=ATTACKS, default=list(ATTACKS),
                        help='Attack kinds for the byzantine experiment')
    parser.add_argument('--placements', nargs='+', default=['random', 'spread', 'clustered'],
                        help='Attacker placements in sessions for the byzantine experiment')
    parser.add_argument('--fractions', nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.3],
                        help='Malicious client fractions for the byzantine experiment')
    parser.add_argument('--attack-scale', type=float, default=10.0,
                        help='Sign-flip multiplier and noise std in honest RMS units')
    parser.add_argument('--clip', type=float, default=None,
                        help='Override the clip bound of the dense and secure configs')
    parser.add_argument('--threads', type=int, default=0,
                        help='TensorFlow intra/inter-op threads (0 = TF default)')
    parser.add_argument('--tag', default=None, help='Suffix for the results directory')
    args = parser.parse_args()

    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    subdir = f'secure_aggregation/{args.experiment}' + (f'_{args.tag}' if args.tag else '')
    report_dir = get_report_dir(args.model, subdir)
    runner = Runner(args.model, args)
    summary = RUNNERS[args.experiment](runner, args, report_dir)

    write_metrics_csv(runner.rows, report_dir, 'rounds.csv')
    write_yaml(report_dir / 'summary.yaml', {
        'model': args.model,
        'experiment': args.experiment,
        'datasets': args.datasets,
        'rounds': args.rounds,
        'local_epochs': args.local_epochs,
        'defaults': {'secure': vars(runner.secure), 'dense': vars(runner.dense),
                     'plain': vars(PLAIN)},
        'seed': SEED,
        **summary,
    })


if __name__ == "__main__":
    main()
