"""Reference scikit-learn classifier on the same Covertype train/test split as covertype-nb,
next to the centralized and federated train.py runs. Writes results/covertype-nb/baselines/."""

import argparse

import numpy as np
from sklearn.naive_bayes import GaussianNB

from common.config import DATASETS_DIR
from ml.models.covertype_nb import VAR_SMOOTHING
from ml.sources.covertype import TEST, TRAIN, CovertypeSource
from ..common.plots import bar_plot
from ..common.reports import get_report_dir, read_run, write_metrics_csv, write_yaml

MODEL = 'covertype-nb'


def split(source: CovertypeSource) -> tuple[np.ndarray, np.ndarray]:
    ids = source.subject_ids()
    x = np.concatenate([source.datapoints(sid) for sid in ids])
    y = np.concatenate([source.labels(sid).argmax(axis=1) for sid in ids])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', default=None, help='Tag of the train.py runs to compare against')
    args = parser.parse_args()

    x_train, y_train = split(CovertypeSource(DATASETS_DIR, 'covertype-wilderness', TRAIN))
    x_test, y_test = split(CovertypeSource(DATASETS_DIR, 'covertype-test', TEST))
    print(f"train={len(y_train)} test={len(y_test)}")

    clf = GaussianNB(var_smoothing=VAR_SMOOTHING).fit(x_train, y_train)
    accuracy = float(np.mean(clf.predict(x_test) == y_test))
    rows = [{'model': 'sklearn GaussianNB', 'accuracy': accuracy}]
    print(f"sklearn GaussianNB: accuracy={accuracy:.4f}")

    for loop, name in (('normal', 'covertype-nb centralized'), ('federated', 'covertype-nb federated')):
        try:
            rows.append({'model': name, 'accuracy': read_run(MODEL, loop, args.tag)['final']['accuracy']})
        except SystemExit as e:
            print(f"no {loop} run: {e}")

    report_dir = get_report_dir(MODEL, 'baselines' + (f'_{args.tag}' if args.tag else ''))
    short = {'sklearn GaussianNB': 'GaussianNB\nscikit-learn',
             'covertype-nb centralized': 'propio\ncentralizado',
             'covertype-nb federated': 'propio\nfederado'}
    names = [short[r['model']] for r in rows]
    bar_plot(report_dir / 'baselines.png', names, [r['accuracy'] for r in rows],
             '', 'exactitud', 'Covertype — covertype-nb vs. scikit-learn')
    write_metrics_csv(rows, report_dir, 'baselines.csv')
    write_yaml(report_dir / 'baselines.yaml', {'train_rows': len(y_train), 'test_rows': len(y_test),
                                               'rows': rows})


if __name__ == "__main__":
    main()
