"""The detector's calibration sweep (report Sec. 5.4): recall and empirical clean
false-positive rate against the expected FPR, with the selected operating point marked.

This script calibrates nothing: it reads the sweep `scripts.figures.calibrate_fpr`
already wrote to results/<model>/calibration.json and fails if there is none.
Produce it with:

    uv run -m scripts.figures.calibrate_fpr <model>
    uv run -m scripts.figures.plot_calibration <model>

The figure is what makes the chosen expected FPR auditable rather than asserted. Each
subject's threshold is the `1 - f` quantile of its own clean errors, so that fraction of
clean windows lies above it by definition — the two curves are recall(f) against the f
line, and the selected point maximizes the gap between them (Youden's J). See
shared/docs/anomalies-and-distillation.md for why J rather than F1.
"""

import argparse
import json

from ml.model_list import MODELS
from ..common.plots import line_plot
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml
from ..common.scoring import CALIBRATION_REPORT


def load_sweep(model: str) -> tuple[float, list[dict], list[str]]:
    path = get_report_dir(model) / CALIBRATION_REPORT
    if not path.exists():
        raise SystemExit(
            f"no calibration report at {path}. Run "
            f"`uv run -m scripts.figures.calibrate_fpr {model}` first — this "
            f"script plots a previous calibration, it does not calibrate.")
    report = json.loads(path.read_text())
    return (float(report['expected_fpr']), report['sweep'],
            report.get('calibration_subjects', []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Calibrated autoencoder to plot')
    args = parser.parse_args()

    expected_fpr, sweep, calibration_subjects = load_sweep(args.model)
    chosen = next(r for r in sweep if r['expected_fpr'] == expected_fpr)
    levels = [r['expected_fpr'] for r in sweep]

    report_dir = get_report_dir(args.model)
    line_plot(report_dir / 'calibration.png', levels,
              {'recall (mixed set)': [r['recall'] for r in sweep],
               'empirical clean FPR': [r['clean_fpr'] for r in sweep],
               "Youden's J = recall - FPR": [r['youden_j'] for r in sweep]},
              'expected FPR (calibrated clean false-positive rate)', 'rate',
              f'{args.model} — detector calibration',
              vline=(expected_fpr, f'selected expected FPR {expected_fpr:g}'), logx=True)

    write_metrics_csv(sweep, report_dir, 'calibration.csv')
    write_yaml(report_dir / 'calibration.yaml', {
        'shows': "Detector calibration sweep: how recall and the empirical clean "
                 "false-positive rate trade off as the expected FPR varies, and the "
                 "operating point selected from it. The expected FPR is the rate at which "
                 "the detector fires on clean signal; a client turns it into a threshold "
                 "at the 1-f quantile of its own clean reconstruction errors, so that "
                 "fraction of clean windows lies above the threshold by definition — the "
                 "parameter is the false-alarm rate, not a proxy for it.",
        'x_axis': {'name': 'expected FPR', 'range': [min(levels), max(levels)],
                   'scale': 'log'},
        'y_axis': {'name': 'rate', 'range': [0, 1]},
        'measured_on': {'subjects': calibration_subjects,
                        'note': 'training subjects — the operating point is chosen '
                                'disjoint from the held-out subjects anomaly_detection scores'},
        'selection': {
            'criterion': "maximum Youden's J (recall - clean FPR)",
            'why': "J is built from two rates each conditioned on a single class, so it "
                   "is independent of the anomaly prevalence of the set it is measured "
                   "on; precision (and therefore F1) mixes the classes and inherits that "
                   "prevalence, so an F1-selected threshold would not transfer to a "
                   "deployment whose prevalence is unknown and subject-varying.",
            'expected_fpr': expected_fpr,
        },
        'headline': {'expected_fpr': expected_fpr, 'recall': chosen['recall'],
                     'precision': chosen['precision'], 'f1': chosen['f1'],
                     'clean_fpr': chosen['clean_fpr'], 'youden_j': chosen['youden_j']},
        'caveats': ["precision and F1 are reported at the selected point but are "
                    "prevalence-dependent: the mixed set is ~50% anomalous by "
                    "construction, and a real deployment's far lower rate would make "
                    "precision worse than shown",
                    "the empirical clean FPR matches the expected FPR exactly on the "
                    "calibration subjects, whose own clean windows set their thresholds; "
                    "on an unseen subject it only lands near it, which is why the "
                    "parameter is 'expected' rather than guaranteed"],
        'source': {'report': f'results/{args.model}/{CALIBRATION_REPORT}',
                   'reproducible': True},
        'backs': 'report Sec. 5.4',
    })


if __name__ == "__main__":
    main()
