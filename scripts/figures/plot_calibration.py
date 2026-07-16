"""The detector's calibration sweep (report Sec. 5.4): recall and clean false-positive
rate against the budget, with the selected operating point marked.

This script calibrates nothing: it reads the sweep `scripts.distillation.distill_calibrate`
already wrote to results/<model>/distill_calibration.json and fails if there is none.
Produce it with:

    uv run -m scripts.distillation.distill_calibrate <model>
    uv run -m scripts.figures.plot_calibration <model>

The figure is what makes the chosen budget auditable rather than asserted. Each subject's
threshold is the `1 - budget` quantile of its own clean errors, so the clean FPR a budget
buys *is* the budget — the two curves are recall(b) against the b line, and the selected
point maximizes the gap between them (Youden's J). See
shared/docs/anomalies-and-distillation.md for why J rather than F1.
"""

import argparse
import json

from ml.model_list import MODELS
from ..common.plots import line_plot
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml
from ..common.scoring import CALIBRATION_REPORT


def load_sweep(model: str) -> tuple[float, list[dict]]:
    path = get_report_dir(model) / CALIBRATION_REPORT
    if not path.exists():
        raise SystemExit(
            f"no calibration report at {path}. Run "
            f"`uv run -m scripts.distillation.distill_calibrate {model}` first — this "
            f"script plots a previous calibration, it does not calibrate.")
    report = json.loads(path.read_text())
    return float(report['budget']), report['sweep']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Calibrated autoencoder to plot')
    args = parser.parse_args()

    budget, sweep = load_sweep(args.model)
    chosen = next(r for r in sweep if r['budget'] == budget)
    budgets = [r['budget'] for r in sweep]

    report_dir = get_report_dir(args.model)
    line_plot(report_dir / 'calibration.png', budgets,
              {'recall (mixed set)': [r['recall'] for r in sweep],
               'clean FPR': [r['clean_fpr'] for r in sweep],
               "Youden's J = recall - FPR": [r['youden_j'] for r in sweep]},
              'budget (allowed clean false-positive rate)', 'rate',
              f'{args.model} — detector calibration',
              vline=(budget, f'selected budget {budget:g}'), logx=True)

    write_metrics_csv(sweep, report_dir, 'calibration.csv')
    write_yaml(report_dir / 'calibration.yaml', {
        'shows': "Detector calibration sweep: how recall and the clean false-positive "
                 "rate trade off as the budget varies, and the operating point selected "
                 "from it. The budget is the share of a subject's own clean windows the "
                 "detector may fire on; its threshold is the 1-budget quantile of that "
                 "subject's clean reconstruction errors, so the clean FPR a budget buys "
                 "is the budget itself.",
        'x_axis': {'name': 'budget', 'range': [min(budgets), max(budgets)],
                   'scale': 'log'},
        'y_axis': {'name': 'rate', 'range': [0, 1]},
        'selection': {
            'criterion': "maximum Youden's J (recall - clean FPR)",
            'why': "J is built from two rates each conditioned on a single class, so it "
                   "is independent of the anomaly prevalence of the set it is measured "
                   "on; precision (and therefore F1) mixes the classes and inherits that "
                   "prevalence, so an F1-selected threshold would not transfer to a "
                   "deployment whose prevalence is unknown and subject-varying.",
            'budget': budget,
        },
        'headline': {'budget': budget, 'recall': chosen['recall'],
                     'precision': chosen['precision'], 'f1': chosen['f1'],
                     'clean_fpr': chosen['clean_fpr'], 'youden_j': chosen['youden_j']},
        'caveats': ["precision and F1 are reported at the selected point but are "
                    "prevalence-dependent: the mixed set is ~50% anomalous by "
                    "construction, and a real deployment's far lower rate would make "
                    "precision worse than shown",
                    "the clean FPR is exact on the calibration subjects, whose own clean "
                    "windows set their thresholds; on an unseen subject it is only "
                    "approximately the budget"],
        'source': {'report': f'results/{args.model}/{CALIBRATION_REPORT}',
                   'reproducible': True},
        'backs': 'report Sec. 5.4',
    })


if __name__ == "__main__":
    main()
