"""Wall time and peak memory of exporting each model's serving artifacts (trainable and
int8 .tflite) from a fresh, untrained model, each in its own process so the peak RSS is
that export's alone. Run it on an otherwise idle machine. Writes results/export_cost/."""

import argparse
import os
import subprocess
import sys
import tempfile
import time

from common.config import MODELS_DIR
from ml.model_list import BASE_MODELS
from ..common.plots import bar_plot
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml

TAG = 'export_cost'


def measure(model: str, timeout: int) -> dict:
    cmd = [sys.executable, '-m', 'scripts.system.train', model, '--epochs', '0', '--tag', TAG]
    with tempfile.TemporaryFile('w+') as log:
        start = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        status = None
        while status is None:
            pid, code, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                status = 'ok' if code == 0 else f'exit {os.waitstatus_to_exitcode(code)}'
            elif time.monotonic() - start > timeout:
                proc.kill()
                _, _, usage = os.wait4(proc.pid, 0)
                status = f'timeout after {timeout}s'
            else:
                time.sleep(0.5)
        seconds = time.monotonic() - start
        if status != 'ok':
            log.seek(0)
            print(log.read()[-2000:], file=sys.stderr)

    model_dir = MODELS_DIR / BASE_MODELS[model].artifact_key
    quantized = (model_dir / f'quantized_{TAG}.tflite').exists()
    for path in model_dir.glob(f'*_{TAG}.*'):
        path.unlink()
    return {'model': model, 'status': status, 'seconds': round(seconds, 1),
            'peak_rss_mib': round(usage.ru_maxrss / 1024, 1), 'int8_exported': quantized}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+', choices=sorted(BASE_MODELS), help='Models to export')
    parser.add_argument('--timeout', type=int, default=1800, help='Seconds before an export is killed')
    args = parser.parse_args()

    rows = []
    for model in args.models:
        row = measure(model, args.timeout)
        rows.append(row)
        print(row, flush=True)

    report_dir = get_report_dir('export_cost')
    write_metrics_csv(rows, report_dir, 'export_cost.csv')
    names = [r['model'] for r in rows]
    bar_plot(report_dir / 'export_seconds.png', names, [r['seconds'] for r in rows],
             'model', 'export wall time (s)', 'Artifact export time')
    bar_plot(report_dir / 'export_peak_rss.png', names, [r['peak_rss_mib'] for r in rows],
             'model', 'peak RSS (MiB)', 'Artifact export peak memory')
    write_yaml(report_dir / 'export_cost.yaml', {'timeout': args.timeout, 'cpus': os.cpu_count(),
                                                 'rows': rows})


if __name__ == "__main__":
    main()
