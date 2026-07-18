"""Systems / footprint table (report Sec. 5.6): one table collating the edge-cost figures
that can be derived from the models and their exported artifacts:

  - parameter count per model (flat trainable-weight vector length);
  - float32 trainable vs. int8 quantized .tflite size + compression ratio;
  - bytes uploaded per round (a dense float32 weight delta) and downloaded (the trainable
    artifact) by a client.

The rows that come from the phone (on-device training time per epoch) and the ESP32 (TFLM
arena size, inference latency, detection quality retained after int8 as measured on-device),
plus aggregation round wall-time, are *pasted in* to the report table from the app/firmware
measurements (see report/planificacion/obtencion-de-resultados.md) — they are listed here as
explicit TODO rows, not measured by this script. Emits CSV + companion text.

    uv run -m scripts.figures.footprint
"""

import argparse

from common.config import DATASETS_DIR, MODELS_DIR, RESULTS_DIR
from ml.model_list import MODELS
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml

# Edge-cost rows the phone/firmware supply; pasted into the report table by hand.
PASTE_IN_ROWS = [
    "on-device training time per epoch (phone, logcat)",
    "aggregation round wall-time (server)",
    "TFLM arena size (ESP32)",
    "int8 inference latency (ESP32)",
    "detection quality retained after int8 (on-device)",
]


def artifact_size(key: str, name: str) -> int | None:
    path = MODELS_DIR / key / name
    return path.stat().st_size if path.exists() else None


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    rows = []
    for key in sorted(MODELS):
        try:
            params = MODELS[key].build_trainer(DATASETS_DIR).model.total_weight_size
        except Exception as e:
            print(f"skipped {key}: {e}")
            continue
        trainable = artifact_size(key, 'trainable.tflite')
        quantized = artifact_size(key, 'quantized.tflite')
        ratio = (trainable / quantized) if trainable and quantized else None
        rows.append({
            'model': key,
            'params': params,
            'trainable_bytes': trainable if trainable is not None else 'N/A',
            'quantized_bytes': quantized if quantized is not None else 'N/A',
            'compression_ratio': f'{ratio:.2f}' if ratio is not None else 'N/A',
            'upload_delta_bytes': params * 4,  # dense float32 weight delta
            'download_bytes': trainable if trainable is not None else 'N/A',
        })
        print(f"{key}: params={params} trainable={trainable} quantized={quantized} "
              f"ratio={ratio}")

    if not rows:
        raise SystemExit("no models could be built (datasets/artifacts missing)")

    report_dir = get_report_dir('footprint')
    write_metrics_csv(rows, report_dir, 'footprint.csv')

    write_yaml(report_dir / 'footprint.yaml', {
        'shows': 'System footprint table: the edge-cost figures derivable from the models '
                 'and their exported artifacts, one row per model.',
        'columns': {
            'params': 'flat trainable-weight count',
            'trainable_bytes': 'on-disk float32 trainable .tflite size',
            'quantized_bytes': 'on-disk int8 quantized .tflite size',
            'compression_ratio': 'trainable_bytes / quantized_bytes',
            'upload_delta_bytes': 'params x 4 — the dense float32 delta a client uploads '
                                  'per round',
            'download_bytes': 'the trainable artifact a client pulls',
        },
        'models': {r['model']: {k: v for k, v in r.items() if k != 'model'} for r in rows},
        'paste_in_rows': {'note': 'measured on the phone/ESP32/server and pasted into the '
                                  'report table, not produced by this script',
                          'todo': PASTE_IN_ROWS},
        'source': {'artifacts': f'{MODELS_DIR}/<model>/',
                   'na_means': 'the artifact was not exported yet (train + seed the model '
                               'first)'},
    })
    print(f"wrote footprint table to {report_dir}/ (results root {RESULTS_DIR})")


if __name__ == "__main__":
    main()
