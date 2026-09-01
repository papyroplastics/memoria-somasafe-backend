"""Systems / footprint table (report Sec. 5.6): one table collating the edge-cost figures
derivable from the models and their exported artifacts — parameter count, float32 trainable
vs. int8 quantized .tflite size, the flat weight buffer, and each of those as the gateway
actually stores and serves it (zstd, same level as common.compression). Rows measured on the
phone/ESP32/server are pasted in by hand and listed here only as TODO placeholders."""

import argparse

import numpy as np

from common.compression import compress
from common.config import DATASETS_DIR, MODELS_DIR, RESULTS_DIR
from ml.model_list import MODELS
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml

PASTE_IN_ROWS = [
    "on-device training time per epoch (phone, logcat)",
    "aggregation round wall-time (server)",
    "TFLM arena size (ESP32)",
    "int8 inference latency (ESP32)",
    "detection quality retained after int8 (on-device)",
]


def artifact_bytes(key: str, name: str) -> bytes | None:
    path = MODELS_DIR / key / name
    return path.read_bytes() if path.exists() else None


def weight_buffer(key: str) -> bytes | None:
    """The flat float32 buffer as the save signature packs it — what a client
    uploads as a delta and pulls back from /model/weights."""
    path = MODELS_DIR / key / 'weights.npy'
    if not path.exists():
        return None
    return np.load(path).astype(np.float32).tobytes()


def sizes(data: bytes | None) -> tuple[int | str, int | str]:
    """(raw, zstd) size of a blob, 'N/A' when it was never exported."""
    if data is None:
        return 'N/A', 'N/A'
    return len(data), len(compress(data))


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    rows = []
    for key in sorted(MODELS):
        if MODELS[key].artifacts_key is not None:
            continue
        try:
            params = MODELS[key].build_model(DATASETS_DIR).total_weight_size
        except Exception as e:
            print(f"skipped {key}: {e}")
            continue
        trainable, trainable_zstd = sizes(artifact_bytes(key, 'trainable.tflite'))
        quantized, quantized_zstd = sizes(artifact_bytes(key, 'quantized.tflite'))
        weights, weights_zstd = sizes(weight_buffer(key))
        rows.append({
            'model': key,
            'params': params,
            'trainable_bytes': trainable,
            'trainable_zstd_bytes': trainable_zstd,
            'quantized_bytes': quantized,
            'quantized_zstd_bytes': quantized_zstd,
            'weights_bytes': weights if weights != 'N/A' else params * 4,
            'weights_zstd_bytes': weights_zstd,
        })
        print(f"{key}: params={params} trainable={trainable}/{trainable_zstd} "
              f"quantized={quantized}/{quantized_zstd} weights={weights}/{weights_zstd}")

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
            'trainable_zstd_bytes': 'the same artifact zstd-compressed, as the gateway '
                                    'stores and serves it — one download per graph change',
            'quantized_bytes': 'on-disk int8 quantized .tflite size',
            'quantized_zstd_bytes': 'the same artifact zstd-compressed, as stored and served',
            'weights_bytes': 'params x 4 — the flat float32 buffer, uploaded uncompressed '
                             'as a client delta once per round',
            'weights_zstd_bytes': 'the same buffer zstd-compressed, as /model/weights '
                                  'serves it back once per round',
        },
        'models': {r['model']: {k: v for k, v in r.items() if k != 'model'} for r in rows},
        'paste_in_rows': {'note': 'measured on the phone/ESP32/server and pasted into the '
                                  'report table, not produced by this script',
                          'todo': PASTE_IN_ROWS},
        'source': {'artifacts': f'{MODELS_DIR}/<model>/',
                   'compression': 'zstd at the level common.compression uses, so the '
                                  'compressed columns are the exact bytes the gateway holds',
                   'na_means': 'the artifact was not exported yet (train + seed the model '
                               'first)'},
    })
    print(f"wrote footprint table to {report_dir}/ (results root {RESULTS_DIR})")


if __name__ == "__main__":
    main()
