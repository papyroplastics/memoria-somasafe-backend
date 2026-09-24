import gzip
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from common.config import SEED

from .common import DataSource

DATASET_URL = 'https://archive.ics.uci.edu/static/public/31/covertype.zip'

N_QUANT = 10
N_WILDERNESS = 4
N_SOIL = 40
N_BINARY = N_WILDERNESS + N_SOIL
N_FEATURES = N_QUANT + N_BINARY
N_CLASSES = 7

QUANT_FILE = 'quant.npy'
BINARY_FILE = 'binary_packed.npy'
LABELS_FILE = 'labels.npy'


class DatasetUnavailibleError(FileNotFoundError):
    def __init__(self, data_dir: str | Path):
        self.message = f"Dataset not found at {data_dir}. Run scripts/get_dataset.py first."
        super().__init__(self.message)


def prepare_covertype(datasets_dir: Path) -> None:
    covertype_dir = datasets_dir / 'covertype'
    files = [QUANT_FILE, BINARY_FILE, LABELS_FILE]
    if all((covertype_dir / f).exists() for f in files):
        print(f"Covertype already present at {covertype_dir}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        archive = tmp_dir / 'covertype.zip'
        print(f"Downloading {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, archive)

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_dir)

        with gzip.open(tmp_dir / 'covtype.data.gz', 'rb') as gz:
            rows = np.loadtxt(gz, delimiter=',', dtype=np.int32)

    quant = rows[:, :N_QUANT].astype(np.int16)
    binary = rows[:, N_QUANT:N_QUANT + N_BINARY].astype(bool)
    labels = rows[:, N_QUANT + N_BINARY].astype(np.uint8)

    covertype_dir.mkdir(parents=True, exist_ok=True)
    np.save(covertype_dir / QUANT_FILE, quant)
    np.save(covertype_dir / BINARY_FILE, np.packbits(binary, axis=1))
    np.save(covertype_dir / LABELS_FILE, labels)
    print(f"Covertype ready at {covertype_dir}: {len(labels)} rows")


class CovertypeSource(DataSource):
    def __init__(self, data_root: Path, key: str):
        self.key = key
        self.data_root = data_root
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._shards: list[np.ndarray] | None = None

    def _ensure_loaded(self) -> None:
        if self._shards is not None:
            return

        covertype_dir = self.data_root / 'covertype'
        quant_path = covertype_dir / QUANT_FILE
        binary_path = covertype_dir / BINARY_FILE
        labels_path = covertype_dir / LABELS_FILE
        if not quant_path.exists():
            raise DatasetUnavailibleError(covertype_dir)

        quant = np.load(quant_path).astype(np.float32)
        binary = np.unpackbits(np.load(binary_path), axis=1)[:, :N_BINARY].astype(np.float32)
        self._features = np.concatenate([quant, binary], axis=1)
        self._labels = np.load(labels_path)

        wilderness = binary[:, :N_WILDERNESS]
        areas = np.argmax(wilderness, axis=1)
        self._shards = [np.where(areas == i)[0] for i in range(N_WILDERNESS)]

    def subject_ids(self) -> list[str]:
        return [f'wilderness{i + 1}' for i in range(N_WILDERNESS)]

    def _shard(self, sid: str) -> np.ndarray:
        self._ensure_loaded()
        return self._shards[int(sid[len('wilderness'):]) - 1]

    def datapoints(self, sid: str) -> np.ndarray:
        self._ensure_loaded()
        return self._features[self._shard(sid)]

    def labels(self, sid: str) -> np.ndarray:
        self._ensure_loaded()
        return np.eye(N_CLASSES, dtype=np.float32)[self._labels[self._shard(sid)] - 1]

    def calibration_data(self, per_subject: int = 100) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        parts = []
        for sid in self.subject_ids():
            points = self.datapoints(sid)
            if len(points):
                parts.append(points[rng.choice(len(points), min(per_subject, len(points)),
                                               replace=False)])
        if not parts:
            raise DatasetUnavailibleError(self.data_root)
        return np.concatenate(parts)
