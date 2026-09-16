from pathlib import Path

import numpy as np
import tensorflow as tf

from common.config import SEED

from .common import DataSource

N_SHARDS = 10
N_CLASSES = 10
DIRICHLET_ALPHA = 0.5
IMAGE_PIXELS = 28 * 28

IID = 'iid'
NONIID = 'noniid'
PARTITIONS = (IID, NONIID)

TRAIN_IMAGES = 'train_images.npy'
TRAIN_LABELS = 'train_labels.npy'
TEST_IMAGES = 'test_images.npy'
TEST_LABELS = 'test_labels.npy'


class DatasetUnavailibleError(FileNotFoundError):
    def __init__(self, data_dir: str | Path):
        self.message = f"Dataset not found at {data_dir}. Run scripts/get_dataset.py first."
        super().__init__(self.message)


def prepare_mnist(datasets_dir: Path) -> None:
    mnist_dir = datasets_dir / 'mnist'
    files = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]
    if all((mnist_dir / f).exists() for f in files):
        print(f"MNIST already present at {mnist_dir}")
        return

    mnist_dir.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    np.save(mnist_dir / TRAIN_IMAGES, x_train.astype(np.uint8))
    np.save(mnist_dir / TRAIN_LABELS, y_train.astype(np.uint8))
    np.save(mnist_dir / TEST_IMAGES, x_test.astype(np.uint8))
    np.save(mnist_dir / TEST_LABELS, y_test.astype(np.uint8))
    print(f"MNIST ready at {mnist_dir}: {len(x_train)} train, {len(x_test)} test")


def iid_partition(labels: np.ndarray, n_shards: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(len(labels))
    return list(np.array_split(order, n_shards))


def dirichlet_partition(labels: np.ndarray, n_shards: int, alpha: float,
                        rng: np.random.Generator) -> list[np.ndarray]:
    shards: list[list[int]] = [[] for _ in range(n_shards)]
    for cls in range(N_CLASSES):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * n_shards)
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for shard, part in zip(shards, np.split(idx, splits)):
            shard.extend(part.tolist())
    return [np.array(shard, dtype=np.int64) for shard in shards]


class MnistShardSource(DataSource):
    def __init__(self, data_root: Path, key: str, partition: str, alpha: float = DIRICHLET_ALPHA):
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition {partition!r}, expected one of {PARTITIONS}")

        self.key = key
        self.data_root = data_root
        self.partition = partition

        mnist_dir = data_root / 'mnist'
        images_path, labels_path = mnist_dir / TRAIN_IMAGES, mnist_dir / TRAIN_LABELS
        if not images_path.exists():
            raise DatasetUnavailibleError(mnist_dir)

        self._images = np.load(images_path)
        self._labels = np.load(labels_path)

        rng = np.random.default_rng([SEED, PARTITIONS.index(partition)])
        self._shards = (iid_partition(self._labels, N_SHARDS, rng) if partition == IID
                        else dirichlet_partition(self._labels, N_SHARDS, alpha, rng))

    def subject_ids(self) -> list[str]:
        return [f'shard{i}' for i in range(N_SHARDS)]

    def _shard(self, sid: str) -> np.ndarray:
        return self._shards[int(sid[len('shard'):])]

    def datapoints(self, sid: str) -> np.ndarray:
        images = self._images[self._shard(sid)].reshape(-1, IMAGE_PIXELS).astype(np.float32)
        return images / 255.0

    def labels(self, sid: str) -> np.ndarray:
        return np.eye(N_CLASSES, dtype=np.float32)[self._labels[self._shard(sid)]]

    def calibration_data(self, per_subject: int = 10) -> np.ndarray:
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


class MnistTestSource(DataSource):
    def __init__(self, data_root: Path, key: str):
        self.key = key
        self.data_root = data_root

        mnist_dir = data_root / 'mnist'
        images_path, labels_path = mnist_dir / TEST_IMAGES, mnist_dir / TEST_LABELS
        if not images_path.exists():
            raise DatasetUnavailibleError(mnist_dir)

        self._images = np.load(images_path)
        self._labels = np.load(labels_path)

    def subject_ids(self) -> list[str]:
        return ['test']

    def datapoints(self, sid: str) -> np.ndarray:
        images = self._images.reshape(-1, IMAGE_PIXELS).astype(np.float32)
        return images / 255.0

    def labels(self, sid: str) -> np.ndarray:
        return np.eye(N_CLASSES, dtype=np.float32)[self._labels]

    def calibration_data(self, per_subject: int = 100) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        points = self.datapoints('test')
        return points[rng.choice(len(points), min(per_subject, len(points)), replace=False)]
