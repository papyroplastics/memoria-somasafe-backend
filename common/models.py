import datetime
from enum import Enum


class ModelPurpose(str, Enum):
    train_only = "train-only"
    embed_infer = "embed-infer"
    app_infer = "app-infer"


# Shared model metadata registry (no TensorFlow). The gateway serves it and
# validates requests against it; the worker maps each key to a trainer builder
# on top of it. A model-version table will replace this once aggregation lands.
models = {
    "feature-mlp": {
        "name": "Feature-based MLP",
        "last_updated": datetime.datetime(2026, 6, 1),
        "purpose": ModelPurpose.train_only,
        "firmware_id": None,
        "app_version": "1.0.0",
        "model_id": 1,
    },
}

model_list = [{"key": key, **meta} for key, meta in models.items()]
