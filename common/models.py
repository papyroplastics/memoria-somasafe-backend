from enum import Enum


class ModelPurpose(str, Enum):
    train_only = "train-only"
    embed_infer = "embed-infer"
    app_infer = "app-infer"


# The model registry lives in ml.model_list (TensorFlow) and is projected into
# the database by scripts.db_seed. This module keeps only the TF-free enum,
# shared by the registry, the DB tables, and the gateway.
