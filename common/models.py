from enum import Enum


class ModelPurpose(str, Enum):
    train_only = "train-only"
    embed_infer = "embed-infer"
    app_infer = "app-infer"


# The model registry itself now lives in the database (ModelDefinition in
# common.db, seeded by scripts.seed). This module keeps only the TF-free enum
# shared by the table definition and the worker.
