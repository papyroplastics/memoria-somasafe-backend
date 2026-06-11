import datetime
from enum import Enum
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import tensorflow as tf

from worker.models.feature_mlp import FeatureMLP, load_feature_dataset, get_rep_dataset_feed
from worker.saving import get_optimized_model

router = APIRouter(prefix="/model")

data_dir = Path('datasets/')
model_dir = Path('results/')
model_file = "post-train-odt.tflite"

batch_size = 1


class ModelPurpose(str, Enum):
    train_only = "train-only"
    embed_infer = "embed-infer"
    app_infer = "app-infer"


feature_mlp_ds = load_feature_dataset(data_dir, batch_size, 1234)[1]
feature_mlp_n_feat = int(next(iter(feature_mlp_ds))[0].shape[-1])

models = {
    "feature-mlp": {
        "meta": {
            "name": "Feature-based MLP",
            "last_updated": datetime.datetime(2026, 6, 1),
            "purpose": ModelPurpose.train_only,
            "firmware_id": None,
            "app_version": "1.0.0",
            "model_id": 1,
        },
        "model": FeatureMLP(
            name='feature_anomaly',
            batch_size=batch_size,
            n_features=feature_mlp_n_feat,
            hidden_dim=32,
            hidden_layers=4,
            learning_rate=1e-3,
        ),
        "rep_dataset": get_rep_dataset_feed(feature_mlp_ds),
    },
}

del feature_mlp_ds, feature_mlp_n_feat

MODEL_LIST = [{"key": key, **entry["meta"]} for key, entry in models.items()]


class ModelWeights(BaseModel):
    parameters: list[float]


@router.get("/list")
async def list_models():
    return MODEL_LIST


@router.get("/trainable/{key}/{id}", response_class=FileResponse)
async def get_model(key: str, id: int):
    entry = models.get(key)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' not found")
    if id != entry["meta"]["model_id"]:
        raise HTTPException(status_code=404, detail=f"Model '{key}' version {id} not found")
    return FileResponse(path=model_dir / key / model_file, filename=f"{key}.tflite")


@router.post("/quantize/{key}/{id}")
async def quantize_model(key: str, id: int, weights: ModelWeights):
    entry = models.get(key)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' not found")
    if id != entry["meta"]["model_id"]:
        raise HTTPException(status_code=404, detail=f"Model '{key}' version {id} not found")

    entry["model"].restore(tf.constant(weights.parameters, dtype=tf.float32))
    buf = get_optimized_model(entry["model"], entry["rep_dataset"])

    return Response(
        content=buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{key}.tflite"'},
    )
