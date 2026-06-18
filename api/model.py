import datetime
from enum import Enum
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import tensorflow as tf

from worker.models.feature_mlp import get_trainer as get_feature_mlp_trainer
from worker.saving import get_optimized_model

router = APIRouter(prefix="/model")

seed = 1234
data_dir = Path('datasets/')
model_dir = Path('results/')
model_file = "post-train-odt.tflite"

class ModelPurpose(str, Enum):
    train_only = "train-only"
    embed_infer = "embed-infer"
    app_infer = "app-infer"

feature_mlp_trainer = get_feature_mlp_trainer(data_dir, seed)
feature_mlp_eval = feature_mlp_trainer.combine(
    feature_mlp_trainer.subject_datasets(data_dir, seed)[1])

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
        "model": feature_mlp_trainer.model,
        "rep_dataset": feature_mlp_trainer.representative_dataset(feature_mlp_eval),
    },
}

del feature_mlp_eval

model_list = [{"key": key, **entry["meta"]} for key, entry in models.items()]


class ModelWeights(BaseModel):
    parameters: list[float]


@router.get("/list")
async def list_models():
    return model_list


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
