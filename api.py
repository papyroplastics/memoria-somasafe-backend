from enum import Enum
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

class ModelType(str, Enum):
    quant = "quantized"
    train = "trainable"

app = FastAPI()

models_dir = Path('models/')
model_file_train = models_dir / "post-train-odt.tflite"
model_file_quant = models_dir / "post-train-opti.tflite"

@app.get("/model/{type}", response_class=FileResponse)
async def main(type: ModelType, version: int | None = None):
    if type == ModelType.train:
        if version == None:
            return FileResponse(path=model_file_train, filename="trainable.tflite")

    if type == ModelType.quant:
        if version == None:
            return FileResponse(path=model_file_quant, filename="quantized.tflite")
