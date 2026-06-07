from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()

models_dir = Path('models/')

MODELS = [
    {
        "key": "feature-mlp",
        "name": "Feature-based MLP",
        "last_updated": "2026-06-01T00:00:00Z",
        "purpose": "train-only",
        "_file": "post-train-odt.tflite",
    },
]

@app.get("/models")
async def list_models():
    return [{k: v for k, v in m.items() if not k.startswith('_')} for m in MODELS]

@app.get("/model/{key}", response_class=FileResponse)
async def get_model(key: str):
    entry = next((m for m in MODELS if m["key"] == key), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' not found")
    return FileResponse(path=models_dir / entry["_file"], filename=f"{key}.tflite")
