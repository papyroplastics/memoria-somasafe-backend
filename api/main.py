from fastapi import FastAPI
from .model import router

app = FastAPI()
app.include_router(router)
