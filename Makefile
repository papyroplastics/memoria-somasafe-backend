get-data:
	uv run -m scripts.get-dataset
train-model:
	uv run -m scripts.train feature-mlp
seed:
	uv run -m scripts.seed

api-run:
	uv run fastapi dev api --host 0.0.0.0
api-test:
	uv run pytest api/test.py

worker-run:
	uv run -m celery -A worker.celery_app worker -B --loglevel=info

services-up:
	podman compose -f compose.yaml up -d
services-down:
	podman compose -f compose.yaml down
