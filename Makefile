ml-get-data:
	uv run -m ml.scripts.get-dataset
ml-train:
	uv run -m ml.scripts.train feature-mlp

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
