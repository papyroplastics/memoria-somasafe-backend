get-data:
	uv run -m scripts.get_dataset
train-model:
	uv run -m scripts.train feature-mlp

nvs_csv := ../firmware/factory_nvs.csv
seed:
	uv run -m scripts.db_seed ${nvs_csv}

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
