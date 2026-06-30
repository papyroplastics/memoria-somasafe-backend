nvs_csv := ../firmware/factory_nvs.csv

proto:
	protoc --proto_path=shared --python_out=scripts/common shared/dataset.proto

get-data:
	uv run -m scripts.get_dataset
seed-db:
	uv run -m scripts.seed_db ${nvs_csv}

api-run:
	uv run fastapi dev api --host 0.0.0.0
api-test:
	uv run pytest api/test/

worker-run:
	uv run -m celery -A worker.celery_app worker -B --loglevel=info

