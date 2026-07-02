shared_repo := https://github.com/papyroplastics/memoria-somasafe-shared.git

shared:
	@if [ -e shared ] || [ -L shared ]; then \
		echo "shared already present"; \
	elif [ -d ../shared ]; then \
		ln -sr ../shared/ .; \
	else \
		git clone ${shared_repo} shared; \
	fi

proto: shared
	protoc --proto_path=shared --python_out=scripts/common shared/dataset.proto

get-data: shared
	uv run -m scripts.get_dataset
seed-db:
	uv run -m scripts.seed_db

api-run:
	uv run fastapi dev api --host 0.0.0.0
api-test:
	uv run pytest api/test/

worker-run:
	uv run -m celery -A worker.celery_app worker -B --loglevel=info

