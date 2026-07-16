shared_repo := https://github.com/papyroplastics/memoria-somasafe-shared.git

shared:
	@if [ -e shared ] || [ -L shared ]; then \
		echo "shared already present"; \
	elif [ -d ../shared ]; then \
		ln -sr ../shared/ .; \
	else \
		git clone ${shared_repo} shared; \
	fi
	$(MAKE) -C shared setup

proto: shared
	protoc --proto_path=shared --python_out=scripts/common shared/dataset.proto

get-data: shared
	uv run -m scripts.system.get_dataset

# Every report result, from a clean slate. Needs the services + api + worker up.
run-all: shared
	./run_all.sh

db-seed: shared
	uv run -m scripts.system.seed_db --assign-device --test-users
db-reseed: shared
	uv run -m scripts.system.seed_db --assign-device --test-users --reseed
db-run:
	podman compose up
db-clean:
	podman compose down
	podman volume rm -a

api-run:
	uv run fastapi dev api --host 0.0.0.0
api-test:
	uv run pytest api/test/

worker-run:
	uv run -m celery -A worker.celery_app worker -B --loglevel=info
worker-monitor:
	uv run -m celery -A worker.celery_app flower


