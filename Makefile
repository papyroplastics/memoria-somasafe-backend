shared_repo := https://github.com/papyroplastics/memoria-somasafe-shared.git

.PHONY: shared ml-data ml-test db-seed db-reseed db-run db-clean api-run api-test worker-run worker-test beat-run worker-monitor
shared:
	@if [ -e shared ] || [ -L shared ]; then \
		echo "shared already present"; \
	elif [ -d ../shared ]; then \
		ln -sr ../shared/ .; \
	else \
		git clone ${shared_repo} shared; \
	fi
	$(MAKE) -C shared setup

ml-data: shared
	uv run -m scripts.system.get_dataset ppg-dalia

ml-test:
	uv run pytest ml/test/

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
	uv run -m celery -A worker.celery_app worker -Q light,heavy --loglevel=info
worker-test:
	uv run pytest worker/test/
beat-run:
	uv run -m celery -A worker.celery_app beat --loglevel=info
worker-monitor:
	uv run -m celery -A worker.celery_app flower


