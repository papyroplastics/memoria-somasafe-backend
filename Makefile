shared_repo := https://github.com/papyroplastics/memoria-somasafe-shared.git

-include .env

.PHONY: shared ml-data ml-test db-seed db-reseed db-run db-clean api-run api-test worker-run worker-2-run worker-test worker-monitor prod-build prod-run prod-x2-run prod-clean
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
	podman compose $(if $(BROKER_PORT),--profile broker) up
db-clean:
	podman compose --profile broker down -v

api-run:
	uv run fastapi dev api --host 0.0.0.0
api-test:
	uv run pytest api/test/

worker-run:
	uv run -m celery --app worker.celery_app worker --queues light,heavy --beat --loglevel=info
worker-2-run:
	uv run -m celery --app worker.celery_app worker --queues heavy --loglevel=info
worker-test:
	uv run pytest worker/test/
worker-monitor:
	uv run -m celery --app worker.celery_app flower

prod_compose := PODMAN_COMPOSE_PROVIDER=podman-compose podman compose -f compose.yaml -f compose.prod.yaml --env-file benchmark/prod.env --profile broker
prod_x1_compose := API_UPSTREAMS="api-1:8000" $(prod_compose)
prod_x2_compose := API_UPSTREAMS="api-1:8000 api-2:8000" $(prod_compose) --profile x2

prod-build:
	$(prod_compose) build
prod-run:
	$(prod_x1_compose) up
prod-run-x2:
	$(prod_x2_compose) up
prod-clean:
	$(prod_x2_compose) down -v

