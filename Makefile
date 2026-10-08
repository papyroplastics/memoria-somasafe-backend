shared_repo := https://github.com/papyroplastics/memoria-somasafe-shared.git

.PHONY: shared ml-data ml-test db-seed db-reseed db-run db-clean prod-db-seed api-run api-test worker-run worker-2-run worker-test worker-monitor prod-build prod-run prod-x2-run prod-clean
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
	podman compose down -v

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

prod_env := prod/local.env
prod_compose := PROD_ENV_FILE=$(prod_env) PODMAN_COMPOSE_PROVIDER=podman-compose podman compose -f compose.yaml -f compose.prod.yaml --env-file $(prod_env)
prod_x1_profiles := edge fastapi-1 celery-1 postgres redis-auth redis-broker
prod_x2_profiles := $(prod_x1_profiles) fastapi-2 celery-2
prod_x1_compose := API_UPSTREAMS="fastapi-1:8000" $(prod_compose) $(addprefix --profile ,$(prod_x1_profiles))
prod_x2_compose := API_UPSTREAMS="fastapi-1:8000 fastapi-2:8000" $(prod_compose) $(addprefix --profile ,$(prod_x2_profiles))

prod-build:
	$(prod_x1_compose) build
prod-run:
	$(prod_x1_compose) up
prod-x2-run:
	$(prod_x2_compose) up
prod-clean:
	$(prod_x2_compose) down -v
prod-db-seed: shared
	set -a && . $(prod_env) && set +a && POSTGRES_HOST=localhost uv run -m scripts.system.seed_db --assign-device --test-users
