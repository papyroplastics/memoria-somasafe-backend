# Prod

A production-like container deployment of the backend. The same images and compose files
run the local stack and the cloud benchmark (see [`../benchmark/`](../benchmark/README.md)
and `plans/cloud-benchmark.md`).

## Stack

`compose.prod.yaml` sits on top of `compose.yaml` (which provides `postgres` and
`redis-auth`) and adds:

- A separate `redis-broker` instance for the Celery broker.
- `fastapi-1`/`fastapi-2` gateways (`api.Containerfile`, no TensorFlow) behind Caddy, which
  round-robins across `API_UPSTREAMS` on port 8000.
- `celery-1` (light + heavy queues, runs beat) and `celery-2` (heavy only), from
  `worker.Containerfile`, with the calibration artifacts baked in. Each writes per-task
  stage timings to the `worker_metrics` volume as `<hostname>-<pid>/<task>.csv`.
- Postgres configured by `postgres.conf` in the cloud, sized for its dedicated data host.
- Prometheus (port 9090, `prometheus.yml`) scraping the gateways, Caddy, celery-exporter,
  the Postgres and Redis exporters, node-exporter and cAdvisor.

## Profiles

Every service belongs to one or more profiles, so each cloud host starts only its own
services from the same files. Profiles are named after the host's main compose service, and
the cloud instances take the same name with an `-inst` suffix (`fastapi-1-inst`, ...):

| Profile | Services |
| --- | --- |
| `fastapi-1`, `fastapi-2` | `fastapi-1`, `fastapi-2` |
| `celery-1`, `celery-2` | `celery-1`, `celery-2` |
| `postgres` | `postgres`, `postgres-exporter` |
| `redis-auth` | `redis-auth`, `redis-auth-exporter` |
| `redis-broker` | `redis-broker`, `redis-broker-exporter` |
| `edge` | `caddy`, `prometheus`, `celery-exporter` |
| `client` | nothing besides the per-host exporters (Locust runs outside compose) |

`node-exporter` and `cAdvisor` are in every profile. Services on different hosts don't
declare `depends_on` on each other: the gateways connect lazily and the workers retry the
broker. The Makefile enables every profile of a topology on one machine:

```bash
make prod-build
make prod-run        # 1x1
make prod-x2-run     # 2x2 (adds fastapi-2 and celery-2)
make prod-db-seed    # once, against the running stack (add --test-users N for more users)
make prod-clean      # tear down, volumes included
```

## Configuration

Every container reads one env file, which is also the compose `--env-file`, so it sets the
compose variables too (ports, `BIND_ADDR`, `WORKER_CONCURRENCY`, `POSTGRES_CONFIG_FILE`).
`compose.prod.yaml` takes its path from `PROD_ENV_FILE`, and both the variable and the flag
must point at the same file:

- `local.env`, used by the Makefile, runs the whole stack on one small machine: services
  reach each other by their compose names, every process runs a single worker, and
  Postgres uses the image's stock config instead of `postgres.conf`.
- `cloud.env` is for the cloud hosts: services reach each other by the `-inst` host names,
  ports are published on every interface, the broker listens on 6379 on its own host,
  and Postgres uses `postgres.conf`, sized for a dedicated 16 GB host. Each host starts
  only its own profile:

  ```bash
  PROD_ENV_FILE=prod/cloud.env podman compose -f compose.yaml -f compose.prod.yaml \
    --env-file prod/cloud.env --profile fastapi-1 up
  ```

Compose only reads the first `--env-file`, so the shared values are duplicated in both
files. Besides credentials, they compress the round cadence so a run of a few minutes sees
several rounds (aggregation every 60 s, secure sessions sealing after 15 s and failing
20 s after sealing) and set a long access-token TTL so no user re-logs in mid-run.
Containers only pick up changes to the env file when they are recreated.

`make prod-db-seed` runs the seed script on the host with `local.env` loaded, so the seeded
credentials match the stack's, connecting to Postgres through its published port.

Start every service of the topology before seeding: bringing up `fastapi-2`/`celery-2` on a
running stack recreates the data services, and celery-exporter misses events for a while
after.
