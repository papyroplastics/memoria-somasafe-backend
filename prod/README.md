# Prod

A production-like container deployment of the backend. The same images and compose files
run the local stack and the cloud benchmark (see [`../benchmark/`](../benchmark/README.md)
and `plans/cloud-benchmark.md`).

## Stack

`compose.prod.yaml` sits on top of `compose.yaml` (which provides `postgres`, `redis-auth`
and `redis-broker`) and adds:

- `fastapi-1`/`fastapi-2` gateways (`api.Containerfile`, no TensorFlow) behind Caddy, which
  round-robins across `API_UPSTREAMS` on port 8000.
- `celery-1` (light + heavy queues, runs beat) and `celery-2` (heavy only), from
  `worker.Containerfile`, with the calibration artifacts baked in. Each writes per-task
  stage timings to the `worker_metrics` volume as `<hostname>-<pid>/<task>.csv`.
- Postgres configured by `postgres.conf`, sized for the cloud's dedicated data host.
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
make db-seed         # once, against the running stack (add --test-users N for more users)
make prod-clean      # tear down, volumes included
```

## Configuration

`prod.env` configures every container. Besides credentials, it compresses the round
cadence so a run of a few minutes sees several rounds: aggregation every 60 s, secure
sessions sealing after 15 s and failing 20 s after sealing. It also sets
`WORKER_CONCURRENCY=4` (the cloud value, too much for a local 2x2 on a small machine;
override it from the shell) and a long access-token TTL so no user re-logs in mid-run.
Containers only pick up changes to it when they are recreated.

`postgres.conf` assumes a dedicated 16 GB host (4 GB of shared buffers). To run the
image's stock config instead, for example on a small local machine, point
`POSTGRES_CONFIG_FILE` at it:

```bash
POSTGRES_CONFIG_FILE=/var/lib/postgresql/18/docker/postgresql.conf WORKER_CONCURRENCY=1 make prod-x2-run
```

Start every service of the topology before seeding: bringing up `fastapi-2`/`celery-2` on a
running stack recreates the data services, and celery-exporter misses events for a while
after.
