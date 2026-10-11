# Prod

A production-like container deployment of the backend. The same images and compose files
run the local stack and the cloud stack that the [benchmark](../benchmark/README.md) drives.

## Stack

`compose.prod.yaml` sits on top of `compose.yaml` (`postgres` and `redis-auth`) and adds:

- `fastapi-1` to `fastapi-4` gateways (`api.Containerfile`, no TensorFlow), behind Caddy
  locally, which round-robins across `API_UPSTREAMS` and health-checks `/healthz`. Locally
  only the first two are used; `fastapi-3`/`fastapi-4` exist for the cloud topologies.
- `celery-1` (light + heavy queues, runs beat) and `celery-2` (heavy only), from
  `worker.Containerfile`, writing per-task stage timings to `results/worker-metrics`.
- Prometheus and celery-exporter, plus Postgres and Redis exporters, node-exporter and
  cAdvisor. The TSDB is bind-mounted at `results/prometheus`.
- A one-shot `bench` container (`../benchmark/bench.Containerfile`) with `results/` mounted.
- A `redis-broker` service that no profile list enables: the Celery broker is db 1 of
  `redis-auth` unless `BROKER_HOST`/`BROKER_PORT` point elsewhere.

Every service belongs to profiles, so each cloud host starts only its own services:

| Profile | Services |
| --- | --- |
| `fastapi-1` to `fastapi-4` | the gateway |
| `celery-1`, `celery-2` | the worker |
| `postgres` | `postgres` |
| `redis-auth` | `redis-auth` |
| `edge` | `caddy` |
| `monitor` | `prometheus` and the Celery, Postgres and Redis exporters |
| `client-1` | only the per-host exporters |
| `bench` | `bench`, only through `run` |

`node-exporter` and `cadvisor` are in every host profile. The Postgres and Redis exporters
reach their services through `POSTGRES_HOST`/`REDIS_HOST`, so they also work against the
managed ones in the cloud, where they skip Redis `CONFIG`, which Memorystore blocks.

## Local

```bash
make prod-build
make prod-run        # 1x1, or prod-x2-run to add fastapi-2 and celery-2
make prod-db-seed    # ARGS=N for N test users
make prod-bench ARGS="benchmark.run 1x1 ..."
make prod-collect RUN=<run_id>
make prod-clean      # volumes included; results/ is left alone
```

Start the whole topology before seeding: adding `fastapi-2`/`celery-2` to a running stack
recreates the data services.

## Configuration

Every container reads one env file, which is also the compose `--env-file` (compose only
reads the first one, so shared values are duplicated). `local.env` runs everything on one
small machine by compose names, one worker process each, stock Postgres config. `cloud.env`
uses the `-inst` host names, publishes ports on every interface and leaves out the per-host
values, which Terraform appends to each host's copy: `POSTGRES_HOST`/`REDIS_HOST` with the
managed services' addresses, `BENCH_API_URL` with every gateway of the topology
(comma-separated), and `WEB_CONCURRENCY`/`WORKER_CONCURRENCY` with the host's vCPU count.
`postgres.conf`, sized for a 4 GB host, is only used locally. Both compress the round
cadence (aggregation every 60 s locally and 30 s in the cloud, short seal and fail timeouts,
a 5 s or 2 s sweep) so a few minutes see several rounds, and set a long token TTL.
Containers only pick up env changes when recreated.

`DB_POOL_SIZE`/`DB_MAX_OVERFLOW` size the SQLAlchemy pool of every process (5 and 10 by
default). The cloud uses 40 and 10, so a gateway process never waits on the pool before its
40 threadpool threads are busy; Cloud SQL allows 1,000 connections.

Host-side commands (`prod-db-seed`, `prod-collect`) load `local.env` with Postgres and Redis
on `localhost`, which is either the local stack or the cloud tunnel.

## Cloud

`terraform/` deploys a stack in `us-east1-c`, one Compute Engine VM per host named
`<host>-inst`, on c4 machines with Hyperdisk Balanced. `-var topology=...` picks one of the
presets in `variables.tf`, each filling the 12 vCPU quota:

| Topology | Gateways | `celery-1` | `client-1` |
| --- | --- | --- | --- |
| `1x1` (default) | 1 × `c4-standard-4` | `c4-standard-4` | `c4-standard-4` |
| `2x1` | 2 × `c4-standard-4` | `c4-standard-2` | `c4-standard-2` |
| `4x1` | 4 × `c4-standard-2` | `c4-standard-2` | `c4-standard-2` |

`client-1` runs `monitor` and is the SSH bastion. There is no Caddy: the benchmark spreads
its users across the gateways itself. `prometheus.cloud.yml` is a Terraform template that
gets the topology's gateways and servers, so it only works once rendered. Hosts have no
external IPv4: SSH goes over IPv6, outgoing traffic through Cloud NAT.

Pass the same `-var topology=...` to every `apply` and `destroy`, since a plain `apply`
goes back to `1x1`. Switching to a topology with more hosts can exceed the quota while the
old VMs still exist: shrink or remove them first with `-target`, then apply the rest.

Postgres and Redis are managed services, which don't count against that quota: Cloud SQL
(Postgres 18, Enterprise Plus, `sql_tier`, `db-perf-optimized-N-8` by default) and Memorystore (Valkey 9.1, cluster mode
disabled, `redis_node_type`), both zonal, without replicas, backups or persistence, and
without auth or TLS. Each is reached through a Private Service Connect endpoint that a
service connection policy creates in the subnet; its address is a Terraform output
(`postgres_host`, `redis_host`). The Cloud SQL instance gets a generated name, since a
deleted one can't be reused for a week. The managed services are left out of
`max_run_hours` and keep running until `terraform destroy`.

Each host's startup script (`terraform/templates/`) writes the compose files, `cloud.env`,
a DNS search domain for containers and a few helpers. The `somasafe` unit then pulls the
host's images (and the server key on `celery-1`) and starts its profiles, retrying every
30 s until they exist. `sudo somasafe-compose ...` runs compose with the host's profiles.

### One-time setup

```bash
gcloud auth login
gcloud auth application-default login
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iam.googleapis.com sqladmin.googleapis.com \
  memorystore.googleapis.com networkconnectivity.googleapis.com --project <project>
echo 'project = "<project>"' > prod/terraform/terraform.tfvars
terraform -chdir=prod/terraform init
```

Your OS Login account needs an SSH key enrolled (`gcloud compute os-login ssh-keys add`)
and loaded in the agent. Hosts run as the `somasafe-host` service account, which can only
read the registry and the key secret.

### Session

```bash
terraform -chdir=prod/terraform apply -var topology=1x1
prod/connect.sh       # own terminal: tunnels Postgres, Redis and Prometheus through client-1 until killed
prod/init.sh 200      # builds and pushes the images, uploads the key, seeds 200 users
```

Per run:

```bash
ssh <user>@<client-1-inst>       # address in `terraform output hosts`
sudo somasafe-compose --profile bench run --rm bench benchmark.run <topology> --processes 2 ...
prod/collect.sh <run_id>         # back on the local machine
```

`collect.sh` pulls the run, runs `prod-collect` (settle, checks, reset), and copies the
worker metrics and a Prometheus snapshot to `results/worker-metrics` and
`results/benchmark/tsdb`.

`terraform destroy` ends the session; VMs also stop after `max_run_hours`. After changing a
deployed file, `apply` updates the startup script in place, and `sudo
google_metadata_script_runner startup && sudo systemctl restart somasafe` on a host
applies it.
