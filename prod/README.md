# Prod

A production-like container deployment of the backend. The same images and compose files
run the local stack and the cloud stack that the [benchmark](../benchmark/README.md) drives.

## Stack

`compose.prod.yaml` sits on top of `compose.yaml` (`postgres` and `redis-auth`) and adds:

- `fastapi-1`/`fastapi-2` gateways (`api.Containerfile`, no TensorFlow), behind Caddy
  locally, which round-robins across `API_UPSTREAMS` and health-checks `/healthz`.
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
| `fastapi-1`, `fastapi-2` | the gateway |
| `celery-1`, `celery-2` | the worker |
| `postgres` | `postgres`, `postgres-exporter` |
| `redis-auth` | `redis-auth`, `redis-auth-exporter` |
| `edge` | `caddy` |
| `monitor` | `prometheus`, `celery-exporter` |
| `client-1` | only the per-host exporters |
| `bench` | `bench`, only through `run` |

`node-exporter` and `cadvisor` are in every host profile.

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
uses the `-inst` host names, publishes ports on every interface and loads `postgres.conf`,
sized for a 4 GB host. Both compress the round cadence (aggregation every 60 s, short seal
and fail timeouts, a 5 s sweep) so a few minutes see several rounds, and set a long token
TTL. Containers only pick up env changes when recreated.

Host-side commands (`prod-db-seed`, `prod-collect`) load `local.env` with Postgres and Redis
on `localhost`, which is either the local stack or the cloud tunnel.

## Cloud

`terraform/` deploys a 1x1 stack on Compute Engine, one VM per host named `<host>-inst`:
`fastapi-1`, `celery-1`, `postgres`, `redis-auth` and `client-1`, which runs `monitor` and
is the SSH bastion. There is no Caddy; the client hits `fastapi-1-inst` directly. Sizes fit
a 12 vCPU quota. Hosts have no external IPv4: SSH goes over IPv6, outgoing traffic through
Cloud NAT.

Each host's startup script (`terraform/templates/`) writes the compose files, `cloud.env`,
a DNS search domain for containers and a few helpers. The `somasafe` unit then pulls the
host's images (and the server key on `celery-1`) and starts its profiles, retrying every
30 s until they exist. `sudo somasafe-compose ...` runs compose with the host's profiles.

### One-time setup

```bash
gcloud auth login
gcloud auth application-default login
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iam.googleapis.com --project <project>
echo 'project = "<project>"' > prod/terraform/terraform.tfvars
terraform -chdir=prod/terraform init
```

Your OS Login account needs an SSH key enrolled (`gcloud compute os-login ssh-keys add`)
and loaded in the agent. Hosts run as the `somasafe-host` service account, which can only
read the registry and the key secret.

### Session

```bash
terraform -chdir=prod/terraform apply
prod/connect.sh       # own terminal: tunnels Postgres, Redis and Prometheus until killed
prod/init.sh 200      # builds and pushes the images, uploads the key, seeds 200 users
```

Per run:

```bash
ssh <user>@<client-1-inst>       # address in `terraform output hosts`
sudo somasafe-compose --profile bench run --rm bench benchmark.run 1x1 --processes 2 ...
prod/collect.sh <run_id>         # back on the local machine
```

`collect.sh` pulls the run, runs `prod-collect` (settle, checks, reset), and copies the
worker metrics and a Prometheus snapshot to `results/worker-metrics` and
`results/benchmark/tsdb`.

`terraform destroy` ends the session; VMs also stop after `max_run_hours`. After changing a
deployed file, `apply` updates the startup script in place, and `sudo
google_metadata_script_runner startup && sudo systemctl restart somasafe` on a host
applies it.
