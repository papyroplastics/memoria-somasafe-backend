# Benchmark

A production-like deployment of the backend plus the tooling to load test it. The same
images and compose files run the local rehearsal and the cloud benchmark (see
`plans/2-local-stress.md` and `plans/3-cloud-benchmark.md`).

## Stack

`compose.prod.yaml` sits on top of `compose.yaml` and adds:

- `api-1`/`api-2` gateways (`api.Containerfile`, no TensorFlow) behind Caddy, which
  round-robins across `API_UPSTREAMS` on port 8000.
- `worker-1` (light + heavy queues, runs beat) and `worker-2` (heavy only), from
  `worker.Containerfile`, with the calibration artifacts baked in. Each writes per-task
  stage timings to the `worker_metrics` volume as `<hostname>-<pid>/<task>.csv`.
- Prometheus (port 9090) scraping the gateways, Caddy, celery-exporter, Postgres/Redis/broker
  exporters, node-exporter and cAdvisor.

The second gateway and worker sit behind the `x2` profile:

```bash
make prod-build
make prod-run        # 1x1
make prod-x2-run     # 2x2
make db-seed         # once, against the running stack
make prod-clean      # tear down, volumes included
```

`prod.env` configures every container. Besides credentials, it compresses the round
cadence so a run of a few minutes sees several rounds: aggregation every 60 s, secure
rounds sealing after 15 s and failing 20 s after sealing. Containers only pick up changes
to it when they are recreated.

## Running

```bash
uv run --group bench -m benchmark.scripts.run 1x1 --submission secure --users 6 --duration 150
uv run -m benchmark.scripts.export <run_id>
```

`run.py` takes the topology the stack was brought up with, picks the models for
`--split`/`--submission`, runs `reset.py`, writes the manifest and drives headless
Locust. Afterwards it saves the
`SecureRound` rows and copies the `worker_metrics` volume. Everything for a run ends up in
`results/benchmark/<run_id>/`. `export.py` then pulls the Prometheus series for the
manifest window and plots them.

## Details worth knowing

- **Host-side scripts and the broker.** `run.py` and `reset.py` run on the host with the
  regular `.env`, which needs `BROKER_PORT=6380` to reach the stack's separate broker.
  Without it the broker reset silently targets the auth Redis instead.
- **Reset, not recreate.** `reset.py` truncates submissions, jobs, rounds and sessions,
  drops every aggregated weight snapshot except each version's seeded one and flushes both
  Redis instances, keeping the Celery queue bindings. The worker metrics volume is never
  reset. The export cuts its rows to the manifest window instead.
- **Users.** Each Locust user is a seeded `test_N` account that owns a device. With
  several Locust processes, user numbers are interleaved by process index
  (`--user-stride`), so the run needs `test_1` up to roughly `users + processes`. Seed more
  with `seed_db --test-users N`. Users past `test_15` share a password but still pay a
  full argon2 verify, so keep `--spawn-rate` low.
- **Iterations.** Each user logs in once. It then repeats one iteration per aggregation
  interval (`constant_pacing`): for every model it downloads the weights and both
  artifacts, then submits through the model's submission type, polling quantize results
until they are ready. Rate limits keep their
  real values, and each finished round clears them. A user whose iteration starts
  before the previous round has finished therefore gets 429s on that model, which is
  expected around round boundaries.
- **Secure path without cryptography.** The server never unmasks individual vectors. It
  only checks lengths and that the aggregate stays finite and within `clip_bound`. So
  clients join with a random `0x04`-prefixed key, poll the descriptor while the round is
  open (409s that Locust and the export count as successes) and submit vectors of ones.
  `--round-failure` is the probability of a round failing: every member seeds a PRNG with
  the model key and round id, so they all agree on which rounds are dropped, and in those
  the member with the lowest user id never submits. The round then fails on the seal
  timeout and the next join opens a fresh one.
- **Request logs.** Every Locust worker process writes `requests_<index>.csv`. Names are
  route templates, never user ids. Each iteration also logs a `schedule_lag` row, which
  stays near zero while the generator holds the offered load.
- **Locust is standalone.** `locustfile.py` imports only Locust and the standard library,
  so the cloud clients can run the upstream `locustio/locust` image with it mounted.
