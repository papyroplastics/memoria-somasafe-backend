# Benchmark

Load-test tooling for the production-like deployment in [`../prod/`](../prod/README.md),
used for the local rehearsal and the cloud benchmark (see `plans/cloud-benchmark.md`). The
scripts run on the host against a stack brought up with `make prod-run` or
`make prod-x2-run` and seeded with enough test users.

## Running

```bash
uv run --group bench -m benchmark.run 1x1 --split small --submission secure --stages 60:3,60:6,60:3
uv run -m benchmark.export <run_id>          # plots
uv run -m benchmark.check <run_id>           # re-run the checks, before the next reset
uv run -m benchmark.snapshot                 # TSDB snapshot, end of session
```

`run.py` takes the topology the stack was brought up with, picks the models for
`--split`/`--submission` (or `--models`), runs `reset.py`, writes the manifest and drives
headless Locust through `--stages`. Afterwards it waits for the stack to settle (at least
one aggregation interval, then empty queues and no unfinished quantize job or secure
round), saves the `SecureRound` rows, copies the `worker_metrics` volume, dumps the
Prometheus series for the manifest window to CSV and runs `check.py`, which writes the
pass/fail results to `checks.json`. Everything for a run ends up in
`results/benchmark/<run_id>/`, and `export.py` plots from that directory alone, querying
Prometheus only if the series are missing or with `--requery`.

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
  (`--user-stride`), so the run needs `test_1` up to roughly the highest stage count
  plus `processes`. Seed more
  with `seed_db --test-users N`. Users past `test_15` share a password but still pay a
  full argon2 verify, so keep `--spawn-rate` low.
- **Stages.** `--stages` is a list of `<seconds>:<active users>`. Every user spawns and
  logs in during the first stage, which is therefore the login burst and must last at
  least `users / spawn-rate`; afterwards a user is active only while its `test_N` number
  is within the current stage's count, and parks otherwise. The Locust processes never
  talk to each other: they all receive the same `--schedule-start`, so they switch stages
  at the same moment. The export marks the stage boundaries on every plot.
- **Iterations.** An active user repeats one iteration per aggregation interval, starting
  at a fixed per-user offset so iterations spread over the interval: for every model it
  downloads the weights and both artifacts, then submits through the model's submission
  type, polling quantize results until they are ready. Rate limits keep their
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
