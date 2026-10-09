# Benchmark

Load-test tooling for the production-like deployment in [`../prod/`](../prod/README.md),
used for the local rehearsal and the cloud benchmark (see `plans/cloud-benchmark.md`). The
scripts run against a stack brought up with `make prod-run` or `make prod-x2-run` and
seeded with enough test users.

## Running

`run.py` and `check.py` run in the `bench` container (`bench.Containerfile`, built by
`make prod-build`), which reads the same env file as the rest of the stack. `make
prod-bench` runs one module in it, passing `ARGS` through:

```bash
make prod-bench ARGS="benchmark.run 1x1 --split small --submission secure --stages 60:3,60:6,60:3"
make prod-bench ARGS="benchmark.check <run_id>"   # re-run the checks, before the next reset
uv run -m benchmark.export <run_id>               # plots, on the host
uv run -m benchmark.snapshot                      # TSDB snapshot, on the host, end of session
```

`run.py` takes the topology the stack was brought up with, picks the models for
`--split`/`--submission` (or `--models`), runs `reset.py`, writes the manifest and drives
headless Locust through `--stages`. Afterwards it waits for the stack to settle (at least
one aggregation interval, then empty queues, no unfinished quantize job, no sealed or
summing secure session, and no session result still waiting for a round), saves the `SecureSession` rows, copies
the worker metrics, dumps the Prometheus series for the manifest window to CSV and runs
`check.py`, which writes the pass/fail results to `checks.json`. Everything for a run ends
up in `results/benchmark/<run_id>/`, and `export.py` plots from that directory alone,
querying Prometheus only if the series are missing or with `--requery`.

## Details worth knowing

- **Configuration.** The container gets every setting from its environment: the database
  and both Redis instances, the aggregation interval and the sweep intervals the checks
  expect, and `BENCH_API_URL`/`BENCH_PROMETHEUS_URL`, the defaults of `--host` and
  `--prometheus`. The worker metrics come from the `worker_metrics` volume, mounted
  read-only at `WORKER_METRICS_DIR`, and the results land in the host's `results/`.
- **Reset, not recreate.** `reset.py` truncates submissions, jobs, secure sessions with
  their seats and partial results, and auth sessions, drops every aggregated weight
  snapshot except each version's seeded one and flushes both Redis instances, keeping the
  Celery queue bindings. The worker metrics volume is never reset. The export cuts its rows
  to the manifest window instead.
- **Users.** Each Locust user is a seeded `test_N` account that owns a device. With
  several Locust processes, user numbers are interleaved by process index
  (`--user-stride`), so the run needs `test_1` up to roughly the highest stage count
  plus `processes`. Seed more with `make prod-db-seed` and `--test-users N`. Users past
  `test_15` share a password but still pay a full argon2 verify, so keep `--spawn-rate`
  low.
- **Stages.** `--stages` is a list of `<seconds>:<active users>`. Every user spawns and
  logs in during the first stage, which is therefore the login burst and must last at
  least `users / spawn-rate`; afterwards a user is active only while its `test_N` number
  is within the current stage's count, and parks otherwise. The Locust processes never
  talk to each other: they all receive the same `--schedule-start`, so they switch stages
  at the same moment. The export marks the stage boundaries on every plot.
- **Iterations.** An active user repeats one iteration per aggregation interval, starting
  at a fixed per-user offset so iterations spread over the interval: for every model it
  downloads the weights and both artifacts, then submits through the model's submission
  type, polling quantize results until they are ready or 80% of the interval has passed.
  Rate limits keep their real values, and each finished round clears them. A user whose
  iteration starts before the previous round has finished therefore gets 429s on that
  model, which is expected around round boundaries.
- **Secure path without cryptography.** The server never unmasks individual vectors. It
  only checks lengths and that each session's mean stays finite and within `clip_bound`.
  So clients join the session of the last weights they downloaded with a random
  `0x04`-prefixed key, poll the descriptor while the session is open and submit vectors of
  ones once it is sealed. They keep polling after submitting: a `404` means the session
  failed and released the seat, and the user joins a fresh one on the same weights. Polling
  `409`s (session open, summing or summed) and `404`s are expected, and Locust and the
  export count them as successes. A seat still pending when the polling window ends is
  kept, and the user's next iteration polls it instead of joining again, so a member never
  walks out of a session that is waiting for it.
- **Session failures.** `--session-failure` is the probability of a session failing:
  every member seeds a PRNG with the model key and session id, so they all agree on which
  sessions are dropped, and in those the member with the lowest user id never submits.
  The session then fails, on the sealed fail timeout or earlier if a round replaces its
  weights, and its other members rejoin. The env files only shorten the timeout that seals
  open sessions with enough members, so the only other failures are open sessions still
  short of members when a round lands; their members rejoin on the new weights.
  `check.py` compares the failed share of sealed sessions against `--session-failure`.
  Open sessions short of members when the load stops are left open, and the settle and
  the checks ignore them.
- **Request logs.** Every Locust worker process writes `requests_<index>.csv`. Names are
  route templates, never user ids. Each iteration also logs a `schedule_lag` row, which
  stays near zero while the generator holds the offered load.
- **Locust is standalone.** `locustfile.py` imports only Locust and the standard library,
  so the cloud clients can run the upstream `locustio/locust` image with it mounted.
