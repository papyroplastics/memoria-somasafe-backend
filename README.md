# SomaSafe backend component

This module is the TensorFlow side of SomaSafe: it defines the anomaly-detection models,
trains them on PPG-DaLiA, and exports artifacts (`SavedModel` + trainable/quantized
`.tflite`) that feed the on-device Android training and the ESP32 inference paths. Every
model is a custom `tf.Module` with explicit `eval` / `train` / `save` / `restore`
signatures so the same graph is LiteRT-trainable on-device and its flattened weights can
be averaged. It also exposes a FastAPI gateway that hands models to the app, accepts
weight-delta uploads, and aggregates them into new global model versions.

## Role in the full thesis system

See [`shared/docs/architecture.md`](../shared/docs/architecture.md) for the full system
design. In short: this is the only module that trains models and the only one that
aggregates federated updates; the app and firmware only ever consume what it exports or
serves.

## Layout

```txt
common/    Shared, TensorFlow-free infra: config, DB tables (SQLModel), Redis, rate
           limiting, secure-aggregation primitives, compression.
api/       FastAPI gateway (no TensorFlow): auth/device/model routers, rate limiting,
           attestation helpers, a pytest suite mirroring the routers.
ml/        TensorFlow models + training, imported by worker + scripts, never by api.
           See [`shared/docs/ml-pipeline.md`](../shared/docs/ml-pipeline.md) for how it's
           structured.
worker/    Celery task layer (models built on first use): quantization, dense and
           secure aggregation, the secure-round sweep, result cleanup.
scripts/   CLI entry points: `system/` (dataset, train, seed, export), `integration/`
           (headless federated/secure runs against the real API), `figures/` (report
           result and figure generators).
```

Evaluation output goes to `results/<model>/`; served `.tflite` artifacts live in
`shared/gen/models/<model>/`.

## Models

See [`shared/docs/model-types.md`](../shared/docs/model-types.md) for what each
architecture is (`FeatureMLP` classifier, waveform and feature autoencoders) and how
per-wearer normalization works, and
[`shared/docs/anomalies-and-distillation.md`](../shared/docs/anomalies-and-distillation.md)
for how synthetic anomaly labels are produced, calibrated into a detector, and distilled
into the small model that ships to the device. The training architecture itself (the
Model/DataSource/Trainer/Loop split, the dataset registry, what's cached vs. derived on
the fly) is in [`shared/docs/ml-pipeline.md`](../shared/docs/ml-pipeline.md).

## Run

```bash
uv run -m scripts.system.get_dataset              # fetch + preprocess PPG-DaLiA (idempotent)
uv run -m scripts.system.train feature-mlp        # train a model, export its artifacts
```

See [`shared/docs/ml-pipeline.md`](../shared/docs/ml-pipeline.md) for the full flag
reference (federated simulation, held-out subjects, tagging runs, transfer learning) and
how to export a subject for the app/firmware test harnesses.

## Server architecture

A FastAPI gateway in front of a Celery worker, backed by PostgreSQL (accounts, models,
submissions, and every served blob) and Redis (Celery broker + rate limiting). The
gateway never runs ML work; the worker quantizes/signs uploads, runs the federated
aggregation rounds and drives the secure-round lifecycle. See
[`shared/docs/server-internals.md`](../shared/docs/server-internals.md) for the
gateway/worker split, storage decisions, the aggregation algorithm, and the rate-limiting
table.

There are multiple upload paths per model (`raw` / `quantize` / `secure`), each with its
own aggregation strategy — see
[`shared/docs/submission-type.md`](../shared/docs/submission-type.md) for what each is
for and [`shared/docs/secure-aggregation.md`](../shared/docs/secure-aggregation.md) for
the masked-aggregation protocol. Model, contract and weights versioning semantics are in
[`shared/docs/versioning.md`](../shared/docs/versioning.md).

## Auth, device attestation, and firmware distribution

Sessions are stateful tokens seeded server-side (no self-registration) — see
[`shared/docs/authentication.md`](../shared/docs/authentication.md). Model downloads and
uploads additionally require a verified device owner — see
[`shared/docs/device-attestation.md`](../shared/docs/device-attestation.md). Firmware
builds are published through `/ota/*` for the app's BLE OTA flow — see "Firmware
distribution" in [`shared/docs/versioning.md`](../shared/docs/versioning.md).

Bootstrap a fresh database with `uv run -m scripts.system.seed_db` (`make db-seed`); see
[`shared/docs/server-internals.md`](../shared/docs/server-internals.md) for `--reseed` and
the test-user setup used by the headless federated scripts.

## Environment

Python `==3.13.*`, TensorFlow `2.21.*`, managed with `uv`. GPU is optional (`uv sync
--extra cuda`). Copy `example.env` to `.env` before running — there are no hardcoded
defaults for DB/Redis credentials. Only Postgres and Redis run in containers
(`compose.yaml`, podman); the gateway and worker run on the host with `uv` (`make
api-run`, `make worker-run`).

## Not yet implemented

Sparse and differential-privacy submission formats are anticipated by the
`submission_type` design (see
[`shared/docs/submission-type.md`](../shared/docs/submission-type.md)) but not
implemented.
