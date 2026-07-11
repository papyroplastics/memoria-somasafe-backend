# SomaSafe backend component

This module is the TensorFlow side of SomaSafe: it defines the anomaly-detection
models, trains them on PPG-DaLiA, and exports artifacts (`SavedModel` +
trainable/quantized `.tflite`) that feed the on-device Android training and the
ESP32 inference paths. Every model is written as a custom `tf.Module` with
explicit `eval` / `train` / `save` / `restore` signatures so the same graph is
LiteRT-trainable on-device and its flattened weights can move through FedAvg.

## Role in the full thesis system

- Defines and trains the candidate anomaly models.
- Exports a `SavedModel` and converts it into a trainable `.tflite` (LiteRT) and
  an int8 `.tflite` (TFLM / ESP32) per model.
- Serves as the source-model path for on-device Android fine-tuning and the
  federated update flow.
- Exposes a FastAPI gateway (`api/`) that hands models to the app and accepts
  weight-delta uploads, running the ML work asynchronously on a Celery worker.

## Generated code

`scripts/export_subject_data.py` writes the capture-import protobuf defined in the
shared schema (`shared/dataset.proto`). Generate (or regenerate, after editing the
schema) its Python stub with the system `protoc` before running the export:

```bash
make proto    # protoc shared/dataset.proto -> scripts/common/dataset_pb2.py (gitignored)
```

## Layout

The codebase splits along its three runtime concerns — the TensorFlow work (`ml/`), the
async task layer (`worker/`), and the HTTP gateway (`api/`) — plus shared, TF-free infra
in `common/` (config, DB tables, the model registry) imported by both api and worker.

```txt
common/    Shared, TensorFlow-free infra imported by api + worker: env-driven config,
           and the SQLModel tables (User, AuthSession, Device, ModelDefinition,
           ModelVersion, GlobalWeights, ClientDeltaSubmission, QuantizationJob, Firmware).
api/       FastAPI gateway (no TensorFlow): routers for auth/device/model (routes/),
           rate-limiting + attestation-challenge helpers (lib/), and a pytest suite
           mirroring the routers (test/).
ml/        TensorFlow models + training, imported by worker + scripts, never by api.
           model_list.py is the registry (key -> metadata + trainer builder), the single
           source of truth. models/ holds one file per architecture (FeatureMLP,
           CNN/LSTM/GRU autoencoders) built on shared bases in common.py. Everything
           else (optimizers, saving/export, training loops incl. fed_avg, dataset
           pipeline, layers, metrics) is model-agnostic and shared across architectures.
           layers.py in particular reimplements a few ops with custom gradients because
           the stock TF gradients only exist as Flex ops the phone's LiteRT runtime
           can't execute.
worker/    Celery task layer (TensorFlow loads at startup): celery_app.py wires the
           broker + beat schedule; tasks.py holds quantize_submission,
           validate_submission, federated_aggregation and cleanup_results; utils/ has
           the TF-free validation/outlier-filtering helpers tasks.py calls into.
scripts/   CLI entry points: dataset fetch/build, train / transfer_learn, seed_db,
           export_subject_data, queue_aggregation, and the autoencoder-distillation tools.
```

Training is split into three layers so any model can be run under any loop:

- **Model** (`TrainableModel`): the graph — `eval` / `train` / `save` / `restore`, plus
  `transfer_from` (copy compatible trainable weights from another instance of the same
  architecture, transferring the overlapping region where a shape differs — used for
  cross-batch-size transfer learning).
- **Trainer** (`Trainer`): everything model-specific — `subject_datasets` (per-subject
  splits), `representative_dataset` (int8 calibration feed), `train_epoch`, and
  `evaluate` (metrics relevant to the model: accuracy for the MLP, reconstruction
  error for the autoencoders). Each trainer declares a `default_batch_size`, and each
  model module exposes `get_trainer(data_root, seed, batch_size=None)` (falling back to
  that default when `batch_size` is `None`).
- **Loop** (`training.py`): orchestration only — `normal_loop` and `federated_loop`
  (simulated FedAvg with an injectable `aggregate` strategy). Loops talk only to the
  `Trainer` interface, so a `(model, trainer)` pair works with either loop and they
  can be compared. `train.py` picks the model and loop and handles export + plotting;
  each run writes its history plot + CSV and eval report under
  `results/<model>/reports/<loop>/` (`normal` or `federated`).

Autoencoder variants (LSTM/GRU/CNN/...) share `TrainableAutoencoder` (reconstruction
train/eval + conditioning) and `AutoencoderTrainer` (windowing + recon-error metrics).
The encoder sees `[BVP, ACC]` but the decoder reconstructs **BVP only** — ACC is
exogenous context that explains motion artifacts and is kept out of the anomaly score.
Every model is **conditioned** on a single `cond` vector — z-scored demographics plus a
causal *activity context* (trailing-2-min mean/std of the ACC). The context is computed
from the **raw** ACC; the whole `cond` (and the `[BVP, ACC]` signal) is fed to the model
**raw**, and the model z-scores it in its `eval`/`train` signatures with baked-in
constants (`context_norm_params.npy` is just the ACC mean/std, so normalizing it equals
the old "normalize ACC, then take trailing stats"). The on-device pipeline feeds raw the
same way. This way the decoder generates the signal *expected for this person at this activity level* rather
than copying its input; a small bottleneck + latent dropout push it to lean on the
condition. The objective is reconstruction MSE plus a first-difference (slope) term that
penalizes a constant "flat line" output.

## Models

See [`shared/docs/model-types.md`](shared/docs/model-types.md) for what each model
architecture is and how conditioning/normalization work; this section covers the
backend-specific dataset and training-pipeline details.

### `FeatureMLP` dataset — synthetic anomaly injection

Labels come from **synthetic anomaly injection**: a window-aligned ~50% mix of anomaly
kinds is injected into the **raw** BVP signal (from `clean-signals/`) on spans of 8–30
windows (64–240 s) and stored in `mixed-signals/` with a per-window binary label bitmap,
so every window is fully clean or fully anomalous. Features are then extracted the same
way the firmware does and saved raw per subject to `datasets/mixed-features/S*/`; the
global `feature_stats.npy` is baked into the model as its z-score constants and also
serialized into the signed quantize payload for the firmware to apply. The same feature
build also runs over the clean signals into `datasets/clean-features/S*/` (every window
normal, label 0) — unused for training, only feeds `export_subject_data.py --clean`. A
separate per-type `anomalous-signals/<kind>/` (each kind applied to every window) lets
`distill_calibrate.py` measure per-kind detection recall in isolation.

Because injection operates on un-normalized signals, perturbations scale with the
signal's own range/std, so they apply at any sensor output range. The five kinds are
signal-integrity artifacts (spike = a sustained baseline step, amplitude blow-up, and a
wavy band-limited noise burst) and rhythm anomalies (timewarp = uniform tachy/brady via
resampling, afib = irregularly-irregular rhythm via a jittered time-warp). Flatline and
baseline wander were dropped: a flatline sits below the AE's reconstruction-error floor
(handle sensor dropout with a signal-quality gate instead) and wander is physiological —
already in the clean signal, so the AE rightly does not flag it.

### Autoencoder evaluation and distillation

The detector is an OR of **several scores**, each oriented higher = more anomalous:
reconstruction MSE — strong on spike/blowup/noise but phase/rate-blind — OR'd with two
cheap, jDSP-portable rhythm indices (in-band spectral entropy, beat-interval
coefficient-of-variation) that catch afib, which reconstruction alone misses. A window is
anomalous if any score crosses its threshold, and the threshold is **per subject** — each
score fires at the `1 - budget` quantile of *that subject's own* clean windows, so a
subject-specific score scale (reconstruction error especially) gives a uniform per-subject
false-alarm rate instead of one dominated by the noisiest subjects. The work splits into
three scripts along **what data each is allowed to see** — mirroring deployment, where
only the budgets are global and everything else is done per-client on unlabeled data:

- **`distill_calibrate.py` (server: labeled, global)** picks the per-score **budgets** —
  the only globally-relevant output, and the only thing that reads the synthetic labels.
  Each budget (the quantile level a client thresholds at) is chosen *independently* as the
  level that maximizes that score's Youden's J (recall minus clean FPR) on the labeled
  data, capped at `--max-budget`. Independent (not a shared combined-FPR ceiling) so a
  low-volume specialist (rr → afib/timewarp) can't be crowded out by a high-volume
  generalist (recon); Youden's J is degeneracy-free, unlike an F1 sweep that flags
  everything on a weakly-separating score. Writes only the budgets to `results/<model>/reports/`.
- **`distill_labels.py` (client: unlabeled)** touches only what a real client has — its
  own clean baseline and the mixed signal + on-device features, **never the true labels or
  the per-anomaly sets**. Reads the budgets, derives each subject's thresholds from its
  *own* clean windows, and emits a **soft** `[0,1]` label per window: the clean-CDF rank
  past each score's threshold, max'd across scores (so `label > 0` reproduces the hard OR),
  then a size-1 temporal **median filter** (real anomalies span many windows, so a lone
  flag is a false positive and a lone gap a false negative — cleaned without tuning to the
  injected span length). Soft targets carry the teacher's confidence — proper knowledge
  distillation, not just pseudo-labeling.
- **`distill_eval.py` (science: unrestricted)** replays the same budgets → per-subject
  thresholds a client uses, then scores the OR detector against the true mixed-window
  labels and the per-type `anomalous-signals/` sets: OR-combined + per-score
  precision/recall/F1, per-anomaly-kind recall, clean FPR. Writes the metrics to
  `results/<model>/reports/`.

The labels land in a datasets-shaped tree (`mixed-features/S*/` with the distilled
`labels.npy`, feature arrays symlinked back to `datasets/`), so the student `FeatureMLP`
trains on them via `train.py --dataset-dir` — the path to validating an unsupervised
teacher that needs no labels on-device. (Uniform-tempo timewarp stays below all three
scores; it needs the activity-expected-HR check on the roadmap.)

## Run

Fetch + preprocess the dataset first (idempotent: skips download/processing if
already present):

```bash
uv run -m ml.scripts.get-dataset
```

Then train any model; artifacts land in `results/<model>/`:

```bash
uv run -m ml.scripts.train feature-mlp                      # synthetic-anomaly classifier
uv run -m ml.scripts.train cnn-ae                           # conditional Conv1D autoencoder (focus)
uv run -m ml.scripts.train lstm-ae                          # conditional LSTM autoencoder
uv run -m ml.scripts.train gru-ae                           # conditional GRU autoencoder
uv run -m ml.scripts.train feature-mlp --loop federated     # simulated FedAvg
uv run -m ml.scripts.train feature-mlp --batch-size 32       # train at a larger batch (GPU-friendly)
```

`--loop` selects the training loop (`normal` by default, or `federated`);
`--epochs` tunes the normal loop and `--local-epochs` the local passes per round
for the federated one. `--batch-size` overrides the model's `default_batch_size`
(useful for GPU throughput — the on-device default batch is often 1). Each run
writes `trainable.tflite` (LiteRT-trainable), `quantized.tflite` (int8, when
supported) and a diagnostic plot into `results/<model>/`; the intermediate
`SavedModel`s only exist in a temp dir during conversion. Models z-score their own
inputs: the `eval`/`train` signatures take
raw inputs and normalize internally (baked z-score constants), so nothing ships or serves
separate normalization params. The int8 `quantized.tflite` is exported from a second
non-normalizing `infer` signature and therefore takes **already-normalized** input — its
per-tensor int8 scale calibrates on normalized values (feeding raw heterogeneous features
through one scale collapses precision). The device applies the params before that model;
they travel to the firmware alongside the signed model (see `shared/docs/model-signing.md`). A
non-default `--batch-size` suffixes those artifacts (`trainable_32.tflite`,
`quantized_32.tflite`, ...) so they don't clobber the canonical default-batch exports.

Because the model's batch size is baked into the `.tflite` input signature, the
GPU-trained large-batch model isn't itself the deliverable. `transfer_learn`
bridges that: it seeds a fresh default-batch model from the large-batch artifact's
weights (via `TrainableModel.transfer_from`) and fine-tunes it for a few epochs.

```bash
uv run -m ml.scripts.train feature-mlp --batch-size 32       # 1) fast GPU training
uv run -m ml.scripts.transfer_learn feature-mlp 32 --epochs 3 # 2) transfer -> default-batch + fine-tune
```

The source batch size must be `>=` the default; `transfer_learn` re-exports the
fine-tuned model under the canonical (unsuffixed) artifact names.

To run the knowledge-distillation round-trip, first calibrate the autoencoder
(`distill_calibrate.py` picks the per-score budgets), then `distill_labels.py` derives
each subject's thresholds and emits the soft-label tree; `distill_eval.py` reports the
detector's per-kind metrics against the ground truth. Then point `feature-mlp` at those
labels with `--dataset-dir`:

```bash
uv run -m scripts.distill_calibrate cnn-ae                                       # budgets -> results/cnn-ae/reports/
uv run -m scripts.distill_eval cnn-ae                                            # metrics -> results/cnn-ae/reports/
uv run -m scripts.distill_labels cnn-ae                                          # teacher -> results/cnn-ae/distilled-labels/
uv run -m scripts.train feature-mlp --dataset-dir results/cnn-ae/distilled-labels  # student on pseudo-labels
```

### Export a subject to the Android app

`export_subject_data.py` packs one subject's windows into the `.ssds` protobuf the app
imports (run `make proto` first). Each window mirrors an ESP sample: raw PPG/ACC, the raw
feature vector, the label in the score field, plus a fake sequence number and contiguous
8 s device-time grid, so imported windows preprocess and train exactly like streamed ones.
The dataset also carries the subject's raw 6-d demographics (`static`, recovered by
de-normalizing `static.npy`), which the app stamps onto the imported group as its
conditioning static.

```bash
uv run -m scripts.export_subject_data 1                              # S1.ssds, every window complete
uv run -m scripts.export_subject_data 1 --include-context           # also embed each window's raw context
uv run -m scripts.export_subject_data 1 --missing-samples 0.7       # keep 70% of windows' signal; drop the rest
uv run -m scripts.export_subject_data 1 --missing-features 0.7      # keep 70% of windows' ML result; phone recomputes the rest
uv run -m scripts.export_subject_data 1 --missing-samples 0.7 --missing-features 0.7  # both, drawn independently
uv run -m scripts.export_subject_data 1 --clean                     # clean (anomaly-free) signals; features from clean-features
```

`--clean` exports the anomaly-free `clean-signals/` instead of `mixed-signals/`; its
feature/label windows come from the `clean-features/` dataset (every window normal,
score 0), precomputed by `get_dataset.py` because on-device extraction is too slow, so
`--missing-features` works with `--clean` too.

The two loss flags assign sequence numbers and timestamps over the full grid *before*
dropping anything, so a removed window leaves a real hole in the sequence numbers. Passed
together they draw the signal and ML-result sets independently, so a window may end up with
signal but no features, features but no signal, or neither (omitted entirely) — exercising
the app's on-device feature/context recovery and its handling of missing samples.

## Environment

- Python `==3.13.*`, TensorFlow `2.21.*`, managed with `uv` (`pyproject.toml`).
- GPU is optional: `uv sync --extra cuda` swaps in CUDA-enabled TensorFlow
  (`tensorflow[and-cuda]`, the nvidia pip wheels); the default install is CPU-only.

## What this module is **not** yet doing

- On-device feature extraction and z-score normalization are implemented in `firmware/main/ml/features.c`. The firmware applies the per-feature normalization params delivered in the signed quantize payload (no longer a baked-in header).
- No on-device LiteRT training wired up from the Android side yet.
- Stateful token auth, Redis-backed per-model rate limiting, ESP32 device
  attestation, and server-side signing of distributed `.tflite`s are implemented
  (see "Auth & rate limiting", "Device attestation" and `shared/docs/model-signing.md`).

## Server architecture

The server is a FastAPI **gateway** in front of a Celery **worker**; the gateway never
runs ML work directly. Both run on the host with `uv` (`make api-run`, `make worker-run`);
only the external services — PostgreSQL and Redis — run in containers via `compose.yaml`
(`make api-serv-up`, podman). Those two are bound to `127.0.0.1` since only the host-run
processes reach them; the API itself binds `0.0.0.0` so the phone can reach it over the LAN.

- **Gateway (`api/`, no TensorFlow):** serves the model artifacts stored on the active
  `GlobalWeights` rows, accepts weight-delta uploads, persists them, enqueues worker jobs, and
  exposes a result endpoint the client polls. Fast to start since it never imports TF.
- **Worker (`worker/tasks.py`):** restores uploaded weights into the model, converts it to
  an int8 `.tflite` against the per-model calibration dataset
  (`ml/saving.py:get_optimized_model`) and signs it, validates submit-only uploads, and
  runs the daily federated aggregation (see "Federated aggregation"). Each forked worker
  child builds every available model (skipping any whose dataset is absent) and caches each
  one's `(model, representative dataset, fingerprint, contract_version, norm bytes)`. This
  build runs post-fork (via the `worker_process_init` signal), never in the parent
  MainProcess: TensorFlow is not fork-safe once its runtime exists, so initializing it before
  the prefork pool forks would deadlock every child on inherited-locked native mutexes.

There are **two upload paths**, and which one a model accepts is a per-model property:
each `ModelVersion` carries a `submission_type` (`raw` / `quantize`, sourced from the code
registry `ml/model_list.py` and seeded into the DB). The `quantize` path accepts only
`quantize`-typed models; the `raw` (submit-only) path accepts both (`quantize`'s dense body
is compatible and submit-only is the least work). A model uploaded on a path it doesn't
accept gets `404` (not `403`, so the path stays unguessable). The type also selects the
aggregation strategy (see "Federated aggregation"); future formats (sparse, DP) will add
their own type + endpoint. Both current paths persist a `ClientDeltaSubmission` that feeds
aggregation; both take the raw little-endian float32 **weight-delta** buffer (Δ = local −
global, the change local training produced against the snapshot it trained from) as the
request body and the `weights_id` of that `GlobalWeights` snapshot (echoed by the download
headers) in the path. Malformed bodies (wrong length, non-finite values) are
rejected with `400`; an unknown `weights_id` is `400`; a `weights_id` belonging to a
frozen (non-latest) model version is `409` — the client must re-download the model first.

Request flow for a `quantize`-typed model:

1. `POST /model/submit/quantize/{key}/{weights_id}` stores the delta as a
   `ClientDeltaSubmission` (tagged with the submitting `user_id`, the `base_weights_id` and its
   `version_id`), creates a `QuantizationJob` (`pending`), enqueues `quantize_submission`,
   and returns `202` with a `job_id` + `status_url`.
2. The worker runs the job: malformedness fails it, but the aggregation-usability verdict
   (MSE gate) is cached silently on the submission and the artifact is produced either
   way — a Byzantine client never learns its update was filtered. The int8 `.tflite` is
   written to the job row along with an ECDSA signature over the canonical model bytes
   (`ml/payload.py`, spec in `shared/docs/model-signing.md`).
3. The client polls `GET /model/quantize/result/{job_id}`: `202` while `pending`/`running`,
   `422` on `failed` (with the error), `200` with the int8 `.tflite` body plus the
   `X-Model-Signature` / `X-Contract-Version` / `X-Norm-Params` headers once `done` — the
   app packages those fields for the ESP32 per its BLE interface version, and the firmware
   re-derives the canonical bytes and verifies the signature before loading. The result is
   scoped to the user who submitted it (resolved via the job's `ClientDeltaSubmission`); another
   user's `job_id` returns `404`.

The submit-only path is `POST /model/submit/raw/{key}/{weights_id}`: same checks and
storage, but nothing comes back — a `validate_submission` task caches the verdict in the
background and the client only ever sees `202`. Fully silent rejection (vs the quantize
path, whose `422` reveals hard failures) and no full-weights artifact round-trip make it
the natural host for future privacy-preserving submission formats (sampled weights,
differential privacy).

**Weights persist indefinitely.** `ClientDeltaSubmission` rows are the substrate federated
aggregation consumes. Only the job's quantized `result` (+ its signature) is ephemeral.

### Federated aggregation

A beat task (`federated_aggregation`, every `FED_AGG_INTERVAL_SECONDS`, default 24 h) runs one
FedAvg round per initialized model:

0. **Strategy:** chosen by the latest version's `submission_type`. `raw` and `quantize` are
   byte-identical dense vectors and share the FedAvg path below; sparse/DP formats will
   branch here. A model whose type has no strategy is skipped.
1. **Window:** submissions created after the model's newest `GlobalWeights` snapshot
   (valid or not, so updates consumed by a later-invalidated round are never re-aggregated),
   filtered to the latest `ModelVersion` — frozen versions never aggregate. One update per
   client: only each user's latest submission in the window counts.
2. **Validation** (`worker/utils/weight_validation.py`): weight count must match the
   model's `total_weight_size`, the buffer must be finite, and — once a previous round
   has set an `mse_threshold` — the delta's magnitude (mean square, which equals its MSE
   from the active global weights) must stay under it. Validation runs once per submission:
   the quantize/validate tasks perform it as uploads arrive and cache the verdict on
   `ClientDeltaSubmission.valid` (never surfacing it to the client); aggregation trusts that
   verdict and validates only rows neither task got to.
3. **Round threshold:** fewer than `FED_MIN_SUBMISSIONS` (default 1) valid submissions skips
   the model until the next round.
4. **Outlier filter:** each submission's L2 distance from the element-wise mean is z-scored;
   rows above the cutoff are dropped (needs ≥ 3 submissions to be meaningful, otherwise all
   are kept).
5. **Averaging:** `ml.training.fed_avg` — the same function the simulated `federated_loop`
   uses, so simulation matches deployment — averages the accepted deltas with uniform
   weighting (submissions carry no sample counts), and the mean is added onto the reference
   global weights (identical to averaging absolute weights when every client shares a base).
   The result is stored as a new `GlobalWeights` row along with the next round's
   `mse_threshold` (a margin over the worst deviation accepted this round).
6. **Artifact baking:** the averaged weights are restored into the cached model and both
   serving artifacts are re-exported onto the new row — the LiteRT-trainable `.tflite` and
   the signed int8 `.tflite` — so a client always pulls a file with the current global
   parameters already inside. If an export fails the row is stored with `valid = false`:
   clients keep pulling the previous snapshot and the window's submissions stay consumed.
7. **Rate-limit reset:** on a successful round the model's download/submission counters are
   cleared for every user (`ratelimit.clear_model_limits`), so clients can immediately
   re-pull the new weights and submit again without waiting out the download cooldown or the
   daily submission caps. (An invalidated round leaves the counters alone — nothing new to
   pull.)

If a round makes the model worse, flip the new row's `valid` flag to false by hand: the
active weights and artifacts (`get_latest_weights`, `/model/download/*`, `/model/list`)
are the latest **valid** snapshot, so clients fall back to the previous round —
artifacts roll back atomically with the weights they were baked from. Schema changes are
handled by wiping the database and re-running the seed script (no production environment,
no migrations).

A round can also be queued by hand, for testing:

```bash
uv run -m scripts.queue_aggregation           # every initialized model
uv run -m scripts.queue_aggregation cnn-ae    # a single model
```

**Headless federated run.** `scripts/fed_client.py` drives the whole stack over the real
HTTP API: for each dataset subject (as user `test_N`) it logs in, pulls the global
trainable artifact, trains one pass through the on-device LiteRT `CompiledModel` runtime,
uploads the update, and logs out; then it queues a round, waits for the new `GlobalWeights`,
and scores it on the held-out subjects, repeating for `--rounds`. The per-round
convergence series is written as a CSV + plot to `results/<model>/reports/fed_client/`.
Seed the accounts first with `scripts.seed_db --test-users` (one `test_N` per subject,
each owning a placeholder device).

```bash
uv run -m scripts.seed_db --test-users                       # one test_N per subject
uv run -m scripts.fed_client --model cnn-ae --rounds 5 --eval-subjects 2
```

**Model versioning.** See [`shared/docs/versioning.md`](shared/docs/versioning.md) for
what `version`, `contract_version`, `fingerprint` and weights (`weights_id` /
`weights_version`) each mean and how a client reacts to each changing. Backend-specific:
`ModelVersion.version` is hand-bumped in the code registry (`ml.model_list.ModelSpec`),
`fingerprint` is derived (`Trainer.arch_fingerprint()`) and enforced as a tripwire only by
`scripts/seed_db.py` (aborts if the fingerprint moved but the version didn't); `/model/list`
reports the latest version only, `/model/versions/{key}` the full history; `/model/download/*`
echoes `X-Model-Fingerprint`, `X-Model-Version`, `X-Weights-ID` and `X-Weights-Timestamp`.

The registry that ties a model `key` to its metadata *and* its TensorFlow trainer builder is
`ml/model_list.py` — the single source of truth consumed by `scripts/train.py` (one trainer),
`worker/tasks.py` (all models + fingerprints, built per worker child), and `scripts/seed_db.py` (publishes
versions + metadata, enforcing the fingerprint tripwire). The api never imports it; it
trusts what the seed wrote to the DB.

**Storage decisions (thesis scope, no production deployment):**

- **PostgreSQL** (via **SQLModel**) holds weight submissions and quantization jobs. Job state
  lives here — it is the single source of truth the poll endpoint reads. Celery's **Redis
  result backend** is used only so callers can await a queued task and read its return value
  (e.g. `queue_aggregation`/`fed_client` block on the aggregation summary instead of polling
  `GlobalWeights`); results expire after `RESULT_TTL_SECONDS`.
- **Redis** is the Celery broker. RabbitMQ was considered but rejected: the workload is a few
  low-frequency jobs, not high-throughput routing, and Redis is a single lightweight service
  that will also host rate-limit counters later — its delivery guarantees are more than enough
  here.
- **No object store.** The `.tflite` files are tiny (hundreds of KB), so quantization
  results live as `BYTEA` on the job row and the serving artifacts as `BYTEA` on their
  `GlobalWeights` row — one consistent store, artifacts can't drift from the weights they
  were baked from, and rollback is a flag flip. The `results/<model>/` files only feed the
  seed script. This drops the planned "distribution worker" and S3/MinIO entirely (MinIO
  remains a drop-in if real object storage is ever wanted).

**Result lifecycle.** A `done` result is streamed on request and stamped `served_at`. A Celery
beat sweep (`cleanup_results`, in-process via `celery worker -B`) nulls the result bytes (and
signature) once a served result is older than `SERVE_GRACE_SECONDS` (5 min) or an unclaimed one
is older than `RESULT_TTL_SECONDS` (1 h), flipping the job to `expired`. Weight submissions are
never reaped.

## Auth & rate limiting

All `/model/*` routes require a logged-in user, and the rate-limited ones
additionally require the user to be a verified device owner (see "Device
attestation"). Accounts are **seeded, not self-registered** — `make seed`
(`uv run -m scripts.seed_db`) bootstraps a fresh DB with the model registry rows
and a default user (`SEED_USER` / `SEED_PASSWORD`, default `somasafe` /
`somasafe`); it is idempotent. Pass a factory NVS CSV (`make seed
nvs_csv=...firmware/factory_nvs.csv`) to also register that device.

Session semantics (stateful tokens, `api/routes/auth.py` endpoints, argon2 password
hashing) are documented in [`shared/docs/authentication.md`](shared/docs/authentication.md).

**Rate limiting is per-user, per-model (`api/lib/ratelimit.py`, Redis db 1).**
The intent: a client can download + quantize every model once in a single pass,
but immediate repeats on the same model are rejected with `429` (+ `Retry-After`).

| Endpoint | Limit |
|----------|-------|
| `GET /model/list`, `GET /model/versions/{key}` | authed only |
| `GET /model/download/{trainable,quantized}/{key}[?version=N]` | device-owner only; one download per model per `DOWNLOAD_COOLDOWN_SECONDS` (default 300 s) |
| `POST /model/submit/quantize/{key}/{weights_id}` | device-owner only; `QUANTIZE_DAILY_LIMIT` (default 2) per model per rolling 24 h; `404` unless the model's `submission_type` is `quantize` |
| `POST /model/submit/raw/{key}/{weights_id}` | device-owner only; `SUBMIT_DAILY_LIMIT` (default 2) per model per rolling 24 h; `404` unless the model's `submission_type` is `raw` or `quantize` |
| `GET /model/quantize/result/{job_id}` | authed; only the user who submitted the job (else `404`) |
| `GET /ota/versions/{interface}` | authed only |
| `GET /ota/download/{interface}/{version}` | device-owner only; one firmware download per interface per `OTA_DOWNLOAD_COOLDOWN_SECONDS` (default 300 s) |

The model-artifact download is a single route with an `Artifact` enum path parameter
(`trainable` / `quantized`), serving the artifact bytes stored on the version's active
`GlobalWeights` row (`?version=` selects a frozen version; default is the latest). It
echoes `X-Model-Fingerprint`, `X-Model-Version`, `X-Weights-ID` and
`X-Weights-Timestamp`; the quantized artifact additionally carries `X-Model-Signature`,
`X-Contract-Version` and `X-Norm-Params` (see `shared/docs/model-signing.md`).

## Firmware distribution (OTA)

The `/ota` routes serve published firmware builds for the BLE OTA path (see
`shared/docs/versioning.md`, "Firmware distribution"). `GET /ota/versions/{interface}`
lists the builds published for a `BLE_INTERFACE_VERSION` (newest first, each with its
version string, supported model-contract list, image size and release date — an unknown
interface yields `[]`); `GET /ota/download/{interface}/{version}` streams the raw image
with the server's ECDSA signature over it in `X-Firmware-Signature`, which the app
forwards to the device for verification against its factory `srv_pub`.

Images are stored as `BYTEA` on the `Firmware` row, like the model artifacts (they are
hard-capped at 1 MB by the OTA partition size, so no object store is warranted).
`scripts/seed_db.py` publishes them: it scans a directory of exports (`--firmware-dir`,
default `shared/gen/firmware/`, populated by `make export-image` in `firmware/`), signs
each image with `SERVER_PRIVATE_KEY_FILE` and inserts any version not already present.
Re-publishing a changed build under an existing version means deleting its row and
re-seeding (no production environment, no migrations).

## Device attestation

See [`shared/docs/device-attestation.md`](shared/docs/device-attestation.md) for the
full ownership-proof flow. Backend-specific: `api/routes/device.py` implements it; a
`Device` row holds the `serial` (PK), the 65-byte uncompressed `public_key`, an optional
`owner_id`, and `last_attested_at`. Devices are seeded ownerless from a factory NVS image
(`scripts/seed_db.py`).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/device/owned` | serials of the devices the caller currently owns |
| POST | `/device/challenge` | `{serial}` → `{instance_id, nonce, server_time, user_id}` |
| POST | `/device/attest` | `{instance_id, signature}` → verify and set the owner |

