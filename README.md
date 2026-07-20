# SomaSafe backend component

This module is the TensorFlow side of SomaSafe: it defines the anomaly-detection
models, trains them on PPG-DaLiA, and exports artifacts (`SavedModel` +
trainable/quantized `.tflite`) that feed the on-device Android training and the
ESP32 inference paths. Every model is written as a custom `tf.Module` with
explicit `eval` / `train` / `save` / `restore` signatures so the same graph is
LiteRT-trainable on-device and its flattened weights can be averaged.

## Role in the full thesis system

- Defines and trains the candidate anomaly models.
- Exports a `SavedModel` and converts it into a trainable `.tflite` (LiteRT) and
  an int8 `.tflite` (TFLM / ESP32) per model.
- Serves as the source-model path for on-device Android fine-tuning and the
  federated update flow.
- Exposes a FastAPI gateway (`api/`) that hands models to the app and accepts
  weight-delta uploads, running the ML work asynchronously on a Celery worker.

## Generated code

`scripts/system/export_subject_data.py` writes the capture-import protobuf defined in the
shared schema (`shared/dataset.proto`). Generate (or regenerate, after editing the
schema) its Python stub with the system `protoc` before running the export:

```bash
make proto    # protoc shared/dataset.proto -> scripts/common/dataset_pb2.py (gitignored)
```

## Layout

The codebase splits along its three runtime concerns — the TensorFlow work (`ml/`), the
async task layer (`worker/`), and the HTTP gateway (`api/`) — plus shared, TF-free infra
in `common/` (config, DB tables, Redis) imported by both api and worker. The model registry
is *not* here: it builds TensorFlow trainers, so it lives in `ml/model_list.py` and the api
never imports it.

```txt
common/    Shared, TensorFlow-free infra imported by api + worker: env-driven config
           (config.py), the secure-aggregation primitives (secure_agg.py), the zstd
           transport wrapper (compression.py), the shared Redis client (redis.py) and the
           two-phase rate-limit primitives over it (ratelimit.py — the worker clears limits
           after a round, so the keying cannot live in api/), the Celery task-name constants
           both sides address (celery_tasks.py), and the SQLModel tables (User, AuthSession,
           Device, ModelDefinition, ModelVersion, GlobalWeights, WeightsArtifact,
           ClientDeltaSubmission, QuantizationJob, QuantizationResult, SecureRound,
           SecureRoundMember, Firmware).
api/       FastAPI gateway (no TensorFlow): routers for auth/device/model (routes/),
           rate-limiting + attestation-challenge helpers (lib/), and a pytest suite
           mirroring the routers (test/).
ml/        TensorFlow models + training, imported by worker + scripts, never by api.
           model_list.py is the registry (key -> metadata + trainer builder), the single
           source of truth. models/ holds one file per architecture (FeatureMLP,
           CNN/LSTM/GRU autoencoders) built on shared bases in common.py. Everything
           else is model-agnostic and shared across architectures: preprocessing.py
           (raw download -> arrays on disk; no TensorFlow) and loading.py (the cached
           tf.data pipelines over them) split the dataset work, and training.py holds
           the loops plus the aggregation rules (average, weighted_average,
           trimmed_mean), alongside optimizers, saving/export, layers, metrics.
           layers.py in particular reimplements a few ops with custom gradients because
           the stock TF gradients only exist as Flex ops the phone's LiteRT runtime
           can't execute.
worker/    Celery task layer (TensorFlow loads at startup): celery_app.py wires the
           broker + beat schedule; tasks.py holds quantize_submission,
           validate_submission, federated_aggregation and cleanup_results (plus the
           structural malformed_reason check they share).
scripts/   CLI entry points, grouped into subpackages by how each relates to the system.
           common/ holds their shared, model-agnostic helpers (api, litert, scoring,
           reports, plots, secure + the generated dataset_pb2); scripts/__init__.py
           seeds the RNG on any submodule import.
             system/       essential to the running pipeline: get_dataset, train (exports the
                           served .tflite artifacts + is the source of the federated
                           convergence curves), transfer_learn, export_subject_data, seed_db.
             integration/  multi-user backend verification over the real HTTP API: fed_client,
                           secure_aggregation, queue_aggregation.
             figures/      report result/figure generators (see RESULTS.md): plot_signals,
                           plot_convergence (reads a previous train.py run — Secs. 5.2+5.3 —
                           and never trains), byzantine + sensitivity (sweeps that do
                           train), footprint, and the autoencoder case studies (a demonstrated
                           use case, not part of the deployed architecture): calibrate_fpr +
                           anomaly_detection (detector eval, Sec. 5.4) and
                           knowledge_distillation (distillation + LOSO personalization).
```

Evaluation/experiment output (histories, reports, figures, distilled labels) goes to
`results/<model>/`; the served `.tflite` artifacts stay in `shared/gen/models/<model>/`.
[`RESULTS.md`](RESULTS.md) lists every result Chapter 5 needs, the order to produce them in, and
the report section it feeds.

Training is split into three layers so any model can be run under any loop:

- **Model** (`TrainableModel`): the graph — `eval` / `train` / `save` / `restore`, plus
  `transfer_from` (copy compatible trainable weights from another instance of the same
  architecture, transferring the overlapping region where a shape differs — used for
  cross-batch-size transfer learning).
- **Trainer** (`Trainer`): only what is model-specific — `subject_dataset` (read one
  subject off disk), `normalize_feed` (the int8 calibration feed), `norm_param_bytes`
  and `eval_metrics` (accuracy for the MLP, reconstruction error for the autoencoders).
  It stores no data and no batch size: `subject_datasets(data_root)` is its single data
  entry point, returning every subject's batched, cached dataset in subject order, which
  `ml.loading.holdout` splits and `ml.loading.pool` merges. Each model class declares a
  `default_batch_size` and each model module exposes `get_trainer(data_root,
  batch_size=None)` (falling back to that default when `batch_size` is `None`).
- **Loop** (`training.py`): orchestration only — `normal_loop` and `federated_loop`
  (simulated FedAvg, aggregating each round's client deltas with `weighted_average`),
  over the generic
  `evaluate` / `train_epoch` steps. Loops talk only to the `Trainer` interface, so a
  `(model, trainer)` pair works with either loop and they can be compared. 
  `train.py` picks the model and loop and handles export + plotting;
  each run writes its history plot + CSV, its `run.yaml` manifest and eval report under
  `results/<model>/<loop>/` (`normal` or `federated`).

Both loops split the data at **subject** granularity: `--eval-subjects` (default `14-15`)
selects which subjects to hold out whole (a single id, an `n-m` range, or an `i,j,k` list),
the centralized loop pools the rest and the federated loop trains those same subjects as
separate clients. So a metric is generalization to an unseen
subject, and the two loops are directly comparable — `scripts.figures.plot_convergence`
plots both curves from the `run.yaml` manifests without retraining anything, and refuses to
overlay runs whose manifests disagree.

Autoencoder variants (LSTM/GRU/CNN/...) share `TrainableAutoencoder` (reconstruction
train/eval) and `AutoencoderTrainer` (windowing + recon-error metrics). They reconstruct
**BVP only, from BVP only**: the signal is the single model input, fed **raw**, and the
model z-scores it in its `eval`/`train` signatures with baked-in constants. The on-device
pipeline feeds raw the same way. ACC never reaches an autoencoder — it exists only as an
input to `FeatureMLP`'s hand-crafted features. The objective is reconstruction MSE plus a
first-difference (slope) term that penalizes a constant "flat line" output.

Since each threshold is a quantile of the subject's own clean errors, the absolute error
scale cancels — only the clean/anomalous overlap matters. See
[`shared/docs/anomalies-and-distillation.md`](../shared/docs/anomalies-and-distillation.md)
for how the reconstruction error becomes a calibrated detector and how its labels are
distilled into the student.

## Models

See [`shared/docs/model-types.md`](shared/docs/model-types.md) for what each model
architecture is and how normalization works; this section covers the backend-specific
dataset and training-pipeline details.

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
`anomaly_detection.py` measure per-kind detection recall in isolation.

Because injection operates on un-normalized signals, perturbations scale with the
signal's own range/std, so they apply at any sensor output range. The five kinds are
signal-integrity artifacts (amplitude blow-up and a wavy band-limited noise burst) and
rhythm anomalies (tachycardia/bradycardia = uniform tempo change via resampling, afib =
irregularly-irregular rhythm via a jittered time-warp). Flatline and baseline wander were
dropped: a flatline sits below the AE's reconstruction-error floor (handle sensor dropout
with a signal-quality gate instead) and wander is physiological — already in the clean
signal, so the AE rightly does not flag it. A sustained-baseline-step "spike" kind was
also dropped.

### Autoencoder evaluation and distillation

Two case studies that put the architecture to use, not parts of the deployed system:
**FPR calibration** turns the autoencoder's reconstruction error into an actual detector by
fixing the one number a client needs to derive a threshold, and **label distillation** shows
how a model with comparable behaviour reaches a light wearable — the heavy unsupervised
teacher emits soft labels that train a small supervised MLP the ESP32 can run.

The detector is the autoencoder's **reconstruction MSE**, oriented higher = more
anomalous. Its threshold is **per subject** — it fires at the `1 - expected_fpr` quantile of
*that subject's own* clean windows, so a subject-specific error scale gives a uniform
per-subject false-alarm rate instead of one dominated by the noisiest subjects. **The server
calibrates the expected FPR; the client computes the threshold.** The `expected_fpr` — the
rate at which the detector fires on clean signal — is a single global number, the only thing
calibration picks. Because each threshold is a quantile of the subject's own clean scores,
that fraction of clean windows lies above it by definition: the parameter *is* the
false-alarm rate rather than a proxy for it, so calibration reduces to maximizing
`recall(f) - f` over a 1-D grid. The rate measured on a given set is its **empirical FPR**
(`clean_fpr`); it matches `f` exactly on the subjects whose own clean windows set the
thresholds, and only lands near it on an unseen subject — hence *expected*.

The soft label a window gets is `sigmoid((error − threshold) / s)`, where `s` is the
standard deviation of that subject's own clean-window scores. Normalizing by the subject's
own clean-error spread makes the ramp calibrated per subject (reconstruction-error scale
varies between people), and `label > 0.5` reproduces the hard decision. Soft targets carry
the teacher's confidence — proper knowledge distillation, not just pseudo-labeling.

The case studies live in three `scripts/figures/` scripts:

- **`calibrate_fpr.py`** picks the expected FPR maximizing Youden's J on the split teacher's
  **training** subjects (a dense grid argmax over Youden's J) and then sweeps +
  plots the whole recall / precision / F1 / clean FPR / Youden's J curve on the **held-out**
  subjects instead — each subject's threshold is a quantile of its own clean scores, so a
  sweep measured on the calibration subjects would show the empirical FPR tracking the
  expected FPR almost exactly by construction, not a generalization number. Writes two
  figures to `results/<model>/calibrate_fpr/`: `calibration.png` (recall/FPR/J vs. expected
  FPR) and `roc.png` (the ROC curve — recall vs. empirical FPR), plus the sweep table
  (`.csv`/`.yaml`). It exists only to justify calibrating on J rather than F1 for the report;
  nothing imports it, and the two scripts below re-pick the FPR internally (it is cheap).
- **`anomaly_detection.py`** takes a model trained **with a split**: it picks the expected
  FPR inline on the training subjects and scores the detector against the true mixed-window
  labels and the per-type `anomalous-signals/` sets on the **held-out** subjects —
  precision/recall/F1, per-anomaly-kind recall and clean FPR. Held-out only, so the numbers
  are generalization to an unseen user.
- **`knowledge_distillation.py`** takes a teacher trained on **all** users (so every
  subject's soft labels are the same, teacher-seen quality) and shows a distilled student can
  be personalized. It distils the soft labels in memory and runs **leave-one-subject-out**:
  per fold a fresh `FeatureMLP` is trained centrally on the *other* subjects' distilled
  labels, fine-tuned on the held-out subject's own labels, and both are scored (float + int8)
  against that subject's **true** labels. Rotating the held-out subject keeps every fold
  leakage-free (the global never trained on the subject it is judged on) and none special.
  A fifth variant, `direct_float`, trains the same student on the other subjects' **true**
  labels instead of the soft ones and is the ceiling: `direct − global` is what distillation
  costs, i.e. what is lost by having no ground truth on the client.

Nothing is written to disk but the metrics; the student never sees a true label, mirroring
the on-device setting where only the expected FPR is global.

## Run

Fetch + preprocess the dataset first (idempotent: skips download/processing if
already present):

```bash
uv run -m scripts.system.get_dataset
```

Then train any model; the served `.tflite` artifacts land in `shared/gen/models/<model>/`
and the training report/plot in `results/<model>/<loop>/`:

```bash
uv run -m scripts.system.train feature-mlp                      # synthetic-anomaly classifier
uv run -m scripts.system.train cnn-ae                           # conditional Conv1D autoencoder (focus)
uv run -m scripts.system.train feature-mlp --loop federated     # simulated FedAvg
uv run -m scripts.system.train feature-mlp --batch-size 32       # train at a larger batch (GPU-friendly)
uv run -m scripts.system.train cnn-ae --eval-subjects 3          # hold out just S3 (LOSO-style)
uv run -m scripts.system.train cnn-ae --eval-subjects 11-15      # hold out S11..S15 (id range)
uv run -m scripts.system.train cnn-ae --eval-subjects 1,7,14     # hold out exactly S1, S7, S14
```

see [`RESULTS.md`](RESULTS.md) for every report result — the file each command emits and the
report section it feeds — and run the pieces you need by hand.

`--loop` selects the training loop (`normal` by default, or `federated`); `--epochs` tunes
the normal loop while `--rounds` and `--local-epochs` tune the federated one's global rounds
and local passes per round. `--eval-subjects` sets which subjects are held out
whole for evaluation (default `14-15`) — it takes either a single id `N` (subject `SN`,
LOSO-style), an inclusive id range `n-m` (subjects `Sn..Sm`), a comma-separated id list
`i,j,k`, or `none` (train on every subject, skip evaluation). The run manifest records the
resolved `train_subjects` and `eval_subjects` **id lists**, not just a count, so an arbitrary
split is reproducible and the case-study scripts score the exact subjects held out.
`--batch-size`
overrides the model's `default_batch_size`
(useful for GPU throughput — the on-device default batch is often 1). Each run
writes `trainable.tflite` (LiteRT-trainable) and `quantized.tflite` (int8, when
supported) into `shared/gen/models/<model>/`, and a diagnostic plot + history + `run.yaml`
into `results/<model>/<loop>/`; the intermediate
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

`--eval-subjects none` trains on **every** subject and skips evaluation (no held-out
set, so no metric plot, reconstruction report, or final metric in the manifest) — how the
all-users teacher for `knowledge_distillation` is produced. Since a run always writes the
canonical `trainable.tflite`, produce that teacher and then **rename** its artifact aside
(`trainable_all.tflite`) so the next split run doesn't clobber it; the case-study scripts take
the teacher by `--weights`, so the filename is free.

Because the model's batch size is baked into the `.tflite` input signature, the
GPU-trained large-batch model isn't itself the deliverable. `transfer_learn`
bridges that: it seeds a fresh default-batch model from the large-batch artifact's
weights (via `TrainableModel.transfer_from`) and fine-tunes it for a few epochs.

```bash
uv run -m scripts.system.train feature-mlp --batch-size 32       # 1) fast GPU training
uv run -m scripts.system.transfer_learn feature-mlp 32 --epochs 3 # 2) transfer -> default-batch + fine-tune
```

The source batch size must be `>=` the default; `transfer_learn` re-exports the
fine-tuned model under the canonical (unsuffixed) artifact names.

For the autoencoder case studies (Sec. 5.4/5.8): `calibrate_fpr` calibrates + plots the FPR
sweep and ROC curve, `anomaly_detection` scores the detector on held-out subjects, and
`subject_roc` lays every subject's ROC on a shared grid — all from the split teacher
`train.py` already produced. `knowledge_distillation` needs a teacher trained on all users,
then runs the leave-one-subject-out personalization end to end (distils the soft labels in
memory — no tree, no `--dataset-dir` student to train). Training the all-users teacher
overwrites the canonical `trainable.tflite`, so copy it aside and retrain the split to restore
the served artifact:

```bash
uv run -m scripts.figures.calibrate_fpr cnn-ae                          # FPR sweep + ROC -> results/cnn-ae/calibrate_fpr/
uv run -m scripts.figures.anomaly_detection cnn-ae                      # detector metrics -> results/cnn-ae/
uv run -m scripts.system.train cnn-ae --eval-subjects none              # all-users teacher (overwrites trainable.tflite)
cp shared/gen/models/cnn-ae/trainable.tflite shared/gen/models/cnn-ae/trainable_all.tflite  # keep it aside
uv run -m scripts.figures.subject_roc cnn-ae --weights shared/gen/models/cnn-ae/trainable_all.tflite  # per-subject spread, every subject on equal footing
uv run -m scripts.figures.knowledge_distillation cnn-ae --student feature-mlp \
    --weights shared/gen/models/cnn-ae/trainable_all.tflite             # LOSO personalization
uv run -m scripts.system.train cnn-ae --eval-subjects 14-15             # restore the canonical split teacher
```

The distilled-vs-direct student comparison (does the distilled student match one trained on
the synthetic ground truth?) is the `direct_float` variant of `knowledge_distillation`.

### Export a subject to the Android app

`export_subject_data.py` packs one subject's windows into the `.ssds` protobuf the app
imports (run `make proto` first). Each window mirrors an ESP sample: raw PPG/ACC, the raw
feature vector, the label in the score field, plus a fake sequence number and contiguous
8 s device-time grid, so imported windows preprocess and train exactly like streamed ones.

```bash
uv run -m scripts.system.export_subject_data 1                              # S1.ssds, every window complete
uv run -m scripts.system.export_subject_data 1 --missing-samples 0.7       # keep 70% of windows' signal; drop the rest
uv run -m scripts.system.export_subject_data 1 --missing-features 0.7      # keep 70% of windows' ML result; phone recomputes the rest
uv run -m scripts.system.export_subject_data 1 --missing-samples 0.7 --missing-features 0.7  # both, drawn independently
uv run -m scripts.system.export_subject_data 1 --clean                     # clean (anomaly-free) signals; features from clean-features
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
only the external services — PostgreSQL and Redis — run in containers via
`compose.yaml` (`make db-run`, podman). Credentials and connection settings are read from a
`.env` file (`common/config.py` loads it via `python-dotenv`, and `compose.yaml` uses it for
variable substitution); copy `example.env` to `.env` and adjust as needed — there are no
hardcoded defaults, so the gateway/worker refuse to start without it. Both are bound to `127.0.0.1` since only the
host-run processes reach them; the API itself binds `0.0.0.0` so the phone can reach it over
the LAN. The gateway and worker do not need a shared filesystem — they only need to reach the
same database, which is what makes the worker actually distributable (Celery's premise)
instead of implicitly assuming a co-located disk.

- **Gateway (`api/`, no TensorFlow):** serves the model artifacts (read from the DB and
  returned as stored, keyed by the active `GlobalWeights` row) — either the whole
  trainable/quantized `.tflite`, or just the snapshot's **flat weight buffer** for a client
  that already holds the graph (`GET /model/weights/{key}`, see "Serving weights on their
  own") — accepts weight-delta uploads, persists them, enqueues worker jobs, and exposes a
  result endpoint the client polls. Fast to start since it never imports TF.
- **Worker (`worker/tasks.py`):** restores uploaded weights into the model, converts it to
  an int8 `.tflite` against the per-model calibration dataset
  (`ml/saving.py:get_optimized_model`) and signs it, validates submit-only uploads, and
  runs the daily federated aggregation (see "Federated aggregation"). Each forked worker
  child builds every available model (skipping any whose dataset is absent) and caches each
  one's `(model, representative dataset, fingerprint, contract_version, norm bytes)`. This
  build runs post-fork (via the `worker_process_init` signal), never in the parent
  MainProcess: TensorFlow is not fork-safe once its runtime exists, so initializing it before
  the prefork pool forks would deadlock every child on inherited-locked native mutexes.

There are **multiple upload paths**, and which one a model accepts is a per-model property:
each `ModelVersion` carries a `submission_type` (`raw` / `quantize` / `secure`, sourced from
the code registry `ml/model_list.py` and seeded into the DB). See
[`shared/docs/submission-type.md`](shared/docs/submission-type.md) for what each type is
*for* — in particular why the `quantize` path exists (delivering a personalized int8 model
to the firmware) and why that quantization runs server-side; this section covers the
backend-specific endpoints and mechanics. The `quantize` path accepts only
`quantize`-typed models; the `raw` (submit-only) path accepts both (`quantize`'s dense body
is compatible and submit-only is the least work). A model uploaded on a path it doesn't
accept gets `404` (not `403`, so the path stays unguessable). The type also selects the
aggregation strategy (see "Federated aggregation"); `secure` carries an incompatible
(masked, non-float32) body and aggregates only inside a sealed round, so it lives entirely
on its own `/model/secure/*` endpoints (see "Secure aggregation"); future formats (sparse,
DP) will likewise add their own type + endpoint. Both dense paths persist a `ClientDeltaSubmission` that feeds
aggregation; both take the raw little-endian float32 **weight-delta** buffer (Δ = local −
global, the change local training produced against the snapshot it trained from) as the
request body and the `weights_id` of that `GlobalWeights` snapshot (echoed by the download
headers) in the path. Malformed bodies (wrong length, non-finite values) are
rejected with `400`; an unknown `weights_id` is `400`; a `weights_id` that isn't the
version's **active** weights — a frozen (non-latest) version, or a snapshot a later round
superseded — is `409`, so a delta is only ever accepted against the base download served
and aggregation reconstructs from; the client must re-download the latest weights first.

Request flow for a `quantize`-typed model:

1. `POST /model/submit/quantize/{key}/{weights_id}` stores the delta as a
   `ClientDeltaSubmission` (tagged with the submitting `user_id` and the `base_weights_id` —
   its model and version are reachable through that snapshot), creates a `QuantizationJob`
   (`pending`), enqueues `quantize_submission` (the job id is set as the Celery task id), and
   returns `202` with the `job_id`.
2. The worker runs the job: a malformed delta fails it (and caches `valid = false` on the
   submission); otherwise `valid = true` is cached and the artifact is produced. The int8
   `.tflite` is written (zstd-compressed) to the job row along with an ECDSA signature over the
   canonical model bytes (`ml/payload.py`, spec in `shared/docs/model-signing.md`; the signature
   covers the raw model, so it is signed before compression).
3. The client polls `GET /model/quantize/result/{job_id}`. The endpoint authorizes and
   answers from the DB row first (the source of truth for the verdict), and only when the
   job is still `pending`/`running` does it **long-poll** — blocking on the Celery task (the
   job id is its task id) for up to `RESULT_POLL_TIMEOUT_SECONDS`, releasing the DB connection
   meanwhile — before re-reading the settled row. It returns `202` while still running,
   `422` on `failed` (with the error), `200` with the zstd-compressed int8 `.tflite` body plus
   the `X-Model-Signature` / `X-Contract-Version` / `X-Norm-Params` headers once `done` — the
   app packages those fields for the ESP32 per its BLE interface version, and the firmware
   re-derives the canonical bytes and verifies the signature before loading. The result is
   scoped to the user who submitted it (resolved via the job's `ClientDeltaSubmission`); another
   user's `job_id` returns `404`, and it never waits on a task it doesn't own.

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
aggregation round per initialized model:

0. **Strategy:** chosen by the latest version's `submission_type`. `raw` and `quantize` are
   byte-identical dense vectors and share the aggregation path below; sparse/DP formats will
   branch here. A model whose type has no strategy is skipped.
1. **Window:** the deltas whose `base_weights_id` is the version's **active** (newest valid)
   `GlobalWeights` snapshot. Matching on the base id guarantees every accepted delta was
   trained against the same weights and never mixes across rounds or frozen versions. One
   update per client: only each user's latest submission counts.
2. **Validation:** a structural check (`malformed_reason` in `worker/tasks.py`) — the weight
   count must match the model's `total_weight_size` and the buffer must be finite. It runs
   once per submission: the quantize/validate tasks perform it as uploads arrive and cache the
   verdict on `ClientDeltaSubmission.valid` (never surfacing it to the client); aggregation
   trusts that verdict and validates only rows neither task got to.
3. **Round threshold:** fewer than `FED_MIN_SUBMISSIONS` (default 1) valid submissions skips
   the model until the next round.
4. **Aggregation:** `ml.training.trimmed_mean` reduces the accepted deltas to one update,
   dropping the `FED_TRIM_RATIO` fraction (default 0.1) of smallest and largest values at
   each coordinate before averaging the rest — this is the round's only Byzantine defense, so
   a handful of gross outliers can't drag the result. The update is added onto the reference
   global weights (identical to aggregating absolute weights when every client shares a base)
   and stored as a new `GlobalWeights` row.

   Weighting by dataset size — the "Avg" in textbook FedAvg, and what the simulated
   `federated_loop` does with its known-honest clients — is deliberately **not** what the
   server does: a submission carries no trustworthy sample count, so an attacker claiming an
   enormous dataset would simply dictate the global model. Uniform weighting removes that
   lever, and the trimmed mean removes the next one, since a single extreme coordinate is
   discarded instead of dragging the mean with it. `ml.training` exposes all three rules
   (`average`, `weighted_average`, `trimmed_mean`) and `scripts.figures.byzantine`
   `--aggregator` compares the two that are sound here.
6. **Artifact baking:** the averaged weights are restored into the cached model and both
   serving artifacts are re-exported as `WeightsArtifact` rows keyed by the new snapshot —
   the LiteRT-trainable `.tflite` and the signed int8 `.tflite` — so a client always pulls a
   file with the current global parameters already inside. They commit in the same transaction
   as the snapshot, so a visible row always has its artifacts. If an export fails the row is
   stored with `valid = false` and no artifacts: clients keep pulling the previous snapshot and
   the window's submissions stay consumed.
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
uv run -m scripts.integration.queue_aggregation           # every initialized model
uv run -m scripts.integration.queue_aggregation cnn-ae    # a single model
```

**Headless federated run.** `scripts/integration/fed_client.py` drives the whole stack over the real
HTTP API: it downloads the trainable artifact once up front (the graph is fixed within a
version), then for each dataset subject (as user `test_N`) it logs in, pulls the current
global weight buffer (`/model/weights`), restores it, trains one pass through the on-device
LiteRT `CompiledModel` runtime, uploads the update, and logs out; then it queues a round,
waits for the new `GlobalWeights`, and scores it on the held-out subjects, repeating for
`--rounds`. It picks the aggregation
strategy from the model's `submission_type`: the dense `raw`/`quantize` path (convergence
series to `results/<model>/fed_client/`) or the masked `secure` path (see "Secure
aggregation" for the extra phases, series to `results/<model>/secure_fed_client/`).
Seed the accounts first with `scripts.system.seed_db --test-users` (one `test_N` per subject,
each owning a placeholder device).

```bash
uv run -m scripts.system.seed_db --test-users                       # one test_N per subject
uv run -m scripts.integration.fed_client --model feature-mlp --rounds 5 --eval-subjects 2   # dense
uv run -m scripts.integration.fed_client --model cnn-ae     --rounds 5 --eval-subjects 2   # secure
```

### Secure aggregation

A model whose `submission_type` is `secure` (currently `cnn-ae`) aggregates weight
updates the server can never read individually: it sees only their sum. The scheme is a
minimal masking protocol for an **honest-but-curious server with no client dropouts and no
Byzantine clients** — the full construction and its invariants are in
[`shared/docs/secure-aggregation.md`](shared/docs/secure-aggregation.md). Unlike the dense
paths' implicit "aggregate whoever submitted since the last snapshot" window, a secure
**round is a first-class object** because the masks require the cohort and its public keys
to be frozen before anyone masks:

1. **Join** (`POST /model/secure/join/{key}`, body `{ka_public_key}`) — a client publishes
   its long-term ECDH public key and takes a seat in the model's `open` `SecureRound`
   (created on the first join, pinned to the version's active `GlobalWeights` as the base
   `W` every member trains against). The key is **snapshotted** into `SecureRoundMember`,
   not looked up later. Returns the `round_id`, the base `weights_id`, and the caller's own
   `user_id` (needed for the add/subtract mask ordering).
2. **Seal** — the roster is frozen, `n` and the fixed-point scale `S = floor(2^31/(n·B))`
   are set, and the round goes `sealed`. There is deliberately **no seal endpoint**: the
   harness (or, in a real deployment, an operator/beat task) does it, so a client can never
   freeze a roster mid-join. Needs `n ≥ SECURE_MIN_MEMBERS` (default 3).
3. **Masked submit** (`POST /model/secure/submit/{round_id}`) — a member fetches the frozen
   descriptor (`GET /model/secure/round/{round_id}`: roster + keys, `n`, `B`, `S`, `R`),
   trains against `W`, clips its delta to ±`B` (`SECURE_CLIP_BOUND`), quantizes into `Z_R`
   (`R = 2^32`), adds each pairwise mask (lower `user_id` adds, higher subtracts), and
   uploads only the `m` little-endian uint32 masked vector. Accepted **exactly once** per
   member (the `(round_id, user_id)` primary key is the structural guard the protocol
   demands — a second vector under the same masks would leak a difference).
4. **Aggregate** (`worker.tasks.secure_aggregation`) — sums the masked vectors in the ring
   (every pairwise mask cancels), dequantizes to the uniform mean delta, adds it onto `W`,
   and bakes a new `GlobalWeights` + serving artifacts via the same path the dense round
   uses. The mean is the only rule the masking admits: a trimmed mean would need to compare
   individual updates coordinate-wise, which is exactly what the server cannot see. If any
   member never submitted, the **round fails wholesale** (masks only cancel over the full
   roster) — no partial recovery. The daily `federated_aggregation` beat skips `secure`
   models (it has no strategy for them); a secure round only runs when its task is queued.

**What secure aggregation costs.** The server never sees an individual update, so the
trimmed mean and the per-client validity verdict are both **structurally impossible**
here — each needs to look at updates individually. This is inherent to the scheme, not a
gap. Only
client-side clipping to `B` and an aggregate-level sanity check (finite, mean delta within
the clip bound) remain. It buys privacy against an honest-but-curious server, not
robustness.

**Headless secure run.** The secure path of `scripts/integration/fed_client.py` (selected when the
model's `submission_type` is `secure`) drives the whole stack over the real HTTP API,
running the four phases above per round for each dataset subject (as user `test_N`), and
additionally verifies client-side that the masks cancel exactly each round. The per-round
convergence series is written to `results/<model>/secure_fed_client/`.

```bash
uv run -m scripts.system.seed_db --test-users                            # one test_N per subject
uv run -m scripts.integration.fed_client --model cnn-ae --rounds 5 --eval-subjects 2
```

**No-training aggregation check.** `scripts/integration/secure_aggregation.py` exercises the same
secure endpoints without any training: each client submits a *random* weight tensor and the
script asserts the global weights the server bakes equal the plaintext mean (up to
quantization + float32 error) — a fast correctness probe of the masking/summation pipeline.

```bash
uv run -m scripts.integration.secure_aggregation --clients 4 --rounds 3
```

**Model versioning.** See [`shared/docs/versioning.md`](shared/docs/versioning.md) for
what `version`, `contract_version`, `fingerprint` and weights (`weights_id` /
`weights_version`) each mean and how a client reacts to each changing. Backend-specific:
`ModelVersion.version` is hand-bumped in the code registry (`ml.model_list.ModelSpec`),
`fingerprint` is derived (`Trainer.arch_fingerprint()`) and enforced as a tripwire only by
`scripts/system/seed_db.py` (aborts if the fingerprint moved but the version didn't); `/model/list`
reports the latest version only, `/model/versions/{key}` the full history; `/model/download/*`
echoes `X-Model-Fingerprint`, `X-Model-Version`, `X-Weights-ID` and `X-Weights-Timestamp`.

The registry that ties a model `key` to its metadata *and* its TensorFlow trainer builder is
`ml/model_list.py` — the single source of truth consumed by `scripts/system/train.py` (one trainer),
`worker/tasks.py` (all models + fingerprints, built per worker child), and `scripts/system/seed_db.py` (publishes
versions + metadata, enforcing the fingerprint tripwire). The api never imports it; it
trusts what the seed wrote to the DB.

**Storage decisions (thesis scope, no production deployment):**

- **PostgreSQL** (via **SQLModel**) holds weight submissions and quantization jobs. Job state
  lives here — it is the single source of truth the poll endpoint reads. Celery's **Redis
  result backend** is used only so callers can await a queued task and read its return value
  (e.g. `queue_aggregation`/`fed_client` block on the aggregation summary instead of polling
  `GlobalWeights`); results expire after `RESULT_TTL_SECONDS`.
- **Redis** is the Celery broker.
- **Every served blob lives in PostgreSQL too.** Model blobs sit in their own table keyed by
  the row that owns it: `WeightsArtifact` (a snapshot's trainable/quantized `.tflite`, keyed by
  `GlobalWeights` + artifact) and `QuantizationResult` (keyed by `QuantizationJob`). A handler
  locates a blob from the row it already loaded, and row existence *is* the presence check — a
  snapshot may legitimately have a trainable artifact and no quantized one. They are separate
  tables rather than columns because the owning rows are read constantly and the blobs almost
  never: aggregation loads `GlobalWeights` for its weights and should not drag a `.tflite`
  along. Firmware keeps its image inline in the `Firmware` row but `/ota/versions` defers the
  `data` column (`list_firmware`), so the listing stays blob-free. Blob columns are
  `STORAGE EXTERNAL` since zstd output is not worth a TOAST compression pass.
- **Why not object storage.** This replaced a local-disk design (`serve/` tree) once the worker
  stopped being assumed to share a filesystem with the API — Celery is a distributed-computing
  tool, so pretending its workers and the gateway share a mount was wrong from the start. The
  obvious fix looked like MinIO, and an earlier revision used it; but the database is *already*
  the shared state both processes reach (the worker reads `GlobalWeights.weights` to aggregate,
  the gateway reads rows on every request), so it satisfies the distributability requirement
  without a second store. What a bucket adds at this scale is a second, untransacted source of
  truth: artifacts had to be uploaded between `flush()` and `commit()`, a failed commit orphaned
  them, the cleanup sweep could delete an object and then fail to null its marker, and the
  download route had to `stat` the bucket because the DB could not say which artifacts existed.
  In Postgres all of that is one transaction and an FK. The blobs stay within what a database
  handles comfortably — `cnn-ae`'s trainable `.tflite` is 13 MB (12 MB zstd, the form served),
  the int8 one 1.2 MB, a couple of MB per firmware image. The trainable artifact grew ~30x when
  `cnn-ae` moved to `latent_dim` 256 (the two bottleneck projections dominate the parameter
  count), so it is the figure to re-check if the architecture grows again. A deployment serving
  many models to many clients would move them behind a
  CDN or object store (the gateway proxies today, so that swap is a storage-layer change, not a
  protocol one); at thesis scale the object store was cost without benefit.
- **The gateway always proxies.** It reads the blob server-side and returns it, so auth and
  per-user rate limiting apply to every byte served, and no storage layer is ever exposed to the
  client. This is what forecloses presigned URLs regardless of backend: the download cooldown is
  spent only once an artifact is actually served (see "Auth & rate limiting"), which a URL handed
  out ahead of time cannot account for, and the response headers carry the signature and contract
  metadata the client needs alongside the bytes. Rollback is still a `valid` flag flip; the
  previous snapshot's artifacts are untouched.
- **Everything served is zstd-compressed.** Blobs are stored compressed and served as-is; the
  client decompresses (any zstd library, a few lines). Signatures cover the *raw* bytes, so the
  server compresses after signing and the client verifies after decompressing — compression is
  a pure transport wrapper, invisible to the signing scheme (`common/compression.py`).

**Result lifecycle.** A `done` result is served on request and stamped `served_at`. A Celery
beat sweep (`cleanup_results`, in-process via `celery worker -B`) deletes the `QuantizationResult`
row and nulls the job's `signature` once a served result is older than
`SERVE_GRACE_SECONDS` (5 min) or an unclaimed one is older than `RESULT_TTL_SECONDS` (1 h),
flipping the job to `expired`. The sweep joins against the result table rather than selecting it,
so it never loads the blobs it is about to drop. Weight submissions are never reaped.

## Auth & rate limiting

All `/model/*` routes require a logged-in user, and the rate-limited ones
additionally require the user to be a verified device owner (see "Device
attestation"). Accounts are **seeded, not self-registered** —
`uv run -m scripts.system.seed_db` (or `make db-seed`, which also passes
`--assign-device --test-users`) bootstraps a fresh DB with the model registry rows
and a default user (`SEED_USER` / `SEED_PASSWORD`, default `somasafe` /
`somasafe`); it is idempotent. Pass a factory NVS CSV as the positional argument
(`uv run -m scripts.system.seed_db firmware/factory_nvs.csv`) to also register that device.

A model's weights are seeded **once** and then owned by aggregation, so re-running the seed
after retraining leaves the old snapshot in place. `--reseed` (or `make db-reseed`)
re-points each seeded model at the artifacts currently on disk: it drops the model's
`GlobalWeights` history along with everything anchored to it — the submissions, quantization
jobs and secure rounds that name a base snapshot they were computed against, and the stored
`WeightsArtifact` rows — then re-seeds from `shared/gen/models/<model>/`. It also skips the
idempotency checks: an existing version's registry row is overwritten in place, **including a
moved architecture fingerprint**, which otherwise aborts the seed (see
[`shared/docs/versioning.md`](../shared/docs/versioning.md)) — that is what makes it the way
to re-seed after changing a model rather than merely retraining one. Model definitions and
versions are untouched.

Session semantics (stateful tokens, `api/routes/auth.py` endpoints, argon2 password
hashing) are documented in [`shared/docs/authentication.md`](shared/docs/authentication.md).

**Rate limiting is per-user, per-resource** (the keying and Redis mechanics in
`common/ratelimit.py`, the gateway's 429-raising wrappers in `api/lib/ratelimit.py`; on the
`REDIS_URL` connection, which the device-attestation challenge store shares and which is
separate from the Celery broker).
The intent: a client can download + quantize every model once in a single pass,
but immediate repeats on the same model are rejected with `429` (+ `Retry-After`).
Each limit is a capped counter over a rolling window (a cooldown is the same with a
cap of 1), keyed per (`RateLimit` action, user, resource — a model key, or a firmware
interface for OTA). Enforcement is **two-phase**: the endpoint checks the limit up
front (`check_limit`) but spends a slot (`record_usage`) only *after* it has done the
work, so a request rejected by the validity checks (or the limit itself) never counts
against the quota — while a request that got as far as doing real work is charged even
if that work then fails.

| Endpoint | Limit |
|----------|-------|
| `GET /model/list`, `GET /model/versions/{key}` | authed only |
| `GET /model/download/{trainable,quantized}/{key}[?version=N]` | device-owner only; one download per model per `DOWNLOAD_COOLDOWN_SECONDS` (default 300 s) |
| `GET /model/weights/{key}[?version=N]` | device-owner only; one pull per model per `DOWNLOAD_COOLDOWN_SECONDS` (default 300 s), on a counter separate from the artifact download |
| `POST /model/submit/quantize/{key}/{weights_id}` | device-owner only; `QUANTIZE_DAILY_LIMIT` (default 2) per model per rolling 24 h; `404` unless the model's `submission_type` is `quantize` |
| `POST /model/submit/raw/{key}/{weights_id}` | device-owner only; `SUBMIT_DAILY_LIMIT` (default 2) per model per rolling 24 h; `404` unless the model's `submission_type` is `raw` or `quantize` |
| `GET /model/quantize/result/{job_id}` | authed; only the user who submitted the job (else `404`) |
| `POST /model/secure/join/{key}` | device-owner only; `404` unless the model's `submission_type` is `secure` |
| `GET /model/secure/round/{round_id}` | authed; only a member of the round (else `404`); `409` until the round is sealed |
| `POST /model/secure/submit/{round_id}` | device-owner + round member; one masked vector per member (structural, `409` on repeat) |
| `GET /ota/versions/{interface}` | authed only |
| `GET /ota/download/{interface}/{version}` | device-owner only; one firmware download per interface per `OTA_DOWNLOAD_COOLDOWN_SECONDS` (default 300 s) |

The model-artifact download is a single route with an `Artifact` enum path parameter
(`trainable` / `quantized`), serving the artifact file keyed by the version's active
`GlobalWeights` row (`?version=` selects a frozen version; default is the latest). It
echoes `X-Model-Fingerprint`, `X-Model-Version`, `X-Weights-ID` and
`X-Weights-Timestamp`; the quantized artifact additionally carries `X-Model-Signature`,
`X-Contract-Version` and `X-Norm-Params` (see `shared/docs/model-signing.md`). The body is
zstd-compressed — the client decompresses before verifying/using it.

### Serving weights on their own

`GET /model/weights/{key}[?version=N]` serves **just the flat weight buffer** of the same
active `GlobalWeights` snapshot the trainable artifact is baked from — the raw
little-endian float32 vector (`GlobalWeights.weights`, exactly what the model's `restore`
signature consumes), stored zstd-compressed and served as stored like every other blob (the
weights are compressed once when the snapshot is baked, never per request). The model graph is
fixed within a `version`, so only the weights change from round to round: a client that
already downloaded the trainable `.tflite` once can pull this much lighter buffer each
round and restore it into the model in hand, instead of re-downloading the whole artifact
(the `cnn-ae` trainable is ~12 MB compressed; its weight buffer is a fraction of that and
carries no graph or optimizer state). It echoes the same
`X-Model-Fingerprint`/`X-Model-Version`/`X-Weights-ID`/`X-Weights-Timestamp` headers as the
artifact download and keys off the same active row, so the `X-Weights-ID` it returns is the
base a subsequent delta submits against. Its cooldown is a separate per-user counter from
the artifact download (both cleared after a federated round), so grabbing the graph once and
then refreshing weights never trips the artifact cooldown and vice versa.
`scripts/integration/fed_client.py` uses it: it downloads the trainable artifact once at
startup and pulls only the weight buffer each round.

## Firmware distribution (OTA)

The `/ota` routes serve published firmware builds for the BLE OTA path (see
`shared/docs/versioning.md`, "Firmware distribution"). `GET /ota/versions/{interface}`
lists the builds published for a `BLE_INTERFACE_VERSION` (newest first, each with its
version string, supported model-contract list, image size and release date — an unknown
interface yields `[]`); `GET /ota/download/{interface}/{version}` streams the raw image
with the server's ECDSA signature over it in `X-Firmware-Signature`, which the app
forwards to the device for verification against its factory `srv_pub`.

The zstd-compressed image is the `Firmware` row's own `data` column, alongside the raw `size`
and the mandatory `signature`; `list_firmware` defers `data`, so listing versions never reads
an image.
`scripts/system/seed_db.py` publishes them: it scans a directory of exports (`--firmware-dir`,
default `shared/gen/firmware/`, populated by `make export-image` in `firmware/`), signs
each image with `SERVER_PRIVATE_KEY_FILE` (a build it can't sign is skipped — the signature
is not optional) and inserts any version not already present. Re-publishing a changed build
under an existing version means deleting its row and re-seeding (no production environment,
no migrations). The download body is zstd-compressed; the client decompresses, then verifies
the signature (over the raw image) before forwarding it to the device.

## Device attestation

See [`shared/docs/device-attestation.md`](shared/docs/device-attestation.md) for the
full ownership-proof flow. Backend-specific: `api/routes/device.py` implements it; a
`Device` row holds the `serial` (PK), the 65-byte uncompressed `public_key`, an optional
`owner_id`, and `last_attested_at`. Devices are seeded ownerless from a factory NVS image
(`scripts/system/seed_db.py`).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/device/owned` | serials of the devices the caller currently owns |
| POST | `/device/challenge` | `{serial}` → `{instance_id, nonce, server_time, user_id}` |
| POST | `/device/attest` | `{instance_id, signature}` → verify and set the owner |

