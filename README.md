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
           SecureRoundMember, Firmware, FirmwareImage).
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
           the loops incl. fed_avg, alongside optimizers, saving/export, layers, metrics.
           layers.py in particular reimplements a few ops with custom gradients because
           the stock TF gradients only exist as Flex ops the phone's LiteRT runtime
           can't execute.
worker/    Celery task layer (TensorFlow loads at startup): celery_app.py wires the
           broker + beat schedule; tasks.py holds quantize_submission,
           validate_submission, federated_aggregation and cleanup_results; utils/ has
           the TF-free validation/outlier-filtering helpers tasks.py calls into.
scripts/   CLI entry points, grouped into subpackages by how each relates to the system.
           common/ holds their shared, model-agnostic helpers (api, litert, scoring, dsp,
           reports, plots, secure + the generated dataset_pb2); scripts/__init__.py
           seeds the RNG on any submodule import.
             system/       essential to the running pipeline: get_dataset, train (exports the
                           served .tflite artifacts + is the source of the federated
                           convergence curves), transfer_learn, export_subject_data, seed_db.
             distillation/ the unsupervised-teacher label pipeline + personalization probe
                           (a demonstrated use case, not part of the deployed architecture):
                           distill_calibrate/labels/eval, personalize_test.
             integration/  multi-user backend verification over the real HTTP API: fed_client,
                           secure_aggregation, queue_aggregation.
             figures/      report result/figure generators (see PLOTS.md): plot_signals,
                           plot_convergence (reads a previous train.py run — Secs. 5.2+5.3 —
                           and never trains), byzantine + sensitivity (sweeps that do
                           train, over datasets ml.loading builds once), footprint.
```

[`TUNING.md`](TUNING.md) records the `cnn-ae` tuning pass and the distillation
simplification — the sweep data behind the current defaults, the negative results (FiLM and
latent dropout were dead weight; the raw ACC channel is a no-op), why calibration maximizes
Youden's J rather than F1, and which anomaly kinds are undetectable by construction.

Evaluation/experiment output (histories, reports, figures, distilled labels) goes to
`results/<model>/`; the served `.tflite` artifacts stay in `shared/gen/models/<model>/`.
[`PLOTS.md`](PLOTS.md) catalogs every figure/report file, the command that produces it, and
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
  (simulated FedAvg with an injectable `aggregate` strategy), over the generic
  `evaluate` / `train_epoch` steps. Loops talk only to the `Trainer` interface, so a
  `(model, trainer)` pair works with either loop and they can be compared. `train.py` picks the model and loop and handles export + plotting;
  each run writes its history plot + CSV, its `run.yaml` manifest and eval report under
  `results/<model>/<loop>/` (`normal` or `federated`).

Both loops split the data at **subject** granularity: `--eval-subjects N` (default 2) holds
out the last N subjects whole, the centralized loop pools the rest and the federated loop
trains those same subjects as separate clients. So a metric is generalization to an unseen
subject, and the two loops are directly comparable — `scripts.figures.plot_convergence`
plots both curves from the `run.yaml` manifests without retraining anything, and refuses to
overlay runs whose manifests disagree.

Autoencoder variants (LSTM/GRU/CNN/...) share `TrainableAutoencoder` (reconstruction
train/eval + conditioning) and `AutoencoderTrainer` (windowing + recon-error metrics).
They reconstruct **BVP only**, from BVP only: ACC as a raw encoder channel measured as a
no-op (identical reconstruction error and identical per-kind recall, to three decimals), so
it reaches the model *solely* through the condition. Every model is **conditioned** on a
single `cond` vector — z-scored demographics plus a causal *activity context* (trailing-2-min
mean/std of the ACC). The context is computed from the **raw** ACC; the whole `cond` (and
the BVP signal) is fed to the model **raw**, and the model z-scores it in its `eval`/`train`
signatures with baked-in constants (`context_norm_params.npy` is just the ACC mean/std, so
normalizing it equals the old "normalize ACC, then take trailing stats"). The on-device
pipeline feeds raw the same way. The objective is reconstruction MSE plus a first-difference
(slope) term that penalizes a constant "flat line" output.

What makes the error separate anomalies is how **tightly the model fits the clean-BVP
manifold**, not how narrow the code is — a sharper fit makes off-manifold input miss by
relatively more. Detection therefore improves with capacity up to `latent_dim` 256 and
degrades past it, and *starving* the code hurts: at 16 it cannot reconstruct clean BVP
either and detection collapses with it. Latent dropout hurts for the same reason and is off
by default. Since each threshold is a quantile of the subject's own clean errors, the
absolute error scale cancels — only the clean/anomalous overlap matters.

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
signal-integrity artifacts (amplitude blow-up and a wavy band-limited noise burst) and
rhythm anomalies (tachycardia/bradycardia = uniform tempo change via resampling, afib =
irregularly-irregular rhythm via a jittered time-warp). Flatline and baseline wander were
dropped: a flatline sits below the AE's reconstruction-error floor (handle sensor dropout
with a signal-quality gate instead) and wander is physiological — already in the clean
signal, so the AE rightly does not flag it. A sustained-baseline-step "spike" kind was
also dropped.

### Autoencoder evaluation and distillation

The detector is the autoencoder's **reconstruction MSE**, oriented higher = more
anomalous. Its threshold is **per subject** — it fires at the `1 - budget` quantile of
*that subject's own* clean windows, so a subject-specific error scale gives a uniform
per-subject false-alarm rate instead of one dominated by the noisiest subjects. The
`budget` (the share of clean windows the detector may fire on) is a single global number,
the only thing calibration picks. Because each threshold is a quantile of the subject's own
clean scores, the clean false-positive rate a budget buys *is* the budget, so the
calibration reduces to maximizing `recall(budget) - budget` over a 1-D grid.

In-band spectral entropy is scored alongside as a hand-crafted **baseline**, thresholded at
the same budget so its precision/recall are directly comparable. It is not part of the
detector and never reaches the distilled labels — `distill_eval.py` reports it so the
learned teacher can be read against a classical index.

The work splits into three scripts along **what data each is allowed to see** — mirroring
deployment, where only the budget is global and everything else is done per-client on
unlabeled data:

- **`distill_calibrate.py` (server: labeled, global)** picks the **budget** — the only
  globally-relevant output, and the only thing that reads the synthetic labels. It is the
  level maximizing the detector's Youden's J (mixed-set recall minus clean FPR) on the
  labeled data. Writes the budget (and the whole sweep, for the report) to
  `results/<model>/`.
- **`distill_labels.py` (client: unlabeled)** touches only what a real client has — its
  own clean baseline and the mixed signal + on-device features, **never the true labels or
  the per-anomaly sets**. Reads the budget, derives each subject's threshold from its
  *own* clean windows, and emits a **soft** `[0,1]` label per window: the clean-CDF rank
  past that threshold (so `label > 0` reproduces the hard decision), then a size-1 temporal
  **median filter** (real anomalies span many windows, so a lone flag is a false positive
  and a lone gap a false negative — cleaned without tuning to the injected span length).
  Soft targets carry the teacher's confidence — proper knowledge distillation, not just
  pseudo-labeling.
- **`distill_eval.py` (science: unrestricted)** replays the same budget → per-subject
  thresholds a client uses, then scores the detector against the true mixed-window labels
  and the per-type `anomalous-signals/` sets: precision/recall/F1, per-anomaly-kind recall
  and clean FPR, for the detector and the spectral baseline alike. Writes the metrics to
  `results/<model>/`.

The labels land in a datasets-shaped tree (`mixed-features/S*/` with the distilled
`labels.npy`, feature arrays symlinked back to `datasets/`), so the student `FeatureMLP`
trains on them via `train.py --dataset-dir` — the path to validating an unsupervised
teacher that needs no labels on-device.

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
uv run -m scripts.system.train lstm-ae                          # conditional LSTM autoencoder
uv run -m scripts.system.train gru-ae                           # conditional GRU autoencoder
uv run -m scripts.system.train feature-mlp --loop federated     # simulated FedAvg
uv run -m scripts.system.train feature-mlp --batch-size 32       # train at a larger batch (GPU-friendly)
uv run -m scripts.system.train cnn-ae --eval-subjects 3          # hold out the last 3 subjects
```

To produce **every** report result in one go — from an empty database and no exported
artifacts, with the services/api/worker already up — run `./run_all.sh` (or `make run-all`).
It sequences training, figures, the distillation round-trip, seeding and the integration
harness in dependency order; it is resumable (each step is skipped when its output exists,
`FORCE=1` redoes them) and tunable via the env vars at the top of the file. See
[`PLOTS.md`](PLOTS.md) for what each step emits.

`--loop` selects the training loop (`normal` by default, or `federated`); `--epochs` tunes
the normal loop while `--rounds` and `--local-epochs` tune the federated one's global rounds
and local passes per round. `--eval-subjects` (default 2) sets how many subjects are held out
whole for evaluation. `--batch-size` overrides the model's `default_batch_size`
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

`--tag NAME` does the same for a *variant* of a model you also train normally: results go to
`results/<model>/<loop>-<name>/` and artifacts are suffixed (`trainable_<name>.tflite`), so
the canonical artifact — the one `seed_db` publishes and the system serves — stays the
untagged run. It exists because the same model key can be trained on different data: the
distillation round-trip below trains `feature-mlp` on both the synthetic ground truth and the
teacher's distilled labels, and without a tag the second run would silently overwrite the
first's results and artifacts.

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

To run the knowledge-distillation round-trip, first calibrate the autoencoder
(`distill_calibrate.py` picks the budget), then `distill_labels.py` derives each subject's
threshold and emits the soft-label tree; `distill_eval.py` reports the detector's per-kind
metrics against the ground truth. Then point `feature-mlp` at those labels with
`--dataset-dir`:

```bash
uv run -m scripts.distillation.distill_calibrate cnn-ae                                       # budget  -> results/cnn-ae/
uv run -m scripts.distillation.distill_eval cnn-ae                                            # metrics -> results/cnn-ae/
uv run -m scripts.distillation.distill_labels cnn-ae                                          # teacher -> results/cnn-ae/distilled-labels/
uv run -m scripts.system.train feature-mlp --dataset-dir results/cnn-ae/distilled-labels \
    --tag distilled                                                                       # student on pseudo-labels
```

`--tag distilled` keeps this run off the canonical `feature-mlp` results and artifacts, so it
can be compared against the same student trained on the direct synthetic labels rather than
replacing it (see "Run"). `personalize_test` fine-tunes the distilled student, so pass
`--weights shared/gen/models/feature-mlp/trainable_distilled.tflite`.

### Export a subject to the Android app

`export_subject_data.py` packs one subject's windows into the `.ssds` protobuf the app
imports (run `make proto` first). Each window mirrors an ESP sample: raw PPG/ACC, the raw
feature vector, the label in the score field, plus a fake sequence number and contiguous
8 s device-time grid, so imported windows preprocess and train exactly like streamed ones.
The dataset also carries the subject's raw 6-d demographics (`static`, recovered by
de-normalizing `static.npy`), which the app stamps onto the imported group as its
conditioning static.

```bash
uv run -m scripts.system.export_subject_data 1                              # S1.ssds, every window complete
uv run -m scripts.system.export_subject_data 1 --include-context           # also embed each window's raw context
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
  returned as stored, keyed by the active `GlobalWeights` row), accepts
  weight-delta uploads, persists them, enqueues worker jobs, and exposes a result endpoint the
  client polls. Fast to start since it never imports TF.
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
   `ClientDeltaSubmission` (tagged with the submitting `user_id`, the `base_weights_id` and its
   `version_id`), creates a `QuantizationJob` (`pending`), enqueues `quantize_submission`
   (the job id is set as the Celery task id), and returns `202` with the `job_id`.
2. The worker runs the job: malformedness fails it, but the aggregation-usability verdict
   (MSE gate) is cached silently on the submission and the artifact is produced either
   way — a Byzantine client never learns its update was filtered. The int8 `.tflite` is
   written (zstd-compressed) to the job row along with an ECDSA signature over the canonical
   model bytes (`ml/payload.py`, spec in `shared/docs/model-signing.md`; the signature covers
   the raw model, so it is signed before compression).
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
HTTP API: for each dataset subject (as user `test_N`) it logs in, pulls the global
trainable artifact, trains one pass through the on-device LiteRT `CompiledModel` runtime,
uploads the update, and logs out; then it queues a round, waits for the new `GlobalWeights`,
and scores it on the held-out subjects, repeating for `--rounds`. It picks the aggregation
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
   (every pairwise mask cancels), dequantizes to the FedAvg mean delta, adds it onto `W`,
   and bakes a new `GlobalWeights` + serving artifacts via the same path FedAvg uses. If any
   member never submitted, the **round fails wholesale** (masks only cancel over the full
   roster) — no partial recovery. The daily `federated_aggregation` beat skips `secure`
   models (it has no strategy for them); a secure round only runs when its task is queued.

**What secure aggregation costs.** The server never sees an individual update, so the
per-submission MSE gate, the z-scored outlier filter, and the per-client validity verdict
are all **structurally impossible** here — this is inherent to the scheme, not a gap. Only
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
- **Every served blob lives in PostgreSQL too**, each in its own table keyed by the row that
  owns it: `WeightsArtifact` (a snapshot's trainable/quantized `.tflite`, keyed by
  `GlobalWeights` + artifact), `FirmwareImage` (keyed by `Firmware`) and `QuantizationResult`
  (keyed by `QuantizationJob`). A handler locates a blob from the row it already loaded, and
  row existence *is* the presence check — a snapshot may legitimately have a trainable artifact
  and no quantized one. They are separate tables rather than columns because the owning rows are
  read constantly and the blobs almost never: aggregation loads `GlobalWeights` for its weights
  and `/ota/versions` lists `Firmware` rows, neither of which should drag a `.tflite` along.
  Blob columns are `STORAGE EXTERNAL` since zstd output is not worth a TOAST compression pass.
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
after retraining leaves the old snapshot in place. `--reset-weights` (or `make db-reseed`)
re-points each seeded model at the artifacts currently on disk: it drops the model's
`GlobalWeights` history along with everything anchored to it — the submissions, quantization
jobs and secure rounds that name a base snapshot they were computed against, and the stored
`WeightsArtifact` rows — then re-seeds from `shared/gen/models/<model>/`. Model definitions and
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

## Firmware distribution (OTA)

The `/ota` routes serve published firmware builds for the BLE OTA path (see
`shared/docs/versioning.md`, "Firmware distribution"). `GET /ota/versions/{interface}`
lists the builds published for a `BLE_INTERFACE_VERSION` (newest first, each with its
version string, supported model-contract list, image size and release date — an unknown
interface yields `[]`); `GET /ota/download/{interface}/{version}` streams the raw image
with the server's ECDSA signature over it in `X-Firmware-Signature`, which the app
forwards to the device for verification against its factory `srv_pub`.

Images live in `FirmwareImage` rows (zstd-compressed, keyed by the `Firmware` row), like the
model artifacts; the `Firmware` row itself keeps only the raw `size` and the mandatory
`signature`, so listing versions never reads an image.
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

