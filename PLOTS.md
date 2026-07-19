# PLOTS — result & figure catalog

Concrete index of every result/figure file the backend produces, the command that produces
it, and the report section it feeds. This is the per-file view; the *what/why* is in
`../report/planificacion/obtencion-de-resultados.md`, and the script layout and results
convention are in `README.md` ("Layout").

Conventions (see `README.md`, "Layout"):

- Evaluation/experiment output goes to **`results/<model>/…`** (`RESULTS_DIR`, gitignored).
  Served `.tflite` artifacts stay in `shared/gen/models/<model>/` and are **not** results.
- Every `<name>.png` has a companion **`<name>.yaml`** — a structured summary (what it
  shows, axes/units, subjects/splits, headline numbers, which report section it backs)
  written via `scripts/common/reports.py:write_yaml`, so the result can be read and
  cited without opening the image. Numeric results also carry a `.csv`. `anomaly_detection`
  and `knowledge_distillation` produce no plot, so their `.yaml` is the whole report
  (`personalization.csv` alongside the latter's, for the per-fold table).
- Convergence numbers for the report come from the **simulated** loops (reproducible, seeded
  from `scripts/__init__.py`), not the headless HTTP client — that is integration
  verification only.
- **The figure scripts split into two kinds.** `plot_convergence` *reads* a previous run
  (`results/<model>/<loop>/run.yaml` + `training.csv`) and fails if there is none — it never
  computes. `byzantine`, `sensitivity` and `knowledge_distillation` *do* train, because they
  sweep configurations no single `train.py` run produces. `calibrate_fpr` is its own case: it
  calibrates (a dense grid argmax over Youden's J) but does not train — see below.
- **Every loop holds out whole subjects** (`--eval-subjects`, default 2 = the last 2; also
  takes an id range `n-m`, a list `i,j,k`, or `none`). The resolved held-out ids are recorded
  in the manifest, so a metric is generalization to an *unseen subject*, and a centralized and
  a federated run at the same `--eval-subjects` train on the same data and score on the same
  subjects — which is what makes the Sec. 5.3 overlay a claim rather than a coincidence.
  `plot_convergence` refuses to draw the overlay if the two manifests' subject lists disagree.
- **The Sec. 5.4 scripts key off that same split.** `anomaly_detection` takes a **split**
  teacher: it picks the operating point on the training subjects (inline) and scores the
  detector on the held-out subjects, so its numbers are generalization to an unseen user
  like every other Chapter 5 metric. `calibrate_fpr` picks the same operating point the same
  way (on the training subjects) but then sweeps and plots the *whole* FPR curve on the
  **held-out** subjects instead — the calibration subjects' empirical FPR would track the
  expected FPR almost exactly by construction (each threshold is a quantile of that
  subject's own clean scores), so sweeping them wouldn't say anything about generalization.
  `knowledge_distillation` instead takes a teacher trained on **all** users (uniform
  per-subject label quality) and runs leave-one-subject-out at the student level, so every
  fold is leakage-free and none is special.

Run scripts from `backend/` with `uv run -m …`. The training runs and the sweeps are
compute-heavy; launch them in the background. This file is the per-file reference for
producing each result by hand, in dependency order (the integration harness needs the
services, api and worker up and a seeded DB).

---

## Chapter 4 — illustrative figures

| File(s) | Command | Section |
|---------|---------|---------|
| `results/<model>/signals.png` + `.yaml`<br>`results/<model>/signals_reconstructed.png` + `.yaml` | `scripts.figures.plot_signals <model> [--seed N]` | 4.1 — synthetic-anomaly windows + autoencoder reconstruction |

## Chapter 5 — validation & results

The 5.2 and 5.3 figures are **plotted from training runs, not by training**. Produce the two
runs once, then draw both figures from them:

```bash
uv run -m scripts.system.train <model> --loop federated --eval-subjects 2
uv run -m scripts.system.train <model> --loop normal    --eval-subjects 2
uv run -m scripts.figures.plot_convergence <model>      # both figures, no training
```

| File(s) | Command | Section / claim |
|---------|---------|-----------------|
| `results/<model>/<loop>/training.png` + `training.csv` + `run.yaml` (+ `reconstruction.png` for AEs, eval report) | `scripts.system.train <model> --loop {normal,federated} --eval-subjects K` | 5.2/5.3 (the underlying runs; the source both figures below read) |
| `results/<model>/federated/convergence.png` + `.csv` + `.yaml` | `scripts.figures.plot_convergence <model>` | **5.2** federated model improves round over round |
| `results/<model>/centralized_vs_federated/centralized_vs_federated.png` + `.csv` + `.yaml` | `scripts.figures.plot_convergence <model>` (same run; `--skip-overlay` to omit) | **5.3** FedAvg ≈ centralized without centralizing raw data (central claim) |
| `results/<model>/anomaly_detection.yaml` | `scripts.figures.anomaly_detection <model>` (split teacher) | **5.4** per-kind recall, clean FPR, detector accuracy·F1 — calibrated on training subjects, scored on the held-out eval subjects |
| `results/<model>/calibrate_fpr/calibration.png` + `.csv` + `.yaml` | `scripts.figures.calibrate_fpr <model>` (split teacher) | **5.4** recall/empirical FPR/Youden's J vs. expected FPR — the sweep backing "J not F1", with the selected operating point marked |
| `results/<model>/calibrate_fpr/roc.png` + `.yaml` | `scripts.figures.calibrate_fpr <model>` (same run) | **5.4** the detector's ROC curve (recall vs. empirical clean FPR) on the held-out subjects, with the selected operating point marked |
| `results/<model>/subject_roc/{roc_by_subject,roc_aggregate}.png` + `.yaml` | `scripts.figures.subject_roc <model> [--weights <all-users teacher>] [--highlight i,j]` | **5.4** per-subject ROC spread + mean±std recall — shows detectability varies by user (point `--weights` at an all-users teacher for equal footing) |
| `results/feature-mlp/personalization/personalization.csv` + `.yaml` | `scripts.figures.knowledge_distillation <ae> --student feature-mlp --weights <all-users teacher>` (trains) | **5.4/5.8** leave-one-subject-out personalization marginal-positive; int8 ≈ float |
| `results/<model>/byzantine/byzantine.png` + `.csv` + `.yaml` | `scripts.figures.byzantine <model> --max-malicious N --rounds R [--aggregator trimmed-mean\|average]` | **5.5** outlier filter holds the round (and no more) — trains |
| `results/footprint/footprint.csv` + `.yaml` | `scripts.figures.footprint` | **5.6** system fits the edge (backend rows; phone/ESP32 rows pasted in) |
| `results/<model>/sensitivity/{participants,local_epochs,loso}.png` + `.csv` + `.yaml` | `scripts.figures.sensitivity <model> [--sweep participants\|local-epochs\|loso\|all]` | **5.7** conclusions robust to configuration (LOSO mean ± std) — trains |

The two sweeps that train (`byzantine`, `sensitivity`) rebuild only the *model* per
configuration — fresh weights, so no run leaks into the next. Every subject's dataset is
built **once** per process and reused across configurations, since it never depends on the
weights (`ml.loading` caches them).

`byzantine` runs its own federated loop rather than `train.py`'s, since it has to append the
malicious clients' updates each round. `--aggregator` picks the rule applied to the
survivors: `trimmed-mean` (default, what the deployed server runs — `--trim` sets the
fraction dropped per side) or the plain `average`. Weighted averaging is not an option:
under this threat model an attacker just claims a huge dataset size, so it is unsound rather
than merely weak. Each aggregator is swept with the z-score filter on and off, which is the
figure's two lines.

## Knowledge distillation + personalization (Secs. 5.4/5.8)

`knowledge_distillation.py` is self-contained: it loads an autoencoder teacher trained on
**all** users, calibrates the expected FPR inline, distils a soft label per window in memory
(sigmoid of the error past the subject's own threshold, scaled by its own clean-error std),
and runs leave-one-subject-out personalization — per fold a fresh FeatureMLP student is
trained centrally on the *other* subjects' distilled labels, fine-tuned on the held-out
subject's own labels, and both are scored (float + int8) against that subject's **true**
labels. Nothing is written to disk but the pooled metrics.

```bash
uv run -m scripts.system.train <ae> --eval-subjects none                 # all-users teacher (overwrites trainable.tflite)
cp shared/gen/models/<ae>/trainable.tflite shared/gen/models/<ae>/trainable_all.tflite
uv run -m scripts.figures.knowledge_distillation <ae> --student feature-mlp \
    --weights shared/gen/models/<ae>/trainable_all.tflite
uv run -m scripts.system.train <ae> --eval-subjects 2                     # restore the canonical split teacher
```

Training the all-users teacher overwrites the canonical `trainable.tflite` (there is no
`--tag` any more), so copy it aside to `trainable_all.tflite` and retrain the split — the
convergence figures and `anomaly_detection`/`calibrate_fpr` want the split teacher back as the
canonical artifact. The scripts take the teacher by `--weights`, so its filename is free. The
distilled-vs-direct student comparison (does distillation reproduce the teacher?) is deferred
— it can be folded into this script later.

## Integration verification (methodology, Sec. 5.1 — not report figures)

Evidence the deployed path works; requires the full stack up (api, worker, redis, postgres)
and a seeded DB (`make db-seed`).

| Output | Command | Purpose |
|--------|---------|---------|
| `results/<model>/fed_client/convergence.{png,csv,yaml}` (dense) or `results/<model>/secure_fed_client/…` (secure) | `scripts.integration.fed_client --model <model> --rounds R --eval-subjects K` | drive the whole federated flow over the real HTTP API for every subject |
| — (asserts masked sum = plaintext mean) | `scripts.integration.secure_aggregation --clients N --rounds R` | secure-endpoint correctness probe |

## Footprint paste-in rows

`footprint.py` fills the backend/artifact-derived rows (param counts, float32 vs int8 sizes +
compression ratio, bytes/round). The remaining rows are measured off-device and pasted into
the report table (see `obtencion-de-resultados.md`): on-device training time per epoch
(phone), aggregation round wall-time (server), TFLM arena size + int8 inference latency
(ESP32), and detection quality retained after int8 as measured on-device.
