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
  cited without opening the image. Numeric results also carry `.csv`/`.json`.
- Convergence numbers for the report come from the **simulated** loops (reproducible, seeded
  from `scripts/__init__.py`), not the headless HTTP client — that is integration
  verification only.
- **The figure scripts split into two kinds.** `plot_convergence` and `plot_calibration`
  *read* a previous run (`results/<model>/<loop>/run.yaml` + `training.csv`, and
  `calibration.json` respectively) and fail if there is none — they never compute.
  `byzantine`, `sensitivity` and `knowledge_distillation` *do* train, because they sweep
  configurations no single `train.py` run produces.
- **Every loop holds out whole subjects** (`--eval-subjects N`, default 2: the last N).
  A metric is therefore generalization to an *unseen subject*, and a centralized and a
  federated run at the same `--eval-subjects` train on the same data and score on the same
  subjects — which is what makes the Sec. 5.3 overlay a claim rather than a coincidence.
  `plot_convergence` refuses to draw the overlay if the two manifests disagree.
- **The Sec. 5.4 scripts key off that same split.** `anomaly_detection` takes a **split**
  teacher: it picks the operating point on the training subjects (inline) and scores the
  detector on the held-out subjects, so its numbers are generalization to an unseen user
  like every other Chapter 5 metric. `calibrate_fpr` calibrates on the same training
  subjects but only dumps the full FPR sweep for the report table. `knowledge_distillation`
  instead takes a teacher trained on **all** users (uniform per-subject label quality) and
  runs leave-one-subject-out at the student level, so every fold is leakage-free and none is
  special.

Run scripts from `backend/` with `uv run -m …`. The training runs and the sweeps are
compute-heavy; launch them in the background.

**To produce everything at once**, `./run_all.sh` (or `make run-all`) sequences every step
below in dependency order, from an empty database and no exported artifacts — it assumes the
services, api and worker are already up. It is resumable (each step is skipped when its
output exists; `FORCE=1` redoes them) and the scale is tunable via the env vars at the top
(`ROUNDS`, `EPOCHS`, `EVAL_SUBJECTS`, `MODELS`, …). The rest of this file is the per-file
reference for running a piece of it by hand.

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
| `results/<model>/anomaly_detection.json` | `scripts.figures.anomaly_detection <model>` (split teacher) | **5.4** per-kind recall, clean FPR, detector vs. spectral baseline / accuracy·F1 — calibrated on training subjects, scored on the held-out eval subjects |
| `results/<model>/calibration.json` | `scripts.figures.calibrate_fpr <model>` (split teacher) | **5.4** the full expected-FPR sweep (table backing "J not F1"); standalone, feeds only the figure below |
| `results/<model>/calibration.png` + `.csv` + `.yaml` | `scripts.figures.plot_calibration <model>` (reads a previous `calibrate_fpr`; never calibrates) | **5.4** recall/empirical FPR vs. expected FPR with the selected operating point — makes the chosen expected FPR auditable |
| `results/feature-mlp/personalization/personalization.csv` + `.json` | `scripts.figures.knowledge_distillation <ae> --student feature-mlp --weights <all-users teacher>` (trains) | **5.4/5.8** leave-one-subject-out personalization marginal-positive; int8 ≈ float |
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
uv run -m scripts.system.train <ae> --tag all --eval-subjects 0          # all-users teacher
uv run -m scripts.figures.knowledge_distillation <ae> --student feature-mlp \
    --weights shared/gen/models/<ae>/trainable_all.tflite
```

The teacher needs `--tag all` so its `trainable_all.tflite` lands beside — not on top of —
the split teacher the convergence figures and `anomaly_detection`/`calibrate_fpr` use. The
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
