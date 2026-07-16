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
  `distill_calibration.json` respectively) and fail if there is none — they never compute.
  `byzantine` and `sensitivity` *do* train, because they sweep configurations no single
  `train.py` run produces.
- **Every loop holds out whole subjects** (`--eval-subjects N`, default 2: the last N).
  A metric is therefore generalization to an *unseen subject*, and a centralized and a
  federated run at the same `--eval-subjects` train on the same data and score on the same
  subjects — which is what makes the Sec. 5.3 overlay a claim rather than a coincidence.
  `plot_convergence` refuses to draw the overlay if the two manifests disagree.

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
| `results/<model>/distill_eval.json` | `scripts.distillation.distill_eval <model>` | **5.4** per-kind recall, clean FPR, detector vs. spectral baseline / accuracy·F1 |
| `results/<model>/calibration.png` + `.csv` + `.yaml` | `scripts.figures.plot_calibration <model>` (reads a previous `distill_calibrate`; never calibrates) | **5.4** recall/FPR vs. budget with the selected operating point — makes the chosen budget auditable |
| `results/<model>/personalization/personalization[_S*].csv` + `.json` | `scripts.distillation.personalize_test --model feature-mlp --teacher <ae>` | **5.4** personalization marginal-positive; int8 ≈ float |
| `results/<model>/byzantine/byzantine.png` + `.csv` + `.yaml` | `scripts.figures.byzantine <model> --max-malicious N --rounds R` | **5.5** outlier filter holds the round (and no more) — trains |
| `results/footprint/footprint.csv` + `.yaml` | `scripts.figures.footprint` | **5.6** system fits the edge (backend rows; phone/ESP32 rows pasted in) |
| `results/<model>/sensitivity/{participants,local_epochs,loso}.png` + `.csv` + `.yaml` | `scripts.figures.sensitivity <model> [--sweep participants\|local-epochs\|loso\|all]` | **5.7** conclusions robust to configuration (LOSO mean ± std) — trains |

The two sweeps that train (`byzantine`, `sensitivity`) rebuild only the *model* per
configuration — fresh weights, so no run leaks into the next. Every subject's dataset is
built **once** per process and reused across configurations, since it never depends on the
weights (`ml.loading` caches them).

## Distillation round-trip (Sec. 5.8) and its inputs

The unsupervised-teacher → student pipeline, end to end:

```bash
uv run -m scripts.distillation.distill_calibrate <ae>   # budget   -> results/<ae>/distill_calibration.json
uv run -m scripts.distillation.distill_eval      <ae>   # metrics  -> results/<ae>/distill_eval.json
uv run -m scripts.distillation.distill_labels    <ae>   # teacher  -> results/<ae>/distilled-labels/
uv run -m scripts.system.train feature-mlp \
    --dataset-dir results/<ae>/distilled-labels \
    --tag distilled                                     # student on the distilled labels
```

Compare the distilled student against the same student trained on the direct synthetic
labels; `distill_eval` reports the detector's metrics against ground truth. Both students are
the same model key, so the distilled run needs **`--tag distilled`**: without it, it
overwrites the direct-label run's `results/feature-mlp/normal/` and its exported artifacts.
Tagged, it lands in `results/feature-mlp/normal-distilled/` and exports
`trainable_distilled.tflite`, leaving the canonical (direct-label) artifact — the one
`seed_db` publishes and the system serves — untouched. `personalize_test` fine-tunes the
distilled student, so point it at that artifact:

```bash
uv run -m scripts.distillation.personalize_test --model feature-mlp --teacher <ae> \
    --weights shared/gen/models/feature-mlp/trainable_distilled.tflite
```

## Integration verification (methodology, Sec. 5.1 — not report figures)

Evidence the deployed path works; requires the full stack up (api, worker, redis, postgres)
and a seeded DB (`make db-seed`).

| Output | Command | Purpose |
|--------|---------|---------|
| `results/<model>/fed_client/convergence.{png,csv,yaml}` (dense) or `results/<model>/secure_fed_client/…` (secure) | `scripts.integration.fed_client --model <model> --rounds R --eval-subjects K` | drive the whole federated flow over the real HTTP API for every subject |
| — (asserts masked sum = plaintext mean) | `scripts.integration.secure_aggregation --clients N --rounds R` | secure-endpoint correctness probe |
| — (prints per-model round summary) | `scripts.integration.queue_aggregation [<model>]` | queue a FedAvg round by hand for testing |

## Footprint paste-in rows

`footprint.py` fills the backend/artifact-derived rows (param counts, float32 vs int8 sizes +
compression ratio, bytes/round). The remaining rows are measured off-device and pasted into
the report table (see `obtencion-de-resultados.md`): on-device training time per epoch
(phone), aggregation round wall-time (server), TFLM arena size + int8 inference latency
(ESP32), and detection quality retained after int8 as measured on-device.
