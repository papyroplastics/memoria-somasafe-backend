#!/usr/bin/env bash
#
# Produce every backend result the report cites, from a clean slate: no generated
# .tflite artifacts and an empty database. Assumes postgres/redis, the api and the
# worker are already up (make db-run / make api-run / make worker-run) and that .env exists.
#
#   ./run_all.sh                 # run everything, skipping steps already done
#   FORCE=1 ./run_all.sh         # redo every step
#   ROUNDS=5 ./run_all.sh        # shorter run (see the tunables below)
#
# Resumable: each step is skipped when its output is already there, so a run that dies
# hours in continues instead of starting over. Nothing here is fast — the training steps
# and the two sweeps dominate. Expect hours.
#
# See PLOTS.md for what each output is and which report section it feeds.

set -euo pipefail
cd "$(dirname "$0")"

# --- tunables ---------------------------------------------------------------------
ROUNDS=${ROUNDS:-15}                       # federated global rounds
EPOCHS=${EPOCHS:-$ROUNDS}                  # centralized epochs; matches ROUNDS so the
                                           # Sec. 5.3 curves are the same length
LOCAL_EPOCHS=${LOCAL_EPOCHS:-2}            # local passes per federated round
EVAL_SUBJECTS=${EVAL_SUBJECTS:-2}          # subjects held out whole, in every run
SWEEP_ROUNDS=${SWEEP_ROUNDS:-10}           # rounds inside the byzantine/sensitivity sweeps
MAX_MALICIOUS=${MAX_MALICIOUS:-4}
FED_CLIENT_ROUNDS=${FED_CLIENT_ROUNDS:-5}  # rounds over the real HTTP API
SECURE_CLIENTS=${SECURE_CLIENTS:-4}
SECURE_ROUNDS=${SECURE_ROUNDS:-3}

AE=${AE:-cnn-ae}                           # focus autoencoder / teacher / secure model
STUDENT=${STUDENT:-feature-mlp}            # distillation student / quantize model
MODELS=${MODELS:-"$AE $STUDENT"}           # models trained and figured

RESULTS=${RESULTS_DIR:-results}
GEN_MODELS=${MODELS_DIR:-shared/gen/models}
DATASETS=${DATASETS_DIR:-shared/gen/datasets}

FORCE=${FORCE:-}
run=(uv run -m)

# --- step harness -----------------------------------------------------------------
step_n=0

# step <name> <output-marker> <command...>
# Skips when the marker exists, unless FORCE is set. An empty marker always runs.
step() {
    local name=$1 marker=$2
    shift 2
    step_n=$((step_n + 1))
    if [ -z "$FORCE" ] && [ -n "$marker" ] && [ -e "$marker" ]; then
        printf '\n[%02d] == skip: %s\n     %s exists (FORCE=1 to redo)\n' "$step_n" "$name" "$marker"
        return
    fi
    printf '\n[%02d] == %s\n     $ %s\n' "$step_n" "$name" "$*"
    "$@"
}

# --- preflight --------------------------------------------------------------------
[ -f .env ] || { echo "no .env — copy example.env and adjust (the api/worker need it)"; exit 1; }

echo "models: $MODELS   teacher: $AE   student: $STUDENT"
echo "rounds: $ROUNDS   epochs: $EPOCHS   local epochs: $LOCAL_EPOCHS   eval subjects: $EVAL_SUBJECTS"

# --- 1. dataset -------------------------------------------------------------------
step "shared submodule" shared make shared
step "download + preprocess PPG-DaLiA" "$DATASETS/clean-features" "${run[@]}" scripts.system.get_dataset

# --- 2. training ------------------------------------------------------------------
# Both loops export to the same shared/gen/models/<model>/trainable.tflite, so the
# federated run goes last: the artifact the DB is seeded from is the federated one, and
# the deployed path continues FedAvg from there.
for model in $MODELS; do
    step "train $model (centralized)" "$RESULTS/$model/normal/run.yaml" \
        "${run[@]}" scripts.system.train "$model" --loop normal \
            --epochs "$EPOCHS" --eval-subjects "$EVAL_SUBJECTS"
    step "train $model (federated)" "$RESULTS/$model/federated/run.yaml" \
        "${run[@]}" scripts.system.train "$model" --loop federated \
            --rounds "$ROUNDS" --local-epochs "$LOCAL_EPOCHS" --eval-subjects "$EVAL_SUBJECTS"
done

# --- 3. figures that only read the runs above (Secs. 5.2 + 5.3) --------------------
for model in $MODELS; do
    step "plot convergence + overlay for $model" \
        "$RESULTS/$model/centralized_vs_federated/centralized_vs_federated.png" \
        "${run[@]}" scripts.figures.plot_convergence "$model"
done

# --- 4. figures that train (Secs. 5.5 + 5.7) --------------------------------------
for model in $MODELS; do
    step "byzantine sweep for $model" "$RESULTS/$model/byzantine/byzantine.png" \
        "${run[@]}" scripts.figures.byzantine "$model" \
            --max-malicious "$MAX_MALICIOUS" --rounds "$SWEEP_ROUNDS" \
            --local-epochs "$LOCAL_EPOCHS" --eval-subjects "$EVAL_SUBJECTS"
    step "sensitivity sweeps for $model" "$RESULTS/$model/sensitivity/loso.png" \
        "${run[@]}" scripts.figures.sensitivity "$model" --sweep all \
            --rounds "$SWEEP_ROUNDS" --local-epochs "$LOCAL_EPOCHS" \
            --eval-subjects "$EVAL_SUBJECTS"
done

# --- 5. illustrative signal figures (Sec. 4.1) ------------------------------------
step "signal + reconstruction figures" "$RESULTS/$AE/signals_reconstructed.png" \
    "${run[@]}" scripts.figures.plot_signals "$AE" --seed 0

# --- 6. distillation round-trip (Secs. 5.4 + 5.8) ---------------------------------
# The student is trained a second time here, on the teacher's labels instead of the
# synthetic ground truth. It is --tag'd so it lands beside the direct-label run rather
# than on top of it: the canonical trainable.tflite stays the direct-label student (the
# one the DB serves), and the distilled variant is what personalize_test fine-tunes.
step "calibrate detector budgets ($AE)" "$RESULTS/$AE/distill_calibration.json" \
    "${run[@]}" scripts.distillation.distill_calibrate "$AE"
step "evaluate detector vs ground truth ($AE)" "$RESULTS/$AE/distill_eval.json" \
    "${run[@]}" scripts.distillation.distill_eval "$AE"
step "distill labels from $AE" "$RESULTS/$AE/distilled-labels" \
    "${run[@]}" scripts.distillation.distill_labels "$AE"
step "train $STUDENT on the distilled labels" "$RESULTS/$STUDENT/normal-distilled/run.yaml" \
    "${run[@]}" scripts.system.train "$STUDENT" --loop normal \
        --dataset-dir "$RESULTS/$AE/distilled-labels" --tag distilled \
        --epochs "$EPOCHS" --eval-subjects "$EVAL_SUBJECTS"
step "personalization probe ($STUDENT from $AE)" \
    "$RESULTS/$STUDENT/personalization/personalization.csv" \
    "${run[@]}" scripts.distillation.personalize_test --model "$STUDENT" --teacher "$AE" \
        --weights "$GEN_MODELS/$STUDENT/trainable_distilled.tflite"

# --- 7. footprint table (Sec. 5.6) ------------------------------------------------
# After every artifact exists, so the size columns are populated.
step "footprint table" "$RESULTS/footprint/footprint.csv" "${run[@]}" scripts.figures.footprint

# --- 8. seed the database ---------------------------------------------------------
# Always runs: idempotent, and --reset-weights re-points each model at the artifacts
# just trained (weights are seeded once and then owned by aggregation, so without it a
# re-run would keep serving the old snapshot).
step "seed database + object store" "" \
    "${run[@]}" scripts.system.seed_db --assign-device --test-users --reset-weights

# --- 9. integration verification over the real HTTP API (Sec. 5.1) ----------------
# Needs the api + worker up. These mutate GlobalWeights (each round bakes a new snapshot),
# so they come after everything that reads the seeded weights. Like the seed, they carry no
# skip marker on purpose: the step above just reset the weight history they build on, so a
# resumed run has to replay them to leave the database in a coherent state.
for model in $MODELS; do
    step "headless federated client ($model)" "" \
        "${run[@]}" scripts.integration.fed_client --model "$model" \
            --rounds "$FED_CLIENT_ROUNDS" --eval-subjects "$EVAL_SUBJECTS"
done
step "secure-aggregation correctness probe" "" \
    "${run[@]}" scripts.integration.secure_aggregation --model "$AE" \
        --clients "$SECURE_CLIENTS" --rounds "$SECURE_ROUNDS"
step "queue one aggregation round per model" "" "${run[@]}" scripts.integration.queue_aggregation

printf '\n== done. results under %s/ — see PLOTS.md for what feeds which section.\n' "$RESULTS"
