"""Train the conditional LSTM autoencoder on PPG-DaLiA via simulated FedAvg.

Pipeline:
  1. (optional) preprocess raw PPG_FieldStudy pickles into per-subject npy caches
  2. build per-client train/eval datasets (one client per subject)
  3. federated-average a ConditionalLSTMAutoencoder across clients
  4. export trainable (`-odt`) and int8 (`-opti`) `.tflite` artifacts
  5. (optional) plot an input window against its reconstruction

Run `get-dalia.py` first so the processed dataset exists under
`datasets/ppg-dalia-processed/`.
Per project policy this script is run manually; it is not part of any test suite.
"""

import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

from worker.dataset import build_subject_dataset, WINDOW_SIZE
from worker.model import ConditionalLSTMAutoencoder
from worker.training import federated_train_eval_loop, reconstruction_eval
from worker.saving import save_odt, save_opti

tf.random.set_seed(1234)

# --- Configuration ---------------------------------------------------------------
# Produced by get-dalia.py; run that first to download + preprocess the dataset.
DATA_DIR = pathlib.Path('datasets') / 'ppg-dalia-processed'
OUTPUT_DIR = pathlib.Path('models')

TRAIN_SPLIT = 0.8

BATCH_SIZE = 32
LOCAL_EPOCHS = 3
GLOBAL_EPOCHS = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def available_subjects() -> list[int]:
    return sorted(
        int(p.name[1:]) for p in DATA_DIR.glob('S*') if p.name[1:].isdigit()
    )


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"Processed dataset not found at {DATA_DIR}. "
            f"Run `uv run python get-dalia.py` first.")

    subjects = available_subjects()
    if not subjects:
        raise SystemExit(
            f"No processed subjects under {DATA_DIR}. "
            f"Run `uv run python get-dalia.py` first.")
    print(f"Using subjects: {subjects}")

    client_train, client_eval = [], []
    for sid in subjects:
        try:
            ds = build_subject_dataset(DATA_DIR, sid)
            n_train = int(len(ds) * TRAIN_SPLIT)
            client_train.append(
                ds.take(n_train).shuffle(n_train).batch(BATCH_SIZE, drop_remainder=True))
            client_eval.append(
                ds.skip(n_train).batch(BATCH_SIZE, drop_remainder=True))
        except (FileNotFoundError, ValueError) as e:
            print(f"   Skipping S{sid}: {e}")

    model = ConditionalLSTMAutoencoder(
        name='dalia_lstm_ae',
        batch_size=BATCH_SIZE,
        seq_len=WINDOW_SIZE,
        n_signals=2,
        n_static=6,
        n_context=2,
        hidden_dim=64,
        latent_dim=32,
        cond_embed_dim=16,
        learning_rate=1e-3,
    )
    print(f"Model parameters: {model.total_parameter_size}")

    federated_train_eval_loop(
        model=model,
        client_train_datasets=client_train,
        client_eval_datasets=client_eval,
        local_epochs=LOCAL_EPOCHS,
        global_epochs=GLOBAL_EPOCHS,
    )

    # Export the trainable model for on-device LiteRT fine-tuning.
    saved_model = save_odt(OUTPUT_DIR, 'pre-train', model)
    print(f"Final global eval loss: "
          f"{tf.reduce_mean([reconstruction_eval(model, ds) for ds in client_eval]):.6f}")

    # Export the int8-quantized model. LSTM int8 conversion has limited op
    # support, so don't let a failure here block the trainable artifact.
    try:
        rep_dataset = client_eval[0].map(
            lambda s, c, st: {'signal': s, 'context': c, 'static': st})
        save_opti(OUTPUT_DIR, 'pre-train', model, rep_dataset)
    except Exception as e:  # noqa: BLE001 - conversion errors are informational here
        print(f"   Quantized export skipped (LSTM int8 unsupported?): {e}")

    # Plot an input window vs. its reconstruction for a quick sanity check.
    for signal, context, static in client_eval[0].take(1):
        recon = saved_model.eval(signal, context, static)['reconstruction']
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(signal[0].numpy())
        axs[0].set_title('Input window [BVP, ACC]')
        axs[1].plot(recon[0].numpy())
        axs[1].set_title('Reconstruction')
        fig.savefig(OUTPUT_DIR / 'reconstruction.png')
        print(f"Saved reconstruction plot to {OUTPUT_DIR / 'reconstruction.png'}")
        break


if __name__ == '__main__':
    main()
