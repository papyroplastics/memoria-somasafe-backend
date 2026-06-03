import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

from worker.dataset import build_subject_dataset
from worker.models import ConditionalLSTMAutoencoder
from worker.training import autoencoder_train_loop
from worker.saving import save_odt, save_opti

tf.random.set_seed(1234)

data_dir = pathlib.Path('datasets') / 'ppg-dalia-processed'
outer_result_dir = pathlib.Path('results')
inner_result_dir = outer_result_dir / 'lstm'
inner_result_dir.mkdir(parents=True, exist_ok=True)

if not data_dir.is_dir():
    raise SystemExit(f"Processed dataset not found at {data_dir}. ")

sample_rate = 64 # hz
window_len = 8  # seconds
window_size = sample_rate * window_len # samples
shift_len = 3  # seconds
shift = sample_rate * shift_len # samples

train_split = 0.975
batch_size = 12
num_slices = 10
num_passes = 1

subject_train_datasets, subject_eval_datasets = [], []
for dir in data_dir.glob('S*'):
    ds = build_subject_dataset(dir, window_size, shift)
    ds = ds.shuffle(len(ds)).batch(batch_size, drop_remainder=True)

    train_count = int(len(ds) * train_split)
    subject_train_datasets.append(ds.take(train_count))
    subject_eval_datasets.append(ds.skip(train_count))

    print(f"Processed {dir.name}")

eval_dataset = tf.data.Dataset.sample_from_datasets(subject_eval_datasets)
del subject_eval_datasets

model = ConditionalLSTMAutoencoder(
    name='dalia_lstm_ae',
    batch_size=batch_size,
    seq_len=window_size,
    n_signals=2,
    n_static=6,
    n_context=2,
    hidden_dim=64,
    latent_dim=32,
    cond_embed_dim=16,
    learning_rate=1e-3,
)

autoencoder_train_loop(
    model=model,
    subject_train_datasets=subject_train_datasets,
    eval_dataset=eval_dataset,
    num_slices=num_slices,
    num_passes=num_passes,
)

print(f"Compiling and saving model")
saved_model, sm_path = save_odt(inner_result_dir, 'pre-train', model)
print(f"Saved model to {sm_path}")

print("Quantizing model")
try:
    rep_dataset = eval_dataset.map(lambda s, c, st: {'signal': s, 'context': c, 'static': st})
    save_opti(inner_result_dir, 'pre-train', model, rep_dataset)
except Exception as e:  # noqa: BLE001 - conversion errors are informational here
    print(f"   Quantized export skipped (LSTM int8 unsupported?): {e}")

# Plot an input window vs. its reconstruction for a quick sanity check.
for signal, context, static in eval_dataset.take(1):
    recon = saved_model.eval(signal, context, static)['reconstruction']
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(signal[0].numpy())
    axs[0].set_title('Input window [BVP, ACC]')
    axs[1].plot(recon[0].numpy())
    axs[1].set_title('Reconstruction')
    fig.savefig(inner_result_dir / 'reconstruction.png')
    print(f"saved reconstruction plot to {inner_result_dir / 'reconstruction.png'}")
    break

