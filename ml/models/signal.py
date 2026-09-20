from pathlib import Path

import tensorflow as tf

from ..metrics import mse_loss, first_difference_loss
from ..sources.dalia import TRAIN_SHIFT, DaliaSignalSource
from .common import ReconstructionTrainer, TrainableAutoencoder


class SignalAutoencoder(TrainableAutoencoder):
    """The waveform variants (LSTM/GRU/CNN): reconstruct the BVP window itself, with a
    first-difference (slope) term alongside the MSE that penalizes a flat-line output."""

    def __init__(self, name: str, batch_size: int | None, seq_len: int, n_signals: int = 1,
                 n_outputs: int = 1, diff_weight: float = 1.0):
        super().__init__(name=name, batch_size=batch_size,
                         input_shape=(seq_len, n_signals))
        self.seq_len = seq_len
        self.n_signals = n_signals
        self.n_outputs = n_outputs
        self.diff_weight = diff_weight

    def _target(self, signal):
        return signal[:, :, :self.n_outputs]

    def _loss(self, reconstruction, target):
        return (mse_loss(reconstruction, target)
                + self.diff_weight * first_difference_loss(reconstruction, target))

    def eval_eager(self, signal: tf.Tensor):
        return self._eval_core(signal)

    def train_eager(self, signal: tf.Tensor):
        return self._train_core(signal)


class AutoencoderTrainer(ReconstructionTrainer):

    dataset_tensors = ['signal']
    training_key = 'ppg-dalia-low-signal-clean'
    calibration_key = 'ppg-dalia-signal-clean'

    def __init__(self, model: SignalAutoencoder, data_root: Path):
        super().__init__(model, data_root)
        self.model: SignalAutoencoder = model # type: ignore
        assert isinstance(self.data, DaliaSignalSource)
        self.data = self.data.with_grid(window=self.model.seq_len, shift=TRAIN_SHIFT)

    def report(self, result_dir, eval_dataset):
        import matplotlib.pyplot as plt
        for batch in eval_dataset.take(1):
            recon = self.model.eval(*batch)['reconstruction']
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(batch[0][0].numpy())
            axs[0].set_title('Input window [BVP]')
            axs[1].plot(recon[0].numpy())
            axs[1].set_title('Reconstruction [BVP]')
            fig.savefig(result_dir / 'reconstruction.png')
            print(f"saved reconstruction plot to {result_dir / 'reconstruction.png'}")
            break
