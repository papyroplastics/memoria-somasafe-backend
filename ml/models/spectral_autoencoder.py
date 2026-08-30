from pathlib import Path

import numpy as np

from ..layers import Dense, relu
from ..preprocessing import BVP_WINDOW
from ..sources.dalia import CLEAN
from ..spectral import DESCRIPTOR_NAMES, N_DESCRIPTORS
from .common import AutoencoderTrainer, DescriptorAutoencoder


class SpectralAutoencoder(DescriptorAutoencoder):
    """Dense autoencoder over the fixed spectral descriptor (``ml.spectral``) of an
    8-second BVP window rather than the waveform, so anomalies that merely smooth or
    slow the signal don't score below the threshold. Four dense layers."""

    def __init__(self, name: str, batch_size: int, hidden_dim: int = 32,
                 latent_dim: int = 6, learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, n_descriptors=N_DESCRIPTORS)

        self.enc_in = Dense(N_DESCRIPTORS, hidden_dim, activation=relu)
        self.to_latent = Dense(hidden_dim, latent_dim, activation=relu)
        self.from_latent = Dense(latent_dim, hidden_dim, activation=relu)
        self.dec_out = Dense(hidden_dim, N_DESCRIPTORS)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, descriptor):
        return self.dec_out(self.from_latent(self.to_latent(self.enc_in(descriptor))))


class SpectralAutoencoderTrainer(AutoencoderTrainer):
    """Feeds descriptors where AutoencoderTrainer feeds windows, on the non-overlapping
    grid the labels live on."""

    dataset_tensors = ['descriptor']

    def __init__(self, model: SpectralAutoencoder, data_root: Path):
        super().__init__(model, data_root, shift=BVP_WINDOW)
        self.model: SpectralAutoencoder = model  # type: ignore

    def subject_arrays(self, sid):
        return (self.data.descriptors(sid, CLEAN, BVP_WINDOW, self.shift),)

    def calibration_arrays(self):
        return self.calibration.calibration_descriptors(BVP_WINDOW, self.shift)

    def report(self, result_dir, eval_dataset):
        import matplotlib.pyplot as plt
        for batch in eval_dataset.take(1):
            target = np.asarray(batch[0])[0]
            recon = np.asarray(self.model.eval(*batch)['reconstruction'])[0]
            index = np.arange(len(DESCRIPTOR_NAMES))
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(index - 0.2, target, 0.4, label='descriptor')
            ax.bar(index + 0.2, recon, 0.4, label='reconstruction')
            ax.set_xticks(index)
            ax.set_xticklabels(DESCRIPTOR_NAMES, rotation=45, ha='right')
            ax.set_ylabel('z-score (per subject)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(result_dir / 'reconstruction.png')
            plt.close(fig)
            print(f"saved reconstruction plot to {result_dir / 'reconstruction.png'}")
            break


def get_model(data_root: Path, batch_size: int | None = None) -> SpectralAutoencoder:
    return SpectralAutoencoder(
        name='dalia_spectral_ae',
        batch_size=batch_size or SpectralAutoencoder.default_batch_size,
    )


def get_trainer(data_root: Path, batch_size: int | None = None) -> AutoencoderTrainer:
    return SpectralAutoencoderTrainer(get_model(data_root, batch_size), data_root)
