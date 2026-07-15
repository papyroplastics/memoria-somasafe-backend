from pathlib import Path

import tensorflow as tf

from ..layers import Conv1D, Dense, relu, upsample2
from ..preprocessing import N_COND, BVP_WINDOW
from .common import TrainableAutoencoder, AutoencoderTrainer, autoencoder_norm_params

class CNNAutoencoder(TrainableAutoencoder):
    """Conditional Conv1D autoencoder for reconstruction-based anomaly detection.

    Strided convolutions downsample the window, then a dense projection collapses the
    remaining time axis into a single ``latent_dim`` code per window; the decoder
    projects back, reshapes, and reconstructs with nearest-neighbour upsampling +
    convolutions. Collapsing time is the point: strided convs alone only shrink the
    time axis, leaving ``(seq_len / 8) * channels`` values — as many numbers as the
    input BVP, enough to copy it verbatim. No recurrence, so it quantizes cleanly for
    TFLM and avoids the LSTM's 1024-step unroll. The condition enters once, joined to
    the code at the bottleneck, so the decoder reconstructs from ``[z, cond]`` jointly.
    Encoder and decoder both see BVP only; ACC reaches the model solely as ``cond``'s
    activity context — as a raw encoder channel it measured as a no-op. ``seq_len`` must
    be divisible by ``2 ** 3``.

    What makes reconstruction error separate anomalies is how *tightly* the model fits
    the clean-BVP manifold, not how narrow the code is: a sharper fit makes off-manifold
    input (blown-up amplitude, elevated rate) miss by relatively more. Detection therefore
    improves with capacity up to ``latent_dim`` 256 and degrades past it, and starving the
    code hurts — at 16 it cannot reconstruct clean BVP either and detection collapses with
    it. The threshold is a quantile of each subject's own clean errors, so the absolute
    error scale cancels and only the clean/anomalous overlap matters. Bradycardia is
    undetectable here by construction: a slowed waveform is *easier* to reconstruct, so its
    error moves the wrong way."""

    def __init__(self, name: str, batch_size: int, seq_len: int,
                 signal_mean, signal_std, cond_mean, cond_std, n_signals: int = 1,
                 n_cond: int = N_COND, hidden_dim: int = 32, latent_dim: int = 256,
                 cond_embed_dim: int = 16, kernel_size: int = 7, n_outputs: int = 1,
                 diff_weight: float = 1.0, learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals, n_cond=n_cond, cond_embed_dim=cond_embed_dim,
                         n_outputs=n_outputs, diff_weight=diff_weight,
                         signal_mean=signal_mean, signal_std=signal_std,
                         cond_mean=cond_mean, cond_std=cond_std)

        self.enc1 = Conv1D(n_signals, hidden_dim, kernel_size, stride=2, activation=relu)
        self.enc2 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=relu)
        self.enc3 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=relu)

        self.enc_steps = seq_len // 8
        self.enc_channels = hidden_dim
        self.enc_flat = self.enc_steps * hidden_dim
        self.to_latent = Dense(self.enc_flat, latent_dim)

        # Concatenating [z, emb] and projecting once is W·[z;emb] = W_z·z + W_c·emb, so
        # the two projections below are that same layer — split to keep tf.concat out of
        # the train signature, whose gradient needs the Flex-only ConcatOffset.
        self.from_latent = Dense(latent_dim, self.enc_flat)
        self.from_cond = Dense(cond_embed_dim, self.enc_flat)

        self.dec1 = Conv1D(hidden_dim, hidden_dim, kernel_size, activation=relu)
        self.dec2 = Conv1D(hidden_dim, hidden_dim, kernel_size, activation=relu)
        self.dec3 = Conv1D(hidden_dim, n_outputs, kernel_size, activation=None)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal, cond):
        x = self.enc3(self.enc2(self.enc1(signal)))

        z = self.to_latent(tf.reshape(x, [-1, self.enc_flat]))
        z = z

        emb = self._embed_cond(cond)
        x = relu(self.from_latent(z) + self.from_cond(emb))
        x = tf.reshape(x, [-1, self.enc_steps, self.enc_channels])

        x = self.dec1(upsample2(x))
        x = self.dec2(upsample2(x))
        x = self.dec3(upsample2(x))
        return x


def get_trainer(data_root: Path, batch_size: int | None = None) -> AutoencoderTrainer:
    sig_mean, sig_std, cond_mean, cond_std = autoencoder_norm_params(data_root)
    model = CNNAutoencoder(
        name='dalia_cnn_ae', batch_size=batch_size or TrainableAutoencoder.default_batch_size,
        seq_len=BVP_WINDOW,
        signal_mean=sig_mean, signal_std=sig_std,
        cond_mean=cond_mean, cond_std=cond_std,
    )
    return AutoencoderTrainer(model)
