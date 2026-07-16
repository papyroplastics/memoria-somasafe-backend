"""Pure protocol tests for common.secure_agg — no server, no TensorFlow.

They pin the two properties the whole scheme rests on: the pairwise masks cancel
exactly across the full cohort, and what survives is the plaintext average up
to quantization error. The dropout case documents *why* a missing member fails the
round rather than degrading gracefully.
"""

import numpy as np
import pytest

from common.secure_agg import (
    compute_scale,
    dequantize,
    generate_keypair,
    mask_vector,
    quantize,
    ring_sum,
)


def _cohort(n, m, clip_bound=1.0, seed=0):
    rng = np.random.default_rng(seed)
    keys = [generate_keypair() for _ in range(n)]              # (sk, pk)
    roster = [(uid, pk) for uid, (_, pk) in enumerate(keys)]
    deltas = [rng.uniform(-clip_bound, clip_bound, m).astype(np.float32)
              for _ in range(n)]
    scale = compute_scale(n, clip_bound)
    return keys, roster, deltas, scale


def test_masks_cancel_and_recover_mean():
    n, m, B = 5, 200, 1.0
    keys, roster, deltas, scale = _cohort(n, m, B, seed=1)

    masked, quantized = [], []
    for uid, (sk, _) in enumerate(keys):
        q = quantize(deltas[uid], B, scale)
        quantized.append(q)
        masked.append(mask_vector(q, uid, roster, sk, round_id=7))

    assert np.array_equal(ring_sum(masked), ring_sum(quantized))

    recovered = dequantize(ring_sum(masked), scale, n)
    plaintext_mean = np.mean(np.stack(deltas), axis=0)
    # quantization error (2/scale) plus the float32 round-trip both outputs carry
    assert np.max(np.abs(recovered - plaintext_mean)) < 2.0 / scale + 1e-6


def test_fresh_round_gives_independent_masks():
    n, m, B = 3, 64, 1.0
    keys, roster, deltas, scale = _cohort(n, m, B, seed=2)
    y_a = mask_vector(quantize(deltas[0], B, scale), 0, roster, keys[0][0], round_id=1)
    y_b = mask_vector(quantize(deltas[0], B, scale), 0, roster, keys[0][0], round_id=2)
    assert not np.array_equal(y_a, y_b)


def test_missing_member_corrupts_the_sum():
    n, m, B = 4, 64, 1.0
    keys, roster, deltas, scale = _cohort(n, m, B, seed=3)
    masked = [mask_vector(quantize(deltas[uid], B, scale), uid, roster, sk, round_id=9)
              for uid, (sk, _) in enumerate(keys)]

    partial = dequantize(ring_sum(masked[:-1]), scale, n - 1)
    survivors_mean = np.mean(np.stack(deltas[:-1]), axis=0)
    assert np.max(np.abs(partial - survivors_mean)) > 1.0


def test_clipping_bounds_influence():
    n, m, B = 3, 32, 0.5
    scale = compute_scale(n, B)
    huge = np.full(m, 100.0, dtype=np.float32)   # a client trying to dominate
    q = quantize(huge, B, scale)
    recovered = dequantize(ring_sum([q]), scale, 1)
    # Clipped to +/-B before quantization, so its per-coordinate contribution is capped.
    assert np.max(recovered) == pytest.approx(B, abs=2.0 / scale)
