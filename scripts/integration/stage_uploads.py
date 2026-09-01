"""Stage 13 plaintext deltas for a model without aggregating them, so that a single
aggregation round can be timed on its own (report Sec. 5.5, tab:costo-borde)."""

import sys

import numpy as np

from ml.model_list import MODELS
from scripts.common.api import (DEFAULT_BASE_URL, download_weights, login, logout,
                                submit_delta)

key = sys.argv[1] if len(sys.argv) > 1 else "feature-ae"
n_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 13

spec = MODELS[key]
base = DEFAULT_BASE_URL
rng = np.random.default_rng(1234)

for i in range(1, n_clients + 1):
    user = f"test_{i}"
    token = login(base, user, user)
    raw, weights_id = download_weights(base, token, key)
    n = len(np.frombuffer(raw, dtype=np.float32))
    delta = (rng.standard_normal(n) * 1e-3).astype(np.float32)
    submit_delta(base, token, key, weights_id, delta.tobytes(), spec.submission_type)
    logout(base, token)
    print(f"staged {i}/{n_clients} ({n} weights)")
