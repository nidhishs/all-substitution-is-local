"""Synthetic regime: 3 binary-state binary-action configs for Experiment 3.

Each config tests a different geometric regime predicted by Theorem 2:
    A_balanced       : prior at threshold, strong signal -> BR_hat > Margin under R1
    A_low_signal     : prior at threshold, weak signal   -> all policies converge low
    A_off_threshold  : prior far from threshold          -> ~100% zero-BR under R2

Channels (M and H) are (S, K) matrices with M[s, k] = P(signal=s | state=k).
For binary K=2 with diagonal-strength theta, M = [[theta, 1-theta], [1-theta, theta]].
"""

from __future__ import annotations

import numpy as np

import core


def _diag_channel(theta: float) -> np.ndarray:
    return np.array([[theta, 1.0 - theta], [1.0 - theta, theta]], dtype=np.float64)


_CONFIGS: dict[str, dict] = {
    "A_balanced": {
        "prior": np.array([0.5, 0.5]),
        "M": _diag_channel(0.85),
        "H": _diag_channel(0.75),
    },
    "A_low_signal": {
        "prior": np.array([0.5, 0.5]),
        "M": _diag_channel(0.70),
        "H": _diag_channel(0.60),
    },
    "A_off_threshold": {
        "prior": np.array([0.8, 0.2]),
        "M": _diag_channel(0.85),
        "H": _diag_channel(0.75),
    },
}


def config_names() -> tuple[str, ...]:
    return tuple(_CONFIGS.keys())


def generate_synthetic_pair(
    name: str, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sample (b_x, b_xh, h, y, meta) for a synthetic config.

    b_x[i] is the model's posterior given the realized model signal m_i.
    b_xh[i] is the joint posterior given (m_i, h_i). h is integer in {0, 1}.
    """
    if name not in _CONFIGS:
        raise ValueError(f"Unknown config {name}; choices: {list(_CONFIGS)}")
    cfg = _CONFIGS[name]
    prior, M, H = cfg["prior"], cfg["M"], cfg["H"]

    y, m, h = core.generate_samples(prior, M, H, n, rng)

    b_x = np.empty((n, 2))
    b_xh = np.empty((n, 2))
    for i in range(n):
        b_x[i] = core.posterior(prior, M[m[i]])
        b_xh[i] = core.joint_posterior(prior, M[m[i]], H[h[i]])

    meta = {
        "dataset": "synthetic",
        "model": "exact_channels",
        "task": name,
        "annotator_id": "synthetic_h",
        "n_classes": 2,
    }
    return b_x, b_xh, h.astype(int), y.astype(int), meta
