"""Shared Bayesian decision-theory utilities for all experiments."""
from __future__ import annotations

import numpy as np

# R[action, class]: action 0 = discharge (negative), 1 = admit (positive)
#                   class  0 = truly negative,        1 = truly positive
R1 = np.array([[1.,  0.], [ 0., 1.]], dtype=float)   # symmetric accuracy
R2 = np.array([[1.,  0.], [-4., 1.]], dtype=float)   # FP penalty (admit negative costs 4)
R3 = np.array([[1., -4.], [ 0., 1.]], dtype=float)   # FN penalty (discharge positive costs 4)
REWARD_MATRICES: dict[str, np.ndarray] = {"R1": R1, "R2": R2, "R3": R3}

# Decision threshold for each reward matrix (binary action: discharge vs admit).
# Derived by solving R[0]·[1-b,b] = R[1]·[1-b,b] for b.
FACET_THRESHOLDS: dict[str, float] = {"R1": 0.5, "R2": 5 / 6, "R3": 1 / 6}


def optimal_value(b: np.ndarray, R: np.ndarray) -> np.ndarray:
    """V(b, R) = max_a R @ b. b: (n, K), R: (A, K). Returns (n,)."""
    return (b @ R.T).max(axis=-1)


def model_action(b: np.ndarray, R: np.ndarray) -> np.ndarray:
    """argmax_a R @ b. b: (n, K), R: (A, K). Returns (n,) int array."""
    return (b @ R.T).argmax(axis=-1)


def boundary_regret(b_x: np.ndarray, b_xh: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Per-instance BR-hat = max(0, V(b_xh,R) - reward of model action evaluated at b_xh).

    b_x, b_xh: (n, K) belief arrays. R: (A, K). Returns (n,) non-negative floats.
    """
    a_x = model_action(b_x, R)          # (n,)
    v_bxh = optimal_value(b_xh, R)      # (n,)
    r_ax = R[a_x]                        # (n, K)
    reward = (r_ax * b_xh).sum(axis=-1)
    return np.maximum(v_bxh - reward, 0.0)


def log_loss_gain(b_x: np.ndarray, b_xh: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log P(y | b_xh) - log P(y | b_x) for binary y in {0, 1}.

    b_x, b_xh: (n,) arrays of P(Y=1). y: (n,) int array.
    """
    b_x  = np.clip(b_x,  1e-7, 1 - 1e-7)
    b_xh = np.clip(b_xh, 1e-7, 1 - 1e-7)
    p_bx  = np.where(y == 1, b_x,  1 - b_x)
    p_bxh = np.where(y == 1, b_xh, 1 - b_xh)
    return np.log(p_bxh) - np.log(p_bx)
