"""Reward matrices and derived indifference thresholds for Experiment 2.

Three binary (2-action, 2-state) rewards:
  R1 — symmetric:        equal cost for FP and FN.
  R2 — FP-penalty (5x): rewards safe inaction when the pathology is absent.
  R3 — FN-penalty (5x): mirror of R2; rewards aggressive detection.

All thresholds are derived from the reward matrices, not hand-set.
"""

from __future__ import annotations

import numpy as np

# R[action, state]: R[0]=negative action, R[1]=positive action; state 0=absent, 1=present.
R1: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
R2: np.ndarray = np.array([[5.0, 0.0], [0.0, 1.0]], dtype=np.float64)
R3: np.ndarray = np.array([[1.0, 0.0], [0.0, 5.0]], dtype=np.float64)

REWARDS: dict[str, np.ndarray] = {"R1": R1, "R2": R2, "R3": R3}


def indifference_threshold(R: np.ndarray) -> float:
    """P(y=1) at which both actions have equal expected reward.

    For 2-action 2-state R: threshold = (R[0,0] - R[1,0]) / ((R[0,0] - R[1,0]) + (R[1,1] - R[0,1]))
    """
    r00, r01 = float(R[0, 0]), float(R[0, 1])
    r10, r11 = float(R[1, 0]), float(R[1, 1])
    denom = (r00 - r10) + (r11 - r01)
    if abs(denom) < 1e-12:
        raise ValueError("Reward matrix actions have equal expected value everywhere.")
    return (r00 - r10) / denom


FACET_THRESHOLDS: dict[str, float] = {
    name: indifference_threshold(R) for name, R in REWARDS.items()
}
