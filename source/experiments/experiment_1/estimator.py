"""Cross-fitted plug-in BR̂ estimator for Experiment 1."""

from __future__ import annotations

import core
import numpy as np
from sklearn.model_selection import KFold

from .conditions import Channel, DecisionProblem


def _estimate_channel(
    signals: np.ndarray,
    labels: np.ndarray,
    n_signals: int,
    n_states: int,
) -> np.ndarray:
    flat = signals * n_states + labels
    counts = (
        np.bincount(flat, minlength=n_signals * n_states).reshape(n_signals, n_states)
        + 1.0
    )
    return counts / counts.sum(axis=0, keepdims=True)


def estimate_br(
    problem: DecisionProblem,
    M: Channel,
    H: Channel,
    n: int,
    rng: np.random.Generator,
    n_folds: int = 5,
) -> dict:
    """Cross-fitted plug-in BR̂ alongside oracle BR under the true model.

    Draws n samples from the generative model, then for each fold estimates
    prior, M, H from training instances and computes posteriors on test instances.
    The oracle uses the true prior and channels on the same (m, h) realization.

    Returns:
        Dict with keys: br_hat, br_oracle, b_x_oracle, b_xh_oracle, y, m, h.
    """
    K = len(problem.prior)
    y, m, h = core.generate_samples(problem.prior, M.kernel, H.kernel, n, rng)

    b_x_hat = np.empty((n, K))
    b_xh_hat = np.empty((n, K))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for train_idx, test_idx in kf.split(np.arange(n)):
        y_tr, m_tr, h_tr = y[train_idx], m[train_idx], h[train_idx]

        prior_hat = np.bincount(y_tr, minlength=K).astype(float) + 1.0
        prior_hat /= prior_hat.sum()
        M_hat = _estimate_channel(m_tr, y_tr, M.kernel.shape[0], K)
        H_hat = _estimate_channel(h_tr, y_tr, H.kernel.shape[0], K)

        n_m, n_h = M_hat.shape[0], H_hat.shape[0]
        bx_table = np.array([core.posterior(prior_hat, M_hat[s]) for s in range(n_m)])
        bxh_table = np.array(
            [
                core.joint_posterior(prior_hat, M_hat[sm], H_hat[sh])
                for sm in range(n_m)
                for sh in range(n_h)
            ]
        ).reshape(n_m, n_h, K)
        b_x_hat[test_idx] = bx_table[m[test_idx]]
        b_xh_hat[test_idx] = bxh_table[m[test_idx], h[test_idx]]

    br_hat = core.boundary_regret(b_x_hat, b_xh_hat, problem.R)

    n_m, n_h = M.kernel.shape[0], H.kernel.shape[0]
    ox_table = np.array(
        [core.posterior(problem.prior, M.kernel[s]) for s in range(n_m)]
    )
    oxh_table = np.array(
        [
            core.joint_posterior(problem.prior, M.kernel[sm], H.kernel[sh])
            for sm in range(n_m)
            for sh in range(n_h)
        ]
    ).reshape(n_m, n_h, K)
    b_x_oracle = ox_table[m]
    b_xh_oracle = oxh_table[m, h]
    br_oracle = core.boundary_regret(b_x_oracle, b_xh_oracle, problem.R)

    return {
        "br_hat": br_hat,
        "br_oracle": br_oracle,
        "b_x_oracle": b_x_oracle,
        "b_xh_oracle": b_xh_oracle,
        "y": y,
        "m": m,
        "h": h,
    }
