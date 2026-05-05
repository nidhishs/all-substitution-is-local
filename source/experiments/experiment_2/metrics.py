"""Per-pair metrics for Experiment 2.

- Top-expertise decile: defined by |h disagreement from model| = 1 - b_x[i, h[i]].
- BR-positive tolerance: scale-aware via core.reward_tolerance(R).
"""

from __future__ import annotations

import numpy as np

import core
from core import spearman_bootstrap_ci

from .rewards import REWARDS

_N_BOOT = 10_000
_BOOT_SEED = 42


def compute_pair_metrics(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute BR̂ dissociation metrics for one (condition, reader) pair.

    Args:
        b_x:  (N, K) model-only beliefs.
        b_xh: (N, K) augmented beliefs.
        h:    (N,) human signal, integer in [0, K).
        y:    (N,) ground-truth labels, integer in [0, K).
        rng:  RNG for bootstrap CI. Defaults to _BOOT_SEED if None.

    Returns:
        Dict with a sub-dict per reward key ("R1", "R2", "R3"), each containing:
            rho_ll_br, rho_ll_br_ci_lo, rho_ll_br_ci_hi -- Spearman rho(lkg, BR^) + BCa 95% CI.
            rho_facet_br                                -- Spearman rho(facet_distance, BR^).
            top_decile_frac_br_zero                     -- fraction of top-disagreement decile with BR^ = 0.
            mean_ll_gain                                -- mean per-instance log-loss gain.
            all_zero_br                                 -- True if BR^ = 0 for all instances.
    """
    if rng is None:
        rng = np.random.default_rng(_BOOT_SEED)
    lkg = core.log_loss_gain(b_x, b_xh, y)

    # Disagreement = how surprised the model is by the human's answer.
    # Computed from raw inputs only; independent of b_xh.
    disagreement = 1.0 - b_x[np.arange(len(h)), h.astype(int)]
    top_decile_mask = disagreement >= np.percentile(disagreement, 90)

    result: dict = {}

    for name, R in REWARDS.items():
        tol = core.reward_tolerance(R)
        br_hat = core.boundary_regret(b_x, b_xh, R)
        all_zero = bool((br_hat <= tol).all())

        if all_zero:
            rho = rho_ci_lo = rho_ci_hi = rho_facet = float("nan")
        else:
            rho, _ = core.spearman_corr(lkg, br_hat)
            rho_ci_lo, rho_ci_hi = spearman_bootstrap_ci(
                lkg,
                br_hat,
                n_boot=_N_BOOT,
                rng=rng,
            )
            rho_facet, _ = core.spearman_corr(core.facet_distance(b_x, R), br_hat)

        frac_zero = float((br_hat[top_decile_mask] <= tol).mean())

        result[name] = {
            "rho_ll_br": rho,
            "rho_ll_br_ci_lo": rho_ci_lo,
            "rho_ll_br_ci_hi": rho_ci_hi,
            "rho_facet_br": rho_facet,
            "top_decile_frac_br_zero": frac_zero,
            "mean_ll_gain": float(lkg.mean()),
            "all_zero_br": all_zero,
        }

    return result
