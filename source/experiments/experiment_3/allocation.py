"""Cross-fitted scoring, per-budget evaluation, and paired BCa bootstrap.

Regime-agnostic: takes (b_x, b_xh, h, y) numpy arrays. Synthetic and real
runners both feed this engine.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

import core
from core import mean_bootstrap_ci

from . import policies

_N_FOLDS = 5
_N_BOOT = 10_000
_BOOT_SEED = 42

# All 7 policies to score, in canonical order.
POLICY_NAMES = ("BR_hat", "Residual", "Margin", "Entropy", "L2D", "Random", "Oracle")


def per_instance_review_gain(
    b_x: np.ndarray, b_xh: np.ndarray, y: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """g(i) = R[a_xh_i, y_i] - R[a_x_i, y_i]. Same as score_oracle."""
    a_x = core.model_action(b_x, R)
    a_xh = core.model_action(b_xh, R)
    return R[a_xh, y].astype(float) - R[a_x, y].astype(float)


def top_q_indices(scores: np.ndarray, q: float) -> np.ndarray:
    """Return indices of the top ceil(q*N) instances by score (descending)."""
    n_select = int(math.ceil(q * len(scores)))
    if n_select <= 0:
        return np.empty(0, dtype=int)
    if n_select >= len(scores):
        return np.argsort(-scores, kind="stable")
    # argpartition is O(N); refine the top set with a secondary sort.
    part = np.argpartition(-scores, n_select - 1)[:n_select]
    return part[np.argsort(-scores[part], kind="stable")]


def utility_gain(per_instance_gain: np.ndarray, selected: np.ndarray) -> float:
    """(1/N) * sum_{i in selected} g(i). N = len(per_instance_gain)."""
    if len(selected) == 0:
        return 0.0
    return float(per_instance_gain[selected].sum() / len(per_instance_gain))


def compute_scores(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
    n_folds: int = _N_FOLDS,
) -> dict[str, np.ndarray]:
    """5-fold cross-fitted scoring for BR_hat, Residual, L2D; direct otherwise.

    Folds are stratified on y to match data.preparation.fit_augmented_beliefs.
    Returns dict {policy_name: (N,) score array}.
    """
    n = len(y)
    out = {name: np.zeros(n) for name in POLICY_NAMES}

    # Direct scorers (no fold-fitting required).
    out["Margin"] = policies.score_margin(b_x, R)
    out["Entropy"] = policies.score_entropy(b_x)
    out["Random"] = policies.score_random(b_x, rng)
    out["Oracle"] = policies.score_oracle(b_x, b_xh, y, R)

    # Cross-fitted scorers.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=_BOOT_SEED)
    # If y is constant, StratifiedKFold fails; fall back to KFold.
    if len(np.unique(y)) < 2:
        skf = KFold(n_splits=n_folds, shuffle=True, random_state=_BOOT_SEED)
        splits = list(skf.split(np.arange(n)))
    else:
        splits = list(skf.split(np.arange(n), y))

    for tr_idx, te_idx in splits:
        b_x_tr, h_tr, y_tr = b_x[tr_idx], h[tr_idx], y[tr_idx]
        b_x_te = b_x[te_idx]
        h_model, y_model = policies.fit_scoring_models(b_x_tr, h_tr, y_tr)

        out["BR_hat"][te_idx] = policies.score_br_hat(b_x_te, R, h_model, y_model)
        out["Residual"][te_idx] = policies.score_residual(b_x_te, h_model, y_model)
        out["L2D"][te_idx] = policies.score_l2d(
            b_x_tr, h_tr, y_tr, R, b_x_test=b_x_te
        )

    return out


def paired_bootstrap_ci(
    diff: np.ndarray,
    n_boot: int = _N_BOOT,
    rng: np.random.Generator | None = None,
    ci: float = 0.95,
) -> tuple[float, float]:
    """BCa CI on the mean of `diff`, vectorized via mean_bootstrap_ci."""
    if rng is None:
        rng = np.random.default_rng(_BOOT_SEED)
    if len(diff) < 2 or float(np.std(diff)) < 1e-15:
        return float("nan"), float("nan")
    return mean_bootstrap_ci(diff, n_boot=n_boot, ci=ci, rng=rng)


def evaluate_budget(
    scores: dict[str, np.ndarray],
    b_x: np.ndarray,
    b_xh: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    q: float,
    baseline_policy: str,
    rng: np.random.Generator,
    n_boot: int = _N_BOOT,
) -> dict[str, dict]:
    """Evaluate every policy at budget q against a fixed baseline policy.

    Returns {policy_name: {utility_gain, n_selected, delta_vs_baseline, delta_ci_lo, delta_ci_hi}}.
    """
    g = per_instance_review_gain(b_x, b_xh, y, R)
    n = len(g)

    base_idx = top_q_indices(scores[baseline_policy], q)
    base_indicator = np.zeros(n)
    base_indicator[base_idx] = 1.0

    out: dict[str, dict] = {}
    for name in POLICY_NAMES:
        sel = top_q_indices(scores[name], q)
        indicator = np.zeros(n)
        indicator[sel] = 1.0
        util = utility_gain(g, sel)

        diff = (indicator - base_indicator) * g
        if name == baseline_policy:
            ci_lo = ci_hi = 0.0
        else:
            ci_lo, ci_hi = paired_bootstrap_ci(diff, n_boot=n_boot, rng=rng)

        out[name] = {
            "utility_gain": util,
            "n_selected": int(len(sel)),
            "delta_vs_baseline": util - utility_gain(g, base_idx),
            "delta_ci_lo": float(ci_lo),
            "delta_ci_hi": float(ci_hi),
        }
    out["_baseline_policy"] = baseline_policy
    return out
