"""Cross-fitted scoring, per-budget evaluation, and paired BCa bootstrap.

Regime-agnostic: takes (b_x, b_xh, h, y) numpy arrays. Synthetic and real
runners both feed this engine.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import KFold, StratifiedKFold

import core
from core import mean_bootstrap_ci

from . import policies

_N_FOLDS = 5
_N_BOOT = 10_000

# All 7 policies to score, in canonical order.
POLICY_NAMES = ("BR_hat", "Residual", "Margin", "Entropy", "L2D", "Random", "Oracle")
BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50)
BASELINE: str = "Margin"


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


def _make_folds(n: int, y: np.ndarray, n_folds: int, seed: int) -> list:
    """Stratified on y when >=2 classes present, else plain KFold."""
    arange = np.arange(n)
    if len(np.unique(y)) < 2:
        return list(
            KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(arange)
        )
    return list(
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(
            arange, y
        )
    )


def compute_scores(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
    n_folds: int = _N_FOLDS,
) -> dict[str, np.ndarray]:
    """5-fold cross-fitted scoring for BR_hat, Residual, L2D; direct for all others.

    Args:
        b_x: (N, K) model-only beliefs.
        b_xh: (N, K) augmented beliefs.
        h: (N,) human signal, integer.
        y: (N,) labels, integer.
        R: Reward matrix of shape (|A|, K).
        rng: NumPy random generator for the Random policy and CV seed.
        n_folds: Number of cross-fitting folds.

    Returns:
        Dict mapping policy name to an (N,) score array.
    """
    n = len(y)

    # Derive cv_seed before any rng consumption so fold splits are independent of N.
    cv_seed = int(rng.integers(0, 2**31 - 1))

    # Direct scorers (no fold-fitting required).
    out: dict[str, np.ndarray] = {
        "Margin": policies.score_margin(b_x, R),
        "Entropy": policies.score_entropy(b_x),
        "Random": policies.score_random(b_x, rng),
        "Oracle": policies.score_oracle(b_x, b_xh, y, R),
    }

    # Cross-fitted scorers — zero-initialize then fill per fold.
    out["BR_hat"] = np.zeros(n)
    out["Residual"] = np.zeros(n)
    out["L2D"] = np.zeros(n)

    for tr_idx, te_idx in _make_folds(n, y, n_folds, cv_seed):
        b_x_tr, h_tr, y_tr = b_x[tr_idx], h[tr_idx], y[tr_idx]
        b_xh_tr = b_xh[tr_idx]
        b_x_te = b_x[te_idx]
        h_model, y_model = policies.fit_scoring_models(b_x_tr, h_tr, y_tr)

        out["BR_hat"][te_idx] = policies.score_br_hat(b_x_te, R, h_model, y_model)
        out["Residual"][te_idx] = policies.score_residual(b_x_te, h_model, y_model)
        out["L2D"][te_idx] = policies.score_l2d(
            b_x_tr, y_tr, b_xh_tr, R, b_x_test=b_x_te
        )

    return out


def paired_bootstrap_ci(
    diff: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = _N_BOOT,
    ci: float = 0.95,
) -> tuple[float, float]:
    """BCa CI on the mean of `diff`, with NaN return for degenerate input.

    Wraps core.mean_bootstrap_ci with a guard for len(diff) < 2 or zero variance,
    cases where the BCa algebra produces meaningless ±inf endpoints.
    """
    if len(diff) < 2 or float(np.std(diff)) < 1e-15:
        return float("nan"), float("nan")
    return mean_bootstrap_ci(diff, n_boot=n_boot, ci=ci, rng=rng)


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided sign-test p-value (H0: P(win) = 0.5).

    Only wins and losses are counted; ties (zero-delta pairs) are not passed in.
    Returns NaN when wins + losses < 2.
    """
    n = wins + losses
    if n < 2:
        return float("nan")
    return float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue)


def _select_indicator(
    scores: np.ndarray, q: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (selected_indices, fresh 0/1 indicator array) for top-q scores."""
    idx = top_q_indices(scores, q)
    indicator = np.zeros(n)
    indicator[idx] = 1.0
    return idx, indicator


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

    base_idx, base_indicator = _select_indicator(scores[baseline_policy], q, n)
    base_gain = utility_gain(g, base_idx)

    def policy_stats(name: str) -> dict:
        sel, indicator = _select_indicator(scores[name], q, n)
        util = utility_gain(g, sel)
        diff = (indicator - base_indicator) * g
        if name == baseline_policy:
            ci_lo = ci_hi = 0.0
        else:
            ci_lo, ci_hi = paired_bootstrap_ci(diff, rng=rng, n_boot=n_boot)
        return {
            "utility_gain": util,
            "n_selected": int(len(sel)),
            "delta_vs_baseline": util - base_gain,
            "delta_ci_lo": float(ci_lo),
            "delta_ci_hi": float(ci_hi),
        }

    return {name: policy_stats(name) for name in POLICY_NAMES}


def _summarize_pair_cells(cells: list[dict]) -> dict:
    """Reduce per-pair (utility_gain, delta_vs_baseline) dicts to cross-pair summary stats."""
    n = len(cells)
    if n == 0:
        nan = float("nan")
        return {
            "mean_utility_gain": nan,
            "mean_delta_vs_baseline": nan,
            "win_rate_vs_baseline": nan,
            "n_wins": 0,
            "n_losses": 0,
            "n_ties": 0,
            "sign_test_p": nan,
        }
    gains = np.fromiter((c["utility_gain"] for c in cells), float, n)
    deltas = np.fromiter((c["delta_vs_baseline"] for c in cells), float, n)
    finite = np.isfinite(deltas)
    wins = int((deltas[finite] > 0).sum())
    losses = int((deltas[finite] < 0).sum())
    ties = int((deltas[finite] == 0).sum())
    return {
        "mean_utility_gain": float(np.nanmean(gains)),
        "mean_delta_vs_baseline": float(np.nanmean(deltas)),
        "win_rate_vs_baseline": wins / n,
        "n_wins": wins,
        "n_losses": losses,
        "n_ties": ties,
        "sign_test_p": sign_test_p(wins, losses),
    }


def evaluate_pair(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    rewards: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    budgets: tuple[float, ...] = BUDGETS,
    baseline: str = BASELINE,
) -> dict:
    """Per-pair metrics: {reward_name: {budget_str: {policy: stats}}}."""
    metrics: dict = {}
    for reward_name, R in rewards.items():
        scores = compute_scores(b_x, b_xh, h, y, R, rng=rng)
        metrics[reward_name] = {
            f"{q:.2f}": evaluate_budget(
                scores,
                b_x,
                b_xh,
                y,
                R,
                q,
                baseline_policy=baseline,
                rng=rng,
            )
            for q in budgets
        }
    return metrics


def aggregate_pairs(
    results: list[dict],
    reward_names: Iterable[str],
    *,
    budgets: tuple[float, ...] = BUDGETS,
    baseline: str = BASELINE,
) -> dict:
    """Cross-pair aggregation per (reward, budget). Mirrors the per-pair shape.

    Each element of `results` must be a dict with a 'metrics' key whose value is
    the output of evaluate_pair (i.e. {reward_name: {budget_str: {policy: stats}}}).
    `reward_names` only needs to be iterable of names; values are not used here.
    """
    q_keys = [f"{q:.2f}" for q in budgets]

    def cell(rn: str, qk: str, p: str) -> dict:
        return _summarize_pair_cells([r["metrics"][rn][qk][p] for r in results])

    return {
        "n_pairs": len(results),
        "baseline_policy": baseline,
        "policies": {
            rn: {qk: {p: cell(rn, qk, p) for p in POLICY_NAMES} for qk in q_keys}
            for rn in reward_names
        },
    }
