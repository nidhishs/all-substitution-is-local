#!/usr/bin/env python3
"""
Experiment 3: From Decision-Value Audit to Review Allocation
=============================================================

Tests whether an ex ante decision-value audit identifies cases where
obtaining human input would improve the deployed decision, under
scarce review budgets.

Setup (proposal §Experiment 3):
  - Every eval instance has (x, h, y) but review policy sees only x-available
    information at allocation time.
  - At budget q, the policy selects cases. Only after selection is h revealed.
  - Reviewed cases take the human-updated action argmax_a r_a · b_{x,h}.
  - Unreviewed cases take model-only action a_x.

Policies:
  1. Boundary regret (BR_pre): rank by E_{h|x}[V(b_{x,h}) - r_{a_x} · b_{x,h}]
  2. Model uncertainty (entropy / margin of b_x)
  3. Residual expertise (estimated predictive-loss gain)
  4. Learning to defer (logistic deferral classifier)
  5. Random
  6. Oracle (hindsight selection using realized utility gain)

Datasets:
  A. CheXpert (binary, from pilot — user supplies data)
  B. CIFAR-10H (10-class, downloaded — see prepare_cifar10h.py)

Usage:
  uv run --with numpy --with scipy --with scikit-learn experiment3_allocation.py \\
      --data data/chexpert_pilot.npz --reward R1

"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from utils import REWARD_MATRICES as _SHARED_REWARDS


# ──────────────────────────────────────────────────────────────
# Core computations (shared with Experiment 1)
# ──────────────────────────────────────────────────────────────

EPS = 1e-15


def terminal_value(R: np.ndarray, b: np.ndarray) -> float:
    """V(b) = max_a r_a · b"""
    return float(np.max(R @ b))


def optimal_action(R: np.ndarray, b: np.ndarray) -> int:
    return int(np.argmax(R @ b))


def boundary_regret(R: np.ndarray, b: np.ndarray, a: int) -> float:
    """BR = V(b) - r_a · b"""
    return terminal_value(R, b) - float(R[a] @ b)


def decision_utility(R: np.ndarray, a: int, y: int) -> float:
    """Realized utility of taking action a when true state is y."""
    return float(R[a, y])


# ──────────────────────────────────────────────────────────────
# Reward matrices
# ──────────────────────────────────────────────────────────────

def make_binary_rewards(name: str, c_review: float = 0.0) -> np.ndarray:
    """Standard binary reward matrices (proposal §Experiment 2).

    Actions: 0 = discharge (negative), 1 = admit (positive).
    States:  0 = truly negative, 1 = truly positive.
    """
    if name in _SHARED_REWARDS:
        return _SHARED_REWARDS[name].copy()
    elif name == "R4":
        # Symmetric with review cost (applied separately)
        return np.array([[1.0, 0.0], [0.0, 1.0]])
    else:
        raise ValueError(f"Unknown reward: {name}")


def make_multiclass_rewards(K: int, name: str) -> np.ndarray:
    """Reward matrices for K-class problems.

    Actions = classes (predict class k).
    States  = true classes.
    """
    if name == "R1":
        # Symmetric accuracy
        return np.eye(K)
    elif name == "R2":
        # Penalize misclassifying class 0 (e.g., rare class)
        R = np.eye(K)
        R[:, 0] = -2.0  # misclassifying a class-0 instance costs 2
        R[0, 0] = 3.0   # correctly classifying class-0 worth 3
        return R
    elif name == "R3":
        # Penalize false positives on class 0
        R = np.eye(K)
        R[0, :] = -1.0  # predicting class 0 when wrong costs 1
        R[0, 0] = 2.0   # predicting class 0 when right worth 2
        return R
    elif name == "R4":
        # Symmetric (review cost applied separately)
        return np.eye(K)
    else:
        raise ValueError(f"Unknown reward: {name}")


# ──────────────────────────────────────────────────────────────
# Allocation scores (policies)
# ──────────────────────────────────────────────────────────────

def score_entropy(b_x: np.ndarray) -> np.ndarray:
    """Shannon entropy of model beliefs. Higher = more uncertain."""
    b_clipped = np.clip(b_x, EPS, 1.0)
    return -np.sum(b_clipped * np.log(b_clipped), axis=1)


def score_margin(b_x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Negative margin to nearest decision facet. Higher = closer to boundary.

    For each instance, compute the gap between the best and second-best
    action values. Negate so that close-to-boundary instances rank highest.
    """
    action_values = b_x @ R.T  # (N, |A|)
    sorted_vals = np.sort(action_values, axis=1)
    margin = sorted_vals[:, -1] - sorted_vals[:, -2]
    return -margin  # negate: small margin → high score


def score_br_pre(
    b_x: np.ndarray,
    h_train: np.ndarray,
    y_train: np.ndarray,
    b_x_train: np.ndarray,
    R: np.ndarray,
    n_folds: int = 5,
    n_h_samples: int = 50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Ex ante boundary regret score (deployable):
    BR_pre(x) = E_{h|x}[V(b_{x,h}) - r_{a_x} · b_{x,h}]

    Estimated by:
    1. Learning P(h|x) from training data (using b_x as features)
    2. Learning P(y|x,h) from training data
    3. Monte Carlo: sample h ~ P(h|x), compute BR for each sample
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = b_x.shape[0]
    K = b_x.shape[1]
    scores = np.zeros(N)

    # Check if h is binary or multi-class
    h_unique = np.unique(h_train)
    n_h_values = len(h_unique)

    # Learn P(h|x) from training data: logistic regression on b_x features
    h_model = LogisticRegression(max_iter=1000, random_state=42)
    features_train = _make_features(b_x_train)
    h_model.fit(features_train, h_train)

    # Learn P(y|x,h) from training data using cross-fitting
    # Fit one augmented model
    aug_features_train = _make_aug_features(b_x_train, h_train, n_h_values)
    y_model = LogisticRegression(max_iter=1000, random_state=42)
    y_model.fit(aug_features_train, y_train)

    features_eval = _make_features(b_x)

    for i in range(N):
        b_i = b_x[i]
        a_x = optimal_action(R, b_i)

        # P(h|x) for this instance
        p_h = h_model.predict_proba(features_eval[i:i+1])[0]

        total_br = 0.0
        for h_idx, h_val in enumerate(h_model.classes_):
            if p_h[h_idx] < EPS:
                continue

            # Estimate b_{x,h} = P(y|x,h)
            aug_feat = _make_aug_features(b_x[i:i+1], np.array([h_val]), n_h_values)
            b_xh = y_model.predict_proba(aug_feat)[0]

            # Ensure b_xh has K classes (pad if needed)
            if len(b_xh) < K:
                b_xh_full = np.zeros(K)
                for ci, c in enumerate(y_model.classes_):
                    b_xh_full[c] = b_xh[ci]
                b_xh = b_xh_full

            br = boundary_regret(R, b_xh, a_x)
            total_br += p_h[h_idx] * br

        scores[i] = total_br

    return scores


def score_residual_expertise(
    b_x: np.ndarray,
    h_train: np.ndarray,
    y_train: np.ndarray,
    b_x_train: np.ndarray,
) -> np.ndarray:
    """
    Estimated predictive-loss gain from observing H.
    E_{h|x}[log P(y|x,h) - log P(y|x)] estimated from training data.

    Uses the same P(h|x) and P(y|x,h) models as BR_pre but computes
    expected log-loss reduction instead of boundary regret.
    """
    N = b_x.shape[0]
    K = b_x.shape[1]
    scores = np.zeros(N)

    h_unique = np.unique(h_train)
    n_h_values = len(h_unique)

    h_model = LogisticRegression(max_iter=1000, random_state=42)
    features_train = _make_features(b_x_train)
    h_model.fit(features_train, h_train)

    aug_features_train = _make_aug_features(b_x_train, h_train, n_h_values)
    y_model = LogisticRegression(max_iter=1000, random_state=42)
    y_model.fit(aug_features_train, y_train)

    features_eval = _make_features(b_x)

    for i in range(N):
        b_i = b_x[i]
        p_h = h_model.predict_proba(features_eval[i:i+1])[0]

        total_gain = 0.0
        for h_idx, h_val in enumerate(h_model.classes_):
            if p_h[h_idx] < EPS:
                continue

            aug_feat = _make_aug_features(b_x[i:i+1], np.array([h_val]), n_h_values)
            b_xh = y_model.predict_proba(aug_feat)[0]

            if len(b_xh) < K:
                b_xh_full = np.zeros(K)
                for ci, c in enumerate(y_model.classes_):
                    b_xh_full[c] = b_xh[ci]
                b_xh = b_xh_full

            # E_y|x,h [log P(y|x,h) - log P(y|x)]
            # = sum_y P(y|x,h) [log P(y|x,h) - log P(y|x)]  (KL divergence)
            for y_idx in range(K):
                p_new = np.clip(b_xh[y_idx], EPS, 1.0)
                p_old = np.clip(b_i[y_idx], EPS, 1.0)
                total_gain += p_h[h_idx] * p_new * (np.log(p_new) - np.log(p_old))

        scores[i] = total_gain

    return scores


def score_l2d(
    b_x: np.ndarray,
    h_train: np.ndarray,
    y_train: np.ndarray,
    b_x_train: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Learning-to-defer score: train a binary classifier to predict whether
    deferring to the human would improve utility.

    Label = 1 if human-updated action has higher utility than model-only action.
    """
    K = b_x_train.shape[1]
    n_h_values = len(np.unique(h_train))

    # Compute deferral labels for training data
    defer_labels = np.zeros(len(b_x_train), dtype=int)
    for i in range(len(b_x_train)):
        a_model = optimal_action(R, b_x_train[i])
        u_model = decision_utility(R, a_model, y_train[i])

        # Simulate human-updated action
        b_xh = _simple_augmented_belief(b_x_train[i], h_train[i], n_h_values, K)
        a_human = optimal_action(R, b_xh)
        u_human = decision_utility(R, a_human, y_train[i])

        defer_labels[i] = int(u_human > u_model)

    # Train deferral classifier
    features_train = _make_features(b_x_train)
    if defer_labels.sum() == 0 or defer_labels.sum() == len(defer_labels):
        # Degenerate case: all same label
        return np.zeros(b_x.shape[0])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(features_train, defer_labels)

    features_eval = _make_features(b_x)
    return clf.predict_proba(features_eval)[:, 1]


def score_oracle(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Oracle score: actual realized utility gain from review (hindsight)."""
    N = b_x.shape[0]
    scores = np.zeros(N)
    for i in range(N):
        a_model = optimal_action(R, b_x[i])
        u_model = decision_utility(R, a_model, y[i])
        a_human = optimal_action(R, b_xh[i])
        u_human = decision_utility(R, a_human, y[i])
        scores[i] = u_human - u_model
    return scores


# ──────────────────────────────────────────────────────────────
# Feature helpers
# ──────────────────────────────────────────────────────────────

def _make_features(b_x: np.ndarray) -> np.ndarray:
    """Features for P(h|x) and deferral models: log-odds of beliefs."""
    b_clipped = np.clip(b_x, EPS, 1.0 - EPS)
    # Use log-odds of each class probability
    log_odds = np.log(b_clipped / (1.0 - b_clipped))
    return log_odds


def _make_aug_features(
    b_x: np.ndarray,
    h: np.ndarray,
    n_h_values: int,
) -> np.ndarray:
    """Features for P(y|x,h): log-odds of beliefs + one-hot h."""
    log_odds = _make_features(b_x)
    h_onehot = np.zeros((len(h), n_h_values))
    for i, hi in enumerate(h):
        h_onehot[i, int(hi)] = 1.0
    return np.hstack([log_odds, h_onehot])


def _simple_augmented_belief(
    b_x: np.ndarray,
    h: int,
    n_h_values: int,
    K: int,
) -> np.ndarray:
    """Simple heuristic augmented belief for L2D training.
    Shift belief toward class h (for multi-class) or toward positive (binary).
    """
    b = b_x.copy()
    if n_h_values == 2:
        # Binary: h=1 means human says positive
        if h == 1:
            b = 0.7 * b + 0.3 * np.array([0.0, 1.0] + [0.0] * (K - 2))[:K]
        else:
            b = 0.7 * b + 0.3 * np.array([1.0, 0.0] + [0.0] * (K - 2))[:K]
    else:
        # Multi-class: shift toward class h
        target = np.zeros(K)
        if h < K:
            target[h] = 1.0
        else:
            target = np.ones(K) / K
        b = 0.7 * b + 0.3 * target
    b = np.clip(b, EPS, None)
    return b / b.sum()


# ──────────────────────────────────────────────────────────────
# Allocation experiment
# ──────────────────────────────────────────────────────────────

@dataclass
class AllocationResult:
    """Results for one (dataset, reward, budget) configuration."""
    dataset: str
    reward_name: str
    budget: float
    # Utility per policy
    utility: dict[str, float] = field(default_factory=dict)
    # Utility gain over model-only per policy
    utility_gain: dict[str, float] = field(default_factory=dict)
    # Gain per review
    gain_per_review: dict[str, float] = field(default_factory=dict)
    # Overlap with oracle
    oracle_overlap: dict[str, float] = field(default_factory=dict)


def run_allocation(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    reward_name: str,
    dataset_name: str,
    budgets: list[float],
    c_review: float = 0.0,
    n_folds: int = 5,
    seed: int = 42,
) -> list[AllocationResult]:
    """
    Run the allocation experiment for one (dataset, reward) pair across budgets.

    Uses cross-fitting: train scoring models on training folds,
    evaluate allocation on test folds.
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    K = b_x.shape[1]

    # Cross-fitted scores
    # We need to compute scores using only training-fold information
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_scores = {
        "BR_pre": np.zeros(N),
        "Entropy": np.zeros(N),
        "Margin": np.zeros(N),
        "Residual": np.zeros(N),
        "L2D": np.zeros(N),
        "Random": np.zeros(N),
        "Oracle": np.zeros(N),
    }

    # Non-parametric scores (don't need training data)
    all_scores["Entropy"] = score_entropy(b_x)
    all_scores["Margin"] = score_margin(b_x, R)
    all_scores["Random"] = rng.random(N)
    all_scores["Oracle"] = score_oracle(b_x, b_xh, y, R)

    # Cross-fitted parametric scores
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(b_x, y)):
        b_x_train = b_x[train_idx]
        h_train = h[train_idx]
        y_train = y[train_idx]
        b_x_test = b_x[test_idx]

        # BR_pre
        all_scores["BR_pre"][test_idx] = score_br_pre(
            b_x_test, h_train, y_train, b_x_train, R, rng=rng
        )

        # Residual expertise
        all_scores["Residual"][test_idx] = score_residual_expertise(
            b_x_test, h_train, y_train, b_x_train
        )

        # L2D
        all_scores["L2D"][test_idx] = score_l2d(
            b_x_test, h_train, y_train, b_x_train, R
        )

    # --- Evaluate allocation at each budget ---
    results = []
    for budget in budgets:
        n_review = max(1, int(N * budget))
        result = AllocationResult(
            dataset=dataset_name,
            reward_name=reward_name,
            budget=budget,
        )

        # Model-only utility (baseline: no review)
        model_only_utility = 0.0
        for i in range(N):
            a_m = optimal_action(R, b_x[i])
            model_only_utility += decision_utility(R, a_m, y[i])
        model_only_utility /= N

        # Oracle selection for overlap computation
        oracle_selected = set(np.argsort(all_scores["Oracle"])[-n_review:])

        for policy_name, scores in all_scores.items():
            # Select top-scoring instances for review
            selected = np.argsort(scores)[-n_review:]
            selected_set = set(selected)

            # Compute utility
            total_utility = 0.0
            for i in range(N):
                if i in selected_set:
                    # Reviewed: take human-updated action
                    a_h = optimal_action(R, b_xh[i])
                    u = decision_utility(R, a_h, y[i])
                    if c_review > 0:
                        u -= c_review
                else:
                    # Unreviewed: take model-only action
                    a_m = optimal_action(R, b_x[i])
                    u = decision_utility(R, a_m, y[i])
                total_utility += u

            avg_utility = total_utility / N
            gain = avg_utility - model_only_utility
            gpr = gain / (n_review / N) if n_review > 0 else 0.0

            result.utility[policy_name] = avg_utility
            result.utility_gain[policy_name] = gain
            result.gain_per_review[policy_name] = gpr
            result.oracle_overlap[policy_name] = (
                len(selected_set & oracle_selected) / n_review
            )

        results.append(result)

    return results


# ──────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────────────────────

def bootstrap_utility_ci(
    b_x: np.ndarray,
    b_xh: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    scores_dict: dict[str, np.ndarray],
    budget: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
    c_review: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap 95% CI for utility gain of each policy.

    Returns dict mapping policy_name -> (mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    n_review = max(1, int(N * budget))

    results = {name: [] for name in scores_dict}

    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        b_x_b = b_x[idx]
        b_xh_b = b_xh[idx]
        y_b = y[idx]

        # Model-only utility
        model_u = sum(
            decision_utility(R, optimal_action(R, b_x_b[i]), y_b[i])
            for i in range(N)
        ) / N

        for name, scores in scores_dict.items():
            s = scores[idx]
            selected = set(np.argsort(s)[-n_review:])

            total_u = 0.0
            for i in range(N):
                if i in selected:
                    a_h = optimal_action(R, b_xh_b[i])
                    u = decision_utility(R, a_h, y_b[i])
                    if c_review > 0:
                        u -= c_review
                else:
                    a_m = optimal_action(R, b_x_b[i])
                    u = decision_utility(R, a_m, y_b[i])
                total_u += u

            results[name].append(total_u / N - model_u)

    ci = {}
    for name, gains in results.items():
        gains = np.array(gains)
        ci[name] = (float(gains.mean()), float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5)))
    return ci


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_data(path: str) -> dict:
    """
    Load dataset from .npz file.

    Expected arrays:
      b_x:  (N, K) calibrated model beliefs
      b_xh: (N, K) human-augmented beliefs (cross-fitted)
      h:    (N,) human signal (integer-coded)
      y:    (N,) ground truth labels (integer-coded)

    Optional:
      dataset_name: string
      K: number of classes
    """
    data = np.load(path, allow_pickle=True)
    return {
        "b_x": data["b_x"],
        "b_xh": data["b_xh"],
        "h": data["h"],
        "y": data["y"],
        "dataset_name": str(data.get("dataset_name", "unknown")),
    }


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────

def print_allocation_results(all_results: list[AllocationResult]) -> None:
    """Print formatted allocation results."""
    print("\n" + "=" * 95)
    print("EXPERIMENT 3: REVIEW ALLOCATION — RESULTS")
    print("=" * 95)

    # Group by (dataset, reward)
    groups = {}
    for r in all_results:
        key = (r.dataset, r.reward_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for (ds, rn), group_results in groups.items():
        print(f"\n── {ds} / {rn} ──\n")

        policies = list(group_results[0].utility.keys())
        budgets = [r.budget for r in group_results]

        # Utility table
        print(f"{'Policy':<12} " + " ".join(f"q={b:.0%}".rjust(10) for b in budgets))
        print("-" * (12 + 11 * len(budgets)))

        for p in policies:
            vals = " ".join(f"{r.utility[p]:>10.4f}" for r in group_results)
            print(f"{p:<12} {vals}")

        # Utility gain table
        print(f"\n{'Utility gain over model-only:'}")
        print(f"{'Policy':<12} " + " ".join(f"q={b:.0%}".rjust(10) for b in budgets))
        print("-" * (12 + 11 * len(budgets)))

        for p in policies:
            vals = " ".join(f"{r.utility_gain[p]:>+10.4f}" for r in group_results)
            print(f"{p:<12} {vals}")

        # Gain per review
        print(f"\n{'Gain per review budget unit:'}")
        print(f"{'Policy':<12} " + " ".join(f"q={b:.0%}".rjust(10) for b in budgets))
        print("-" * (12 + 11 * len(budgets)))

        for p in policies:
            vals = " ".join(f"{r.gain_per_review[p]:>+10.4f}" for r in group_results)
            print(f"{p:<12} {vals}")

        # Oracle overlap
        print(f"\n{'Oracle overlap:'}")
        print(f"{'Policy':<12} " + " ".join(f"q={b:.0%}".rjust(10) for b in budgets))
        print("-" * (12 + 11 * len(budgets)))

        for p in policies:
            if p == "Oracle":
                continue
            vals = " ".join(f"{r.oracle_overlap[p]:>10.1%}" for r in group_results)
            print(f"{p:<12} {vals}")


def save_results_json(all_results: list[AllocationResult], path: str) -> None:
    """Save results to JSON."""
    data = []
    for r in all_results:
        d = {
            "dataset": r.dataset,
            "reward_name": r.reward_name,
            "budget": r.budget,
            "utility": r.utility,
            "utility_gain": r.utility_gain,
            "gain_per_review": r.gain_per_review,
            "oracle_overlap": r.oracle_overlap,
        }
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ──────────────────────────────────────────────────────────────
# Synthetic demo (runs without external data)
# ──────────────────────────────────────────────────────────────

def generate_synthetic_demo(
    N: int = 2000,
    K: int = 2,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic binary dataset that mimics the CheXpert pilot structure.

    - Model: calibrated probability b_x ~ Beta-distributed
    - Human: binary signal correlated with y but not identical to model
    - Ground truth: binary
    """
    rng = np.random.default_rng(seed)

    # Generate ground truth: ~40% positive (like Cardiomegaly)
    prevalence = 0.42
    y = rng.binomial(1, prevalence, size=N)

    # Model belief b_x = P(Y=1|x): correlated with y but noisy
    b_x_pos = np.clip(rng.beta(4, 2, size=N), 0.05, 0.95)   # y=1 instances
    b_x_neg = np.clip(rng.beta(2, 4, size=N), 0.05, 0.95)   # y=0 instances
    b_x_1d = np.where(y == 1, b_x_pos, b_x_neg)

    b_x = np.column_stack([1 - b_x_1d, b_x_1d])  # (N, 2)

    # Human signal: binary, correlated with y (like bc1)
    # P(h=1 | y=1) = 0.80, P(h=1 | y=0) = 0.25
    h = np.zeros(N, dtype=int)
    h[y == 1] = rng.binomial(1, 0.80, size=(y == 1).sum())
    h[y == 0] = rng.binomial(1, 0.25, size=(y == 0).sum())

    # Augmented belief b_{x,h}: shift b_x toward h's information
    b_xh_1d = np.zeros(N)
    for i in range(N):
        if h[i] == 1:
            b_xh_1d[i] = np.clip(b_x_1d[i] * 2.5 / (b_x_1d[i] * 2.5 + (1 - b_x_1d[i])), 0.02, 0.98)
        else:
            b_xh_1d[i] = np.clip(b_x_1d[i] * 0.4 / (b_x_1d[i] * 0.4 + (1 - b_x_1d[i])), 0.02, 0.98)

    b_xh = np.column_stack([1 - b_xh_1d, b_xh_1d])

    return {
        "b_x": b_x,
        "b_xh": b_xh,
        "h": h,
        "y": y,
        "dataset_name": "synthetic_binary",
    }


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Review Allocation")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz data file (b_x, b_xh, h, y)")
    parser.add_argument("--reward", type=str, nargs="+", default=["R1", "R2", "R3"],
                        help="Reward matrices to test")
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20, 0.50],
                        help="Review budgets as fractions")
    parser.add_argument("--c-review", type=float, default=0.0,
                        help="Per-case review cost (for R4)")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Number of bootstrap samples for CIs (0 = skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true",
                        help="Run on synthetic demo data (no data file needed)")
    args = parser.parse_args()

    if args.data is None and not args.demo:
        print("No data file provided. Running synthetic demo.")
        print("Use --data path/to/data.npz for real data, or --demo explicitly.\n")
        args.demo = True

    # Load data
    if args.demo:
        data = generate_synthetic_demo(N=2000, seed=args.seed)
    else:
        data = load_data(args.data)

    b_x = data["b_x"]
    b_xh = data["b_xh"]
    h = data["h"]
    y = data["y"]
    ds_name = data["dataset_name"]
    K = b_x.shape[1]

    print(f"Dataset: {ds_name}")
    print(f"N = {len(y)}, K = {K}")
    print(f"Prevalence: {y.mean():.1%}" if K == 2 else f"Class distribution: {np.bincount(y)}")
    print(f"Rewards: {args.reward}")
    print(f"Budgets: {args.budgets}")
    print()

    all_results = []

    for rn in args.reward:
        if K == 2:
            R = make_binary_rewards(rn)
        else:
            R = make_multiclass_rewards(K, rn)

        c_rev = args.c_review if rn == "R4" else 0.0

        results = run_allocation(
            b_x, b_xh, h, y, R,
            reward_name=rn,
            dataset_name=ds_name,
            budgets=args.budgets,
            c_review=c_rev,
            seed=args.seed,
        )
        all_results.extend(results)

    print_allocation_results(all_results)

    # Bootstrap CIs if requested
    if args.bootstrap > 0:
        print("\n── Bootstrap 95% CIs (utility gain) ──\n")
        for rn in args.reward:
            if K == 2:
                R = make_binary_rewards(rn)
            else:
                R = make_multiclass_rewards(K, rn)

            # Recompute scores for bootstrap
            rng = np.random.default_rng(args.seed)
            scores_dict = {
                "Entropy": score_entropy(b_x),
                "Margin": score_margin(b_x, R),
                "Random": rng.random(len(y)),
            }

            for budget in args.budgets:
                ci = bootstrap_utility_ci(
                    b_x, b_xh, y, R, scores_dict, budget,
                    n_bootstrap=args.bootstrap, seed=args.seed
                )
                print(f"{rn} / q={budget:.0%}:")
                for name, (mean, lo, hi) in ci.items():
                    print(f"  {name:<12} {mean:>+.4f}  [{lo:>+.4f}, {hi:>+.4f}]")
                print()

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    save_results_json(all_results, str(out_dir / f"experiment3_{ds_name}.json"))


if __name__ == "__main__":
    main()
