#!/usr/bin/env python3
"""
Experiment 1: Estimator and Geometry Validation
================================================

Validates that the boundary-regret estimator recovers known BR values
under controlled finite-action Bayesian decision problems.

Five conditions (proposal §Experiment 1):
  C1: Human signal predictive, all posteriors stay in model action's region  (BR = 0)
  C2: Human signal crosses a decision boundary                              (BR > 0)
  C3: Human improves log-loss but activates no rival action                 (BR = 0)
  C4: Model & human cross same boundary   (ASIL substitutability)
  C5: Model & human cross different boundaries (ASIL complementarity)

Metrics:
  - Spearman rank correlation between BR-hat and true BR
  - Calibration error (ECE) between BR-hat and true BR
  - False positive rate when true BR = 0
  - Sensitivity to calibration degradation in b_x and b_{x,h}
  - ASIL sign accuracy (C4, C5 only)

Usage:
  uv run --with numpy --with scipy experiment1_synthetic.py

"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

# ──────────────────────────────────────────────────────────────
# Core data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class DecisionProblem:
    """Finite-state, finite-action Bayesian decision problem."""
    R: np.ndarray          # (|A|, K) reward matrix
    prior: np.ndarray      # (K,) prior over states

    @property
    def n_actions(self) -> int:
        return self.R.shape[0]

    @property
    def n_states(self) -> int:
        return self.R.shape[1]

    def terminal_value(self, b: np.ndarray) -> float:
        return float(np.max(self.R @ b))

    def optimal_action(self, b: np.ndarray) -> int:
        return int(np.argmax(self.R @ b))

    def action_value(self, b: np.ndarray, a: int) -> float:
        return float(self.R[a] @ b)

    def boundary_regret(self, b: np.ndarray, a: int) -> float:
        """BR = V(b) - r_a . b"""
        return self.terminal_value(b) - self.action_value(b, a)


@dataclass
class Channel:
    """Information channel P(signal | state)."""
    kernel: np.ndarray     # (n_signals, K) likelihood matrix

    @property
    def n_signals(self) -> int:
        return self.kernel.shape[0]

    @property
    def n_states(self) -> int:
        return self.kernel.shape[1]


# ──────────────────────────────────────────────────────────────
# Bayesian computations (exact)
# ──────────────────────────────────────────────────────────────

EPS = 1e-15

def posterior(prior: np.ndarray, likelihood_row: np.ndarray) -> np.ndarray:
    """P(y | signal) ∝ P(signal | y) P(y)"""
    joint = likelihood_row * prior
    s = joint.sum()
    if s < EPS:
        return prior.copy()
    return joint / s


def joint_posterior(
    prior: np.ndarray,
    lik_m: np.ndarray,
    lik_h: np.ndarray,
) -> np.ndarray:
    """P(y | m, h) ∝ P(m|y) P(h|y) P(y), assuming M ⊥ H | Y."""
    joint = lik_m * lik_h * prior
    s = joint.sum()
    if s < EPS:
        return prior.copy()
    return joint / s


def signal_probability(prior: np.ndarray, likelihood_row: np.ndarray) -> float:
    """P(signal) = sum_y P(signal|y) P(y)"""
    return float(likelihood_row @ prior)


# ──────────────────────────────────────────────────────────────
# Exact boundary-regret computation
# ──────────────────────────────────────────────────────────────

def exact_expected_br(
    problem: DecisionProblem,
    b_model: np.ndarray,
    H: Channel,
) -> float:
    """
    E_h[BR(x,h)] = sum_h P(h|b_m) [V(b_{m,h}) - r_{a_m} . b_{m,h}]

    Uses the model belief b_model as the conditioning belief.
    Human posteriors are computed by updating b_model with H.
    """
    a_m = problem.optimal_action(b_model)

    total = 0.0
    for h_idx in range(H.n_signals):
        # P(h | b_model) = H.kernel[h] . b_model
        p_h = signal_probability(b_model, H.kernel[h_idx])
        if p_h < EPS:
            continue

        # Posterior after human signal: P(y | x, h) ∝ P(h|y) b_model(y)
        b_mh = posterior(b_model, H.kernel[h_idx])
        br = problem.boundary_regret(b_mh, a_m)
        total += p_h * br

    return total


def exact_log_loss_gain(
    b_model: np.ndarray,
    H: Channel,
    b_model_prior: np.ndarray,
) -> float:
    """
    Expected log-loss gain from observing H:
    E_{h,y}[log P(y|x,h) - log P(y|x)]

    Computed exactly using the generative model.
    b_model is the model belief (used as both P(y|x) and the conditioning belief).
    """
    total = 0.0
    for h_idx in range(H.n_signals):
        p_h = signal_probability(b_model, H.kernel[h_idx])
        if p_h < EPS:
            continue

        b_mh = posterior(b_model, H.kernel[h_idx])

        # E_y|m,h [log P(y|m,h) - log P(y|m)]
        for y_idx in range(len(b_model)):
            p_y_given_mh = b_mh[y_idx]
            p_y_given_m = b_model[y_idx]
            if p_y_given_mh < EPS or p_y_given_m < EPS:
                continue
            # P(y | m, h) * [log P(y|m,h) - log P(y|m)]
            # weighted by P(h|m)
            total += p_h * p_y_given_mh * (np.log(p_y_given_mh) - np.log(p_y_given_m))

    return total


# ──────────────────────────────────────────────────────────────
# ASIL: Delta-VoI computation for C4/C5
# ──────────────────────────────────────────────────────────────

def value_of_information(
    problem: DecisionProblem,
    b: np.ndarray,
    ch: Channel,
) -> float:
    """VoI(ch | b) = E_s[V(posterior_s)] - V(b)"""
    ev = 0.0
    for s in range(ch.n_signals):
        p_s = signal_probability(b, ch.kernel[s])
        if p_s < EPS:
            continue
        b_s = posterior(b, ch.kernel[s])
        ev += p_s * problem.terminal_value(b_s)
    return ev - problem.terminal_value(b)


def delta_voi(
    problem: DecisionProblem,
    b: np.ndarray,
    M: Channel,
    H: Channel,
) -> float:
    """
    Delta-VoI(H | M, b) = E_m[VoI(H | b_m)] - VoI(H | b)

    Positive = complements, Negative = substitutes.
    """
    voi_h_before = value_of_information(problem, b, H)

    voi_h_after = 0.0
    for m_idx in range(M.n_signals):
        p_m = signal_probability(b, M.kernel[m_idx])
        if p_m < EPS:
            continue
        b_m = posterior(b, M.kernel[m_idx])
        voi_h_after += p_m * value_of_information(problem, b_m, H)

    return voi_h_after - voi_h_before


# ──────────────────────────────────────────────────────────────
# Finite-sample estimation
# ──────────────────────────────────────────────────────────────

def generate_samples(
    prior: np.ndarray,
    M: Channel,
    H: Channel,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (y, m, h) triples from the joint distribution."""
    K = len(prior)
    y = rng.choice(K, size=n_samples, p=prior)
    m = np.array([rng.choice(M.n_signals, p=M.kernel[:, yi]) for yi in y])
    h = np.array([rng.choice(H.n_signals, p=H.kernel[:, yi]) for yi in y])
    return y, m, h


def estimate_channel_from_samples(
    signals: np.ndarray,
    labels: np.ndarray,
    n_signals: int,
    n_states: int,
    alpha: float = 1.0,  # Dirichlet smoothing
) -> np.ndarray:
    """Estimate P(signal | state) from (signal, label) pairs with Laplace smoothing."""
    counts = np.full((n_signals, n_states), alpha)
    for s, y in zip(signals, labels):
        counts[s, y] += 1.0
    return counts / counts.sum(axis=0, keepdims=True)


def estimate_br_from_samples(
    problem: DecisionProblem,
    M: Channel,
    H: Channel,
    n_samples: int,
    rng: np.random.Generator,
    n_folds: int = 5,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate BR from finite samples using cross-fitting.

    Returns:
        model_beliefs: (n_samples, K) estimated model beliefs
        br_true:       (n_samples,) exact BR for each instance
        br_hat:        (n_samples,) estimated BR for each instance
        ll_gain:       (n_samples,) estimated log-loss gain for each instance
    """
    prior = problem.prior
    K = problem.n_states
    y_all, m_all, h_all = generate_samples(prior, M, H, n_samples, rng)

    # Cross-fitting: split into folds
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    model_beliefs = np.zeros((n_samples, K))
    augmented_beliefs = np.zeros((n_samples, K))

    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        # Estimate channels from training fold
        M_hat = estimate_channel_from_samples(
            m_all[train_idx], y_all[train_idx], M.n_signals, K
        )
        H_hat = estimate_channel_from_samples(
            h_all[train_idx], y_all[train_idx], H.n_signals, K
        )

        # Estimate prior from training fold
        prior_hat = np.bincount(y_all[train_idx], minlength=K).astype(float)
        prior_hat = (prior_hat + 1.0) / (prior_hat.sum() + K)  # smoothed

        # For each test instance, compute estimated posteriors
        for i in test_idx:
            mi, hi = m_all[i], h_all[i]

            # Model belief: P(y | m)
            b_m = posterior(prior_hat, M_hat[mi])

            # Apply temperature scaling (for calibration degradation tests)
            if temperature != 1.0:
                log_b = np.log(np.clip(b_m, EPS, None))
                log_b /= temperature
                log_b -= log_b.max()  # numerical stability
                b_m = np.exp(log_b)
                b_m /= b_m.sum()

            model_beliefs[i] = b_m

            # Augmented belief: P(y | m, h)
            b_mh = joint_posterior(prior_hat, M_hat[mi], H_hat[hi])
            if temperature != 1.0:
                log_b = np.log(np.clip(b_mh, EPS, None))
                log_b /= temperature
                log_b -= log_b.max()
                b_mh = np.exp(log_b)
                b_mh /= b_mh.sum()

            augmented_beliefs[i] = b_mh

    # Compute BR-hat and true BR for each instance
    br_hat = np.zeros(n_samples)
    br_true = np.zeros(n_samples)
    ll_gain = np.zeros(n_samples)

    for i in range(n_samples):
        mi, hi, yi = m_all[i], h_all[i], y_all[i]

        # --- Estimated quantities ---
        b_m_hat = model_beliefs[i]
        b_mh_hat = augmented_beliefs[i]
        a_m_hat = problem.optimal_action(b_m_hat)
        br_hat[i] = problem.boundary_regret(b_mh_hat, a_m_hat)

        # Log-loss gain (estimated)
        ll_gain[i] = np.log(np.clip(b_mh_hat[yi], EPS, None)) - np.log(np.clip(b_m_hat[yi], EPS, None))

        # --- Exact quantities ---
        b_m_exact = posterior(prior, M.kernel[mi])
        b_mh_exact = joint_posterior(prior, M.kernel[mi], H.kernel[hi])
        a_m_exact = problem.optimal_action(b_m_exact)
        br_true[i] = problem.boundary_regret(b_mh_exact, a_m_exact)

    return model_beliefs, br_true, br_hat, ll_gain


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation and p-value."""
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0, 1.0
    r, p = sp_stats.spearmanr(x, y)
    return float(r), float(p)


def false_positive_rate(
    br_true: np.ndarray,
    br_hat: np.ndarray,
    threshold: float = 1e-6,
) -> float:
    """Fraction of true-zero-BR instances where BR-hat > threshold."""
    true_zero = br_true < threshold
    if true_zero.sum() == 0:
        return float('nan')
    return float((br_hat[true_zero] > threshold).mean())


def calibration_error_ece(
    br_true: np.ndarray,
    br_hat: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error between BR-hat and true BR (binned)."""
    if len(br_hat) == 0:
        return float('nan')

    # Bin by BR-hat value
    bin_edges = np.linspace(br_hat.min() - EPS, br_hat.max() + EPS, n_bins + 1)
    ece = 0.0
    total = len(br_hat)

    for b in range(n_bins):
        mask = (br_hat >= bin_edges[b]) & (br_hat < bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        mean_hat = br_hat[mask].mean()
        mean_true = br_true[mask].mean()
        ece += mask.sum() / total * abs(mean_hat - mean_true)

    return float(ece)


# ──────────────────────────────────────────────────────────────
# Condition generators
# ──────────────────────────────────────────────────────────────

def make_channel(kernel: np.ndarray) -> Channel:
    """Create channel, normalizing rows to sum to 1 per state (columns)."""
    k = kernel.copy()
    col_sums = k.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums < EPS, 1.0, col_sums)
    k = k / col_sums
    return Channel(kernel=k)


def design_condition_c1(K: int, n_actions: int, rng: np.random.Generator):
    """
    C1: Human signal is predictive but all posteriors stay in the model action's region.
    BR = 0 everywhere.

    Strategy: Use a reward matrix where action 0 dominates in a large region,
    place the prior deep inside, and make H a weak channel that only slightly
    perturbs beliefs.
    """
    # Reward: action 0 strongly preferred in state 0, others balanced
    R = rng.uniform(0.5, 2.0, size=(n_actions, K))
    R[0, 0] = 10.0  # action 0 dominates when b_0 is large

    # Prior concentrated on state 0 (deep inside action 0's region)
    prior = np.full(K, 0.5 / (K - 1))
    prior[0] = 0.5
    prior = prior / prior.sum()

    problem = DecisionProblem(R=R, prior=prior)

    # Model channel: moderately informative
    M_kernel = rng.dirichlet(np.ones(K) * 2.0, size=3).T  # 3 signals, transpose to (signals, states)
    M_kernel = M_kernel.T
    M = make_channel(M_kernel)

    # Human channel: weakly informative (slight perturbations)
    # Small off-diagonal entries so posteriors barely move
    H_kernel = np.eye(K) * 0.4 + 0.6 / K
    # Add slight noise
    H_kernel += rng.uniform(0, 0.05, size=H_kernel.shape)
    H = make_channel(H_kernel)

    return problem, M, H, "C1: Predictive signal, no boundary crossing"


def design_condition_c2(K: int, n_actions: int, rng: np.random.Generator):
    """
    C2: Human signal crosses a decision boundary. BR > 0 for some instances.

    Strategy: Place belief near a boundary. Make H a strong binary channel
    that pushes posteriors clearly across.
    """
    # Reward matrix with clear boundaries
    R = np.zeros((n_actions, K))
    for a in range(n_actions):
        R[a, a % K] = 5.0  # each action specializes in one state
        for k in range(K):
            if k != a % K:
                R[a, k] = 1.0

    # Prior near boundary between action 0 and action 1
    prior = np.full(K, 1.0 / K)
    # Slightly favor state 0 so we're near the 0/1 boundary
    prior[0] += 0.05
    prior = prior / prior.sum()

    problem = DecisionProblem(R=R, prior=prior)

    # Model: moderate informativeness
    M_kernel = np.eye(K) * 0.5 + 0.5 / K
    M = make_channel(M_kernel)

    # Human: strong channel that clearly distinguishes state 0 vs 1
    H_kernel = np.eye(K) * 0.7 + 0.3 / K
    H = make_channel(H_kernel)

    return problem, M, H, "C2: Human signal crosses decision boundary"


def design_condition_c3(K: int, n_actions: int, rng: np.random.Generator):
    """
    C3: Human improves log-loss significantly but activates no rival action.
    High residual expertise, zero decision value.

    Strategy: A dominant "safe" action with high uniform reward. Specialized
    actions beat safe only if posterior on their state exceeds 0.96.
    M is very weakly informative (diagonal 0.05), so model beliefs ≈ prior.
    H is strongly informative (diagonal 0.85), pushing posteriors to ~0.90.
    Combined M+H max posterior stays below 0.96 → safe action always optimal.
    Log-loss gain is large: beliefs go from near-uniform to ~0.90 on one state.
    """
    R = np.zeros((n_actions, K))
    # Action 0: uniformly good (safe action) — high threshold to beat
    R[0, :] = 19.0
    # Other actions: specialized — beats safe only if b_k > 19/20 = 0.95
    for a in range(1, n_actions):
        R[a, a % K] = 20.0
        for k in range(K):
            if k != a % K:
                R[a, k] = 0.0

    prior = np.full(K, 1.0 / K)
    problem = DecisionProblem(R=R, prior=prior)

    # Model: very weakly informative — beliefs stay near prior
    M_kernel = np.eye(K) * 0.05 + 0.95 / K
    M = make_channel(M_kernel)

    # Human: strongly informative — concentrates posterior on correct state
    # Max combined M+H posterior ≈ 0.91 (K=3) to 0.90 (K=8), well below 0.95
    H_kernel = np.eye(K) * 0.85 + 0.15 / K
    H = make_channel(H_kernel)

    return problem, M, H, "C3: High log-loss gain, zero boundary regret"


def design_condition_c4(K: int, n_actions: int, rng: np.random.Generator):
    """
    C4: Model and human cross the same boundary (ASIL substitutability).
    Both channels move beliefs toward distinguishing the same pair of states.
    ΔVoI(H|M) < 0.

    Strategy: Both M and H are informative about state 0 vs not-state-0.
    They provide redundant information.
    """
    R = np.zeros((n_actions, K))
    for a in range(n_actions):
        R[a, a % K] = 5.0
        for k in range(K):
            if k != a % K:
                R[a, k] = 1.0

    prior = np.full(K, 1.0 / K)
    problem = DecisionProblem(R=R, prior=prior)

    # Both M and H distinguish state 0 from others (redundant)
    base = np.full((2, K), 0.3 / (K - 1))
    base[0, 0] = 0.7   # signal 0 → state 0 likely
    base[1, 0] = 0.3 / (K - 1)
    base[1, :] = 0.7 / (K - 1)
    base[1, 0] = 0.3 * (K - 1) / K

    # M: distinguishes state 0
    M_kernel = np.zeros((2, K))
    M_kernel[0, 0] = 0.8
    M_kernel[0, 1:] = 0.2 / (K - 1)
    M_kernel[1, 0] = 0.2
    M_kernel[1, 1:] = 0.8 / (K - 1)
    M = make_channel(M_kernel)

    # H: also distinguishes state 0 (same boundary → substitutes)
    H_kernel = np.zeros((2, K))
    H_kernel[0, 0] = 0.75
    H_kernel[0, 1:] = 0.25 / (K - 1)
    H_kernel[1, 0] = 0.25
    H_kernel[1, 1:] = 0.75 / (K - 1)
    H = make_channel(H_kernel)

    return problem, M, H, "C4: Same-boundary crossing (substitutability)"


def design_condition_c5(K: int, n_actions: int, rng: np.random.Generator):
    """
    C5: Model and human cross different boundaries (ASIL complementarity).
    ΔVoI(H|M) > 0.

    NOTE: With |A|=2 actions, complementarity is theoretically impossible
    (ASIL theorem: 2-action problems force weak substitutability).
    For |A|=2, this condition is expected to show ΔVoI ≤ 0.

    For |A|≥3: uses the exact ASIL paper parameters (verified: ΔVoI = 5/64 > 0
    at b2 = (1/4, 1/6, 7/12)).
    - R = [[12,0,3],[0,12,3],[3,3,9]] (specialized + safe action)
    - M (channel_i): distinguishes s1 from {s2,s3}
    - H (channel_j): distinguishes s2 from {s1,s3}
    - Prior = (1/4, 1/6, 7/12) — the ASIL complement region
    Extended to K>3 states and |A|>3 actions by adding neutral background.
    """
    if n_actions < 3:
        # With 2 actions, complementarity is impossible.
        R = np.zeros((n_actions, K))
        for a in range(n_actions):
            R[a, a % K] = 5.0
            for k in range(K):
                if k != a % K:
                    R[a, k] = 1.0
        prior = np.full(K, 1.0 / K)
        problem = DecisionProblem(R=R, prior=prior)
        M_kernel = np.eye(K)[:2] * 0.7 + 0.3 / K
        M = make_channel(M_kernel)
        H_kernel = np.eye(K)[:2][::-1] * 0.7 + 0.3 / K
        H = make_channel(H_kernel)
        return problem, M, H, "C5: Different-boundary (|A|=2, complement impossible)"

    # ── |A| ≥ 3: Exact ASIL paper geometry ──

    # Core 3×3 reward from ASIL paper (information_model.py)
    R = np.zeros((n_actions, K))
    R[0, 0] = 12.0;  R[0, 1] = 0.0;  R[0, 2 % K] = 3.0   # a1: specialized to s1
    R[1, 0] = 0.0;   R[1, 1] = 12.0; R[1, 2 % K] = 3.0   # a2: specialized to s2
    R[2, 0] = 3.0;   R[2, 1] = 3.0;  R[2, 2 % K] = 9.0   # a3: safe action

    # Extend to K > 3 states: neutral reward (3.0) in extra states
    for k in range(3, K):
        R[0, k] = 3.0
        R[1, k] = 3.0
        R[2, k] = 3.0

    # Extend to |A| > 3 actions: extra safe-like actions
    for a in range(3, n_actions):
        R[a, :] = 3.0
        if a < K:
            R[a, a] = 9.0  # slightly specialized

    # Prior: ASIL paper's b2 = (1/4, 1/6, 7/12) — verified complement region
    prior = np.zeros(K)
    prior[0] = 1.0 / 4.0
    prior[1] = 1.0 / 6.0
    if K >= 3:
        prior[2] = 7.0 / 12.0
    if K > 3:
        # Redistribute a small fraction to background states
        scale = 0.92
        prior[:3] *= scale
        prior[3:] = (1.0 - scale * (1.0 / 4 + 1.0 / 6 + 7.0 / 12)) / max(K - 3, 1)
        # Ensure proper normalization
        prior[3:] = max(0.01, prior[3])
    prior = prior / prior.sum()

    problem = DecisionProblem(R=R, prior=prior)

    # ASIL channel_i (M): distinguishes s1 from {s2, s3, ...}
    # Exact parameters from information_model.py
    M_kernel = np.zeros((2, K))
    M_kernel[0, 0] = 0.75   # signal 0: s1 more likely
    M_kernel[0, 1] = 0.25
    M_kernel[1, 0] = 0.25   # signal 1: s1 less likely
    M_kernel[1, 1] = 0.75
    if K >= 3:
        M_kernel[0, 2] = 0.25
        M_kernel[1, 2] = 0.75
    for k in range(3, K):
        M_kernel[0, k] = 0.25   # neutral on background states
        M_kernel[1, k] = 0.75
    M = make_channel(M_kernel)

    # ASIL channel_j (H): distinguishes s2 from {s1, s3, ...}
    # Different boundary than M → complementarity
    H_kernel = np.zeros((2, K))
    H_kernel[0, 0] = 0.25   # signal 0: s2 more likely
    H_kernel[0, 1] = 0.75
    H_kernel[1, 0] = 0.75   # signal 1: s2 less likely
    H_kernel[1, 1] = 0.25
    if K >= 3:
        H_kernel[0, 2] = 0.25
        H_kernel[1, 2] = 0.75
    for k in range(3, K):
        H_kernel[0, k] = 0.25
        H_kernel[1, k] = 0.75
    H = make_channel(H_kernel)

    return problem, M, H, "C5: Different-boundary crossing (complementarity)"


# ──────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────

@dataclass
class ConditionResult:
    name: str
    K: int
    n_actions: int
    n_samples: int
    spearman_rho: float
    spearman_p: float
    fpr: float
    ece: float
    mean_br_true: float
    mean_br_hat: float
    frac_br_positive_true: float
    frac_br_positive_hat: float
    spearman_ll_vs_br: float
    spearman_ll_vs_br_p: float
    # ASIL metrics (C4/C5 only)
    delta_voi_value: Optional[float] = None
    delta_voi_sign_correct: Optional[bool] = None
    # Calibration sensitivity
    sensitivity: dict = field(default_factory=dict)


def run_condition(
    condition_fn,
    K: int,
    n_actions: int,
    n_samples: int,
    seed: int,
    temperatures: list[float] | None = None,
    is_asil: bool = False,
    expected_sign: int = 0,  # +1 for complement, -1 for substitute
) -> ConditionResult:
    """Run a single condition and return metrics."""
    if temperatures is None:
        temperatures = [0.5, 0.8, 1.0, 1.2, 2.0]

    rng = np.random.default_rng(seed)
    problem, M, H, name = condition_fn(K, n_actions, rng)

    # --- Run estimation at T=1.0 (no miscalibration) ---
    rng_est = np.random.default_rng(seed + 1000)
    _, br_true, br_hat, ll_gain = estimate_br_from_samples(
        problem, M, H, n_samples, rng_est, temperature=1.0
    )

    rho, rho_p = spearman_corr(br_true, br_hat)
    fpr = false_positive_rate(br_true, br_hat)
    ece = calibration_error_ece(br_true, br_hat)
    rho_ll, rho_ll_p = spearman_corr(ll_gain, br_true)

    result = ConditionResult(
        name=name,
        K=K,
        n_actions=n_actions,
        n_samples=n_samples,
        spearman_rho=rho,
        spearman_p=rho_p,
        fpr=fpr,
        ece=ece,
        mean_br_true=float(br_true.mean()),
        mean_br_hat=float(br_hat.mean()),
        frac_br_positive_true=float((br_true > 1e-6).mean()),
        frac_br_positive_hat=float((br_hat > 1e-6).mean()),
        spearman_ll_vs_br=rho_ll,
        spearman_ll_vs_br_p=rho_ll_p,
    )

    # --- ASIL sign check (C4/C5) ---
    if is_asil:
        dv = delta_voi(problem, problem.prior, M, H)
        result.delta_voi_value = dv
        result.delta_voi_sign_correct = (
            (dv < 0 and expected_sign < 0) or
            (dv > 0 and expected_sign > 0) or
            (abs(dv) < 1e-10 and expected_sign == 0)
        )

    # --- Calibration sensitivity ---
    sensitivity = {}
    for T in temperatures:
        rng_t = np.random.default_rng(seed + int(T * 1000))
        _, br_true_t, br_hat_t, _ = estimate_br_from_samples(
            problem, M, H, n_samples, rng_t, temperature=T
        )
        rho_t, _ = spearman_corr(br_true_t, br_hat_t)
        fpr_t = false_positive_rate(br_true_t, br_hat_t)
        sensitivity[f"T={T:.1f}"] = {"spearman": rho_t, "fpr": fpr_t}

    result.sensitivity = sensitivity

    return result


def run_all_experiments(
    n_samples: int = 10000,
    base_seed: int = 42,
) -> list[ConditionResult]:
    """Run all conditions across (K, |A|) configurations."""

    configs = [
        (3, 2), (3, 3),
        (5, 3), (5, 5),
        (8, 3), (8, 5),
    ]

    conditions = [
        (design_condition_c1, False, 0),
        (design_condition_c2, False, 0),
        (design_condition_c3, False, 0),
        (design_condition_c4, True, -1),   # substitutes: ΔVoI < 0
        (design_condition_c5, True, +1),   # complements: ΔVoI > 0 (only for |A|≥3)
    ]

    results = []
    total = len(configs) * len(conditions)
    done = 0

    for K, n_actions in configs:
        for cond_idx, (cond_fn, is_asil, expected_sign) in enumerate(conditions):
            # C5 with |A|=2: complementarity is impossible, expect ΔVoI ≤ 0
            actual_expected_sign = expected_sign
            if cond_fn == design_condition_c5 and n_actions < 3:
                actual_expected_sign = -1  # substitutes expected

            seed = base_seed + K * 100 + n_actions * 10 + cond_idx
            done += 1
            print(f"  [{done}/{total}] K={K}, |A|={n_actions}, ", end="", flush=True)

            result = run_condition(
                cond_fn, K, n_actions, n_samples, seed,
                is_asil=is_asil,
                expected_sign=actual_expected_sign,
            )
            print(f"{result.name}")
            results.append(result)

    return results


# ──────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────

def print_results(results: list[ConditionResult]) -> None:
    """Print formatted results tables."""

    # Group by condition
    condition_names = []
    seen = set()
    for r in results:
        cname = r.name.split(":")[0]
        if cname not in seen:
            condition_names.append(cname)
            seen.add(cname)

    print("\n" + "=" * 90)
    print("EXPERIMENT 1: ESTIMATOR AND GEOMETRY VALIDATION — RESULTS")
    print("=" * 90)

    # ── Table 1: Core metrics per condition (averaged across configs) ──
    print("\n── Table 1: Core Metrics (averaged across K, |A| configurations) ──\n")
    print(f"{'Condition':<8} {'Spearman ρ':>12} {'FPR':>8} {'ECE':>8} "
          f"{'%BR>0 true':>12} {'%BR>0 hat':>12} {'ρ(LL,BR)':>10}")
    print("-" * 80)

    for cname in condition_names:
        cond_results = [r for r in results if r.name.startswith(cname)]
        avg_rho = np.mean([r.spearman_rho for r in cond_results])
        avg_fpr = np.nanmean([r.fpr for r in cond_results])
        avg_ece = np.mean([r.ece for r in cond_results])
        avg_pos_true = np.mean([r.frac_br_positive_true for r in cond_results])
        avg_pos_hat = np.mean([r.frac_br_positive_hat for r in cond_results])
        avg_ll = np.mean([r.spearman_ll_vs_br for r in cond_results])
        print(f"{cname:<8} {avg_rho:>12.3f} {avg_fpr:>8.3f} {avg_ece:>8.4f} "
              f"{avg_pos_true:>12.1%} {avg_pos_hat:>12.1%} {avg_ll:>10.3f}")

    # ── Table 2: Detailed per-configuration results ──
    print("\n── Table 2: Detailed Results Per Configuration ──\n")
    print(f"{'Condition':<5} {'K':>3} {'|A|':>4} {'ρ(BR̂,BR)':>10} {'p-val':>10} "
          f"{'FPR':>8} {'ECE':>8} {'μ(BR)':>8} {'μ(BR̂)':>8} {'ΔVoI':>8} {'Sign✓':>6}")
    print("-" * 95)

    for r in results:
        cname = r.name.split(":")[0]
        dvoi = f"{r.delta_voi_value:.4f}" if r.delta_voi_value is not None else "—"
        sign = "✓" if r.delta_voi_sign_correct is True else ("✗" if r.delta_voi_sign_correct is False else "—")
        print(f"{cname:<5} {r.K:>3} {r.n_actions:>4} {r.spearman_rho:>10.3f} {r.spearman_p:>10.2e} "
              f"{r.fpr:>8.3f} {r.ece:>8.4f} {r.mean_br_true:>8.4f} {r.mean_br_hat:>8.4f} "
              f"{dvoi:>8} {sign:>6}")

    # ── Table 3: Calibration sensitivity ──
    print("\n── Table 3: Calibration Sensitivity (Spearman ρ at different temperatures) ──\n")
    temps = ["T=0.5", "T=0.8", "T=1.0", "T=1.2", "T=2.0"]
    header = f"{'Condition':<5} {'K':>3} {'|A|':>4} " + " ".join(f"{t:>8}" for t in temps)
    print(header)
    print("-" * len(header))

    for r in results:
        cname = r.name.split(":")[0]
        vals = " ".join(f"{r.sensitivity.get(t, {}).get('spearman', float('nan')):>8.3f}" for t in temps)
        print(f"{cname:<5} {r.K:>3} {r.n_actions:>4} {vals}")

    # ── Table 4: FPR sensitivity ──
    print("\n── Table 4: FPR Sensitivity at Different Temperatures ──\n")
    header = f"{'Condition':<5} {'K':>3} {'|A|':>4} " + " ".join(f"{t:>8}" for t in temps)
    print(header)
    print("-" * len(header))

    for r in results:
        cname = r.name.split(":")[0]
        vals = " ".join(f"{r.sensitivity.get(t, {}).get('fpr', float('nan')):>8.3f}" for t in temps)
        print(f"{cname:<5} {r.K:>3} {r.n_actions:>4} {vals}")

    # ── Summary verdicts ──
    print("\n── Summary Verdicts ──\n")

    # C1/C3: should have low FPR
    c1_results = [r for r in results if r.name.startswith("C1")]
    c3_results = [r for r in results if r.name.startswith("C3")]
    zero_br_results = c1_results + c3_results
    avg_fpr_zero = np.nanmean([r.fpr for r in zero_br_results])
    print(f"C1+C3 (true BR≈0): avg FPR = {avg_fpr_zero:.3f}  "
          f"{'✓ PASS (<0.10)' if avg_fpr_zero < 0.10 else '✗ FAIL (≥0.10)'}")

    # C2: should have high Spearman and positive BR
    c2_results = [r for r in results if r.name.startswith("C2")]
    avg_rho_c2 = np.mean([r.spearman_rho for r in c2_results])
    print(f"C2 (boundary crossing): avg ρ = {avg_rho_c2:.3f}  "
          f"{'✓ PASS (>0.50)' if avg_rho_c2 > 0.50 else '✗ needs investigation'}")

    # C3: high LL gain but low BR
    avg_ll_c3 = np.mean([r.spearman_ll_vs_br for r in c3_results])
    avg_pos_c3 = np.mean([r.frac_br_positive_true for r in c3_results])
    print(f"C3 (expertise ≠ decision value): avg %BR>0 = {avg_pos_c3:.1%}, "
          f"avg ρ(LL,BR) = {avg_ll_c3:.3f}")

    # C4: ΔVoI should be negative (substitutes)
    c4_results = [r for r in results if r.name.startswith("C4")]
    c4_sign_ok = all(r.delta_voi_sign_correct for r in c4_results if r.delta_voi_sign_correct is not None)
    print(f"C4 (substitutability): ΔVoI < 0 in all configs: "
          f"{'✓ PASS' if c4_sign_ok else '✗ FAIL'}")

    # C5: ΔVoI should be positive for |A|≥3 (complements)
    c5_results_3plus = [r for r in results if r.name.startswith("C5") and r.n_actions >= 3]
    c5_results_2 = [r for r in results if r.name.startswith("C5") and r.n_actions < 3]
    c5_sign_ok = all(r.delta_voi_sign_correct for r in c5_results_3plus if r.delta_voi_sign_correct is not None)
    c5_2act_ok = all(
        r.delta_voi_value is not None and r.delta_voi_value <= 0
        for r in c5_results_2
    )
    print(f"C5 (complementarity, |A|≥3): ΔVoI > 0 in all configs: "
          f"{'✓ PASS' if c5_sign_ok else '✗ FAIL'}")
    print(f"C5 (|A|=2, substitution expected): ΔVoI ≤ 0: "
          f"{'✓ PASS (ASIL theorem)' if c5_2act_ok else '✗ unexpected'}")

    # Calibration degradation
    print(f"\nCalibration degradation: check Table 3 — ρ should degrade monotonically "
          f"as T departs from 1.0.")


def save_results_json(results: list[ConditionResult], path: str) -> None:
    """Save results to JSON for paper figures."""
    data = []
    for r in results:
        d = {
            "name": r.name,
            "K": r.K,
            "n_actions": r.n_actions,
            "n_samples": r.n_samples,
            "spearman_rho": r.spearman_rho,
            "spearman_p": r.spearman_p,
            "fpr": r.fpr if not np.isnan(r.fpr) else None,
            "ece": r.ece,
            "mean_br_true": r.mean_br_true,
            "mean_br_hat": r.mean_br_hat,
            "frac_br_positive_true": r.frac_br_positive_true,
            "frac_br_positive_hat": r.frac_br_positive_hat,
            "spearman_ll_vs_br": r.spearman_ll_vs_br,
            "delta_voi_value": r.delta_voi_value,
            "delta_voi_sign_correct": r.delta_voi_sign_correct,
            "sensitivity": r.sensitivity,
        }
        data.append(d)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Experiment 1: Estimator and Geometry Validation")
    print("=" * 50)
    print(f"N = 10,000 samples per condition")
    print(f"Configurations: K ∈ {{3,5,8}}, |A| ∈ {{2,3,5}}")
    print(f"Cross-fitting: 5 folds")
    print(f"Temperatures: {{0.5, 0.8, 1.0, 1.2, 2.0}}")
    print()

    results = run_all_experiments(n_samples=10000, base_seed=42)
    print_results(results)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    save_results_json(results, str(out_dir / "experiment1_results.json"))
