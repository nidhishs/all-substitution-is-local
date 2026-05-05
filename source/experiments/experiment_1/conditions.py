"""Condition designs for Experiment 1.

Three roles: C1 positive control (BR > 0), C2 weak null (BR near 0 via prior),
C3 strong null (BR = 0 by construction despite informative h).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import core


@dataclass
class Channel:
    kernel: np.ndarray  # (S, K), columns sum to 1


@dataclass
class DecisionProblem:
    prior: np.ndarray  # (K,)
    R: np.ndarray  # (|A|, K)


def _group_of_state(K: int, G: int) -> np.ndarray:
    """Integer group index for each of K states split into G equal groups."""
    return np.arange(K) * G // K


def _group_channel_kernel(K: int, G: int, epsilon: float) -> np.ndarray:
    """(G, K) channel kernel where P(signal=g | state=k) = 1-ε if group(k)==g else ε/(G-1).

    Requires G >= 2. Columns sum to 1.
    """
    groups = _group_of_state(K, G)
    off = epsilon / (G - 1)
    kernel = np.full((G, K), off)
    for g in range(G):
        kernel[g, groups == g] = 1.0 - epsilon
    return kernel


def _group_reward_matrix(
    K: int, G: int, on: float = 5.0, off: float = 1.0
) -> np.ndarray:
    """(G, K) reward matrix: R[g, k] = on if group(k)==g else off."""
    groups = _group_of_state(K, G)
    R = np.full((G, K), off)
    for g in range(G):
        R[g, groups == g] = on
    return R


def _max_reachable_group_posterior(
    prior: np.ndarray, M: Channel, H: Channel, G: int
) -> float:
    """Max group mass over all reachable (m, h) joint posteriors."""
    K = len(prior)
    groups = _group_of_state(K, G)
    max_mass = 0.0
    for m_s in range(M.kernel.shape[0]):
        for h_s in range(H.kernel.shape[0]):
            b = core.joint_posterior(prior, M.kernel[m_s], H.kernel[h_s])
            for g in range(G):
                max_mass = max(max_mass, float(b[groups == g].sum()))
    return max_mass


def design_c1(K: int, G: int) -> tuple[DecisionProblem, Channel, Channel]:
    """C1 positive control: augmented posteriors cross decision boundaries."""
    prior = np.ones(K) / K
    M = Channel(_group_channel_kernel(K, G, epsilon=0.30))
    H = Channel(_group_channel_kernel(K, G, epsilon=0.15))
    R = _group_reward_matrix(K, G)
    return DecisionProblem(prior=prior, R=R), M, H


def design_c2(K: int, G: int) -> tuple[DecisionProblem, Channel, Channel]:
    """C2 weak null: concentrated prior + weak channels keep posteriors in group 0's region."""
    groups = _group_of_state(K, G)
    prior = np.zeros(K)
    n_g0 = int((groups == 0).sum())
    prior[groups == 0] = 0.97 / n_g0
    n_other = K - n_g0
    prior[groups != 0] = 0.03 / n_other
    M = Channel(_group_channel_kernel(K, G, epsilon=0.40))
    H = Channel(_group_channel_kernel(K, G, epsilon=0.35))
    R = _group_reward_matrix(K, G)
    return DecisionProblem(prior=prior, R=R), M, H


def design_c3(K: int, G: int) -> tuple[DecisionProblem, Channel, Channel]:
    """C3 strong null: safe action dominates everywhere despite highly informative h.

    τ is derived from the channel geometry: midpoint between max reachable group
    mass and 1. This certifies that no (m, h) posterior can make any risky action
    beat the safe action, so BR ≡ 0 by construction.
    """
    prior = np.ones(K) / K
    M = Channel(_group_channel_kernel(K, G, epsilon=0.20))
    H = Channel(_group_channel_kernel(K, G, epsilon=0.01))

    max_reachable = _max_reachable_group_posterior(prior, M, H, G)
    tau = 0.5 * (1.0 + max_reachable)

    groups = _group_of_state(K, G)
    R = np.zeros((G + 1, K))
    R[0, :] = tau
    for g in range(G):
        R[g + 1, groups == g] = 1.0

    return DecisionProblem(prior=prior, R=R), M, H
