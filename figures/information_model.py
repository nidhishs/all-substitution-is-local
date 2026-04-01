"""Decision-theoretic computations for the interior complementarity paper.
The module implements reward matrices, Bayesian updates, VoI calculations, and the Bregman force decomposition.
"""

from dataclasses import dataclass

import numpy as np

EPS = 1e-10

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InformationProblem:
    """A decision problem with two information channels over a finite state space.

    Attributes:
        reward_matrix: Shape (num_actions, num_states). Entry [a, s] is the
            payoff of action a in state s.
        channel_i_kernel: Shape (num_signals_i, num_states). Row k is the
            likelihood P(signal_k | state) for channel i.
        channel_j_kernel: Shape (num_signals_j, num_states). Row k is the
            likelihood P(signal_k | state) for channel j.
    """

    reward_matrix: np.ndarray
    channel_i_kernel: np.ndarray
    channel_j_kernel: np.ndarray


PROBLEM = InformationProblem(
    reward_matrix=np.array(
        [
            [12.0, 0.0, 3.0],  # a1: specialized to s1
            [0.0, 12.0, 3.0],  # a2: specialized to s2
            [3.0, 3.0, 9.0],  # a3: safe action
        ]
    ),
    channel_i_kernel=np.array(
        [
            [0.75, 0.25, 0.25],  # signal 0: s1 more likely
            [0.25, 0.75, 0.75],  # signal 1: s1 less likely
        ]
    ),
    channel_j_kernel=np.array(
        [
            [0.25, 0.75, 0.25],  # signal 0: s2 more likely
            [0.75, 0.25, 0.75],  # signal 1: s2 less likely
        ]
    ),
)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def terminal_value(belief: np.ndarray) -> float:
    """V(b) = max_a  r_a . b  ; the value of the best action at this belief."""
    return float(np.max(PROBLEM.reward_matrix @ belief))



def compute_posteriors(
    belief: np.ndarray,
    kernel: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    """Bayesian Update: for each signal, compute posterior belief and its probability.

    Returns (posteriors, signal_probabilities), one entry per signal with P > 0.
    """
    posteriors: list[np.ndarray] = []
    signal_probabilities: list[float] = []

    for likelihood_row in kernel:
        signal_prob = float(likelihood_row @ belief)
        if signal_prob <= EPS:
            continue  # skip signals that never fire; avoids div-by-zero
        posteriors.append(likelihood_row * belief / signal_prob)
        signal_probabilities.append(signal_prob)

    return posteriors, signal_probabilities


def expected_value_after(belief: np.ndarray, kernel: np.ndarray) -> float:
    """E[V(posterior)] = sum_s P(signal) * V(posterior), expected terminal value after observing a channel."""
    posteriors, weights = compute_posteriors(belief, kernel)
    return sum(w * terminal_value(p) for w, p in zip(weights, posteriors))


def value_of_information(belief: np.ndarray, kernel: np.ndarray) -> float:
    """VoI of a channel at belief b: E[V(posterior)] - V(b)."""
    return expected_value_after(belief, kernel) - terminal_value(belief)


def delta_voi(belief: np.ndarray) -> float:
    """Delta-VoI(j | i, b): how much consulting channel i changes channel j's value.

    Positive = complements (i makes j more valuable).
    Negative = substitutes (i makes j less valuable).
    """
    voi_j_before = value_of_information(belief, PROBLEM.channel_j_kernel)

    posteriors_i, weights_i = compute_posteriors(belief, PROBLEM.channel_i_kernel)
    voi_j_after = sum(
        w * value_of_information(p, PROBLEM.channel_j_kernel)
        for w, p in zip(weights_i, posteriors_i)
    )

    return voi_j_after - voi_j_before


def bregman_forces(belief: np.ndarray) -> tuple[float, float]:
    """Decompose delta-VoI into complement force E[D_g] and substitute force E[D_h].

    delta-VoI = complement_force - substitute_force.

    complement_force = E_i[g(posterior)] - g(b)   (Jensen gap of g)
    substitute_force = E_i[V(posterior)] - V(b)   (Jensen gap of h = V)
    """
    posteriors_i, weights_i = compute_posteriors(belief, PROBLEM.channel_i_kernel)

    substitute_force = (
        sum(w * terminal_value(p) for w, p in zip(weights_i, posteriors_i))
        - terminal_value(belief)
    )

    ev_j_at_prior = expected_value_after(belief, PROBLEM.channel_j_kernel)
    complement_force = (
        sum(
            w * expected_value_after(p, PROBLEM.channel_j_kernel)
            for w, p in zip(weights_i, posteriors_i)
        )
        - ev_j_at_prior
    )
    return complement_force, substitute_force


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_known_values() -> None:
    """Assert exact values from the paper's worked examples."""

    # b1 = (1/11, 2/11, 8/11): Theorem 2 -- channel i is decision-irrelevant
    belief_thm2 = np.array([1 / 11, 2 / 11, 8 / 11])
    assert abs(delta_voi(belief_thm2) - 3 / 176) < EPS
    voi_i = value_of_information(belief_thm2, PROBLEM.channel_i_kernel)
    assert abs(voi_i) < EPS, "VoI(i) should be 0"
    complement, substitute = bregman_forces(belief_thm2)
    assert abs(substitute) < EPS, "E[D_h] should be 0"
    assert complement > 0, "E[D_g] should be > 0"

    # b2 = (1/4, 1/6, 7/12): Decomposition example
    belief_star = np.array([1 / 4, 1 / 6, 7 / 12])
    assert abs(delta_voi(belief_star) - 5 / 64) < EPS
    assert (
        abs(value_of_information(belief_star, PROBLEM.channel_j_kernel) - 1 / 16)
        < EPS
    )

    # b3 = (5/12, 5/12, 1/6): Substitution example
    belief_sub = np.array([5 / 12, 5 / 12, 1 / 6])
    assert abs(delta_voi(belief_sub) - (-77 / 32)) < EPS

    # Triple point: all actions tied at (1/3, 1/3, 1/3)
    vals = PROBLEM.reward_matrix @ np.array([1 / 3, 1 / 3, 1 / 3])
    assert abs(vals[0] - vals[1]) < EPS
    assert abs(vals[0] - vals[2]) < EPS

    print("All verification checks passed.")
