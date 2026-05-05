"""Tests for core boundary-regret primitives."""

import math

import numpy as np
import pytest

from core import (
    bootstrap_ci,
    boundary_regret,
    facet_distance,
    false_positive_rate,
    generate_samples,
    joint_posterior,
    log_loss_gain,
    model_action,
    optimal_value,
    posterior,
    signal_prob,
    spearman_corr,
    value_of_information,
)

# Canonical reward matrices used across tests
R_SYM = np.array([[1.0, 0.0], [0.0, 1.0]])  # symmetric, threshold 0.5
R_FP = np.array([[1.0, 0.0], [-4.0, 1.0]])  # FP penalty, threshold 5/6
R_FN = np.array([[1.0, -4.0], [0.0, 1.0]])  # FN penalty, threshold 1/6


# ---------------------------------------------------------------------------
# optimal_value
# ---------------------------------------------------------------------------


def test_optimal_value_symmetric():
    b = np.array([[0.6, 0.4], [0.3, 0.7]])
    v = optimal_value(b, R_SYM)
    np.testing.assert_allclose(v, [0.6, 0.7])


def test_optimal_value_shape():
    b = np.random.default_rng(0).dirichlet([1, 1], size=10)
    assert optimal_value(b, R_SYM).shape == (10,)


def test_optimal_value_requires_2d():
    with pytest.raises(ValueError, match="N, K"):
        optimal_value(np.array([0.5, 0.5]), R_SYM)


# ---------------------------------------------------------------------------
# model_action
# ---------------------------------------------------------------------------


def test_model_action_symmetric():
    b = np.array([[0.6, 0.4], [0.3, 0.7]])
    np.testing.assert_array_equal(model_action(b, R_SYM), [0, 1])


def test_model_action_fp_penalty_threshold():
    # R_FP threshold = 5/6 ≈ 0.833
    above = np.array([[0.05, 0.95]])  # b1 > 5/6 → admit (1)
    below = np.array([[0.50, 0.50]])  # b1 < 5/6 → discharge (0)
    assert model_action(above, R_FP)[0] == 1
    assert model_action(below, R_FP)[0] == 0


def test_model_action_requires_2d():
    with pytest.raises(ValueError):
        model_action(np.array([0.5, 0.5]), R_SYM)


# ---------------------------------------------------------------------------
# boundary_regret
# ---------------------------------------------------------------------------


def test_boundary_regret_zero_when_beliefs_equal():
    b = np.random.default_rng(1).dirichlet([1, 1], size=20)
    np.testing.assert_allclose(boundary_regret(b, b, R_SYM), 0.0, atol=1e-12)


def test_boundary_regret_positive_when_action_flips():
    # b_x → action 0, b_xh → action 1 (action flips → BR > 0)
    b_x = np.array([[0.6, 0.4]])
    b_xh = np.array([[0.3, 0.7]])
    # V(b_xh) = 0.7, r_{a_x=0} · b_xh = R_SYM[0] · [0.3, 0.7] = 0.3 → BR = 0.4
    np.testing.assert_allclose(boundary_regret(b_x, b_xh, R_SYM), [0.4], atol=1e-9)


def test_boundary_regret_non_negative():
    rng = np.random.default_rng(2)
    b_x = rng.dirichlet([1, 1], size=100)
    b_xh = rng.dirichlet([1, 1], size=100)
    assert (boundary_regret(b_x, b_xh, R_FP) >= 0).all()


def test_boundary_regret_zero_when_action_unchanged_and_no_gain():
    # b_xh deep inside the same decision region as b_x → BR = 0
    b_x = np.array([[0.9, 0.1]])  # strongly → action 0 under R_FP (b1 << 5/6)
    b_xh = np.array([[0.85, 0.15]])  # still → action 0
    br = boundary_regret(b_x, b_xh, R_FP)
    # V(b_xh) under R_FP: action 0 = 0.85, action 1 = -4*0.85 + 0.15 = -3.25 → action 0 wins
    # r_{a_x=0} · b_xh = 1*0.85 + 0*0.15 = 0.85 = V(b_xh) → BR = 0
    np.testing.assert_allclose(br, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# log_loss_gain
# ---------------------------------------------------------------------------


def test_log_loss_gain_positive_when_b_xh_closer_to_truth():
    y = np.array([1])
    b_x = np.array([[0.4, 0.6]])
    b_xh = np.array([[0.2, 0.8]])  # more confident on true class
    assert log_loss_gain(b_x, b_xh, y)[0] > 0


def test_log_loss_gain_negative_when_b_xh_worse():
    y = np.array([1])
    b_x = np.array([[0.3, 0.7]])
    b_xh = np.array([[0.6, 0.4]])  # overconfident on wrong class
    assert log_loss_gain(b_x, b_xh, y)[0] < 0


def test_log_loss_gain_zero_when_beliefs_equal():
    y = np.array([0, 1])
    b = np.array([[0.7, 0.3], [0.2, 0.8]])
    np.testing.assert_allclose(log_loss_gain(b, b, y), 0.0, atol=1e-12)


def test_log_loss_gain_shape():
    rng = np.random.default_rng(3)
    b_x = rng.dirichlet([1, 1], size=30)
    b_xh = rng.dirichlet([1, 1], size=30)
    y = rng.integers(0, 2, size=30)
    assert log_loss_gain(b_x, b_xh, y).shape == (30,)


# ---------------------------------------------------------------------------
# facet_distance
# ---------------------------------------------------------------------------


def test_facet_distance_zero_on_boundary():
    b = np.array([[0.5, 0.5]])  # exactly on the R_SYM boundary
    np.testing.assert_allclose(facet_distance(b, R_SYM), 0.0, atol=1e-12)


def test_facet_distance_at_extreme():
    b = np.array([[1.0, 0.0]])
    # scores: [1.0, 0.0] → margin = 1.0
    np.testing.assert_allclose(facet_distance(b, R_SYM), [1.0], atol=1e-9)


def test_facet_distance_non_negative():
    rng = np.random.default_rng(4)
    b = rng.dirichlet([1, 1], size=200)
    assert (facet_distance(b, R_FP) >= 0).all()


def test_facet_distance_decreases_near_boundary():
    # For R_SYM, threshold at 0.5; b closer to 0.5 should have smaller margin
    b_far = np.array([[0.9, 0.1]])  # far from 0.5
    b_near = np.array([[0.6, 0.4]])  # near 0.5
    assert facet_distance(b_far, R_SYM)[0] > facet_distance(b_near, R_SYM)[0]


# ---------------------------------------------------------------------------
# posterior
# ---------------------------------------------------------------------------


def test_posterior_basic():
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.8, 0.2])
    p = posterior(prior, likelihood)
    np.testing.assert_allclose(p, [0.8, 0.2], atol=1e-9)


def test_posterior_sums_to_one():
    prior = np.array([0.3, 0.5, 0.2])
    likelihood = np.array([0.1, 0.4, 0.5])
    assert abs(posterior(prior, likelihood).sum() - 1.0) < 1e-12


def test_posterior_fallback_on_impossible_signal():
    prior = np.array([0.4, 0.6])
    likelihood = np.array([0.0, 0.0])
    np.testing.assert_array_equal(posterior(prior, likelihood), prior)


# ---------------------------------------------------------------------------
# joint_posterior
# ---------------------------------------------------------------------------


def test_joint_posterior_matches_sequential_update():
    # M ⊥ H | Y: joint == sequential is equivalent under this assumption
    prior = np.array([0.5, 0.5])
    lik_m = np.array([0.8, 0.2])
    lik_h = np.array([0.6, 0.4])
    joint = joint_posterior(prior, lik_m, lik_h)
    sequential = posterior(posterior(prior, lik_m), lik_h)
    np.testing.assert_allclose(joint, sequential, atol=1e-9)


def test_joint_posterior_sums_to_one():
    prior = np.array([0.3, 0.7])
    p = joint_posterior(prior, np.array([0.9, 0.1]), np.array([0.5, 0.5]))
    assert abs(p.sum() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# signal_prob
# ---------------------------------------------------------------------------


def test_signal_prob_known_value():
    # P(signal) = 0.8*0.5 + 0.2*0.5 = 0.5
    p = signal_prob(np.array([0.5, 0.5]), np.array([0.8, 0.2]))
    np.testing.assert_allclose(p, 0.5, atol=1e-9)


def test_signal_prob_sums_correctly_over_signals():
    # Sum over all signals must equal 1
    prior = np.array([0.4, 0.6])
    kernel = np.array([[0.7, 0.3], [0.3, 0.7]])  # (2, 2) channel
    total = sum(signal_prob(prior, kernel[s]) for s in range(2))
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# value_of_information
# ---------------------------------------------------------------------------


def test_value_of_information_non_negative():
    rng = np.random.default_rng(5)
    prior = rng.dirichlet([1, 1])
    # (S=3, K=2): dirichlet([1,1,1], size=2) gives (2, 3) rows-sum-to-1; .T gives (3, 2) cols-sum-to-1
    kernel = rng.dirichlet([1, 1, 1], size=2).T
    assert value_of_information(prior, kernel, R_SYM) >= -1e-9


def test_value_of_information_zero_for_uninformative_channel():
    # Uniform channel: P(s|y) = 1/S for all y → signal carries no information
    prior = np.array([0.4, 0.6])
    kernel = np.full((4, 2), 0.25)  # (4 signals, 2 states) uniform
    np.testing.assert_allclose(
        value_of_information(prior, kernel, R_SYM), 0.0, atol=1e-9
    )


def test_value_of_information_positive_for_informative_channel():
    prior = np.array([0.5, 0.5])
    kernel = np.eye(2)  # perfect channel: signal reveals state
    voi = value_of_information(prior, kernel, R_SYM)
    # With perfect info V(b_s) = 1.0 always, V(prior) = 0.5 → VoI = 0.5
    np.testing.assert_allclose(voi, 0.5, atol=1e-9)


# ---------------------------------------------------------------------------
# generate_samples
# ---------------------------------------------------------------------------


def test_generate_samples_output_shapes():
    rng = np.random.default_rng(42)
    prior = np.array([0.3, 0.4, 0.3])
    M = np.array([[0.7, 0.2, 0.1], [0.3, 0.8, 0.9]])  # (2, 3)
    H = np.array([[0.6, 0.1, 0.3], [0.4, 0.9, 0.7]])  # (2, 3)
    y, m, h = generate_samples(prior, M, H, n=200, rng=rng)
    assert y.shape == m.shape == h.shape == (200,)


def test_generate_samples_valid_ranges():
    rng = np.random.default_rng(0)
    prior = np.array([0.5, 0.5])
    M = H = np.array([[0.8, 0.2], [0.2, 0.8]])  # (2, 2)
    y, m, h = generate_samples(prior, M, H, n=500, rng=rng)
    assert (y >= 0).all() and (y < 2).all()
    assert (m >= 0).all() and (m < 2).all()
    assert (h >= 0).all() and (h < 2).all()


def test_generate_samples_marginal_matches_prior():
    rng = np.random.default_rng(99)
    prior = np.array([0.2, 0.5, 0.3])
    M = H = np.eye(3)  # perfect channel; marginal of y = prior
    y, _, _ = generate_samples(prior, M, H, n=20_000, rng=rng)
    empirical = np.bincount(y, minlength=3) / len(y)
    np.testing.assert_allclose(empirical, prior, atol=0.02)


# ---------------------------------------------------------------------------
# spearman_corr
# ---------------------------------------------------------------------------


def test_spearman_corr_perfectly_monotone():
    x = np.arange(10, dtype=float)
    rho, _ = spearman_corr(x, x)
    np.testing.assert_allclose(rho, 1.0, atol=1e-9)


def test_spearman_corr_anti_monotone():
    x = np.arange(10, dtype=float)
    rho, _ = spearman_corr(x, -x)
    np.testing.assert_allclose(rho, -1.0, atol=1e-9)


def test_spearman_corr_constant_input_returns_safe_fallback():
    rho, p = spearman_corr(np.ones(10), np.arange(10, dtype=float))
    assert rho == 0.0 and p == 1.0


# ---------------------------------------------------------------------------
# false_positive_rate
# ---------------------------------------------------------------------------


def test_false_positive_rate_zero():
    br_true = np.array([0.0, 0.0, 1.0])
    br_hat = np.array([0.0, 0.0, 0.5])
    np.testing.assert_allclose(
        false_positive_rate(br_true, br_hat, threshold=0.01), 0.0
    )


def test_false_positive_rate_partial():
    br_true = np.array([0.0, 0.0, 1.0])
    br_hat = np.array([1.0, 0.0, 0.5])  # one of two true-zeros is flagged
    np.testing.assert_allclose(
        false_positive_rate(br_true, br_hat, threshold=0.01), 0.5
    )


def test_false_positive_rate_nan_when_no_true_zeros():
    br_true = np.array([1.0, 2.0])
    br_hat = np.array([0.5, 1.0])
    assert math.isnan(false_positive_rate(br_true, br_hat, threshold=0.01))


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_mean_coverage():
    # 95% CI on the mean of a Gaussian should cover the true mean ~95% of the time
    rng = np.random.default_rng(7)
    covered = 0
    n_trials = 200
    for _ in range(n_trials):
        data = rng.normal(0.0, 1.0, size=60)
        lo, hi = bootstrap_ci(np.mean, data, n_boot=400, ci=0.95, rng=rng)
        covered += lo <= 0.0 <= hi
    # Generous tolerance: coverage should be at least 85%
    assert covered >= 170, f"Coverage too low: {covered}/{n_trials}"


def test_bootstrap_ci_paired_positive_correlation():
    # CI on Spearman ρ of positively correlated data should be entirely positive
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=80)
    y = x + rng.normal(0, 0.3, size=80)
    lo, hi = bootstrap_ci(
        lambda a, b: spearman_corr(a, b)[0], x, y, n_boot=500, rng=rng
    )
    assert lo > 0 and hi <= 1.0


def test_bootstrap_ci_lo_le_hi():
    rng = np.random.default_rng(11)
    data = rng.normal(0, 1, size=50)
    lo, hi = bootstrap_ci(np.mean, data, n_boot=200, rng=rng)
    assert lo <= hi
