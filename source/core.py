"""Boundary-regret theory primitives."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import norm as sp_norm

_EPS = 1e-15


# ---------------------------------------------------------------------------
# Decision theory - vectorised over (N, K) belief arrays
# ---------------------------------------------------------------------------


def optimal_value(b: np.ndarray, R: np.ndarray) -> np.ndarray:
    """V(b) = max_a R[a] . b.

    Args:
        b: Belief array of shape (N, K).
        R: Reward matrix of shape (|A|, K).

    Returns:
        Array of shape (N,) containing the optimal value for each instance.
    """
    if b.ndim != 2:
        raise ValueError(f"b must be (N, K), got shape {b.shape}")
    return (b @ R.T).max(axis=-1)


def model_action(b: np.ndarray, R: np.ndarray) -> np.ndarray:
    """argmax_a R[a] . b.

    Args:
        b: Belief array of shape (N, K).
        R: Reward matrix of shape (|A|, K).

    Returns:
        Integer array of shape (N,) containing the optimal action index for each instance.
    """
    if b.ndim != 2:
        raise ValueError(f"b must be (N, K), got shape {b.shape}")
    return (b @ R.T).argmax(axis=-1)


def reward_tolerance(R: np.ndarray) -> float:
    """Scale-aware tolerance for "BR is essentially zero" decisions.

    Returns max(1e-12, 1e-8 * reward_range) so that callers comparing per-instance
    BR against `tol` get the same threshold regardless of the reward matrix scale.
    """
    return max(1e-12, 1e-8 * float(R.max() - R.min()))


def boundary_regret(b_x: np.ndarray, b_xh: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Per-instance BR = max(0, V(b_xh) - r_{a_x} . b_xh).

    a_x = model_action(b_x, R) is the model-only action before observing h.

    Args:
        b_x: Model-only belief array of shape (N, K).
        b_xh: Joint model-and-human belief array of shape (N, K).
        R: Reward matrix of shape (|A|, K).

    Returns:
        Array of shape (N,) containing per-instance boundary regret, >= 0.
    """
    a_x = model_action(b_x, R)  # (N,)
    v_xh = optimal_value(b_xh, R)  # (N,)
    r_ax = R[a_x]  # (N, K)
    reward = (r_ax * b_xh).sum(axis=-1)  # (N,)
    return np.maximum(v_xh - reward, 0.0)


def log_loss_gain(b_x: np.ndarray, b_xh: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log P(y | b_xh) - log P(y | b_x). Positive means b_xh is more informative.

    Args:
        b_x: Model-only belief array of shape (N, K).
        b_xh: Joint model-and-human belief array of shape (N, K).
        y: True label array of shape (N,), integer values in [0, K).

    Returns:
        Array of shape (N,) containing per-instance log-loss gain.
    """
    idx = np.arange(len(y))
    p_x = np.clip(b_x[idx, y], _EPS, 1.0)
    p_xh = np.clip(b_xh[idx, y], _EPS, 1.0)
    return np.log(p_xh) - np.log(p_x)


def facet_distance(b: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Margin from the nearest reward-induced decision boundary.

    Equal to best_action_score - second_best_action_score for each instance.
    Zero means b lies exactly on a decision boundary facet.

    Args:
        b: Belief array of shape (N, K).
        R: Reward matrix of shape (|A|, K).

    Returns:
        Array of shape (N,) containing per-instance facet distance, >= 0.
    """
    if b.ndim != 2:
        raise ValueError(f"b must be (N, K), got shape {b.shape}")
    scores = b @ R.T  # (N, |A|)
    N = len(b)
    a_star = scores.argmax(axis=-1)  # (N,)
    best = scores[np.arange(N), a_star]  # (N,)
    masked = scores.copy()
    masked[np.arange(N), a_star] = -np.inf
    second_best = masked.max(axis=-1)  # (N,)
    return np.maximum(best - second_best, 0.0)


# ---------------------------------------------------------------------------
# Bayesian primitives - single-instance
# ---------------------------------------------------------------------------


def posterior(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """P(y | signal) proportional to likelihood . prior. Falls back to prior if P(signal) ~= 0."""
    joint = likelihood * prior
    total = joint.sum()
    return joint / total if total >= _EPS else prior.copy()


def joint_posterior(
    prior: np.ndarray, lik_m: np.ndarray, lik_h: np.ndarray
) -> np.ndarray:
    """P(y | m, h) proportional to P(m|y) P(h|y) P(y). Requires M independent H | Y."""
    joint = lik_m * lik_h * prior
    total = joint.sum()
    return joint / total if total >= _EPS else prior.copy()


def signal_prob(prior: np.ndarray, likelihood: np.ndarray) -> float:
    """P(signal) = sum_y P(signal|y) P(y)."""
    return float(likelihood @ prior)


def value_of_information(
    b: np.ndarray, channel_kernel: np.ndarray, R: np.ndarray
) -> float:
    """VoI(channel | b) = E_s[V(posterior(b, kernel[s]))] - V(b).

    Args:
        b: Current belief vector of shape (K,).
        channel_kernel: Kernel matrix of shape (S, K) where kernel[s, k] = P(signal=s | state=k).
        R: Reward matrix of shape (|A|, K).

    Returns:
        Scalar float representing the value of information for the channel given belief b.
    """
    v_prior = float((R @ b).max())
    ev = 0.0
    for s in range(channel_kernel.shape[0]):
        p_s = signal_prob(b, channel_kernel[s])
        if p_s >= _EPS:
            b_s = posterior(b, channel_kernel[s])
            ev += p_s * float((R @ b_s).max())
    return ev - v_prior


# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------


def generate_samples(
    prior: np.ndarray,
    M: np.ndarray,
    H: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw (y, m, h) from P(y, m, h) = P(y) P(m|y) P(h|y).

    Args:
        prior: Prior distribution of shape (K,).
        M: Model signal kernel of shape (S_m, K) where M[s, k] = P(model_signal=s | state=k).
        H: Human signal kernel of shape (S_h, K) where H[s, k] = P(human_signal=s | state=k).
        n: Number of samples to draw.
        rng: NumPy random generator instance.

    Returns:
        Tuple (y, m, h) where each element is an integer array of shape (n,).
    """
    K = len(prior)
    y = rng.choice(K, size=n, p=prior)
    m = np.array([rng.choice(M.shape[0], p=M[:, yi]) for yi in y])
    h = np.array([rng.choice(H.shape[0], p=H[:, yi]) for yi in y])
    return y, m, h


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and two-sided p-value. Returns (0.0, 1.0) for constant inputs."""
    if np.std(x) < _EPS or np.std(y) < _EPS:
        return 0.0, 1.0
    result = sp_stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def false_positive_rate(
    br_true: np.ndarray, br_hat: np.ndarray, threshold: float
) -> float:
    """Fraction of true-zero-BR instances where BR_hat > threshold.

    Returns nan when there are no true-zero instances.
    threshold should be chosen relative to the reward scale.

    Args:
        br_true: Ground-truth boundary regret array of shape (N,).
        br_hat: Estimated boundary regret array of shape (N,).
        threshold: Decision threshold relative to the reward scale.

    Returns:
        Scalar float false positive rate, or nan if there are no true-zero instances.
    """
    true_zero = br_true < threshold
    if true_zero.sum() == 0:
        return float("nan")
    return float((br_hat[true_zero] > threshold).mean())


def _bca_ci(
    theta_hat: float,
    boot: np.ndarray,
    jack: np.ndarray,
    ci: float,
) -> tuple[float, float]:
    """BCa bias-correction, acceleration, and percentile interval from boot/jack arrays."""
    jack_mean = jack.mean()
    diff = jack_mean - jack
    denom = (diff**2).sum()
    a = (diff**3).sum() / (6.0 * denom**1.5) if denom > _EPS else 0.0

    frac = float(np.clip((boot < theta_hat).mean(), _EPS, 1.0 - _EPS))
    z0 = float(sp_norm.ppf(frac))

    alpha = 1.0 - ci
    z_lo = float(sp_norm.ppf(alpha / 2.0))
    z_hi = float(sp_norm.ppf(1.0 - alpha / 2.0))

    def _p(z_a: float) -> float:
        w = z0 + z_a  # BCa inner argument (Efron 1987)
        return float(sp_norm.cdf(z0 + w / (1.0 - a * w)))

    return float(np.percentile(boot, 100.0 * _p(z_lo))), float(
        np.percentile(boot, 100.0 * _p(z_hi))
    )


def _batch_pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r for each row pair in x (B, N) and y (B, N).

    Returns array of shape (B,). Rows with zero variance return 0.
    """
    xm = x - x.mean(axis=1, keepdims=True)
    ym = y - y.mean(axis=1, keepdims=True)
    num = (xm * ym).sum(axis=1)
    denom = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1))
    return np.where(denom > _EPS, num / denom, 0.0)


def spearman_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """BCa bootstrap CI for Spearman rho, vectorized.

    Equivalent to bootstrap_ci(lambda a, b: spearman_corr(a, b)[0], x, y, ...) for
    non-tied data. For tied data the bootstrap replicates use ordinal ranking while
    the point estimate uses average ranking — the two may diverge slightly.
    Generates all bootstrap index rows in one call and computes Pearson r on
    re-ranked samples in batch. Jackknife LOO correlations are computed via a
    rank-adjustment matrix rather than N sequential scipy calls.

    Args:
        x: First array of shape (N,).
        y: Second array of shape (N,).
        n_boot: Number of bootstrap replicates.
        ci: Coverage level (e.g. 0.95 for 95%).
        rng: NumPy random generator. If None, a default generator is created.

    Returns:
        Tuple (lower, upper) bounds of the ci-level BCa confidence interval.
    """
    from scipy.stats import rankdata as _rankdata

    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    rx = _rankdata(x).astype(np.float64)
    ry = _rankdata(y).astype(np.float64)

    # Point estimate via Pearson r on ranks.
    theta_hat = float(_batch_pearsonr(rx[None], ry[None])[0])

    # --- Vectorized bootstrap ---
    # Consuming the RNG in one call produces the same sequence as n_boot sequential
    # calls of size n (PCG bit-stream is order-independent of array shape).
    idx = rng.integers(0, n, size=(n_boot, n))  # (n_boot, n)
    x_b = x[idx]  # (n_boot, n) — resample original values then re-rank
    y_b = y[idx]
    # Ordinal ranking via double-argsort; matches scipy average ranking for non-tied data.
    rx_b = x_b.argsort(axis=1).argsort(axis=1).astype(np.float64) + 1
    ry_b = y_b.argsort(axis=1).argsort(axis=1).astype(np.float64) + 1
    boot = _batch_pearsonr(rx_b, ry_b)  # (n_boot,)

    # --- Vectorized jackknife ---
    # When element i is removed, rank of j shifts down by 1 iff rx[j] > rx[i].
    rx_adj = rx[None, :] - (rx[None, :] > rx[:, None]).astype(np.float64)  # (n, n)
    ry_adj = ry[None, :] - (ry[None, :] > ry[:, None]).astype(np.float64)
    eye = np.eye(n, dtype=bool)
    rx_loo = rx_adj[~eye].reshape(n, n - 1)  # (n, n-1)
    ry_loo = ry_adj[~eye].reshape(n, n - 1)
    jack = _batch_pearsonr(rx_loo, ry_loo)  # (n,)

    return _bca_ci(theta_hat, boot, jack, ci)


def mean_bootstrap_ci(
    x: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """BCa bootstrap CI for the mean of x, fully vectorized.

    Drop-in replacement for bootstrap_ci(lambda a: float(a.mean()), x, ...).
    Bootstrap means are computed in one matrix operation. The LOO jackknife
    mean has a closed form — jack[i] = (N·mean - x[i]) / (N-1) — so the
    N-iteration loop is eliminated entirely.

    Args:
        x: Input array of shape (N,).
        n_boot: Number of bootstrap replicates.
        ci: Coverage level (e.g. 0.95 for 95%).
        rng: NumPy random generator. If None, a default generator is created.

    Returns:
        Tuple (lower, upper) bounds of the ci-level BCa confidence interval.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    theta_hat = float(x.mean())

    # --- Vectorized bootstrap ---
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = x[idx].mean(axis=1)  # (n_boot,)

    # --- Closed-form jackknife LOO means ---
    jack = (n * theta_hat - x) / (n - 1)  # (n,)

    return _bca_ci(theta_hat, boot, jack, ci)


def bootstrap_ci(
    stat_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """BCa bootstrap confidence interval.

    All arrays are resampled jointly on the same row indices.

    Args:
        stat_fn: Statistic function; stat_fn(*arrays) must return a scalar float.
        *arrays: One or more arrays to resample jointly. All must share the same
            first-axis length.
        n_boot: Number of bootstrap replicates.
        ci: Coverage level for the confidence interval (e.g. 0.95 for 95%).
        rng: NumPy random generator instance. If None, a default generator is created.

    Returns:
        Tuple (lower, upper) bounds of the ci-level confidence interval.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(arrays[0])
    theta_hat = stat_fn(*arrays)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = stat_fn(*[a[idx] for a in arrays])

    jack = np.empty(n)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        jack[i] = stat_fn(*[a[mask] for a in arrays])
        mask[i] = True

    return _bca_ci(theta_hat, boot, jack, ci)
