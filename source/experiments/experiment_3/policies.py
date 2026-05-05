"""Pure scoring functions for Experiment 3 review-allocation policies.

All scorers return shape (N,) arrays where higher = higher review priority.
Cross-fitting is the caller's responsibility; the fitter helpers
`fit_scoring_models` (P(h|b_x) and P(y|b_x, h)) operate on a single
training fold's arrays.

Policy list:
    score_br_hat       : ex-ante boundary regret E_{h|x}[BR(b_x, b_xh, R)]
    score_residual     : expected KL E_{h|x}[KL(b_xh || b_x)]  (residual expertise)
    score_margin       : -facet_distance(b_x, R)   (close-to-boundary)
    score_entropy      : Shannon entropy of b_x
    score_l2d          : logistic-proxy P(should defer | b_x)
    score_random       : uniform random
    score_oracle       : realized R[a_xh, y] - R[a_x, y]   (hindsight; uses y)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

import core
from utils import logit

_EPS = 1e-9


def _hx_features(b_x: np.ndarray) -> np.ndarray:
    """P(h | b_x) features: [logit(b_x[:, 1])] for binary K=2."""
    if b_x.shape[1] != 2:
        raise ValueError(f"Only binary K=2 supported; got K={b_x.shape[1]}")
    return logit(b_x[:, 1]).reshape(-1, 1)


def _yxh_features(b_x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """P(y | b_x, h) features: [logit(b_x[:, 1]), h, logit(b_x[:, 1]) * h].

    Mirrors data.preparation.fit_augmented_beliefs feature set exactly.
    """
    z = logit(b_x[:, 1])
    h_f = h.astype(float)
    return np.stack([z, h_f, z * h_f], axis=1)


def fit_scoring_models(
    b_x_train: np.ndarray, h_train: np.ndarray, y_train: np.ndarray
) -> tuple[LogisticRegression, LogisticRegression]:
    """Fit P(h | b_x) and P(y | b_x, h) on a training fold.

    Returns (h_model, y_model). Both are sklearn LogisticRegression(max_iter=1000).
    The y_model uses features identical to data.preparation.fit_augmented_beliefs.

    Raises sklearn.exceptions on single-class training data; the upstream
    KFold/StratifiedKFold guard in compute_scores covers globally-constant y,
    but a per-fold constant h is left to surface as a fit-time error rather
    than masked.
    """
    h_model = LogisticRegression(max_iter=1000)
    h_model.fit(_hx_features(b_x_train), h_train)

    y_model = LogisticRegression(max_iter=1000)
    y_model.fit(_yxh_features(b_x_train, h_train), y_train)

    return h_model, y_model


def _aligned_proba(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Predict_proba reshaped to (N, K) with columns indexed by class label.

    Sklearn orders columns by `model.classes_`, which can differ from [0, 1, ...].
    Single-class fits are not handled — sklearn raises before returning such a model.
    """
    proba = model.predict_proba(X)
    K = int(model.classes_.max()) + 1
    out = np.zeros((X.shape[0], K))
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, col]
    return out


def _predict_b_xh_for_h(
    b_x_test: np.ndarray, h_value: int, y_model: LogisticRegression
) -> np.ndarray:
    """Predict (N, 2) augmented belief assuming h = h_value for every row."""
    h_filled = np.full(b_x_test.shape[0], h_value, dtype=int)
    return _aligned_proba(y_model, _yxh_features(b_x_test, h_filled))


def _h_marginal(b_x_test: np.ndarray, h_model: LogisticRegression) -> np.ndarray:
    """Return (N, 2) array of P(h_value | b_x_test) for binary h."""
    return _aligned_proba(h_model, _hx_features(b_x_test))


def score_br_hat(
    b_x: np.ndarray,
    R: np.ndarray,
    h_model: LogisticRegression,
    y_model: LogisticRegression,
) -> np.ndarray:
    """E_{h|x}[BR(b_x, b_xh, R)] = ex-ante boundary regret."""
    p_h = _h_marginal(b_x, h_model)  # (N, 2)
    out = np.zeros(b_x.shape[0])
    for h_val in (0, 1):
        b_xh = _predict_b_xh_for_h(b_x, h_val, y_model)
        out += p_h[:, h_val] * core.boundary_regret(b_x, b_xh, R)
    return out


def score_residual(
    b_x: np.ndarray,
    h_model: LogisticRegression,
    y_model: LogisticRegression,
) -> np.ndarray:
    """E_{h|x}[KL(b_xh || b_x)] -- residual expertise (mutual info I(h; y | x))."""
    p_h = _h_marginal(b_x, h_model)  # (N, 2)
    out = np.zeros(b_x.shape[0])
    log_b_x = np.log(np.clip(b_x, _EPS, 1.0))
    for h_val in (0, 1):
        b_xh = _predict_b_xh_for_h(b_x, h_val, y_model)
        log_b_xh = np.log(np.clip(b_xh, _EPS, 1.0))
        kl = (b_xh * (log_b_xh - log_b_x)).sum(axis=1)  # (N,)
        out += p_h[:, h_val] * kl
    return out


def score_margin(b_x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Higher score = closer to the reward-induced facet."""
    return -core.facet_distance(b_x, R)


def score_entropy(b_x: np.ndarray) -> np.ndarray:
    """Shannon entropy of b_x. Reward-blind."""
    return -(b_x * np.log(np.clip(b_x, _EPS, 1.0))).sum(axis=1)


def score_l2d(
    b_x_train: np.ndarray,
    y_train: np.ndarray,
    b_xh_train: np.ndarray,
    R: np.ndarray,
    b_x_test: np.ndarray | None = None,
) -> np.ndarray:
    """Logistic-proxy learning-to-defer: predict P(should defer | b_x).

    Deferral label uses realized augmented beliefs: defer_i = 1[ R[a_xh_i, y_i] > R[a_x_i, y_i] ]
    where a_x = argmax over b_x_train, a_xh = argmax over b_xh_train.
    Second-stage classifier maps logit(b_x) -> defer.
    """
    if b_x_test is None:
        b_x_test = b_x_train

    a_x = core.model_action(b_x_train, R)
    a_xh = core.model_action(b_xh_train, R)
    defer = (R[a_xh, y_train] > R[a_x, y_train]).astype(int)

    if defer.max() == defer.min():
        return np.full(b_x_test.shape[0], float(defer[0]))

    clf = LogisticRegression(max_iter=1000)
    clf.fit(_hx_features(b_x_train), defer)
    return _aligned_proba(clf, _hx_features(b_x_test))[:, 1]


def score_random(b_x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random scores."""
    return rng.random(b_x.shape[0])


def score_oracle(
    b_x: np.ndarray, b_xh: np.ndarray, y: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Hindsight realised gain: R[a_xh, y] - R[a_x, y]. Uses y; not deployable."""
    a_x = core.model_action(b_x, R)
    a_xh = core.model_action(b_xh, R)
    return R[a_xh, y] - R[a_x, y]
